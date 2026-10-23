from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, DiffusionPipeline, StableDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModel


def _scalar_schedule(value, epoch, global_step):
    del epoch, global_step
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f'Expected a scalar schedule value, got {type(value).__name__}.')


def merge_dW_to_unet(pipe, dW_dict, alpha=1.0):
    state = pipe.unet.state_dict()
    for key, delta in dW_dict.items():
        state[key] += delta * alpha
    pipe.unet.load_state_dict(state, strict=False)
    return pipe


def load_model_delta(pipe_rf, tuned_model, base_model, alpha=1.0):
    base_pipe = DiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, safety_checker=None
    )
    base_state = base_pipe.unet.state_dict()

    tuned_pipe = DiffusionPipeline.from_pretrained(
        tuned_model, torch_dtype=torch.float16, safety_checker=None
    )
    tuned_state = tuned_pipe.unet.state_dict()
    delta = {key: tuned_state[key] - base_state[key] for key in tuned_state}
    merge_dW_to_unet(pipe_rf, delta, alpha=alpha)
    pipe_rf.vae = tuned_pipe.vae
    pipe_rf.text_encoder = tuned_pipe.text_encoder
    return delta


class RFDSLoss(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.weights_dtype = torch.float16

        rf_model = args.rf_model_path
        pipe_kwargs = {
            'tokenizer': None,
            'safety_checker': None,
            'feature_extractor': None,
            'requires_safety_checker': False,
            'torch_dtype': self.weights_dtype,
        }

        @dataclass
        class SubModules:
            pipe: StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(rf_model, **pipe_kwargs)
        if args.tuned_model_path:
            load_model_delta(
                pipe,
                tuned_model=args.tuned_model_path,
                base_model=args.base_sd_model_path,
                alpha=args.model_delta_alpha,
            )
        pipe = pipe.to(self.device)
        self.submodules = SubModules(pipe=pipe)

        self.tokenizer = AutoTokenizer.from_pretrained(rf_model, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(
            rf_model, subfolder='text_encoder'
        ).to(self.device)
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)
        for parameter in self.unet.parameters():
            parameter.requires_grad_(False)

        self.scheduler = DDPMScheduler.from_pretrained(
            rf_model, subfolder='scheduler', torch_dtype=self.weights_dtype
        )
        self.scheduler_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.scheduler = self.scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.text_embeddings = None
        self.text_plus_uncond_embeddings = None
        self.set_min_max_steps()

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    def embed_text(self, description):
        text_input = self.tokenizer(
            description, padding='max_length', max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors='pt'
        )
        uncond_input = self.tokenizer(
            [''], padding='max_length', max_length=text_input.input_ids.shape[-1], return_tensors='pt'
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        self.text_embeddings = text_embeddings.repeat_interleave(self.args.batch_size, 0)
        self.text_plus_uncond_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        self.text_plus_uncond_embeddings = self.text_plus_uncond_embeddings.repeat_interleave(
            self.args.batch_size, 0
        )
        torch.cuda.empty_cache()
        return self.text_plus_uncond_embeddings

    def forward_unet(self, unet, latents, t, encoder_hidden_states):
        with torch.cuda.amp.autocast():
            input_dtype = latents.dtype
            return unet(
                latents.to(self.weights_dtype), t.to(self.weights_dtype),
                encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            ).sample.to(input_dtype)

    def compute_grad_rfds_rev(self, latents, text_embeddings_vd):
        batch_size = latents.shape[0]
        with torch.no_grad():
            t = torch.randint(
                self.min_step, self.max_step + 1, [batch_size], dtype=torch.long, device=self.device
            )
            noise = torch.randn_like(latents)
            tau = t[:, None, None, None] / self.num_train_timesteps
            latents_noisy = tau * latents + (1 - tau) * noise
            velocity = self.forward_unet(
                self.unet, latents_noisy, t, encoder_hidden_states=text_embeddings_vd[:batch_size]
            )
            noise = noise + (latents - noise - velocity)
            latents_noisy = tau * latents + (1 - tau) * noise
            velocity = self.forward_unet(
                self.unet, torch.cat([latents_noisy] * 2), torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings_vd
            )
        velocity_text, velocity_null = velocity.chunk(2)
        guided = velocity_null + self.args.diffusion_guidance_scale * (velocity_text - velocity_null)
        return guided, noise

    def encode_images(self, imgs):
        with torch.cuda.amp.autocast():
            input_dtype = imgs.dtype
            imgs = imgs * 2.0 - 1.0
            posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
            latents = posterior.sample() * self.vae.config.scaling_factor
            return latents.to(input_dtype)

    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        num_steps = self.scheduler.config.num_train_timesteps
        self.min_step = int(num_steps * min_step_percent)
        self.max_step = int(num_steps * max_step_percent)

    def update_step(self, epoch, global_step, on_load_weights=False):
        del on_load_weights
        self.set_min_max_steps(
            min_step_percent=_scalar_schedule(self.args.min_step_percent, epoch, global_step),
            max_step_percent=_scalar_schedule(self.args.max_step_percent, epoch, global_step),
        )

    def get_loss(self, x, description):
        x = x * 2.0 - 1.0
        latent_z = self.encode_images(x)
        text_embeddings = self.embed_text(description)
        grad, _ = self.compute_grad_rfds_rev(latent_z, text_embeddings)
        grad_z = torch.nan_to_num(grad)
        loss = (grad_z.clone() * latent_z).sum(1).mean()
        del grad_z
        return loss
