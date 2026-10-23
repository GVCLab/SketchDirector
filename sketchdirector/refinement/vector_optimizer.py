import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from . import refine_utils
from . import sketch_utils as utils
from .rfds_loss import RFDSLoss
from .painter_progressive import (
    Painter,
    PainterOptimizerProgress,
)
def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_strokes, args=args, num_segments=args.num_segments, device=args.device)
    renderer = renderer.to(args.device)
    return renderer

def get_target(args):
    if args.target_is_dict:
        if os.path.splitext(os.path.basename(args.target))[-1] == '.npy':
            input_dict = np.load(args.target, allow_pickle='TRUE').item()
        else:
            input_dict = utils.load_compressed_npz(args.target)
        target = input_dict['image']
        target = target.resize((args.render_size, args.render_size))
        if 'mask' in input_dict:
            mask = input_dict['mask']
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (args.render_size, args.render_size))[0][0]
        else:
            mask = utils.get_mask(target, args.device)
        if not args.caption and 'caption' in input_dict:
            args.caption = input_dict['caption']
        target = utils.create_masked_image(target, mask)
        if 'attn_map' in input_dict:
            attn = input_dict['attn_map']
            attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), (args.render_size, args.render_size))[0][0]
            args.attn_from_dict = attn
    else:
        target = Image.open(args.target)
        if target.mode == 'RGBA':
            new_image = Image.new('RGBA', target.size, 'WHITE')
            new_image.paste(target, (0, 0), target)
            target = new_image
        target = target.convert('RGB')
        mask = utils.get_mask(target, args.device)
        target = utils.create_masked_image(target, mask)
        if args.fix_scale:
            target, mask = utils.fix_image_mask_scale(target, mask)
        target = target.resize((args.render_size, args.render_size))
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (args.render_size, args.render_size))[0][0]
    im_np = np.array(target)
    test_mask = mask.clone()
    test_mask[test_mask < 0.5] = 0
    test_mask[test_mask >= 0.5] = 1
    w, h = (target.size[0], target.size[1])
    x0, x1, y0, y1 = utils.get_obj_bb(test_mask)
    im_width, im_height = (x1 - x0, y1 - y0)
    max_size = max(im_width, im_height)
    target_size = int(args.render_size * args.object_size_ratio)
    if max_size > target_size:
        if im_width > im_height:
            new_width, new_height = (target_size, int(target_size / im_width * im_height))
        else:
            new_width, new_height = (int(target_size / im_height * im_width), target_size)
        mask_np3 = np.stack([test_mask] * 3, axis=-1)
        mask = utils.cut_and_resize(mask_np3, x0, x1, y0, y1, new_height, new_width, 'mask')
        mask = torch.from_numpy(mask[:, :, 0])
        target_np = im_np / im_np.max()
        im_np = utils.cut_and_resize(target_np, x0, x1, y0, y1, new_height, new_width, 'image')
        im_np_final = (im_np / im_np.max() * 255).astype(np.uint8)
        target = Image.fromarray(im_np_final)
        args.new_width = new_width
        args.new_height = new_height
        args.obj_bb = (x0, x1, y0, y1)
        args.original_center_y = (y0 + (y1 - y0) / 2) / h
        args.original_center_x = (x0 + (x1 - x0) / 2) / w
        args.scale_w = new_width / im_width
        args.scale_h = new_height / im_height
    args.input_image = target
    args.mask = mask
    target.save(f'{args.output_dir}/input.png')
    if args.use_wandb:
        wandb.log({'input': wandb.Image(target)})
    utils.save_mask(mask, args.output_dir, args.use_wandb)
    data_transform = transforms.ToTensor()
    target = data_transform(target).unsqueeze(0).to(args.device)
    return (target, mask)

def load_mask(mask_path, device=None):
    import torchvision.transforms as transforms
    mask = Image.open(mask_path).convert('L')
    mask = transforms.ToTensor()(mask)
    mask = mask.unsqueeze(0)
    if device is not None:
        mask = mask.to(device)
    return mask

def run_progress_vector_optimization(args, svg_path, description, incremental_prompt, bg_path, round_idx, object_id, layout_meta, white_mask_path):
    print('Starting object-level vector optimization...', flush=True)
    args.output_dir = os.path.join(args.output_dir, f'round_{round_idx:02}_{object_id}')
    white_mask = load_mask(white_mask_path)
    if white_mask.dim() == 2:
        white_mask = white_mask.unsqueeze(0).unsqueeze(0)
    elif white_mask.dim() == 3:
        white_mask = white_mask.unsqueeze(0)
    white_mask = white_mask.float().to(args.device)
    jpg_log_dir = os.path.join(args.output_dir, 'jpg_logs')
    svg_log_dir = os.path.join(args.output_dir, 'svg_logs')
    svg_png_dir = os.path.join(args.output_dir, 'svg_to_png')
    bg_combined_dir = os.path.join(args.output_dir, 'sketches_with_bg')
    os.makedirs(jpg_log_dir, exist_ok=True)
    os.makedirs(svg_log_dir, exist_ok=True)
    os.makedirs(svg_png_dir, exist_ok=True)
    os.makedirs(bg_combined_dir, exist_ok=True)
    txt_save_path = os.path.join(args.output_dir, 'prompts_log.txt')
    with open(txt_save_path, 'w', encoding='utf-8') as f:
        f.write(f'==================================================\n')
        f.write(f'ROUND: {round_idx} | OBJECT ID: {object_id}\n')
        f.write(f'==================================================\n\n')
        f.write(f'[1] Isolated Object Description (For loss_sketch):\n')
        f.write(f'{description}\n\n')
        f.write(f'--------------------------------------------------\n\n')
        f.write(f'[2] Incremental Context Prompt (For loss_bg):\n')
        f.write(f'{incremental_prompt}\n\n')
        f.write(f'--------------------------------------------------\n')
        f.write(f'Background Canvas Source: {bg_path}\n')
    bg_pil = Image.open(bg_path).convert('RGB').resize((512, 512))
    transform = transforms.ToTensor()
    bg_img = transform(bg_pil).to(args.device)
    print(
        f"[Refine] Keep original object position. "
        f"Layout={layout_meta}"
    )
    
    renderer = load_renderer(args)
    rfds_loss = RFDSLoss(args, args.device)
    optimizer = PainterOptimizerProgress(args, renderer)
    img = renderer.init_image(svg_path)
    print("renderer image:", img.shape)
    optimizer.init_optimizers()
    print('Starting optimization...', flush=True)
    epoch_range = tqdm(range(args.num_iter + 1))
    n_accumulation_steps = 2
    num_steps = 200
    for epoch in tqdm(range(num_steps * n_accumulation_steps + 1)):
        sketches = renderer.get_image().to(args.device)
        sketches_transformed = sketches
        white_mask_transformed = torch.nn.functional.interpolate(
            white_mask,
            size=(512,512),
            mode="nearest"
        )
        canvas_after_white = bg_img.unsqueeze(0) * (1 - white_mask_transformed) + torch.ones_like(bg_img.unsqueeze(0)) * white_mask_transformed
        alpha = 1.0 - sketches_transformed.mean(dim=1, keepdim=True)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        sketches_bg = sketches_transformed * alpha + canvas_after_white * (1 - alpha)
        point_var = renderer.parameters()
        loss_sketch = rfds_loss.get_loss(sketches, description)
        loss_bg = rfds_loss.get_loss(sketches_bg, incremental_prompt)
        loss = 0.5 * loss_sketch + 0.5 * loss_bg
        loss.backward()
        if (epoch + 1) % n_accumulation_steps == 0:
            actual_step = (epoch + 1) // n_accumulation_steps
            rfds_loss.update_step(epoch=0, global_step=actual_step)
            optimizer.step_()
            optimizer.zero_grad_()
            if actual_step % args.save_interval == 0:
                renderer.save_svg(f'{args.output_dir}/svg_logs', f'svg_iter{epoch}', object_id)
                if not os.path.exists(f'{args.output_dir}/svg_to_png'):
                    os.mkdir(f'{args.output_dir}/svg_to_png')
                path_svg = f'{args.output_dir}/svg_logs/svg_iter{epoch}.svg'
                sketch_iter = utils.read_svg(path_svg, args.device, multiply=True, args=args).cpu().numpy()
                sketch_iter = Image.fromarray((sketch_iter * 255).astype('uint8'), 'RGB')
                sketch_iter.save('{0}/{1}/iter_{2:04}.png'.format(args.output_dir, 'svg_to_png', int(epoch)))
                bg_combined_np = sketches_bg.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
                bg_combined_np = np.clip(bg_combined_np, 0.0, 1.0)
                bg_combined_pil = Image.fromarray((bg_combined_np * 255).astype('uint8'), 'RGB')
                bg_combined_path = os.path.join(bg_combined_dir, f'combined_iter_{epoch:04}.png')
                bg_combined_pil.save(bg_combined_path)
                if args.use_wandb:
                    sketch_array = np.array(sketch_iter)
                    wandb.log({'cur_sketch': wandb.Image(sketch_array)}, step=epoch)
            if epoch == 0 and args.use_init_method:
                utils.plot_initial_points(renderer.get_attn_map_to_plot(), renderer.get_clustered_mask(), inputs, renderer.get_inds(), args.use_wandb, '{}/{}.jpg'.format(args.output_dir, 'initial_points'))
    if hasattr(args, 'scale_w') or hasattr(args, 'scale_h'):
        utils.increase_object_size(renderer, args.scale_w, args.scale_h, args.original_center_x, args.original_center_y)
    if args.output_svg_size != 512:
        utils.resize_svg(renderer, args.output_svg_size, args.output_svg_size)
    print(layout_meta)
    print(renderer.parameters())
    renderer.save_svg(args.output_dir, 'final_svg', object_id)
    final_svg_path = f'{args.output_dir}/final_svg.svg'
    final_sketch_num = utils.read_svg(f'{args.output_dir}/final_svg.svg', args.device, multiply=True, args=None).cpu().numpy()
    final_sketch = Image.fromarray((final_sketch_num * 255).astype('uint8'), 'RGB')
    final_sketch.save(f'{args.output_dir}/final_sketch.png')
    print(f'Final sketch saved to: {args.output_dir}/final_sketch.png')
    if args.use_wandb:
        final_sketch = np.array(final_sketch)
        wandb.log({f'final sketch': wandb.Image(final_sketch)})
    utils.make_video(args.output_dir)
    if args.use_wandb:
        video_path = f'{args.output_dir}/sketch.mp4'
        wandb.log({'sketch_video': wandb.Video(video_path, format='mp4')})
    return (final_svg_path, point_var)

def save_global_canvas(global_canvas, save_dir, epoch):
    canvas_np = global_canvas.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    canvas_pil = Image.fromarray((np.clip(canvas_np, 0.0, 1.0) * 255).astype('uint8'), 'RGB')
    canvas_pil.save(os.path.join(save_dir, f'global_canvas_iter_{epoch:04}.png'))

def build_layout_history(objects_list, epoch, renderers):
    layout_history = []
    for obj_data in objects_list:
        obj_id = obj_data['object_id']
        meta = obj_data['layout_meta']
        renderer = renderers[obj_id]

        obj_svg_dir = os.path.join('temp_svg', obj_id)
        os.makedirs(obj_svg_dir, exist_ok=True)

        svg_path = os.path.join(obj_svg_dir, f'iter_{epoch}.svg')
        renderer.save_svg(obj_svg_dir, f'iter_{epoch}', obj_id)

        refine_bg_svg_path = os.path.join( 'temp_svg', f'bg_refine_{obj_id}.svg')
        refine_utils.inject_pure_vector_white_box(
            svg_path=svg_path,
            output_svg_path=refine_bg_svg_path,
            padding=15.0
        )

        layout_history.append(
            {
                'object_id': obj_id,
                'svg_path': svg_path,
                'bg_svg_path': refine_bg_svg_path,
                'x': meta['x'],
                'y': meta['y'],
                'scale': meta['scale'],
                'depth': obj_data['depth'],
                'rotation': meta['rotation']
            }
        )

    return layout_history