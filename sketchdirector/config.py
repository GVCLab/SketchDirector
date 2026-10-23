import argparse
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def build_parser():
    parser = argparse.ArgumentParser(description='SketchDirector multi-concept scene sketching')

    # Inputs / outputs
    parser.add_argument('--concept_to_draw', type=str, default='cat')
    parser.add_argument('--concept_file', type=str, default=None, help='One scene prompt per line')
    parser.add_argument('--background_path', type=str, default=None)
    parser.add_argument('--path2save', type=str, default='results')
    parser.add_argument('--output_dir', type=str, default='output_sketches')
    parser.add_argument('--memory_dir', type=str, default='memory')

    # Model API
    parser.add_argument('--model', type=str, default='gpt-5.5')
    parser.add_argument('--openai_base_url', type=str, default=None)
    parser.add_argument('--gen_mode', type=str, default='generation', choices=['generation', 'completion'])
    parser.add_argument('--seed_mode', type=str, default='deterministic', choices=['deterministic', 'stochastic'])

    # Grid / vector generation
    parser.add_argument('--res', type=int, default=64)
    parser.add_argument('--cell_size', type=int, default=12)
    parser.add_argument('--stroke_width', type=float, default=7.0)
    parser.add_argument('--num_strokes', type=int, default=100)
    parser.add_argument('--width', type=float, default=2.5)
    parser.add_argument('--control_points_per_seg', type=int, default=4)
    parser.add_argument('--num_segments', type=int, default=1)

    # Runtime
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render_size', type=int, default=512)
    parser.add_argument('--output_svg_size', type=int, default=512)
    parser.add_argument('--save_interval', type=int, default=100)

    # Refinement
    parser.add_argument('--skip_refinement', action='store_true')
    parser.add_argument('--refine_steps', type=int, default=200)
    parser.add_argument('--grad_accum_steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--diffusion_guidance_scale', type=float, default=100.0)
    parser.add_argument('--min_step_percent', type=float, default=0.02)
    parser.add_argument('--max_step_percent', type=float, default=0.98)
    parser.add_argument('--rf_model_path', type=str, default=os.getenv('SKETCHDIRECTOR_RF_MODEL_PATH', './2_rectified_flow_from_sd_1_5'))
    parser.add_argument('--base_sd_model_path', type=str, default=os.getenv('SKETCHDIRECTOR_BASE_SD_MODEL_PATH', './stable-diffusion-v1-5'))
    parser.add_argument('--tuned_model_path', type=str, default=os.getenv('SKETCHDIRECTOR_TUNED_MODEL_PATH', 'yykkL123/simple_style_sd1.5'))
    parser.add_argument('--model_delta_alpha', type=float, default=1.0)
    parser.add_argument('--skip_video', action='store_true')

    parser.add_argument('--num_iter', type=int, default=2000)
    parser.add_argument('--object_size_ratio', type=float, default=0.75)
    parser.add_argument('--use_init_method', type=int, default=1)
    parser.add_argument('--object_name', type=str, default='')
    parser.add_argument('--attn_model', type=str, default='diffusion', choices=['diffusion', 'clip'])
    parser.add_argument('--fix_scale', type=int, default=0)
    parser.add_argument('--use_wandb', type=int, default=0)
    parser.add_argument('--wandb_user', type=str, default='')
    parser.add_argument('--wandb_name', type=str, default='default')
    parser.add_argument('--wandb_project_name', type=str, default='')

    parser.add_argument('--depth_threshold', type=float, default=0.0)
    parser.add_argument('--mask_dilation', type=int, default=10)
    parser.add_argument('--sample_radius_ratio', type=float, default=0.01)
    parser.add_argument('--sample_point_ratio', type=float, default=0.05)
    parser.add_argument('--depth_encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--depth_model_path', type=str, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--target_is_dict', type=int, default=0)
    parser.add_argument('--save_svg_in_dict', type=int, default=1)
    parser.add_argument('--diffusion_model', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--diffusion_timesteps', type=int, default=1000)
    parser.add_argument('--caption', type=str, default='')
    parser.add_argument('--conditioning_scale', type=float, default=0.15)
    parser.add_argument('--condition', type=str, default='depth', choices=['depth','hed','scribble','seg','canny','normal'])
    return parser


def parse_arguments(argv=None):
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    args.caption = args.concept_to_draw
    args.grid_size = (args.res + 1) * args.cell_size
    args.device = torch.device('cpu' if args.use_cpu or not torch.cuda.is_available() else 'cuda')

    # pydiffvg is needed only by structure-aware refinement. Keep it optional for --skip_refinement.
    try:
        import pydiffvg
        pydiffvg.set_use_gpu(args.device.type == 'cuda')
        pydiffvg.set_device(args.device)
    except ImportError:
        if not args.skip_refinement:
            print('Warning: pydiffvg is not installed; refinement will not be available.')
    return args
