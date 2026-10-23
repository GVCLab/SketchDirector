import os
import shutil
import subprocess

import pydiffvg
import torch


def read_svg(path_svg, device, multiply=False, args=None):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(path_svg)
    if args is not None and (hasattr(args, 'scale_w') or hasattr(args, 'scale_h')):
        w, h = args.scale_w, args.scale_h
        for path in shapes:
            path.points = path.points / canvas_width
            path.points = 2 * path.points - 1
            path.points[:, 0] /= w
            path.points[:, 1] /= h
            path.points = 0.5 * (path.points + 1.0) * canvas_width
            center_x, center_y = canvas_width / 2, canvas_height / 2
            path.points[:, 0] += args.original_center_x * canvas_width - center_x
            path.points[:, 1] += args.original_center_y * canvas_height - center_y

    if multiply:
        canvas_width *= 2
        canvas_height *= 2
        for path in shapes:
            path.points *= 2
            path.stroke_width *= 2

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    img = pydiffvg.RenderFunction.apply(
        canvas_width, canvas_height, 2, 2, 0, None, *scene_args
    )
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
        img.shape[0], img.shape[1], 3, device=device
    ) * (1 - img[:, :, 3:4])
    return img[:, :, :3]


def increase_object_size(renderer, scale_w, scale_h, original_center_x, original_center_y):
    with torch.no_grad():
        canvas_width = canvas_height = 512
        for path in renderer.shapes:
            path.points = path.points / canvas_width
            path.points = 2 * path.points - 1
            path.points[:, 0] /= scale_w
            path.points[:, 1] /= scale_h
            path.points = 0.5 * (path.points + 1.0) * canvas_width
            center_x, center_y = canvas_width / 2, canvas_height / 2
            path.points[:, 0] += original_center_x * canvas_width - center_x
            path.points[:, 1] += original_center_y * canvas_height - center_y


def resize_svg(renderer, target_width, target_height):
    width_scale = target_width / 512
    height_scale = target_height / 512
    renderer.canvas_width = target_width
    renderer.canvas_height = target_height
    with torch.no_grad():
        for shape in renderer.shapes:
            shape.points[:, 0] *= width_scale
            shape.points[:, 1] *= height_scale
            shape.stroke_width = torch.tensor(float(shape.stroke_width) * width_scale)
    return renderer


def make_video(cur_path):
    if shutil.which('ffmpeg') is None:
        print('ffmpeg not found; skipping optimization video.')
        return False
    output = os.path.join(cur_path, 'sketch.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '10', '-pattern_type', 'glob', '-i',
        f'{cur_path}/svg_to_png/iter_*.png', '-vb', '20M', '-vf', 'scale=224:224', output
    ], check=False)
    return os.path.exists(output)
