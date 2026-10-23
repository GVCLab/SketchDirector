import os
import xml.etree.ElementTree as ET

import pydiffvg
import torch


class Painter(torch.nn.Module):
    def __init__(self, args, num_strokes=4, num_segments=4, device=None):
        super().__init__()
        self.args = args
        self.render_size = args.render_size
        self.num_paths = num_strokes
        self.num_segments = num_segments
        self.width = args.width
        self.control_points_per_seg = args.control_points_per_seg
        self.device = device
        self.canvas_width = args.render_size
        self.canvas_height = args.render_size
        self.shapes = []
        self.shape_groups = []
        self.points_vars = []
        self.optimize_flag = []

    def init_image(self, svg_path):

        canvas_width, canvas_height, shapes, shape_groups = \
            pydiffvg.svg_to_scene(svg_path)

        print("Original SVG size:", canvas_width, canvas_height)

        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()

            xml_paths = root.findall(
                './/{http://www.w3.org/2000/svg}path'
            )

            if not xml_paths:
                xml_paths = root.findall('.//path')

            for idx, (shape, path_elem) in enumerate(
                    zip(shapes, xml_paths)
            ):
                shape.semantic_class = path_elem.get(
                    'data-class',
                    f'stroke_part_{idx}'
                )

        except Exception as exc:
            print(
                f"Warning: could not recover SVG semantic labels: {exc}"
            )


        target_size = 512.0

        scale_x = target_size / float(canvas_width)
        scale_y = target_size / float(canvas_height)


        print(
            f"Scale x={scale_x:.6f}, y={scale_y:.6f}"
        )


        for shape in shapes:

            if hasattr(shape, "points"):

                # x coordinate
                shape.points[:, 0] *= scale_x

                # y coordinate
                shape.points[:, 1] *= scale_y


            if hasattr(shape, "stroke_width"):

                if torch.is_tensor(shape.stroke_width):

                    # use average scale for stroke
                    shape.stroke_width *= (
                        scale_x + scale_y
                    ) / 2.0


        self.canvas_width = int(target_size)
        self.canvas_height = int(target_size)


        self.shapes = shapes
        self.shape_groups = shape_groups

        self.optimize_flag = [
            True
        ] * len(self.shapes)


        print(
            "Normalized renderer canvas:",
            self.canvas_width,
            self.canvas_height
        )


        return self.get_image()

    def render_warp(self):
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups
        )
        return pydiffvg.RenderFunction.apply(
            self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args
        )

    def get_image(self):
        img = self.render_warp()
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=self.device
        ) * (1 - opacity)
        return img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

    def parameters(self, recurse=True):
        del recurse
        self.points_vars = []
        for should_optimize, path in zip(self.optimize_flag, self.shapes):
            if should_optimize and hasattr(path, 'points'):
                path.points.requires_grad_(True)
                self.points_vars.append(path.points)
        return self.points_vars

    def save_svg(self, output_dir, name, object_id):
        os.makedirs(output_dir, exist_ok=True)
        svg_path = os.path.join(output_dir, f"{name}.svg")

        default_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
        for group in self.shape_groups:
            if group.stroke_color is None:
                group.stroke_color = default_color.clone()

        for shape in self.shapes:
            if hasattr(shape, 'stroke_width'):
                width = float(shape.stroke_width.detach().cpu()) if torch.is_tensor(shape.stroke_width) else float(shape.stroke_width)
                if width < 1.5:
                    shape.stroke_width = torch.tensor(1.5, dtype=torch.float32)

        pydiffvg.save_svg(svg_path, self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)

        tree = ET.parse(svg_path)
        root = tree.getroot()
        groups = root.findall('.//{http://www.w3.org/2000/svg}g') or root.findall('.//g')
        if groups:
            groups[0].set('id', str(object_id))

        paths = root.findall('.//{http://www.w3.org/2000/svg}path') or root.findall('.//path')
        for i, (path_elem, shape) in enumerate(zip(paths, self.shapes)):
            path_elem.set('data-class', str(getattr(shape, 'semantic_class', f'stroke_part_{i}')))
        tree.write(svg_path, encoding='utf-8', xml_declaration=True)


class PainterOptimizerProgress:
    def __init__(self, args, renderer):
        self.renderer = renderer
        self.points_lr = args.lr
        self.points_optim = None

    def init_optimizers(self):
        self.points_optim = torch.optim.Adam(
            self.renderer.parameters(), lr=self.points_lr, betas=(0.9, 0.9), eps=1e-6
        )

    def zero_grad_(self):
        if self.points_optim is not None:
            self.points_optim.zero_grad()

    def step_(self):
        if self.points_optim is not None:
            self.points_optim.step()
