import os
from typing import Any, Dict

import torch
from PIL import Image

from . import refine_utils
from .vector_optimizer import run_progress_vector_optimization


class RefinementOptimizer:
    def __init__(self, args, sketch_app):
        self.args = args
        self.sketch_app = sketch_app
        self.output_dir = self.args.output_dir
        self.refine_history = []
        self.global_optimize_batch = []

        os.makedirs(self.output_dir, exist_ok=True)

    def run_optimize_progress(self, results: Dict[str, Any]) -> Dict[str, Any]:
        refine_layout_history = []
        object_states = results.get("object_states", {})

        current_canvas_path = f"{self.args.output_dir}/canvas_step_0.png"
        current_svg_path = f"{self.args.output_dir}/canvas_step_0.svg"

        Image.new("RGBA", (512, 512), (255, 255, 255, 255)).save(
            current_canvas_path, "PNG"
        )
        with open(current_svg_path, "w", encoding="utf-8") as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>'
                '<svg xmlns="http://www.w3.org/2000/svg" '
                'width="512" height="512" viewBox="0 0 512 512"></svg>'
            )

        objects_history = []
        for obj_id, state in object_states.items():
            objects_history.append(
                {
                    "object_id": obj_id,
                    "description": state.get("description", ""),
                    "incremental_prompt": state.get("incremental_prompt", ""),
                    "svg_path": state.get("svg_path", ""),
                    "bg_svg_path": state.get("bg_svg_path", ""),
                    "visual_complexity": state.get("visual_complexity", 1.0),
                    "depth": state.get("depth", 1),
                    "x": state.get("x", 0),
                    "y": state.get("y", 0),
                    "scale": state.get("scale", 1.0),
                    "rotation": state.get("rotation", 0.0),
                    "raw_w": state.get("raw_w", 100),
                    "raw_h": state.get("raw_h", 100),
                }
            )
            print(
                f"object_id:{obj_id},x:{state.get('x', 0)}, y:{state.get('y', 0)}"
            )

        print(
            f"\n[Refinement] Starting diffusion refinement for "
            f"{len(objects_history)} objects."
        )

        layout_history = []
        cleaned_svg_dir = os.path.join(
            os.path.dirname(self.args.output_dir),
            "cleaned_objects",
        )
        os.makedirs(cleaned_svg_dir, exist_ok=True)

        for step_idx, target_object in enumerate(objects_history):
            obj_id = target_object.get("object_id")
            print(
                f"\n[Refinement] Processing object "
                f"{step_idx + 1}/{len(objects_history)}: ID={obj_id}"
            )

            layout_meta = {
                "x": target_object["x"] / 780.0,
                "y": target_object["y"] / 780.0,
                "scale": target_object["scale"],
                "rotation": target_object.get("rotation",0.0)
            }
            print(
                f"object_id:{obj_id},layout_meta:{layout_meta}"
            )

            complexity = target_object.get("visual_complexity", 1.0)
            current_num_strokes = max(4, min(int(60 * complexity), 1024))

            raw_initial_svg = target_object.get("svg_path", "")
            raw_bg_svg = target_object.get("bg_svg_path", "")
            resampled_initial_svg_path = os.path.join(
                self.args.output_dir,
                f"global_{obj_id}_resampled.svg",
            )

            refine_utils.resample_svg_to_fixed_cubic_beziers(
                input_svg_path=raw_initial_svg,
                output_svg_path=resampled_initial_svg_path,
                num_target_strokes=current_num_strokes,
            )

            white_mask_path = os.path.join(
                self.args.output_dir,
                f"{obj_id}_white_mask.png",
            )

            refine_utils.extract_white_box_mask(
                bg_svg_path=raw_bg_svg,
                output_path=white_mask_path,
            )

            optimized_svg_path, point_var = run_progress_vector_optimization(
                args=self.args,
                svg_path=resampled_initial_svg_path,
                description=target_object.get("description", ""),
                incremental_prompt=target_object.get("incremental_prompt", ""),
                bg_path=current_canvas_path,
                round_idx=step_idx + 1,
                object_id=obj_id,
                layout_meta=layout_meta,
                white_mask_path=white_mask_path,
            )

            filter_results = refine_utils.filter_short_curves(
                svg_path=optimized_svg_path,
                length_threshold=15.0,
                debug=True,
            )

            optimized_svg_path = filter_results["filtered_svg"]
            point_var = filter_results["points_var"]
            point_var_clean = torch.from_numpy(point_var).float()

            print(
                f"[Refinement] Filtered curves saved. "
                f"New point count: {len(point_var)}"
            )

            bg_svg_path = os.path.join(cleaned_svg_dir, f"{obj_id}_with_bg.svg")
            refine_utils.inject_pure_vector_white_box(
                svg_path=optimized_svg_path,
                output_svg_path=bg_svg_path,
                padding=15.0,
            )

            print(f"[Refinement] White background SVG generated: {bg_svg_path}")

            current_canvas_path, current_svg_path = (
                self.sketch_app.overlay_object_on_canvas(
                    bg_svg_path,
                    current_canvas_path,
                    current_svg_path,
                    layout_meta,
                    output_png_path=os.path.join(
                        self.args.output_dir,
                        f"canvas_step_{step_idx + 1}.png",
                    ),
                    output_svg_path=os.path.join(
                        self.args.output_dir,
                        f"canvas_step_{step_idx + 1}.svg",
                    ),
                )
            )

            layout_history.append(
                {
                    "object_id": target_object["object_id"],
                    "x": layout_meta["x"],
                    "y": layout_meta["y"],
                    "scale": layout_meta["scale"],
                    "rotation": layout_meta.get("rotation", 0.0),
                }
            )

            print(
                f"[Refinement] Added '{target_object['object_id']}' "
                f"to layout history. Current count: {len(layout_history)}"
            )
            print(f"[Refinement] Updated canvas PNG: {current_canvas_path}")
            print(f"[Refinement] Updated canvas SVG: {current_svg_path}")

        print(
            f"\n[Refinement] Progressive generation completed. "
            f"Final canvas saved at: {current_canvas_path}"
        )

    def filter_result(self, optimized_results, layout_history):
        for obj_id, res in optimized_results.items():
            filter_results = refine_utils.filter_short_curves(
                svg_path=res["final_svg_path"],
                length_threshold=15.0,
                debug=False,
            )

            refine_bg_svg_path = os.path.join(
                self.args.path2save,
                f"bg_refine_{obj_id}_final.svg",
            )
            refine_utils.inject_pure_vector_white_box(
                svg_path=filter_results["filtered_svg"],
                output_svg_path=refine_bg_svg_path,
                padding=15.0,
            )

            for obj in layout_history:
                if obj.get("object_id") == obj_id:
                    obj["svg_path"] = filter_results["filtered_svg"]
                    obj["bg_svg_path"] = refine_bg_svg_path
                    break

        return layout_history