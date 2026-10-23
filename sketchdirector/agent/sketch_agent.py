import base64
import copy
import io
import json
import os
import re
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cairosvg
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from .. import pipeline_utils as utils
from ..refinement import refine_utils
from ..prompts import system_prompt

from .memory import ExperienceMemory


class SketchAgent:
    def __init__(self, sketch_app, args, memory: ExperienceMemory):
        self.sketch_app = sketch_app
        self.args = args
        self.memory = memory
        
        self.res = args.res
        self.cell_size = args.cell_size
        self.grid_size = args.grid_size
        self.canvas_size = args.grid_size
        
        self.cells_to_pixels_map = utils.cells_to_pixels(
            self.res, 
            self.cell_size, 
            header_size=self.cell_size
        )
        
        self.current_task = {
            "concept": "",
            "objects": [],
            "layout": {},
            "operation_history": [],
            "current_step": 0,
            "max_iterations": 100,
            "failure_count": 0,
            "max_failures": 5,
            "scene_signature": "",
            "similar_experiences": [],
            "planned_code": ""
        }
        
        self.object_states = {}
        
        self.available_operations = [
            "move", "scale", "rotate", "delete", "layered", "modify"
        ]
        
        self.visualization = {
            "step_images": [],
            "step_svgs": [],
            "current_canvas_png": None,
            "current_canvas_svg": None,
            "step_counter": 0
        }
        
        self._init_canvas()
    
    
    def _init_canvas(self):
        canvas_dir = os.path.join(self.args.output_dir, "visualization_steps")
        os.makedirs(canvas_dir, exist_ok=True)
        
        blank_svg = os.path.join(canvas_dir, "step_000_initial.svg")
        with open(blank_svg, "w", encoding="utf-8") as f:
            f.write(f'''<?xml version="1.0" encoding="utf-8"?>
    <svg xmlns="http://www.w3.org/2000/svg" 
        width="{self.grid_size}" 
        height="{self.grid_size}" 
        viewBox="0 0 {self.grid_size} {self.grid_size}">
        <rect width="100%" height="100%" fill="white"/>
    </svg>''')
        
        blank_png = os.path.join(canvas_dir, "step_000_initial.png")
        
        temp_png = blank_png.replace('.png', '_temp.png')
        
        from PIL import Image
        blank_img = Image.new('RGB', (self.grid_size, self.grid_size), color='white')
        blank_img.save(temp_png)
        
        self._overlay_grid_on_image(temp_png, blank_png)
        
        if os.path.exists(temp_png):
            os.remove(temp_png)
        
        self.visualization["current_canvas_png"] = blank_png
        self.visualization["current_canvas_svg"] = blank_svg
        self.visualization["step_images"].append(blank_png)
        self.visualization["step_svgs"].append(blank_svg)
        self.visualization["step_counter"] = 0

    def _pixel_to_grid(self, x_pixel: float, y_pixel: float) -> Tuple[int, int]:
        cell_size = self.cell_size
        grid_col = int(x_pixel // cell_size)
        grid_row = int(y_pixel // cell_size)
        grid_col = max(0, min(self.res - 1, grid_col))
        grid_row = max(0, min(self.res - 1, grid_row))
        return grid_col, grid_row

    def _grid_to_pixel(self, grid_col: int, grid_row: int) -> Tuple[float, float]:
        cell_size = self.cell_size
        x_pixel = grid_col * cell_size + cell_size / 2
        y_pixel = grid_row * cell_size + cell_size / 2
        return x_pixel, y_pixel

    def _grid_to_pixel_point(self, grid_point: str) -> Tuple[float, float]:
        return utils.grid_point_to_pixel(grid_point, self.cells_to_pixels_map)

    def _pixel_to_grid_point(self, x_pixel: float, y_pixel: float) -> str:
        grid_col, grid_row = self._pixel_to_grid(x_pixel, y_pixel)
        return f"x{grid_col}y{grid_row}"
    
    def analyze_scene(self, concept: str) -> Dict:
        print(f"\n🧠 Agent analyzing scene: {concept}")
        
        scene_plan_raw = self.sketch_app.call_scene_planner()
        scene_objects = self.sketch_app.parse_scene_plan(scene_plan_raw)
        
        scene_signature = self._generate_scene_signature(scene_objects)
        
        similar_experiences = self.memory.retrieve_similar_experiences(scene_signature)
        if similar_experiences:
            print(f"📚 Found {len(similar_experiences)} similar historical experiences; they will be used as references")
        
        self.current_task["concept"] = concept
        self.current_task["objects"] = scene_objects
        self.current_task["scene_signature"] = scene_signature
        self.current_task["similar_experiences"] = similar_experiences
        
        return {
            "objects": scene_objects,
            "scene_signature": scene_signature,
            "similar_experiences": similar_experiences
        }
    
    def _generate_scene_signature(self, objects: List[Dict]) -> str:
        object_types = [obj.get("object_id", "") for obj in objects]
        object_count = len(objects)
        depth_distribution = [obj.get("depth", 1) for obj in objects]
        
        signature = f"count_{object_count}_types_{'_'.join(sorted(object_types))}_depths_{'_'.join(map(str, sorted(depth_distribution)))}"
        return signature
    
    def generate_raw_sketches_parallel(self, objects: List[Dict]) -> List[Dict]:
        print(f"\n🎨 [Agent] Generating raw sketches for all objects...")
        
        os.makedirs(self.args.path2save, exist_ok=True)
        
        def generate_single(obj):
            try:
                print(f"  📝 Generating {obj['object_id']}...")
                
                sketching_raw = self.sketch_app.call_model_for_sketch_generation(obj["description"])
                
                initial_svg = self.sketch_app.parse_model_to_svg(sketching_raw, description=obj['object_id'])
                
                raw_svg_path = os.path.join(self.args.path2save, f"{obj['object_id']}_raw.svg")
                with open(raw_svg_path, "w", encoding='utf-8') as f:
                    f.write(initial_svg)

                
                bg_svg_path = os.path.join(self.args.path2save, f"{obj['object_id']}_bg.svg")
              
                refine_utils.inject_pure_vector_white_box(
                    svg_path=raw_svg_path,  # Use the normalized file
                    output_svg_path=bg_svg_path,
                    padding=15.0
                )

                
                try:
                    bbox = refine_utils.get_bbox(bg_svg_path)
                    raw_w = bbox.get("w", 100)
                    raw_h = bbox.get("h", 100)
                    raw_x = bbox.get("x", 100)
                    raw_y = bbox.get("y", 100)

                except Exception as e:
                    print(f"  ⚠️ Failed to extract bounding box: {e}, using default size")
                    raw_w = 100
                    raw_h = 100
                
                print(f"     ✅ {obj['object_id']} completed: {raw_w:.1f}x{raw_h:.1f}")
                
                return {
                    "object_id": obj["object_id"],
                    "description": obj["description"],
                    "incremental_prompt": obj["incremental_prompt"],
                    "svg_path": raw_svg_path,
                    "bg_svg_path": bg_svg_path,
                    "raw_w": raw_w,
                    "raw_h": raw_h,
                    "depth": obj.get("depth", 1),
                    "visual_complexity": obj.get("visual_complexity", 1),
                    "success": True
                }
            except Exception as e:
                import traceback
                print(f"  ❌ {obj.get('object_id', 'unknown')} failed: {e}")
                print(f"     {traceback.format_exc()}")
                return {
                    "object_id": obj.get("object_id", "unknown"),
                    "success": False,
                    "error": str(e)
                }
        
        
        results = []
        for obj in objects:

            result = generate_single(obj)
            results.append(result)
        
        success_count = sum(1 for r in results if r["success"])
        print(f"\n📊 Generation summary: {success_count}/{len(objects)} succeeded")
        for r in results:
            if not r["success"]:
                print(f"  ❌ {r['object_id']}: {r.get('error', 'Unknown error')}")
        
        for result in results:
            if result["success"]:
                obj_id = result["object_id"]
                self.object_states[obj_id] = {
                    "raw_w": result["raw_w"],
                    "raw_h": result["raw_h"],
                    "svg_path": result["svg_path"],
                    "bg_svg_path": result["bg_svg_path"],
                    "depth": result["depth"],
                    "visual_complexity": result["visual_complexity"],
                    "x": None,
                    "y": None,
                    "scale": 1.0,
                    "rotation": 0.0,
                    "description": result["description"],
                    "incremental_prompt": result["incremental_prompt"]
                }
        
        return results
    
    def _get_current_canvas_for_vlm(self) -> str:
        import base64
        
        current_png = self.visualization.get("current_canvas_png")
        if not current_png or not os.path.exists(current_png):
            blank_png = os.path.join(self.args.output_dir, "blank_canvas_with_grid.png")
            os.makedirs(os.path.dirname(blank_png), exist_ok=True)
            
            from PIL import Image
            temp_png = blank_png.replace('.png', '_temp.png')
            blank_img = Image.new('RGB', (self.grid_size, self.grid_size), color='white')
            blank_img.save(temp_png)
            self._overlay_grid_on_image(temp_png, blank_png)
            current_png = blank_png
        
        try:
            with open(current_png, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"⚠️ Failed to read canvas image: {e}")
            from PIL import Image
            import io
            temp_png = os.path.join(self.args.output_dir, "temp_grid.png")
            blank_img = Image.new('RGB', (self.grid_size, self.grid_size), color='white')
            blank_img.save(temp_png)
            self._overlay_grid_on_image(temp_png, temp_png)
            with open(temp_png, "rb") as f:
                base64_str = base64.b64encode(f.read()).decode('utf-8')
            return base64_str

    def _check_boundaries_with_details(self) -> Dict:
        canvas_size = 512
        out_of_bounds = []
        all_objects_info = []
        
        for obj_id, state in self.object_states.items():
            if state.get("x") is None or state.get("y") is None:
                continue
            
            raw_w = state.get("raw_w", 100)
            raw_h = state.get("raw_h", 100)
            scale = state.get("scale", 1.0)
            x = state["x"]
            y = state["y"]
            
            svg_path = state.get("svg_path")
            if svg_path and os.path.exists(svg_path):
                try:
                    bbox = refine_utils.get_bbox(svg_path)
                    raw_cx = bbox.get("x", 0) + bbox.get("w", 0) / 2
                    raw_cy = bbox.get("y", 0) + bbox.get("h", 0) / 2
                    raw_w = bbox.get("w", 100)
                    raw_h = bbox.get("h", 100)
                except:
                    raw_cx = raw_w / 2
                    raw_cy = raw_h / 2
            else:
                raw_cx = raw_w / 2
                raw_cy = raw_h / 2
            
            total_scale = scale
            
            left = x - raw_cx * total_scale
            top = y - raw_cy * total_scale
            right = x + (raw_w - raw_cx) * total_scale
            bottom = y + (raw_h - raw_cy) * total_scale
            
            obj_info = {
                "object_id": obj_id,
                "center_x": x,
                "center_y": y,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": right - left,
                "height": bottom - top,
                "scale": scale,
                "is_out_of_bounds": False
            }
            
            if left < 0 or top < 0 or right > canvas_size or bottom > canvas_size:
                obj_info["is_out_of_bounds"] = True
                obj_info["suggestion"] = self._suggest_fix_for_out_of_bounds(
                    left, top, right, bottom, canvas_size, obj_id
                )
                out_of_bounds.append(obj_info)
            
            all_objects_info.append(obj_info)
        
        summary = ""
        if out_of_bounds:
            summary = f"There are {len(out_of_bounds)} objects outside the canvas boundary:\n"
            for obj in out_of_bounds:
                summary += f"  - {obj['object_id']}: position({obj['left']:.0f}, {obj['top']:.0f}) size{obj['width']:.0f}x{obj['height']:.0f}\n"
                if obj.get("suggestion"):
                    summary += f"    Suggestion: {obj['suggestion']}\n"
        
        return {
            "out_of_bounds": len(out_of_bounds) > 0,
            "details": out_of_bounds,
            "all_objects": all_objects_info,
            "summary": summary
        }


    def _suggest_fix_for_out_of_bounds(self, left, top, right, bottom, canvas_size, obj_id):
        suggestions = []
        operations = []
        
        if left < 0:
            delta_x = -left + 10 
            operations.append(f'move("{obj_id}", x + {delta_x:.0f}, y)  # move right by {delta_x:.0f}px')
            suggestions.append(f"move right by {delta_x:.0f} pixels")
        elif right > canvas_size:
            delta_x = -(right - canvas_size) - 10
            operations.append(f'move("{obj_id}", x + {delta_x:.0f}, y)  # move left by {-delta_x:.0f}px')
            suggestions.append(f"move left by {right - canvas_size + 10:.0f} pixels")
        
        if top < 0:
            delta_y = -top + 10
            operations.append(f'move("{obj_id}", x, y + {delta_y:.0f})  # move down by {delta_y:.0f}px')
            suggestions.append(f"move down by {delta_y:.0f} pixels")
        elif bottom > canvas_size:
            delta_y = -(bottom - canvas_size) - 10
            operations.append(f'move("{obj_id}", x, y + {delta_y:.0f})  # move up by {-delta_y:.0f}px')
            suggestions.append(f"move up by {bottom - canvas_size + 10:.0f} pixels")
        
        width = right - left
        height = bottom - top
        if width > canvas_size * 0.9 or height > canvas_size * 0.9:
            new_scale = min(canvas_size / width, canvas_size / height) * 0.85
            operations.append(f'scale("{obj_id}", {new_scale:.2f})  # scale down to {new_scale:.2f}x')
            suggestions.append(f"scale down to {new_scale:.2f}x")
        
        if operations:
            return " or ".join(suggestions) + f"\n   Suggested operations: {', '.join(operations)}"
        else:
            return "Adjust position or scale"

    def plan_layout_code(self) -> str:
        print(f"\n📝 [Agent] Planning layout strategy...")
        
        grid_context = self._get_grid_context()
            
        objects = self.current_task["objects"]
        similar_exp = self.current_task.get("similar_experiences", [])
        
        current_canvas_base64 = self._get_current_canvas_for_vlm()
        
        object_svg_map = {}
        for obj_id, state in self.object_states.items():
            object_svg_map[obj_id] = {
                "description": state.get("description", ""),
                "svg_path": state.get("svg_path", ""),
                "bbox": {
                    "x": state.get("x", 0),
                    "y": state.get("y", 0),
                    "w": state.get("raw_w", 0),
                    "h": state.get("raw_h", 0)
                }
            }
        
        boundary_info = self._check_boundaries_with_details()
        out_of_bounds = boundary_info["details"]
        
        boundary_context = ""
        if out_of_bounds:
            boundary_context = f"\n=== ⚠️ Objects out of bounds (Grid: {self.res}x{self.res}) ===\n"
            for obj in out_of_bounds:
                grid_col, grid_row = self._pixel_to_grid(
                    obj.get("center_x", 0), 
                    obj.get("center_y", 0)
                )
                boundary_context += f"- {obj['object_id']}: grid ({grid_col}, {grid_row})"
                if "suggestion" in obj:
                    boundary_context += f"  Suggested fix: {obj['suggestion']}\n"
            boundary_context += "\nResolve these issues in the generated code.\n"
        
        additional_context = grid_context + "\n" + boundary_context
    
        layout_code = self.sketch_app.call_vlm_for_layout_code(
            concept=self.current_task["concept"],
            objects=objects,
            current_canvas_base64=current_canvas_base64,
            object_svg_map=object_svg_map,
            similar_experiences=similar_exp,
            available_operations=self.available_operations,
            additional_context=additional_context
        )
        
        self.current_task["planned_code"] = layout_code
        return layout_code
    
    
    def execute_layout_code(self, code: str, iteration: int) -> Dict:
        print(f"\n⚙️ [Agent] Executing layout code (iteration #{iteration})...")
        
        try:
            exec_globals = {
                "objects": {obj["object_id"]: self.object_states[obj["object_id"]] 
                        for obj in self.current_task["objects"] 
                        if obj["object_id"] in self.object_states},
                "move": lambda obj_id, x, y: self._move(obj_id, iteration, x, y),
                "scale": lambda obj_id, factor: self._scale(obj_id, iteration, factor),
                "rotate": lambda obj_id, degrees: self._rotate(obj_id, iteration, degrees),
                "delete": lambda obj_id: self._delete(obj_id, iteration),
                "layered": lambda obj_id, layer: self._layered(obj_id, iteration, layer),
                "modify": lambda obj_id, new_description=None, regenerate=True: self._modify(obj_id, iteration, new_description, regenerate),
                "torch": torch,
                "print": print,
                "len": len,
                "range": range
            }
            
            exec(code, exec_globals)
            
            self.current_task["operation_history"].append({
                "step": len(self.current_task["operation_history"]),
                "iteration": iteration,
                "code": code,
                "result": "success"
            })
            
            return {"success": True, "code": code}
            
        except Exception as e:
            print(f"❌ Layout code execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e), "code": code}
    
    def _move(self, obj_id: str, iteration: int, x: float, y: float):
        if obj_id in self.object_states:
            x = max(0, min(self.res - 1, x))
            y = max(0, min(self.res - 1, y))
            
            target_x, target_y = self._grid_to_pixel(x, y)
            state = self.object_states[obj_id]
            
            raw_w = state.get("raw_w", 100)
            raw_h = state.get("raw_h", 100)
            factor = state.get("scale", 1.0)
            state["x"] = target_x - raw_w * factor / 2
            state["y"] = target_y - raw_h * factor / 2
            
            self.memory.record_operation({
                "operation": "move", "object_id": obj_id,
                "params": {"grid_x": x, "grid_y": y, "grid_pixel_x": target_x, "grid_pixel_y": target_y}
            })

            self._render_current_canvas("move", iteration, {"obj": obj_id, "grid": f"x{int(x)}y{int(y)}"})


    def _scale(self, obj_id: str, iteration: int, factor: float):
        if obj_id in self.object_states:
            state = self.object_states[obj_id]
            x = state.get("x", 100)
            y = state.get("y", 100)
            old_factor = state.get("scale", 1.0)
            old_w = state.get("raw_w", 100) * old_factor
            old_h = state.get("raw_h", 100) * old_factor

            new_w = state.get("raw_w", 100) * factor
            new_h = state.get("raw_h", 100) * factor

            state["scale"] = max(0.3, factor)
            
            state["x"] = x
            state["y"] = y
            
            self.memory.record_operation({
                "operation": "scale", "object_id": obj_id, "params": {"factor": factor}
            })


            self._render_current_canvas("scale", iteration, {"obj": obj_id, "factor": factor})
    
    def _rotate(self, obj_id: str, iteration: int, degrees: float):
        if obj_id in self.object_states:
            self.object_states[obj_id]["rotation"] = degrees
            
            self.memory.record_operation({
                "operation": "rotate",
                "object_id": obj_id,
                "params": {"degrees": degrees}
            })
            
            print(f"  🔄 rotate({obj_id}, {degrees:.1f}°)")
            png_path = self._render_current_canvas("rotate", iteration, {"obj": obj_id, "degrees": degrees})

    
    def _delete(self, obj_id: str, iteration: int):
        if obj_id in self.object_states:
            del self.object_states[obj_id]
            
            self.memory.record_operation({
                "operation": "delete",
                "object_id": obj_id,
                "params": {}
            })
            
            print(f"  🗑️ delete({obj_id})")
            png_path = self._render_current_canvas("delete", iteration, {"obj": obj_id})

    
    def _layered(self, obj_id: str, iteration: int, layer: str):
        if obj_id in self.object_states:
            depth_map = {"bg": 0, "mid": 1, "fg": 2, "front": 3}
            self.object_states[obj_id]["depth"] = depth_map.get(layer, 1)
            
            self.memory.record_operation({
                "operation": "layered",
                "object_id": obj_id,
                "params": {"layer": layer}
            })
            
            print(f"  📚 layered({obj_id}, {layer})")
            png_path = self._render_current_canvas("layered", iteration, {"obj": obj_id, "layer": layer})

    
    def _modify(self, obj_id: str, iteration: int, new_description: str = None, regenerate: bool = True):
        if obj_id not in self.object_states:
            print(f"  ❌ Object {obj_id} does not exist")
            return
        
        print(f"  ✏️ modify({obj_id}) - regenerating raw sketch...")
        
        obj = next((o for o in self.current_task["objects"] if o["object_id"] == obj_id), None)
        if not obj:
            print(f"  ❌ Could not find original information for object {obj_id}")
            return
        
        description = new_description if new_description else obj["description"]
        
        try:
            sketching_raw = self.sketch_app.call_model_for_sketch_generation(description)
            new_svg = self.sketch_app.parse_model_to_svg(sketching_raw, description=obj_id)
            
            svg_path = os.path.join(self.args.path2save, f"{obj_id}_regenerated.svg")
            
            with open(svg_path, "w") as f:
                f.write(new_svg)
            
            bg_svg_path = os.path.join(self.args.path2save, f"{obj_id}_regenerated_bg.svg")
            refine_utils.inject_pure_vector_white_box(
                svg_path=svg_path,
                output_svg_path=bg_svg_path,
                padding=15.0
            )
            
            bbox = refine_utils.get_bbox(bg_svg_path)
            
            self.object_states[obj_id]["svg_path"] = svg_path
            self.object_states[obj_id]["bg_svg_path"] = bg_svg_path
            self.object_states[obj_id]["raw_w"] = bbox["w"]
            self.object_states[obj_id]["raw_h"] = bbox["h"]
            
            self.memory.record_operation({
                "operation": "modify",
                "object_id": obj_id,
                "params": {
                    "new_description": description,
                    "new_bbox": {"w": bbox["w"], "h": bbox["h"]}
                }
            })
            
            print(f"  ✅ modify({obj_id}) completed, new size: {bbox['w']}x{bbox['h']}")
            png_path = self._render_current_canvas("modify", iteration, {"obj": obj_id})

            
        except Exception as e:
            print(f"  ❌ modify({obj_id}) failed: {e}")
    
    def _overlay_grid_on_image(self, input_png: str, output_png: str):
        from PIL import Image, ImageDraw
        
        img = Image.open(input_png)
        draw = ImageDraw.Draw(img)
        
        canvas_size = self.grid_size
        cell_size = self.cell_size
        res = self.res
        
        for i in range(res + 1):
            pos = i * cell_size
            draw.line([(pos, 0), (pos, canvas_size)], fill='#e0e0e0', width=1)
            draw.line([(0, pos), (canvas_size, pos)], fill='#e0e0e0', width=1)
        
        draw.rectangle([(0, 0), (canvas_size-1, canvas_size-1)], outline='#999999', width=2)
        
        for i in range(0, res, 2):
            x = i * cell_size + cell_size / 2
            draw.text((x-3, canvas_size-12), str(i), fill='#bbbbbb', font=None)
            
            y_pos = i * cell_size + cell_size / 2
            draw.text((3, y_pos-4), str(i), fill='#bbbbbb', font=None)
        
        img.save(output_png)

    def _render_png_with_grid(self, svg_path: str, output_png_path: str):
        import cairosvg
        from PIL import Image, ImageDraw
        
        temp_png = output_png_path.replace('.png', '_temp.png')
        
        try:
            cairosvg.svg2png(
                url=svg_path, 
                write_to=temp_png,
                dpi=96,
                background_color='white'
            )

        except Exception as e:
            print(f"  ⚠️ CairoSVG rendering failed: {e}")
            from PIL import Image
            img = Image.new('RGB', (self.grid_size, self.grid_size), color='white')
            img.save(temp_png)

        
        if os.path.exists(temp_png):
            self._overlay_grid_on_image(temp_png, output_png_path)
            os.remove(temp_png)
        else:
            print(f"  ❌ Temporary PNG does not exist: {temp_png}")

    def _render_current_canvas(self, operation_name: str = "", iteration: int = 1, params: dict = None) -> str:
        if not self.object_states:
            return self.visualization["current_canvas_png"]
            
        layout_history = []
        for obj_id, state in self.object_states.items():
            if state.get("x") is None or state.get("y") is None:
                continue
                
            item = {
                "object_id": obj_id,
                "svg_path": state.get("svg_path"),
                "bg_svg_path": state.get("bg_svg_path"),
                "x": state.get("x"),
                "y": state.get("y"),
                "scale": state.get("scale"),
                "rotation": state.get("rotation", 0.0),
                "depth": state.get("depth", 1)
            }
            layout_history.append(item)
            
        if not layout_history:
            return self.visualization["current_canvas_png"]
            
        layout_history.sort(key=lambda item: item["depth"])
        self.visualization["step_counter"] += 1
        step_num = self.visualization["step_counter"]
        op_desc = f"{operation_name}" if operation_name else f"step_{step_num}"
        if params:
            param_str = "_".join([f"{k}_{v}" for k, v in params.items() if v is not None])
            op_desc = f"{op_desc}_{param_str}"
            
        step_dir = os.path.join(self.args.output_dir, "visualization_steps")
        os.makedirs(step_dir, exist_ok=True)
        png_path = os.path.join(step_dir, f"iter_{iteration:03d}_step_{step_num:03d}_{op_desc}.png")
        svg_path = os.path.join(step_dir, f"iter_{iteration:03d}_step_{step_num:03d}_{op_desc}.svg")
        
        try:
            self.sketch_app.rebuild_official_canvas(
                layout_history,
                output_png_path=None,
                output_svg_path=svg_path,
                canvas_size=self.grid_size
            )
            
            self._render_png_with_grid(svg_path, png_path)
            
            self.visualization["current_canvas_png"] = png_path
            self.visualization["current_canvas_svg"] = svg_path
            self.visualization["step_images"].append(png_path)
            self.visualization["step_svgs"].append(svg_path)
            
            if "iteration_steps" not in self.visualization:
                self.visualization["iteration_steps"] = {}

            
            if iteration not in self.visualization["iteration_steps"]:
                self.visualization["iteration_steps"][iteration] = []

            
            self.visualization["iteration_steps"][iteration].append({
                "step": step_num,
                "operation": operation_name,
                "png_path": png_path,
                "svg_path": svg_path,
                "code": self.current_task.get("planned_code", ""),
                "object_states": copy.deepcopy(self.object_states)
            })
            

            return png_path
            
        except Exception as e:
            print(f"  ❌ Canvas rendering failed: {e}")
            import traceback
            traceback.print_exc()
            return self.visualization["current_canvas_png"]
    
    def verify_layout(self) -> Dict:
        print(f"\n✅ [Agent] Verify the layout...")
        
        issues = []
        warnings = []
        scores = {}
        
        print("  📐 Geometric verification...")
        
        boundary_issues = self._check_boundaries_strict()
        if boundary_issues:
            issues.extend(boundary_issues)
            print(f"    ❌ Boundary issues: {boundary_issues}")
        else:
            print(f"    ✅ Boundary check passed")
        
        size_issues, size_score = self._check_object_sizes()
        if size_issues:
            warnings.extend(size_issues)
            print(f"    ⚠️ Size issues: {len(size_issues)}")
        else:
            print(f"    ✅ Size check passed")
        scores['size_score'] = size_score
        
        distribution_score = self._check_distribution()
        scores['distribution_score'] = distribution_score
        print(f"    📊 Distribution score: {distribution_score:.2f}")
        
        print("  🧠 Semantic verification (VLM)...")
        objects_collage_base64 = self._get_objects_collage()
        semantic_result = self._verify_with_vlm(objects_collage_base64)
        
        semantic_score = semantic_result.get('overall_score', 0.5)
        semantic_passed = semantic_result.get('passed', False)
        scores['semantic_score'] = semantic_score
        scores['semantic_passed'] = semantic_passed
        
        if semantic_passed:
            print(f"    ✅ Semantic check passed (score: {semantic_score:.2f})")
        else:
            print(f"    ❌ Semantic check failed (score: {semantic_score:.2f})")
            if semantic_result.get('issues'):
                warnings.extend(semantic_result['issues'])
        
        passed = (len(boundary_issues) == 0 and semantic_passed)
        
        overall_score = (size_score * 0.30 + 
                        distribution_score * 0.3 + 
                        semantic_score * 0.40)
        
        if len(boundary_issues) > 0:
            overall_score *= 0.5
        
        result = {
            "passed": passed,
            "overall_score": overall_score,
            "scores": scores,
            "issues": issues,
            "warnings": warnings,
            "object_count": len(self.object_states),
            "current_canvas": self.visualization["current_canvas_png"],
            "needs_refinement": not passed or overall_score < 0.7
        }
        
        print(f"\n  📊 Overall evaluation:")
        print(f"    - Passed: {passed}")
        print(f"    - Overall score: {overall_score:.2f}")
        print(f"    - Issues: {len(issues)}, Warnings: {len(warnings)}")
        
        return result


    def _check_boundaries_strict(self) -> List[Dict]:
        issues = []
        canvas_size = self.grid_size
        margin = 5  
        
        for obj_id, state in self.object_states.items():
            if state.get("x") is None or state.get("y") is None:
                continue
            
            raw_w = state["raw_w"] 
            raw_h = state["raw_h"] 
            factor = state.get("scale", 1.0)
            w = raw_w * factor
            h = raw_h * factor
            x, y = state["x"], state["y"]
            

            left = x
            top = y
            right = x + w
            bottom = y + h
            
            if left < margin or right > canvas_size - margin or top < margin or bottom > canvas_size - margin:
                grid_col, grid_row = self._pixel_to_grid(x, y)
                issues.append({
                    "type": "boundary",
                    "object_id": obj_id,
                    "message": f"Object {obj_id} at grid ({grid_col}, {grid_row}) exceeds canvas boundary",
                    "details": {
                        "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
                        "canvas_size": canvas_size
                    }
                })
        
        return issues


    def _check_overlaps_soft(self) -> Tuple[List[Dict], float]:
        issues = []
        obj_ids = list(self.object_states.keys())
        total_overlap_ratio = 0
        overlap_count = 0
        
        for i in range(len(obj_ids)):
            for j in range(i + 1, len(obj_ids)):
                obj1 = self.object_states[obj_ids[i]]
                obj2 = self.object_states[obj_ids[j]]
                
                if obj1.get("x") is None or obj2.get("x") is None:
                    continue
                
                w1 = obj1["raw_w"] * obj1.get("scale", 1.0)
                h1 = obj1["raw_h"] * obj1.get("scale", 1.0)
                w2 = obj2["raw_w"] * obj2.get("scale", 1.0)
                h2 = obj2["raw_h"] * obj2.get("scale", 1.0)
                
                overlap_x = max(0, min(obj1["x"] + w1/2, obj2["x"] + w2/2) - max(obj1["x"] - w1/2, obj2["x"] - w2/2))
                overlap_y = max(0, min(obj1["y"] + h1/2, obj2["y"] + h2/2) - max(obj1["y"] - h1/2, obj2["y"] - h2/2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                min_area = min(area1, area2)
                overlap_ratio = overlap_area / min_area if min_area > 0 else 0
                
                total_overlap_ratio += overlap_ratio
                overlap_count += 1
                
                if overlap_ratio > 0.5:
                    issues.append({
                        "type": "overlap",
                        "object1": obj_ids[i],
                        "object2": obj_ids[j],
                        "overlap_ratio": overlap_ratio,
                        "overlap_area": overlap_area,
                        "severity": "high" if overlap_ratio > 0.7 else "medium"
                    })
                elif overlap_ratio > 0.2:
                    issues.append({
                        "type": "overlap_warning",
                        "object1": obj_ids[i],
                        "object2": obj_ids[j],
                        "overlap_ratio": overlap_ratio,
                        "overlap_area": overlap_area,
                        "severity": "low"
                    })
        
        if overlap_count > 0:
            avg_overlap = total_overlap_ratio / overlap_count
            overlap_score = max(0, 1 - avg_overlap * 1.5) 
        else:
            overlap_score = 1.0
        
        return issues, overlap_score


    def _check_object_sizes(self) -> Tuple[List[Dict], float]:
        issues = []
        canvas_size = self.grid_size
        
        for obj_id, state in self.object_states.items():
            w = state["raw_w"] * state.get("scale", 1.0)
            h = state["raw_h"] * state.get("scale", 1.0)
            
            if w < canvas_size * 0.03 or h < canvas_size * 0.03:
                issues.append({
                    "type": "too_small",
                    "object_id": obj_id,
                    "message": f"Object {obj_id} is too small ({w:.1f}x{h:.1f})"
                })
            
            if w > canvas_size * 0.8 or h > canvas_size * 0.8:
                issues.append({
                    "type": "too_large",
                    "object_id": obj_id,
                    "message": f"Object {obj_id} is too large ({w:.1f}x{h:.1f})"
                })
        
        size_score = 1.0
        for issue in issues:
            if issue["type"] == "too_small":
                size_score *= 0.8
            elif issue["type"] == "too_large":
                size_score *= 0.8
        
        return issues, min(1.0, size_score)


    def _check_distribution(self) -> float:
        if len(self.object_states) < 2:
            return 1.0
        
        positions = [(s["x"], s["y"]) for s in self.object_states.values() 
                    if s.get("x") is not None and s.get("y") is not None]
        
        if not positions:
            return 0.0
        
        center_x = sum(p[0] for p in positions) / len(positions)
        center_y = sum(p[1] for p in positions) / len(positions)
        
        canvas_center_x = self.grid_size / 2
        canvas_center_y = self.grid_size / 2
        
        center_dist = ((center_x - canvas_center_x)**2 + (center_y - canvas_center_y)**2)**0.5
        max_dist = self.grid_size * 0.3  
        
        distribution_score = max(0, 1 - center_dist / max_dist)
        
        return min(1.0, distribution_score)


    def _verify_with_vlm(self, objects_collage_base64) -> Dict:
        current_canvas_base64 = self._get_current_canvas_for_vlm()
        
        concept = self.current_task.get("concept", "")
        objects = self.current_task.get("objects", [])
        
        object_descriptions = []
        for obj in objects:
            obj_id = obj.get("object_id", "")
            obj_desc = obj.get("description", "")
            object_descriptions.append(f"- {obj_id}: {obj_desc}")
        objects_str = "\n".join(object_descriptions)
        
        system_prompt = """You are an expert scene composition validator and art critic.

    Your task is to analyze:
    1. The overall composition canvas (the full scene)
    2. The individual object thumbnails (each object isolated with its ID label)

    By comparing both, you can accurately determine what objects are actually present and how well they match the concept.

    You MUST respond with a valid JSON object.

    Output format:
    {
        "semantic_match": 0.0-1.0,
        "composition_quality": 0.0-1.0,
        "recognizability": 0.0-1.0,
        "spatial_relationship": 0.0-1.0,
        "size_proportion": 0.0-1.0,
        "present_objects": ["object_id1", "object_id2"],
        "missing_objects": ["object_id3"],
        "issues": [],
        "feedback": "brief summary"
    }
    """
        
        user_prompt = """
    === SCENE CONCEPT ===
    "{concept}"

    === EXPECTED OBJECTS ===
    {objects_str}

    === INPUT IMAGES ===
    - **Image 1**: Full composition canvas (the complete scene)
    - **Image 2**: Object collage - each individual object is shown separately with its object_id label on top

    === EVALUATION CRITERIA ===
    1. **Semantic Match** (0-1): How well does the visual scene match the concept description?
    - Compare the full scene against the concept
    - Check if all expected objects are present (use the object collage to verify)
    - 1.0: Perfect match, all expected objects present and recognizable
    - 0.7: Good match, most objects present but some details missing
    - 0.5: Partial match, some objects missing or incorrect
    - 0.3: Poor match, major objects missing
    - 0.0: No match

    2. **Composition Quality** (0-1): Is the arrangement visually appealing?
    - Consider: balance, spacing, focal point, no awkward cutoffs
    - Objects should not overlap significantly (>50%)
    - Objects should not be too close to edges

    3. **Recognizability** (0-1): Are the individual objects recognizable?
    - Use the object collage to clearly see each object
    - Can you tell what each object is supposed to be?
    - Are any objects completely unidentifiable?

    4. **Spatial Relationship** (0-1): Are the positional relationships between objects correct and reasonable?
    - Does the spatial layout follow logic and common sense (e.g., a cup should be on a table, not floating in mid-air; the sun should be in the sky, not on the ground)?
    - Are the front-back, up-down, and left-right positional relationships between objects reasonable and consistent with the scene concept?

    5. **Size Proportion** (0-1): Are the relative sizes and scales between objects reasonable?
    - Does the scaling between objects follow real-world logic (e.g., a chair should not be larger than a house, a cat should not be larger than an elephant)?
    - Are the relative proportions of objects in the same scene coordinated and natural?

    **IMPORTANT**: Use the object collage (Image 2) to accurately identify which objects are present. The collage shows each object with its ID label, so you can definitively say whether an object exists in the scene.

    Analyze the images and respond with JSON only.
    """
        
        import json
        
        try:
            response = self.sketch_app.query_multimodal_model_with_image_and_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_base64=[current_canvas_base64, objects_collage_base64]
            )
            
            clean_response = response.strip()
            clean_response = clean_response.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(clean_response)
            
            default_result = {
                "semantic_match": result.get("semantic_match", 0.5),
                "composition_quality": result.get("composition_quality", 0.5),
                "recognizability": result.get("recognizability", 0.5),
                "spatial_relationship": result.get("spatial_relationship", 0.5),
                "size_proportion": result.get("size_proportion", 0.5),
                "present_objects": result.get("present_objects", []),
                "missing_objects": result.get("missing_objects", []),
                "issues": result.get("issues", []),
                "feedback": result.get("feedback", "No feedback provided")
            }
            
            default_result["overall_score"] = (
                default_result["semantic_match"] * 0.30 +
                default_result["spatial_relationship"] * 0.30 +
                default_result["size_proportion"] * 0.15 +
                default_result["composition_quality"] * 0.15 +
                default_result["recognizability"] * 0.10
            )
            if default_result["overall_score"] >= 0.7:
                default_result["passed"] = True
            else:
                default_result["passed"] = False
            
            print(f"    📊 VLM scores: semantic={default_result['semantic_match']:.2f}, "
                f"spatial={default_result['spatial_relationship']:.2f}, "
                f"proportion={default_result['size_proportion']:.2f}, "
                f"composition={default_result['composition_quality']:.2f}, "
                f"Overall score={default_result['overall_score']:.2f}")
            
            if default_result.get("missing_objects"):
                print(f"    ❌ Missing objects: {default_result['missing_objects']}")
            
            return default_result
            
        except json.JSONDecodeError as e:
            print(f"    ⚠️ JSON parsing failed: {e}")
            print(f"    Raw response: {response[:200]}...")
            return {
                "semantic_match": 0.5,
                "composition_quality": 0.5,
                "recognizability": 0.5,
                "spatial_relationship": 0.5,
                "size_proportion": 0.5,
                "overall_score": 0.5,
                "passed": False,
                "present_objects": [],
                "missing_objects": [],
                "issues": [{"type": "parse_error", "description": str(e), "severity": "high"}],
                "feedback": f"VLM response parse failed: {e}"
            }
            
        except Exception as e:
            print(f"    ⚠️ VLM verification failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "semantic_match": 0.5,
                "composition_quality": 0.5,
                "recognizability": 0.5,
                "spatial_relationship": 0.5,
                "size_proportion": 0.5,
                "overall_score": 0.5,
                "passed": False,
                "present_objects": [],
                "missing_objects": [],
                "issues": [{"type": "validation_error", "description": str(e), "severity": "high"}],
                "feedback": f"VLM validation failed: {e}"
            }

    def _get_objects_collage(self, save_path: str = None) -> str:
        from PIL import Image, ImageDraw, ImageFont
        import io
        import base64
        import os
        from datetime import datetime
        
        objects = self.current_task.get("objects", [])
        if not objects:
            blank_img = Image.new('RGB', (100, 100), color='white')
            buffered = io.BytesIO()
            blank_img.save(buffered, format="PNG")
            
            if save_path is None:
                save_path = f"collage_blank_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            blank_img.save(save_path, format="PNG")

            
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        thumb_size = 150
        padding = 10
        label_height = 30
        label_padding = 5
        
        cols = min(4, len(objects))
        rows = (len(objects) + cols - 1) // cols
        
        title_height = 40
        total_width = cols * (thumb_size + padding) + padding
        total_height = title_height + rows * (thumb_size + padding + label_height) + padding
        
        collage = Image.new('RGB', (total_width, total_height), color='white')
        draw = ImageDraw.Draw(collage)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            small_font = ImageFont.truetype("arial.ttf", 11)
            title_font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        title = f"Objects in Scene ({len(objects)} objects)"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text((total_width//2 - title_width//2, 10), title, fill=(50, 50, 150), font=title_font)
        
        draw.line([(20, title_height-5), (total_width-20, title_height-5)], fill=(200, 200, 200), width=1)
        
        for idx, obj in enumerate(objects):
            obj_id = obj.get("object_id", f"obj_{idx}")
            obj_desc = obj.get("description", "")[:30]  # Truncate the description
            
            obj_image = self._get_object_thumbnail(obj_id, size=thumb_size)
            
            row = idx // cols
            col = idx % cols
            
            x = padding + col * (thumb_size + padding)
            y = title_height + padding + row * (thumb_size + padding + label_height)
            
            draw.rectangle([x-2, y-2, x+thumb_size+2, y+thumb_size+2], 
                        outline=(200, 200, 200), width=1)
            
            if obj_image:
                collage.paste(obj_image, (x, y))
            else:
                draw.rectangle([x, y, x+thumb_size, y+thumb_size], 
                            fill=(240, 240, 240), outline=(180, 180, 180))
                draw.text((x+thumb_size//2-30, y+thumb_size//2-7), "No image", 
                        fill=(150,150,150), font=small_font)
            
            label_y = y + thumb_size + 5
            
            draw.rectangle([x, label_y, x+thumb_size, label_y + label_height - 5], 
                        fill=(60, 80, 200), outline=(40, 60, 180))
            
            text_bbox = draw.textbbox((0, 0), obj_id, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x + (thumb_size - text_width) // 2
            draw.text((text_x, label_y + 5), obj_id, fill='white', font=font)
            
            if obj_desc and False: 
                desc_y = label_y + label_height - 5
                draw.text((x+5, desc_y), obj_desc[:20], fill=(100,100,100), font=small_font)
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if hasattr(self, 'args') and hasattr(self.args, 'output_dir'):
                save_path = os.path.join(self.args.output_dir, f"objects_collage_{timestamp}.png")
            else:
                save_path = f"objects_collage_{timestamp}.png"
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
        collage.save(save_path, format="PNG", quality=95, optimize=True)

        
        buffered = io.BytesIO()
        collage.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        if not hasattr(self, '_collage_paths'):
            self._collage_paths = []
        self._collage_paths.append(save_path)
        
        return base64_str

    def _get_object_thumbnail(self, object_id: str, size: int = 150) -> Image.Image:
        from PIL import Image, ImageDraw
        import io
        import os
        
        if hasattr(self, 'object_states') and object_id in self.object_states:
            obj_state = self.object_states[object_id]
            bg_svg_path = obj_state.get('bg_svg_path')
            
            if bg_svg_path and os.path.exists(bg_svg_path):
                try:
                    import cairosvg
                    png_data = cairosvg.svg2png(
                        url=bg_svg_path, 
                        output_width=size, 
                        output_height=size
                    )
                    img = Image.open(io.BytesIO(png_data))
                    
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    
                    white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
                    white_bg.paste(img, (0, 0), img)
                    img = white_bg.convert('RGB')
                    return img
                    
                except ImportError:
                    print(f"      ⚠️ cairosvg not installed")
                except Exception as e:
                    print(f"      ⚠️ SVG render failed ({bg_svg_path}): {e}")
            else:
                print(f"      ⚠️ SVG file not found: {bg_svg_path}")
        
        img = Image.new('RGB', (size, size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        draw.text((size//2 - 30, size//2 - 7), f"❌{object_id}", fill=(200, 200, 200), font=font)
        draw.rectangle([0, 0, size-1, size-1], outline=(220, 220, 220), width=1)
        
        return img
    
    
    def diagnose_failure(self, verification_result: Dict) -> Dict:
        """Diagnose the cause of failure"""
        print(f"\n🔍 [Agent] Diagnosing failure...")
        
        issues = verification_result.get("issues", [])
        
        if not issues:
            return {"needs_fix": False}
        
        overlap_issues = [i for i in issues if i["type"] == "overlap"]
        boundary_issues = [i for i in issues if i["type"] == "boundary"]
        
        diagnosis = {
            "needs_fix": True,
            "overlap_count": len(overlap_issues),
            "boundary_count": len(boundary_issues),
            "error_type": "unknown"
        }
        
        if overlap_issues:
            diagnosis["error_type"] = "overlap_conflict"
            diagnosis["recommended_fix"] = "Adjust the position or scale of overlapping objects to increase spacing"
            diagnosis["affected_objects"] = list(set(
                [i["object1"] for i in overlap_issues] + [i["object2"] for i in overlap_issues]
            ))
        elif boundary_issues:
            diagnosis["error_type"] = "boundary_violation"
            diagnosis["recommended_fix"] = "Move out-of-bounds objects back inside the canvas or reduce their scale"
            diagnosis["affected_objects"] = [i["object_id"] for i in boundary_issues]
        
        # Record failure in memory
        self.memory.record_failure({
            "scene_signature": self.current_task.get("scene_signature", ""),
            "operation": self.current_task.get("planned_code", ""),
            "error_type": diagnosis["error_type"],
            "step": len(self.current_task["operation_history"]),
            "recommended_fix": diagnosis["recommended_fix"],
            "verification_result": verification_result
        })
        
        return diagnosis
    
    def repair_code(self, diagnosis: Dict, failed_code: str) -> str:
        print(f"\n🔧 [Agent] Repairing code...")
        
        repair_prompt = f"""
        The original code execution failed. Failure type: {diagnosis.get('error_type', 'unknown')}
        Affected objects: {diagnosis.get('affected_objects', [])}
        Recommended fix: {diagnosis.get('recommended_fix', '')}

        Failed code:{failed_code}
        
        Fix the code by rewriting from the step where the problem occurred.
        """
        
        repaired_code = self.sketch_app.call_vlm_for_code_repair(
            repair_prompt=repair_prompt,
            failed_code=failed_code,
            diagnosis=diagnosis
        )
        
        return repaired_code
    
    def _clean_code_response(self, response: str) -> str:
        pattern = r'```(?:python)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            cleaned = matches[0].strip()
        else:
            cleaned = response.strip()
        
        if cleaned.startswith("python\n"):
            cleaned = cleaned[7:]
        
        return cleaned
    
    def plan_layout_code_with_history(self, attempt_history: List[Dict], iteration: int = 1) -> str:
        print(f"\n📝 [Agent] Generating a new layout plan from historical experience (iteration #{iteration})...")
        
        objects = self.current_task["objects"]
        similar_exp = self.current_task.get("similar_experiences", [])
        
        current_canvas_base64 = self._get_current_canvas_for_vlm()
        
        object_svg_map = {}
        for obj_id, state in self.object_states.items():
            object_svg_map[obj_id] = {
                "description": state.get("description", ""),
                "svg_path": state.get("svg_path", ""),
                "bbox": {
                    "x": state.get("x", 0),
                    "y": state.get("y", 0),
                    "w": state.get("raw_w", 0),
                    "h": state.get("raw_h", 0)
                },
                "grid_position": self._pixel_to_grid(
                    state.get("x", 0), 
                    state.get("y", 0)
                ) if state.get("x") is not None else None
            }
        
        history_reflection  = self._build_history_reflection(attempt_history, iteration)

        boundary_info = self._check_boundaries_with_details()
        grid_context = self._get_grid_context()
        
        boundary_context = ""
        if boundary_info["out_of_bounds"]:
            boundary_context = "\n=== ⚠️ Current objects outside the canvas ===\n"
            for obj in boundary_info["details"]:
                grid_col, grid_row = self._pixel_to_grid(
                    obj.get("center_x", 0), 
                    obj.get("center_y", 0)
                )
                boundary_context += f"- {obj['object_id']}: grid ({grid_col}, {grid_row})\n"
        
        layout_code = self.sketch_app.call_vlm_for_layout_code_with_history(
            concept=self.current_task["concept"],
            objects=objects,
            current_canvas_base64=current_canvas_base64,
            object_svg_map=object_svg_map,
            similar_experiences=similar_exp,
            available_operations=self.available_operations,
            additional_context=grid_context + "\n" + boundary_context,
            history_context=history_reflection   # 🆕 Pass attempt history
        )
        
        self.current_task["planned_code"] = layout_code
        return layout_code


    def _build_history_context(self, attempt_history: List[Dict]) -> str:
        if not attempt_history:
            return "No previous attempts yet."
        
        context = "\n=== Previous Attempts (for learning and reference) ===\n"
        
        recent_attempts = attempt_history[-5:]
        
        for attempt in recent_attempts:
            context += f"\nAttempt #{attempt['attempt']}:\n"
            context += f"  Result: {attempt['result']}\n"
            
            if 'score' in attempt:
                context += f"  Score: {attempt['score']:.2f}\n"
            
            if 'issues' in attempt and attempt['issues']:
                context += f"  Issues:\n"
                for issue in attempt['issues'][:3]:
                    issue_type = issue.get('type', 'unknown')
                    issue_msg = issue.get('message', '')
                    if isinstance(issue, dict):
                        context += f"    - {issue_type}: {issue_msg}\n"
                    else:
                        context += f"    - {issue}\n"
            
            if 'warnings' in attempt and attempt['warnings']:
                context += f"  Warnings:\n"
                for warning in attempt['warnings'][:2]:
                    if isinstance(warning, dict):
                        context += f"    - {warning.get('type', 'unknown')}\n"
                    else:
                        context += f"    - {warning}\n"
            
            if attempt['result'] == 'failed':
                context += f"  Lesson: this plan is not viable and needs to be redesigned\n"
            elif attempt['result'] == 'passed':
                context += f"  Lesson: this plan is viable.\n"
            elif attempt['result'] == 'execution_failed':
                context += f"  Lesson: the code contains a syntax error; check function names and arguments\n"
        
        context += "\nLearn from the history above, analyze why previous attempts failed, and redesign a viable layout with corresponding improvements."
        context += "Do not copy failed code; understand the problems and redesign the layout."
        context += "Each iteration should be a new attempt."

        return context


    def _get_grid_context(self) -> str:
        return f"""
    === GRID INFORMATION ===
    - Grid size: {self.res}x{self.res} cells
    - Canvas size: {self.grid_size}x{self.grid_size} pixels
    - Cell size: {self.cell_size} pixels
    - Grid center: ({self.res//2}, {self.res//2})
    - IMPORTANT: Use grid coordinates (0 to {self.res-1}) in move() calls!
    - Grid (0, 0) is top-left corner
    - Grid ({self.res}, {self.res}) is bottom-right corner
    """

    def _reset_to_initial_layout(self):
        import random
        random.seed(42)  
        
        canvas_size = self.grid_size
        margin = self.cell_size * 0.5
        
        total = len(self.object_states)
        if total == 0:
            print("⚠️ No objects to layout")
            return
        
        sorted_items = list(self.object_states.items())
        
        for idx, (obj_id, state) in enumerate(sorted_items):
            # Random size
            raw_w = state.get("raw_w", 100)
            raw_h = state.get("raw_h", 100)
            
            scale_factor = random.uniform(0.3, 1.0)
            state["scale"] = scale_factor
            
            scaled_w = raw_w * scale_factor
            scaled_h = raw_h * scale_factor
            
            max_x = canvas_size - scaled_w - margin
            max_y = canvas_size - scaled_h - margin
            
            min_x = margin
            min_y = margin
            
            if max_x < min_x:
                max_x = min_x
            if max_y < min_y:
                max_y = min_y
            
            x = random.uniform(min_x + scaled_w/2, max_x + scaled_w/2)
            y = random.uniform(min_y + scaled_h/2, max_y + scaled_h/2)
            
            state["x"] = x
            state["y"] = y
            
            # Reset operation history
            self.current_task["operation_history"] = []
            
            grid_col, grid_row = self._pixel_to_grid(x, y)

    def _build_summary_context(self, attempt_history: List[Dict], verification: Dict) -> str:
        context = f"""
    === Task Concept ===
    {self.current_task.get('concept', '')}

    === Object List ===
    {self.current_task.get('objects', [])}

    === Attempt History ===
    """
        
        for attempt in attempt_history:
            context += f"\nAttempt #{attempt.get('attempt', 0)}:\n"
            context += f"  Result: {attempt.get('result', 'unknown')}\n"
            if attempt.get('score') is not None:
                context += f"  Score: {attempt.get('score', 0):.2f}\n"
            
            if attempt.get('result') == 'failed':
                issues = attempt.get('issues', [])
                if issues:
                    context += "  Failure reasons:\n"
                    for issue in issues:
                        if isinstance(issue, dict):
                            context += f"    - {issue.get('type', 'unknown')}: {issue.get('message', '')}\n"
            
            code = attempt.get('code', '')
            if code:
                import re
                moves = re.findall(r"move\([^)]+\)", code)
                scales = re.findall(r"scale\([^)]+\)", code)
                layers = re.findall(r"layered\([^)]+\)", code)
                if moves or scales or layers:
                    context += "  Key operations:\n"
                    for move in moves[:3]:
                        context += f"    - {move}\n"
        
        context += f"""
    === Final Result ===
    Verification passed.
    Overall score: {verification.get('overall_score', 0):.2f}
    """
        
        return context

    def _get_previous_canvas_steps_grid(self, last_attempt: Dict, iteration: int, save_png: bool = True) -> Optional[str]:
        attempt_num = last_attempt.get('attempt', 0)
        previous_iter = attempt_num
        
        if not hasattr(self, 'visualization') or "iteration_steps" not in self.visualization:
            return None
        
        steps = self.visualization["iteration_steps"].get(previous_iter, [])
        if not steps:
            return None
        
        from PIL import Image, ImageDraw, ImageFont
        import io
        import base64
        
        CANVAS_SIZE = 300
        CODE_HEIGHT = 32
        PADDING = 10
        COLS = 4
        FONT_SIZE = 11
        
        cell_w = CANVAS_SIZE + PADDING * 2
        cell_h = CANVAS_SIZE + CODE_HEIGHT + PADDING * 2
        total_steps = len(steps)
        rows = (total_steps + COLS - 1) // COLS
        total_w = COLS * cell_w + PADDING
        total_h = rows * cell_h + PADDING + 40  
        
        grid = Image.new('RGB', (total_w, total_h), color='white')
        draw = ImageDraw.Draw(grid)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            small_font = ImageFont.truetype("arial.ttf", 11)
            title_font = ImageFont.truetype("arial.ttf", 16)
            font_bold = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            font_bold = font
        
        title = f"Layout Generation Steps - Iteration {iteration}"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text((total_w//2 - title_width//2, 10), title, fill=(50, 50, 150), font=title_font)  # Blue color
        
        draw.line([(20, 35), (total_w - 20, 35)], fill=(200, 200, 200), width=1)
        
        operations = []
        if steps and steps[0].get("code"):
            lines = steps[0]["code"].split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and any(op in line for op in ['move(', 'scale(', 'layered(', 'rotate(']):
                    operations.append(line)
        
        for idx, step in enumerate(steps):
            row = idx // COLS
            col = idx % COLS
            if row % 2 == 1:
                col = COLS - 1 - col
            
            x = PADDING + col * cell_w
            y = 40 + PADDING + row * cell_h
            
            draw.rectangle([(x-2, y-2), (x + cell_w + 2, y + cell_h + 2)], 
                        outline=(200, 200, 200), width=1)
            
            png_path = step.get("png_path")
            if png_path and os.path.exists(png_path):
                try:
                    canvas = Image.open(png_path)
                    if canvas.mode != 'RGBA':
                        canvas = canvas.convert('RGBA')
                    white_bg = Image.new('RGBA', canvas.size, (255, 255, 255, 255))
                    white_bg.paste(canvas, (0, 0), canvas)
                    canvas = white_bg.convert('RGB')
                    canvas = canvas.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.LANCZOS)
                except:
                    canvas = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), color='white')
                    draw_canvas = ImageDraw.Draw(canvas)
                    draw_canvas.text((CANVAS_SIZE//2 - 30, CANVAS_SIZE//2 - 8), f"Step {idx+1}", fill=(200, 200, 200), font=small_font)
            else:
                canvas = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), color='white')
                draw_canvas = ImageDraw.Draw(canvas)
                draw_canvas.text((CANVAS_SIZE//2 - 30, CANVAS_SIZE//2 - 8), f"Step {idx+1}", fill=(200, 200, 200), font=small_font)
            
            grid.paste(canvas, (x + PADDING, y + PADDING))
            
            badge_x = x + PADDING + 5
            badge_y = y + PADDING + 5
            
            draw.ellipse([(badge_x, badge_y), (badge_x + 22, badge_y + 22)], 
                        fill=(60, 80, 200))  
            draw.text((badge_x + 6, badge_y + 3), f"{idx+1}", fill='white', font=font_bold)
            
            if idx == 0:
                code_text = "Initial Layout"
                color = (50, 50, 50)  
            elif idx - 1 < len(operations):
                code_text = operations[idx - 1][:40] + ('...' if len(operations[idx - 1]) > 40 else '')
                color = (50, 50, 50)
            else:
                code_text = step.get("operation", f"Step {idx+1}")
                color = (50, 50, 50)
            
            text_x = x + PADDING + 5
            text_y = y + PADDING + CANVAS_SIZE + 6
            draw.text((text_x, text_y), code_text, fill=color, font=font)
        
        if save_png:
            step_dir = os.path.join(self.args.output_dir, "reflection_images")
            os.makedirs(step_dir, exist_ok=True)
            png_path = os.path.join(step_dir, f"iter_{iteration}_canvas_all.png")
            grid.save(png_path, format="PNG", quality=95, optimize=True)

        buffered = io.BytesIO()
        grid.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _build_history_reflection(self, attempt_history: List[Dict], iteration: int) -> str:
        if not attempt_history:
            return "This is the first attempt; no historical experience is available."
        
        last_attempt = attempt_history[-1]

        if last_attempt.get('result') == 'passed':
            return "✅ The previous attempt succeeded. Continue in this direction."
        
        grid_image_base64 = self._get_previous_canvas_steps_grid(last_attempt, iteration-1)

        reflection_context = self._build_reflection_context(last_attempt)
        
        existing_experiences = self.kb.get_formatted_experiences() if hasattr(self, 'kb') else "No previous experiences"
        
        reflection_prompt = f"""
        You are a layout optimization expert. Carefully inspect the collage below, which shows every execution step from the previous attempt.

        === Image Description ===
        - Each tile shows the canvas state after one operation.
        - The operation code executed at that step is shown below the canvas.
        - The collage starts from step 1 and uses a snake layout with four tiles per row.
        - You can observe the complete evolution from the initial to the final layout.

        === Previous Attempt ===
        {reflection_context}

        === Existing Experiences (reference only) ===
        {existing_experiences}

        === Task ===
        Carefully inspect each change in the collage and analyze freely:

        1. What do you observe in the collage, and what changes at each step?
        - What is the initial state (first image)?
        - What operation was executed at each step? Refer to the code below each canvas.
        - What is wrong with the final state (last image)?
        - At which step did the problem begin?

        2. Why did the previous attempt fail, and what is the root cause?
        - Was it caused by a specific operation?
        - Or was the overall strategy flawed?

        3. What concrete improvements would you recommend?
        - Which objects should be moved, and to what positions? Provide specific grid coordinates.
        - Which object scales should be adjusted?
        - Which object layers should be changed?
        - Which step should be modified or removed?

        4. Can any existing experience be reused, or is a new strategy required?

        Respond in clear natural language without using a rigid template.

        === ⚠️ Important: Experience Extraction ===
        At the end of your reflection, if no similar existing experience is available, summarize one reusable experience using the format below:

        Experience Summary:[Describe the experience in one sentence]
        Applicable Scenario:[When this experience is useful]
        Key Points:[Concrete operation suggestions, numbered]

        If the current attempt provides no reusable experience, output:
        No New Experience
        """
        
        try:
            if grid_image_base64:
                reflection = self.sketch_app.query_multimodal_model_with_image_and_text(
                    system_prompt="You are a layout optimization expert who learns from failures and provides constructive suggestions. Carefully inspect every step in the collage.",
                    user_prompt=reflection_prompt,
                    image_base64=grid_image_base64
                )
            else:
                # Use text-only mode when no image is available
                reflection = self.sketch_app.query_model_text_only(
                    system_prompt="You are a layout optimization expert who learns from failures and provides constructive suggestions.",
                    user_prompt=reflection_prompt
                )
            
            reflection = reflection.strip()
            
            extracted_experience = self._extract_experience_from_reflection(reflection)
            if extracted_experience:
                similar_exp = self._find_similar_experience(extracted_experience)
                if similar_exp:
                    self.kb.update_experience(similar_exp["id"], {
                        "content": extracted_experience.get("content", ""),
                        "applicable_scenario": extracted_experience.get("applicable_scenario", "General"),
                        "verified_count": similar_exp.get("verified_count", 0) + 1
                    })
                    print(f"  ✅ Updated existing experience: {similar_exp['id']}")
                else:
                    self.kb.add_experience(extracted_experience)
                    print(f"  ✅ Added a new experience to the knowledge base")
            
            if existing_experiences and existing_experiences != "No previous experiences":
                reflection += f"\n\n📚 Existing experiences for reference:\n{existing_experiences}"
            
            return f"\n=== 🔍 Agent Reflection ===\n\n{reflection}\n"
            
        except Exception as e:
            print(f"⚠️ VLM reflection failed: {e}, using fallback")
            return self._fallback_reflection(last_attempt)


    def _extract_experience_from_reflection(self, reflection: str) -> Optional[Dict]:
        import re
        
        if "No New Experience" in reflection:
            return None
        
        patterns = [
            r'Experience Summary:\s*(.+?)(?=\n\s*Applicable Scenario|$)',
            r'Reusable Experience:\s*(.+?)(?=\n\s*Applicable Scenario|$)',
            r'Experience:\s*(.+?)(?=\n\s*Applicable Scenario|$)',
            r'Reusable Experience:\s*(.+?)(?=\n\s*Applicable Scenario|$)',
        ]
        
        content = None
        for pattern in patterns:
            match = re.search(pattern, reflection, re.DOTALL)
            if match:
                content = match.group(1).strip()
                # Remove possible trailing markers
                content = re.sub(r'\n\s*Applicable Scenario.*$', '', content, flags=re.DOTALL)
                break
        
        if not content:
            paragraphs = reflection.split('\n\n')
            for para in reversed(paragraphs):
                if 'Experience' in para and len(para) > 20:
                    if 'Applicable Scenario' in para:
                        match = re.search(r'^(.*?)(?=\s*Applicable Scenario)', para, re.DOTALL)
                        if match:
                            content = match.group(1).strip()
                            break
                    else:
                        content = para.strip()
                        break
        
        if not content or len(content) < 20:
            return None
        
        scenario_match = re.search(r'Applicable Scenario:\s*(.+?)(?=\n\s*Key Points|\n\s*$)', reflection, re.DOTALL)
        applicable_scenario = scenario_match.group(1).strip() if scenario_match else "General"
        
        key_points = []
        key_points_match = re.search(r'Key Points:\s*(.+?)(?=\n\s*Experience Summary|\n\s*$)', reflection, re.DOTALL)
        if key_points_match:
            points_text = key_points_match.group(1).strip()
            for line in points_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    key_points.append(line.lstrip('0123456789.-• ').strip())
            if not key_points:
                for sent in points_text.split('.'):
                    sent = sent.strip()
                    if sent and len(sent) > 10:
                        key_points.append(sent)
        
        if not key_points and len(content) > 30:
            # Try splitting into sentences
            for sent in content.split('.')[:3]:
                sent = sent.strip()
                if sent and len(sent) > 10:
                    key_points.append(sent)
        
        if not key_points:
            key_points = [content[:100]]
        
        experience = {
            "content": content,
            "applicable_scenario": applicable_scenario,
            "key_points": key_points,
            "verified_count": 1,
            "failed_count": 0
        }
        
        return experience


    def _find_similar_experience(self, new_exp: Dict) -> Optional[Dict]:
        existing_exps = self.kb.get_experiences()
        new_content = new_exp.get("content", "").lower()
        
        for exp in existing_exps:
            existing_content = exp.get("content", "").lower()
            new_words = set(new_content.split())
            existing_words = set(existing_content.split())
            common_words = new_words & existing_words
            
            if len(common_words) >= 5 and len(common_words) / max(len(new_words), len(existing_words)) > 0.2:
                return exp
        
        return None


    def _get_previous_canvas_for_reflection(self, last_attempt: Dict) -> List[Dict]:
        attempt_num = last_attempt.get('attempt', 0)
        previous_iter = attempt_num - 1
        
        steps_info = []
        
        if hasattr(self.visualization, "iteration_steps"):
            steps = self.visualization["iteration_steps"].get(previous_iter, [])
            
            for s in steps:
                png_path = s.get("png_path")
                if png_path and os.path.exists(png_path):
                    import base64
                    with open(png_path, "rb") as f:
                        png_base64 = base64.b64encode(f.read()).decode('utf-8')
                    
                    steps_info.append({
                        "step": s.get("step", 0),
                        "operation": s.get("operation", ""),
                        "png_base64": png_base64,
                        "code": s.get("code", ""),
                        "object_states": s.get("object_states", {})
                    })
        
        return steps_info


    def _build_reflection_context(self, last_attempt: Dict) -> str:
        context = f"""
        Attempt: #{last_attempt.get('attempt', 0)}
        Result: {last_attempt.get('result', 'unknown')}
        Score: {last_attempt.get('score', 0):.2f}

        === Verification Issues ===
        """
        issues = last_attempt.get('issues', [])
        if issues:
            for issue in issues:
                if isinstance(issue, dict):
                    context += f"- Type: {issue.get('type', 'unknown')}\n"
                    context += f"  Object: {issue.get('object_id', 'unknown')}\n"
                    context += f"  Description: {issue.get('message', '')}\n"
                    if issue.get('details'):
                        context += f"  Details: {issue.get('details')}\n"
                    context += "\n"
        else:
            context += "（No specific issues recorded）\n"

        context += "\n=== Warnings ===\n"
        warnings = last_attempt.get('warnings', [])
        if warnings:
            for warning in warnings[:3]:
                if isinstance(warning, dict):
                    context += f"- {warning.get('message', str(warning))}\n"
                else:
                    context += f"- {warning}\n"
        else:
            context += "（No warnings）\n"

        context += "\n=== Previous Attempt Code ===\n"
        code = last_attempt.get('code', '')
        if code:
            lines = code.split('\n')
            relevant_lines = []
            for line in lines:
                if any(op in line for op in ['move(', 'scale(', 'layered(', 'rotate(']):
                    relevant_lines.append(line.strip())
            
            if relevant_lines:
                context += "Key operations:\n"
                for line in relevant_lines[:10]:
                    context += f"  {line}\n"
                if len(relevant_lines) > 10:
                    context += f"  ... ({len(relevant_lines)} operations total)\n"
            else:
                context += code[:500] + "...\n"

        context += "\n=== Current Object States ===\n"
        for obj_id, state in self.object_states.items():
            context += f"- {obj_id}:\n"
            context += f"  Description: {state.get('description', '')}\n"
            context += f"  position: ({state.get('x', 0):.1f}, {state.get('y', 0):.1f})\n"
            context += f"  size: {state.get('raw_w', 0):.1f}x{state.get('raw_h', 0):.1f}\n"
            if state.get('svg_path'):
                context += f"  SVG path: {state.get('svg_path')}\n"

        return context

    def _summarize_experience(self, attempt_history: List[Dict], verification: Dict):
        print("\n📝 Summarizing experience...")
        
        summary_context = self._build_summary_context(attempt_history, verification)
        
        existing_experiences = self.kb.get_formatted_experiences()
        
        summary_prompt = f"""
    You are an expert at summarizing layout experience. Based on the complete attempt history below, summarize one reusable experience.

    === Existing Experiences (may be revised or extended) ===
    {existing_experiences}

    === Current Attempt History ===
    {summary_context}

    === Task ===
    Summarize one experience using the following format:
    1. Experience description: one or two concise sentences describing what to do and when
    2. Applicable scenario: what kind of scene this experience applies to
    3. Key points: concrete operational suggestions

    Notes:
    - If a similar experience already exists, revise or extend it instead of creating a duplicate.
    - If an existing experience does not apply to the current scene, clearly state what needs adaptation.
    - Keep the experience concrete and actionable rather than overly abstract.

    Return the following JSON format:
    {{
        "content": "Experience description",
        "applicable_scenario": "Applicable Scenario",
        "key_points": ["point 1", "point 2", "point 3"],
        "related_to": "Use the related experience ID if applicable; otherwise use 'new'"
    }}
    """
        
        try:
            result_str = self.sketch_app.query_model_text_only(
                system_prompt="You are an expert at summarizing layout experience. Output JSON only.",
                user_prompt=summary_prompt
            )
            
            import json
            import re
            json_match = re.search(r'\{.*\}', result_str, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                related_to = result.get('related_to', 'new')
                if related_to != 'new':
                    success = self.kb.update_experience(related_to, {
                        "content": result.get('content', ''),
                        "applicable_scenario": result.get('applicable_scenario', ''),
                        "key_points": result.get('key_points', []),
                        "verified_count": self._get_experience_verified_count(related_to) + 1
                    })
                    if success:
                        print(f"✅ Updated experience #{related_to}")
                    else:
                        self.kb.add_experience(result)
                        print("✅ Added new experience")
                else:
                    self.kb.add_experience({
                        "content": result.get('content', ''),
                        "applicable_scenario": result.get('applicable_scenario', ''),
                        "key_points": result.get('key_points', []),
                        "verified_count": 1,
                        "failed_count": 0
                    })
                    print("✅ Added new experience")
            else:
                print("⚠️ Could not parse the experience summary; skipping")
                
        except Exception as e:
            print(f"⚠️ Experience summarization failed: {e}")
            self._manual_summarize(attempt_history)


    def _get_experience_verified_count(self, exp_id: str) -> int:
        for exp in self.kb.knowledge["experiences"]:
            if exp.get("id") == exp_id:
                return exp.get("verified_count", 0)
        return 0


    def _manual_summarize(self, attempt_history: List[Dict]):
        failure_reasons = []
        for attempt in attempt_history:
            if attempt.get('result') == 'failed':
                issues = attempt.get('issues', [])
                for issue in issues:
                    if isinstance(issue, dict):
                        msg = issue.get('message', '')
                        if msg:
                            failure_reasons.append(msg)
        
        from collections import Counter
        if failure_reasons:
            common_reason = Counter(failure_reasons).most_common(1)
            if common_reason:
                reason = common_reason[0][0]
                exp = {
                    "content": f"Avoid {reason}, and keep all objects within the canvas boundary",
                    "applicable_scenario": "Scenes with large objects",
                    "key_points": [
                        "Use a conservative scale factor (0.5-0.7)",
                        "Place objects within grid coordinates 5-58",
                        "Use layered() to manage depth ordering"
                    ],
                    "verified_count": 1,
                    "failed_count": 0
                }
                self.kb.add_experience(exp)
                print("✅ Generated fallback experience")


    def extract_layout_functions(self, text: str) -> str:
        
        pattern = r"(?:move|scale|rotate|delete|layered|modify)\([^)]*\)"

        extracted = re.findall(pattern, text)

        return "\n".join(extracted)




    def run_loop(self, concept: str) -> Dict:
        print(f"\n{'='*60}")
        print(f"🚀 Starting SketchDirector sketch generation loop")
        print(f"Concept: {concept}")
        print(f"Grid: {self.res}x{self.res} cells, {self.grid_size}x{self.grid_size}px")
        print(f"{'='*60}")
        
        scene_info = self.analyze_scene(concept)
        objects = scene_info["objects"]
        
        sketch_results = self.generate_raw_sketches_parallel(objects)
        successful_sketches = [r for r in sketch_results if r["success"]]
        print(f"✅ Successfully generated {len(successful_sketches)}/{len(objects)} Raw Sketches")
        
        
        attempt_history = []
        best_result = None
        best_score = -1
        best_object_states = None
        best_operation_history = None
        
        iteration = 0
        max_iterations = self.current_task.get("max_iterations", 10)
        
        from .experience_knowledge import ExperienceKnowledge
        self.kb = ExperienceKnowledge("experience_knowledge.json")

        while iteration < max_iterations:
            iteration += 1

            print(f"🔄 Agent iteration #{iteration}/{max_iterations}")

            import copy
            self._previous_object_states = copy.deepcopy(self.object_states)
        
            print(f"\n📐 Resetting to the initial random layout (iteration #{iteration})...")
            self._reset_to_initial_layout() 
            
            self.visualization["step_counter"] = 0
            
            self._render_current_canvas("initial_layout", iteration)
            
            print(f"\n📝 Generating a new layout plan (attempt #{iteration})...")

            layout_code = self.plan_layout_code_with_history(attempt_history, iteration)
            print(f"📊 Generated code:\n{layout_code[:500]}...")
            layout_code = self.extract_layout_functions(layout_code)

            exec_result = self.execute_layout_code(layout_code, iteration)
            
            if not exec_result["success"]:
                print(f"❌ Execution failed: {exec_result.get('error', 'Unknown error')}")
                attempt_history.append({
                    "attempt": iteration,
                    "code": layout_code,
                    "result": "execution_failed",
                    "error": exec_result.get('error', ''),
                    "canvas_base64": self._get_current_canvas_for_vlm()
                })
                continue
            
            verification = self.verify_layout()
            

            current_score = verification.get('overall_score', 0)
            attempt_history.append({
                "attempt": iteration,
                "code": layout_code,
                "result": "passed" if verification['passed'] else "failed",
                "score": current_score,
                "issues": verification.get('issues', []),
                "warnings": verification.get('warnings', []),
                "canvas_base64": self._get_current_canvas_for_vlm()
            })
            
            if current_score > best_score:
                best_score = current_score
                best_result = {
                    "code": layout_code,
                    "verification": verification,
                    "attempt": iteration
                }
                import copy
                best_object_states = copy.deepcopy(self.object_states)
                best_operation_history = copy.deepcopy(self.current_task["operation_history"])
                print(f"  ⭐ New best plan! Score: {best_score:.2f}")
            
            if verification["passed"]:
                print("✅ Layout verification passed.")
                self._summarize_experience(attempt_history, verification)
                last_attempt = attempt_history[-1]
                grid_image_base64 = self._get_previous_canvas_steps_grid(last_attempt, iteration)

                self.memory.record_success({
                    "scene_signature": self.current_task.get("scene_signature", ""),
                    "operation_sequence": self.current_task["operation_history"],
                    "key_parameters": {
                        "objects": list(self.object_states.keys()),
                        "final_positions": {k: {"x": v["x"], "y": v["y"]} 
                                        for k, v in self.object_states.items()}
                    },
                    "score": verification.get("overall_score", 0.8),
                    "attempts": iteration
                })
                
                video_path = self._generate_evolution_video()
                final_result = self._build_final_canvas()
                final_result["evolution_video"] = video_path
                final_result["total_steps"] = len(self.visualization["step_images"])
                final_result["attempts"] = iteration
                final_result["best_score"] = best_score
                
                return final_result
            
            if current_score >= 0.85 and len(verification['issues']) == 0:
                print(f"✅ Layout quality is sufficient (score={current_score:.2f}), accepting the current plan")
                last_attempt = attempt_history[-1]
                grid_image_base64 = self._get_previous_canvas_steps_grid(last_attempt, iteration)

                self.memory.record_success({
                    "scene_signature": self.current_task.get("scene_signature", ""),
                    "operation_sequence": self.current_task["operation_history"],
                    "key_parameters": {
                        "objects": list(self.object_states.keys()),
                        "final_positions": {k: {"x": v["x"], "y": v["y"]} 
                                        for k, v in self.object_states.items()}
                    },
                    "score": current_score,
                    "attempts": iteration
                })
                
                video_path = self._generate_evolution_video()
                final_result = self._build_final_canvas()
                final_result["evolution_video"] = video_path
                final_result["total_steps"] = len(self.visualization["step_images"])
                final_result["attempts"] = iteration
                final_result["best_score"] = best_score
                return final_result
            
            # Use the best result when the maximum number of iterations is reached
            if iteration >= max_iterations:
                print(f"⚠️ Maximum iterations reached; using the best plan (score={best_score:.2f})")
                last_attempt = attempt_history[-1]
                grid_image_base64 = self._get_previous_canvas_steps_grid(last_attempt, iteration)

                if best_result and best_object_states:
                    print(f"  📌 Restored state from attempt #{best_result['attempt']}")
                    self.object_states = copy.deepcopy(best_object_states)
                    self.current_task["operation_history"] = copy.deepcopy(best_operation_history)
                    # Re-render the best plan
                    self._render_current_canvas("best_solution", iteration, best_result['attempt'])
                
                if self.object_states:
                    video_path = self._generate_evolution_video()
                    result = self._build_final_canvas()
                    result["evolution_video"] = video_path
                    result["total_steps"] = len(self.visualization["step_images"])
                    result["attempts"] = iteration
                    result["best_score"] = best_score
                    result["best_attempt"] = best_result["attempt"] if best_result else None
                    return result
                else:
                    return {"success": False, "error": "Failed to generate a valid layout"}
        
        if self.object_states:
            video_path = self._generate_evolution_video()
            result = self._build_final_canvas()
            result["evolution_video"] = video_path
            result["total_steps"] = len(self.visualization["step_images"])
            result["attempts"] = iteration
            result["best_score"] = best_score
            return result
        else:
            return {"success": False, "error": "Failed to generate a valid layout"}
    
    
    def _generate_evolution_video(self, output_path: str = None) -> str:
        if not self.visualization["step_images"]:
            print("⚠️ No step images are available")
            return None
        
        if output_path is None:
            output_path = os.path.join(self.args.output_dir, "layout_evolution.mp4")
        
        print(f"🎬 Generating layout-evolution video: {output_path}")
        
        try:
            with imageio.get_writer(output_path, fps=2) as writer:
                for img_path in self.visualization["step_images"]:
                    if os.path.exists(img_path):
                        # 🔑 Open in RGBA mode first and composite onto a white background
                        img = Image.open(img_path)
                        
                        # Convert transparent background to white for RGBA images
                        if img.mode == 'RGBA':
                            # Create a white background
                            white_bg = Image.new('RGB', img.size, (255, 255, 255))
                            # Composite the RGBA image onto the white background
                            white_bg.paste(img, mask=img.split()[3])  # Use the alpha channel as the mask
                            img = white_bg
                        else:
                            # Convert directly to RGB when the image is not RGBA
                            img = img.convert('RGB')
                        
                        writer.append_data(np.array(img))
            
            print(f"✅ Video generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Video generation failed: {e}")
            return None
    
    def _build_final_canvas(self) -> Dict:
        print(f"\n🎨 [Agent] Building final canvas...")
        
        layout_history = []
        for obj_id, state in self.object_states.items():
            layout_history.append({
                "object_id": obj_id,
                "svg_path": state.get("svg_path"),
                "bg_svg_path": state.get("bg_svg_path"),
                "x": state.get("x", self.grid_size / 2),
                "y": state.get("y", self.grid_size / 2),
                "scale": state.get("scale", 1.0),
                "rotation": state.get("rotation", 0.0),
                "depth": state.get("depth", 1)
            })
        
        layout_history.sort(key=lambda item: item["depth"])
        
        svg_path = os.path.join(self.args.output_dir, "agent_final_canvas.svg")
        self.sketch_app.rebuild_official_canvas(
            layout_history,
            output_png_path=None,
            output_svg_path=svg_path,
            canvas_size=self.grid_size
        )
        
        png_path = os.path.join(self.args.output_dir, "agent_final_canvas.png")
        self._render_png_with_grid(svg_path, png_path)
        
        clean_png_path = os.path.join(self.args.output_dir, "agent_final_canvas_clean.png")
        try:
            import cairosvg
            cairosvg.svg2png(url=svg_path, write_to=clean_png_path, dpi=96, background_color='white')
        except:
            pass
        
        return {
            "success": True,
            "canvas_png": png_path,
            "canvas_svg": svg_path,
            "canvas_clean_png": clean_png_path,
            "object_states": self.object_states,
            "iteration_count": len(self.current_task["operation_history"])
        }
