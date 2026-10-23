import ast
import base64
import copy
import io
import json
import os
import re
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cairosvg
import torch
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw

from .. import pipeline_utils as utils
from ..prompts import (
    LAYOUT_CRITIC_PROMPT,
    LAYOUT_REVISER_PROMPT,
    SINGLE_LAYOUT_PROMPT,
    anchor_planner_prompt,
    gt_sketches_example,
    raw_sketch_prompt,
    raw_system_prompt,
    scene_planner_prompt,
    system_prompt,
)

try:
    from ..prompts import layout_planner_prompt
except ImportError:
    layout_planner_prompt = SINGLE_LAYOUT_PROMPT


class SketchApp:
    def __init__(self, args):
        # General
        self.path2save = args.path2save
        self.target_concept = args.concept_to_draw
        self.background_path = args.background_path
        self.object_res=args.res

        self.res = args.res
        self.num_cells = args.res
        self.cell_size = args.cell_size
        self.grid_size = (args.grid_size, args.grid_size)
        self.init_canvas, self.positions = utils.create_grid_image(res=args.res, cell_size=args.cell_size, header_size=args.cell_size)
        
        self.init_canvas_str = self.prepare_visual_input(self.background_path)
        self.cells_to_pixels_map = utils.cells_to_pixels(args.res, args.cell_size, header_size=args.cell_size)

        self.stroke_width = args.stroke_width
        
        self.composited_svg_elements = []
        
        
        self.cache = False
        self.max_tokens = 5000

        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        self.model = getattr(args, "model", "gpt-5.5")
        self.gen_mode = args.gen_mode
        self.seed_mode = args.seed_mode
        
    def create_object_canvas(self):
        canvas = Image.new("RGB", (self.object_res, self.object_res), "white")
        return canvas
    def prepare_visual_input(self, background_path):
        bg_img = Image.open(background_path).convert("RGB")

        grid_bg_img = utils.overlay_grid_on_image(
            bg_img,
            res=self.res
        )

        save_path = f"{self.path2save}/debug_grid_input.png"
        grid_bg_img.save(save_path)

        buffered = BytesIO()
        grid_bg_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"

    def append_object_to_scene_svg(
        self,
        object_svg_path,
        object_id
    ):
        with open(object_svg_path, "r") as f:
            svg_content = f.read()

        inner = re.search(
            r"<svg[^>]*>(.*)</svg>",
            svg_content,
            re.DOTALL
        ).group(1)

        group_svg = f'''
        <g id="{object_id}">
            {inner}
        </g>
        '''

        self.composited_svg_elements.append(group_svg)

        final_svg = f'''
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width="{self.grid_size[0]}"
            height="{self.grid_size[1]}"
            viewBox="0 0 {self.grid_size[0]} {self.grid_size[1]}"
        >

        {"".join(self.composited_svg_elements)}

        </svg>
        '''

        composited_svg_path = f"{self.path2save}/composited.svg"

        with open(composited_svg_path, "w") as f:
            f.write(final_svg)

        return composited_svg_path

    def call_llm(self, system_message, other_msg, additional_args):

        if isinstance(system_message, str):
            messages = [
                {"role": "system", "content":  f"please respond with JSON format.{system_message}"}
            ] + other_msg
        else:
            messages = system_message + other_msg

        if "top_k" in additional_args:
            del additional_args["top_k"]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            # max_tokens=self.max_tokens,
            # **additional_args
        )

        return response.choices[0].message.content

    def query_model_text_only(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    
    def define_input_to_llm(self, msg_history, init_canvas_str, msg):
        other_msg = msg_history 
        
        content = []
        
        if init_canvas_str is not None and init_canvas_str != "None":
       
            clean_base64 = init_canvas_str.strip().replace('\n', '').replace('\r', '')
            
            if not clean_base64.startswith('data:image/'):
                image_url = f"data:image/png;base64,{clean_base64}"
            else:
                image_url = clean_base64
                
                
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }})
        
        content.append({"type": "text", "text": msg})
        other_msg = other_msg + [{"role": "user", "content": content}]
        return other_msg
    
    def call_vlm_for_layout_code_with_history(self, concept, objects, similar_experiences=None, 
                                                available_operations=None, additional_context="",
                                                current_canvas_base64=None, object_svg_map=None,
                                                history_context=""):
        import base64
        import json

        objects_info = []
        for obj in objects:
            obj_id = obj.get("object_id") if isinstance(obj, dict) else getattr(obj, "object_id", str(obj))
            obj_desc = obj.get("description") if isinstance(obj, dict) else getattr(obj, "description", "")
            obj_depth = obj.get("depth") if isinstance(obj, dict) else getattr(obj, "depth", 1)
            obj_complexity = obj.get("visual_complexity", 1.0) if isinstance(obj, dict) else getattr(obj, "visual_complexity", 1.0)
            
            raw_w = obj.get("raw_w") if isinstance(obj, dict) else getattr(obj, "raw_w", None)
            raw_h = obj.get("raw_h") if isinstance(obj, dict) else getattr(obj, "raw_h", None)
            
            size_info = f" (size: {raw_w}x{raw_h})" if raw_w and raw_h else " (size determined after generation)"
            
            objects_info.append(
                f"  - ID: '{obj_id}', depth: {obj_depth}"
            )
        
        objects_context = "\n".join(objects_info)
        
        svg_mapping_context = ""
        if object_svg_map:
            svg_mapping_context = "\nObject-to-SVG mapping:\n"
            for obj_id, svg_info in object_svg_map.items():
                bbox = svg_info.get('bbox', {})
                svg_mapping_context += f"- {obj_id}: | "
                svg_mapping_context += f"bbox: ({bbox.get('x',0):.0f},{bbox.get('y',0):.0f}) "
                svg_mapping_context += f"{bbox.get('w',0):.0f}x{bbox.get('h',0):.0f}\n"
        
        ops_desc = ", ".join(available_operations) if available_operations else "move, scale, rotate, delete, layered, modify"
        
        system_prompt = (
            "You are an expert Python code generator for sketch composition.\n\n"
            
            "=== CRITICAL: LEARN FROM HISTORY ===\n"
            "You MUST analyze the provided history_context carefully.\n"
            "Your code must incorporate lessons learned from previous failures.\n"
            "Each attempt should be a FRESH design, NOT a patch of previous code.\n\n"
            
            f"=== GRID SYSTEM ===\n"
            f"- Grid: {self.res}x{self.res} cells (0 to {self.res-1})\n"
            f"- Canvas: {self.grid_size}x{self.grid_size} pixels\n"
            f"- Cell size: {self.cell_size} pixels\n"
            f"- Center: ({self.res//2}, {self.res//2})\n"
            f"- ALWAYS use grid coordinates (0-{self.res-1}) for move(), NEVER pixel coordinates!\n\n"
            
            "=== SAMPLE CODE STRUCTURE (showing how to use each function) ===\n"
            f"# Grid size: {self.res}x{self.res}, center at ({self.res//2}, {self.res//2})\n\n"

            "# move() - places an object at a specific grid position\n"
            f"move('object_id', 15, 32)  # Move to x15y32\n\n"

            "# scale() - changes the size of an object\n"
            f"scale('object_id', 0.7)  # Scale to 70% of original size\n\n"
            f"# Note: factor must be >= 0.3, values below will be clamped to 0.3\n\n"

            "# rotate() - rotates an object\n"
            f"rotate('object_id', 15)  # Rotate 15 degrees clockwise\n\n"

            "# layered() - adjusts the depth/layer of an object (z-order)\n"
            f"layered('object_id', 'bg')   # Send to background\n"
            f"layered('object_id', 'mid')  # Middle layer\n"
            f"layered('object_id', 'fg')   # Bring to foreground\n"
            f"layered('object_id', 'front')# Very front\n\n"

            "# delete() - removes an object from the scene\n"
            f"delete('object_id')  # Remove this object\n\n"

            "# modify() - regenerates a sketch with a new description\n"
            f"modify('object_id', 'a more detailed description')  # Regenerate\n\n"

            "# Note: You can combine multiple operations\n"
            f"# Example: move('table', 9, 30)\n"
            f"#          scale('table', 0.8)\n"
            f"#          layered('table', 'bg')\n\n"

            "# IMPORTANT: layering is about z-order (which objects appear in front of others)\n"
            "# - Objects with 'bg' layer appear behind all others\n"
            "# - Objects with 'front' layer appear in front of all others\n"
            "# - Use layering to create proper depth relationships between objects\n"
            
            "=== RULES ===\n"
            "1. Output ONLY executable Python code, no explanations, no markdown\n"
            "2. Use exact object IDs from the scene\n"
            "3. Keep objects at least 1 cell from edges (1 to {self.res-2})\n"
            "4. Avoid overlaps, consider visual balance\n"
            "5. Use comments to explain key decisions, especially how you learned from history\n"
            "6. DO NOT import modules or use print()\n"
        )

        has_history = history_context and len(history_context.strip()) > 0
        
        history_section = ""
        if has_history:
            history_section = f"""
    === 📖 HISTORY ANALYSIS (MUST READ AND APPLY) ===
    {history_context}

    Based on the above analysis, your NEW code MUST:
    1. Fix the specific problems identified in the history
    2. Apply the suggested improvements
    3. Learn from what went wrong
    4. NOT repeat the same mistakes

    """
        else:
            history_section = """
    === 📖 NO HISTORY YET ===
    This is the first attempt. Generate a reasonable initial layout.
    Make sure all objects are placed properly on the canvas.

    """
        
        scene_section = f"""
    === SCENE ===
    Concept: "{concept}"

    Objects:
    {objects_context}

    Current canvas state:
    {svg_mapping_context}

    === TASK ===
    Generate Python code that creates a well-composed layout for this scene.
    {f"Use the history analysis above to improve the layout." if has_history else "This is the first attempt, create a clean initial layout."}

    Key requirements:
    1. All objects must be placed within canvas bounds (grid 0-{self.res-1})
    2. Objects should not overlap unnecessarily
    3. Use appropriate scale factors
    4. Set proper layers (z-order)
    5. Consider the visual relationship between objects
    {f"6. Your code MUST reflect the lessons from history_context" if has_history else "6. Keep it simple and balanced"}
    """
        
        user_prompt = history_section + scene_section
        
        try:
            if current_canvas_base64:
                json_str = self.query_multimodal_model_with_image_and_text(
                    system_prompt, 
                    user_prompt, 
                    current_canvas_base64
                )
            else:
                json_str = self.query_model_text_only(system_prompt, user_prompt)
        except AttributeError:
            if current_canvas_base64:
                json_str = self.query_multimodal_model(system_prompt, user_prompt, current_canvas_base64)
            else:
                from PIL import Image
                import io
                blank_img = Image.new('RGB', (512, 512), color='white')
                buffered = io.BytesIO()
                blank_img.save(buffered, format="PNG")
                encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                json_str = self.query_multimodal_model(system_prompt, user_prompt, encoded_image)
        
        code_response = self._clean_code_response(json_str)
        
        return code_response


    def get_response_from_llm(
        self,
        msg,
        system_message,
        msg_history=[],
        init_canvas_str=None,
        prefill_msg=None,
        seed_mode="stochastic",
        stop_sequences=None,
        gen_mode="generation"
    ):  
        additional_args = {}
        if seed_mode == "deterministic":
            additional_args["temperature"] = 0.0
            additional_args["top_k"] = 1

        if self.cache:
            system_message = [{
                "type": "text",
                "text": system_message,
                "cache_control": {"type": "ephemeral"}
            }]

        other_msg = self.define_input_to_llm(msg_history, init_canvas_str, msg) 

        if gen_mode == "completion":
            if prefill_msg:
                other_msg = other_msg + [{"role": "assistant", "content": f"{prefill_msg}"}]
            
        if stop_sequences:
            additional_args["stop"]= [stop_sequences]
        else:
            additional_args["stop"]= ["</answer>"]
 
        content = self.call_llm(system_message, other_msg, additional_args)
        
        if gen_mode == "completion":
            other_msg = other_msg[:-1] # remove initial assistant prompt
            content = f"{prefill_msg}{content}" 

        if self.path2save is not None:
            system_message_json = [{"role": "system", "content": system_message}]
            new_msg_history = other_msg + [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                        }
                    ],
                }
            ]    
            with open(f"{self.path2save}/experiment_log.json", 'w') as json_file:
                json.dump(system_message_json + new_msg_history, json_file, indent=4)
            print(f"Data has been saved to [{self.path2save}/experiment_log.json]")
        return content

    def map_skeleton_to_original(nodes, target_res, original_w, original_h):
        mapped_nodes = {}
        for node_id, pos in nodes.items():
            
            norm_x = pos[0] / target_res
            norm_y = pos[1] / target_res
            
            final_x = norm_x * original_w
            final_y = (1.0 - norm_y) * original_h # Flip when the origin is at the top-left.
            
            mapped_nodes[node_id] = [final_x, final_y]
        return mapped_nodes


    def call_scene_planner(self):
        print("=== Stage 1: Scene Planning ===")

        msg_history = []

        other_msg = self.define_input_to_llm(
            msg_history=msg_history,
            init_canvas_str=self.init_canvas_str,
            msg=f"Target scene: {self.target_concept}"
        )

        try:

            response = self.call_llm(
                system_message=scene_planner_prompt,
                other_msg=other_msg,
                additional_args={}
            )

            print("FULL RESPONSE:")
            print(response)

            clean_response = re.sub(
                r"<think>.*?</think>",
                "",
                response,
                flags=re.DOTALL
            ).strip()

            clean_response = clean_response.replace("```json", "").replace("```", "").strip()

            match = re.search(
                r"(\[[\s\S]*\]|\{[\s\S]*\})",
                clean_response
            )

            if not match:
                raise ValueError("No JSON found in response")

            content = match.group(1)

            print("CONTENT:")
            print(content)


            return content

        except Exception as e:
            print(type(e))
            print(repr(e))

            if hasattr(e, "response"):
                print("response:")
                print(e.response)

            traceback.print_exc()
    def call_scene_planner_without_image(self):
        print("=== Stage 1: Scene Planning (Text Only) ===")
        
        msg = f"""
        Target scene: {self.target_concept}
        
        Please decompose this scene into individual objects.
        Return JSON array with fields: object_id, description, scene_role, layer
        """
        
        messages = [
            {"role": "system", "content": f"please respond with JSON format.{scene_planner_prompt}"},
            {"role": "user", "content": msg}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            return None
        except Exception as e:
            print(f"Error in text-only planner: {e}")
            return None

    def parse_scene_plan(self, content):
        try:
            scene_objects = json.loads(content)

            scene_objects = sorted(
                scene_objects,
                key=lambda x: x["depth"]
            )

            return scene_objects

        except Exception as e:
            print("Scene parsing failed")
            print(content)
            raise e


    def predict_object_layout(self, target_object, current_canvas_path, layout_history):
        print(f"  [Layout] Predicting layout for '{target_object['object_id']}'...")
        
        history_str = json.dumps(layout_history, indent=2) if layout_history else "None (First object)"

        user_content = (
            f"--- Placed Objects Layout History (Normalized) ---\n{history_str}\n\n"
            f"--- New Object to Insert ---\n"
            f"Object ID: {target_object['object_id']}\n"
            f"Object Description: {target_object['description']}\n"
            f"Incremental Context: {target_object['incremental_prompt']}\n\n"
            f"Predict the JSON output {{x, y, scale, rotation}} complying with boundary constraints."
        )

        layout_raw = self.call_multimodal_llm(
            system_prompt=SINGLE_LAYOUT_PROMPT,
            text_prompt=user_content,
            image_path=current_canvas_path
        )
        
        try:
            clean_json_str = layout_raw.replace("```json", "").replace("```", "").strip()
            layout_meta = json.loads(clean_json_str)
            
            if "rotation" not in layout_meta:
                layout_meta["rotation"] = 0.0
                print(f"  ⚠️ [Layout] Rotation not specified, defaulting to 0.0.")
        except Exception as e:
            print(f"  ⚠️ [Layout] Parse failed, error: {e}. Using safe fallback.")
            layout_meta = {"x": 0.5, "y": 0.5, "scale": 0.5, "rotation": 0.0}

        return layout_meta


    def call_multimodal_llm(self, system_prompt, text_prompt, image_path):
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,  
                response_format={ "type": "json_object" }, 
                messages=[
                    {
                        "role": "system",
                        "content": f"please respond with JSON format.{system_prompt}"
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": text_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1, 
            )
            
            result_text = response.choices[0].message.content
            return result_text

        except Exception as e:
            print(f"❌ [VLM API Error] Multimodal model request failed: {e}")
            return "{}"

    def call_layout_planner(
        self,
        object_info,
        current_scene_path
    ):

        print(f"=== Layout Planning: {object_info['object_id']} ===")

        if current_scene_path is None:
            current_canvas_str = self.init_canvas_str
        else:
            current_canvas_str = self.prepare_visual_input(
                current_scene_path
            )

        msg = f"""
    Object ID:
    {object_info['object_id']}

    Object Description:
    {object_info['description']}

    Scene Role:
    {object_info['scene_role']}
    """

        msg_history = []

        other_msg = self.define_input_to_llm(
            msg_history=msg_history,
            init_canvas_str=current_canvas_str,
            msg=msg
        )

        response = self.call_llm(
            system_message=layout_planner_prompt,
            other_msg=other_msg,
            additional_args={}
        )

        print(response)

        clean_response = re.sub(
            r"<think>.*?</think>",
            "",
            response,
            flags=re.DOTALL
        ).strip()

        clean_response = clean_response.replace("```json", "").replace("```", "").strip()

        match = re.search(
            r"(\[[\s\S]*\]|\{[\s\S]*\})",
            clean_response
        )

        if not match:
            raise ValueError("No JSON found in response")

        content = match.group(1)

        print("CONTENT:")
        print(content)

        return content
                
    def parse_layout(self, content):
        try:
            layout = json.loads(content)
            return layout

        except Exception as e:
            print("Layout parsing failed")
            print(content)
            raise e
            
    def call_anchor_planner_object(
        self,
        object_info
    ):

        print(f"=== Object Anchor Planning (64x64): {object_info['object_id']} ===")

        canvas = self.create_object_canvas()
        canvas_str = utils.image_to_str(canvas)

        msg = f"""
    Object ID:
    {object_info['object_id']}

    Object Description:
    {object_info['description']}

    IMPORTANT:
    - This is a single-object canonical canvas (64x64)
    - The object should occupy full canvas
    - No global layout constraints
    """

        other_msg = self.define_input_to_llm(
            msg_history=[],
            init_canvas_str=canvas_str,
            msg=msg
        )

        system_message = anchor_planner_prompt.replace(
            "{res}",
            str(self.object_res)
        )

        response = self.call_llm(
            system_message=system_message,
            other_msg=other_msg,
            additional_args={}
        )

        print(response)

        # Remove ``<think>`` content.
        clean_response = re.sub(
            r"<think>.*?</think>",
            "",
            response,
            flags=re.DOTALL
        ).strip()

        # Remove Markdown JSON fences.
        clean_response = clean_response.replace("```json", "").replace("```", "").strip()

        # Extract JSON in either list or object form.
        match = re.search(
            r"(\[[\s\S]*\]|\{[\s\S]*\})",
            clean_response
        )

        if not match:
            raise ValueError("No JSON found in response")

        content = match.group(1)

        print("CONTENT:")
        print(content)

        return content

    def parse_anchors(self, content):
        try:
            anchors = json.loads(content)
            return anchors

        except Exception as e:
            print("Anchor parsing failed")
            print(content)
            raise e

    def anchors_to_svg(
        self,
        anchors_json,
        object_id
    ):

        svg_elements = []

        for anchor in anchors_json["anchors"]:

            anchor_name = anchor["id"]
            print("anchor_name===", anchor_name)

            for point_str in anchor["points"]:

                px, py = utils.grid_point_to_pixel(
                    point_str,
                    self.cells_to_pixels_map
                )

                circle_svg = f'''
                <circle
                    cx="{px}"
                    cy="{py}"
                    r="{self.stroke_width}"
                    fill="black"
                    data-class="{anchor_name}"
                />
                '''

                svg_elements.append(circle_svg)

        svg_content = f'''
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width="{self.grid_size[0]}"
            height="{self.grid_size[1]}"
            viewBox="0 0 {self.grid_size[0]} {self.grid_size[1]}"
        >
            <g id="{object_id}">
                {"".join(svg_elements)}
            </g>
        </svg>
        '''

        svg_path = f"{self.path2save}/{object_id}.svg"

        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)

        return svg_path


    def render_svg_to_png(
        self,
        svg_path,
        png_path
    ):

        cairosvg.svg2png(
            url=svg_path,
            write_to=png_path,
            background_color="white"
        )

    def get_svg_bbox(self, svg_path):

        tree = ET.parse(svg_path)
        root = tree.getroot()

        xs=[]
        ys=[]

        paths = root.findall('.//{http://www.w3.org/2000/svg}path')

        for p in paths:
            d = p.get("d")

            if d is None:
                continue

            nums = re.findall(
                r'-?\d+\.?\d*',
                d
            )

            nums=list(map(float,nums))

            for i in range(0,len(nums)-1,2):
                xs.append(nums[i])
                ys.append(nums[i+1])


        return (
            min(xs),
            min(ys),
            max(xs),
            max(ys)
        )
        
    def overlay_object_on_canvas(
            self,
            bg_svg_path,
            current_canvas_png_path,
            current_canvas_svg_path,
            layout_meta,
            output_png_path,
            output_svg_path
    ):
        print("🎨 [Overlay] Embedding SVG asset into global canvas...")

        GLOBAL_SIZE = 780
        LOCAL_SIZE = 512

        target_x = layout_meta.get("x", 0.5)
        target_y = layout_meta.get("y", 0.5)

        scale = layout_meta.get("scale",0.5)
        rotation = layout_meta.get("rotation", 0.0)

        print("layout normalized:", target_x,target_y)
        # layout_meta:
        # normalized coordinate on 780 canvas
        pixel_cx = target_x * LOCAL_SIZE
        pixel_cy = target_y * LOCAL_SIZE

        print("780 canvas target:",pixel_cx,pixel_cy)


        xmin,ymin,xmax,ymax = self.get_svg_bbox(bg_svg_path)

        obj_cx = (xmin+xmax)/2
        obj_cy = (ymin+ymax)/2


        tx = pixel_cx - xmin * scale
        ty = pixel_cy - ymin * scale
        
        transform_str = (
            f"translate({tx},{ty}) "
            f"scale({scale})"
        )

        print("SVG bbox:", xmin,ymin,xmax,ymax)

        print(
        "bbox center:",
        obj_cx,obj_cy
        )

        print(
        "target:",
        pixel_cx,pixel_cy
        )

        if rotation != 0:
            transform_str += (
                f" rotate({rotation},{LOCAL_SIZE/2},{LOCAL_SIZE/2})"
            )


        print(
            f"[Overlay] target pixel: ({pixel_cx:.2f},{pixel_cy:.2f})"
        )

        print(
            f"[Overlay] transform: {transform_str}"
        )


        try:

            ET.register_namespace(
                '',
                "http://www.w3.org/2000/svg"
            )

            obj_tree = ET.parse(bg_svg_path)
            obj_root = obj_tree.getroot()

            obj_elements = list(obj_root)


            group = ET.Element(
                "{http://www.w3.org/2000/svg}g"
            )

            group.set(
                "transform",
                transform_str
            )


            for elem in obj_elements:
                obj_root.remove(elem)
                group.append(elem)

            if (
                current_canvas_svg_path
                and os.path.exists(current_canvas_svg_path)
            ):

                canvas_tree = ET.parse(
                    current_canvas_svg_path
                )

                canvas_root = canvas_tree.getroot()

                canvas_root.append(group)


            else:

                obj_root.clear()

                obj_root.set(
                    "width",
                    str(GLOBAL_SIZE)
                )

                obj_root.set(
                    "height",
                    str(GLOBAL_SIZE)
                )

                obj_root.set(
                    "viewBox",
                    f"0 0 {GLOBAL_SIZE} {GLOBAL_SIZE}"
                )

                obj_root.append(group)

                canvas_tree = obj_tree


            canvas_tree.write(
                output_svg_path,
                encoding="utf-8",
                xml_declaration=True
            )


        except Exception as e:

            print(
                f"❌ [Overlay SVG Error]: {e}"
            )

            return (
                current_canvas_png_path,
                current_canvas_svg_path
            )

        temp_fg_png = bg_svg_path.replace(
            ".svg",
            "_temp.png"
        )

        temp_svg = bg_svg_path.replace(
            ".svg",
            "_temp.svg"
        )


        try:

            temp_root = ET.Element(
                "{http://www.w3.org/2000/svg}svg",
                {
                    "width":str(GLOBAL_SIZE),
                    "height":str(GLOBAL_SIZE),
                    "viewBox":
                    f"0 0 {GLOBAL_SIZE} {GLOBAL_SIZE}"
                }
            )


            temp_group = ET.Element(
                "{http://www.w3.org/2000/svg}g"
            )

            temp_group.set(
                "transform",
                transform_str
            )


            for elem in list(obj_root):

                temp_group.append(
                    elem
                )


            temp_root.append(
                temp_group
            )


            ET.ElementTree(
                temp_root
            ).write(
                temp_svg,
                encoding="utf-8",
                xml_declaration=True
            )


            cairosvg.svg2png(
                url=temp_svg,
                write_to=temp_fg_png,
                output_width=GLOBAL_SIZE,
                output_height=GLOBAL_SIZE
            )


            foreground = Image.open(
                temp_fg_png
            ).convert("RGBA")


            if (
                current_canvas_png_path
                and os.path.exists(current_canvas_png_path)
            ):

                base_canvas = Image.open(
                    current_canvas_png_path
                ).convert("RGBA")

            else:

                base_canvas = Image.new(
                    "RGBA",
                    (GLOBAL_SIZE, GLOBAL_SIZE),
                    (255,255,255,255)
                )


            base_canvas.alpha_composite(
                foreground
            )


            base_canvas.convert("RGB").save(
                output_png_path,
                "PNG"
            )


        except Exception as e:

            print(
                f"❌ [Overlay PNG Error]: {e}"
            )


        finally:

            for p in [
                temp_fg_png,
                temp_svg
            ]:
                if os.path.exists(p):
                    os.remove(p)


        return (
            output_png_path,
            output_svg_path
        )

    def overlay_object(
        self,
        png_path,
        output_path
    ):

        foreground = Image.open(png_path).convert("RGBA")

        base = self.init_canvas.convert("RGBA")

        base.alpha_composite(foreground)

        base.save(output_path)



    def render_composited_scene(
        self,
        composited_svg_path
    ):

        output_png = f"{self.path2save}/composited.png"

        cairosvg.svg2png(
            url=composited_svg_path,
            write_to=output_png,
            background_color="white"
        )

        return output_png

    def generate_sketch_from_commands(self):
        description = self.target_concept
        sketching_commands = self.call_model_for_sketch_generation(description)
        model_strokes_svg = self.parse_model_to_svg(sketching_commands, description=description)
        
        words = self.target_concept.split()[:5]  
        short_name = "_".join(words).replace("/", "_").replace(".", "_")
    
        if len(short_name) > 50:
            short_name = short_name[:50]
        base_filename = f"{short_name}_{uuid.uuid4().hex[:6]}"
        
        svg_path = f"{self.path2save}/{base_filename}.svg"
        with open(svg_path, "w") as svg_file:
            svg_file.write(model_strokes_svg)

        cairosvg.svg2png(url=f"{self.path2save}/{base_filename}.svg", write_to=f"{self.path2save}/{base_filename}.png", background_color="white")
        
        output_png_path = f"{self.path2save}/{base_filename}_canvas.png"
        cairosvg.svg2png(url=f"{self.path2save}/{base_filename}.svg", write_to=output_png_path)
        foreground = Image.open(output_png_path)
        self.init_canvas.paste(Image.open(output_png_path), (0, 0), foreground) 
        self.init_canvas.save(output_png_path)
        return svg_path, description

    def call_model_for_sketch_generation(self, description):
        print("Calling LLM...")
        
        add_args = {}
        add_args["stop_sequences"] = ["</answer>"] 

        msg_history = []
        init_canvas_str = self.init_canvas_str 

        formatted_system_prompt = raw_sketch_prompt.format(
            res=self.res,
        )
        raw_first_prompt = raw_system_prompt.format(concept=description, res=self.res, gt_sketches_example=gt_sketches_example)
        

        all_llm_output = self.get_response_from_llm(
            msg=raw_first_prompt,
            system_message=formatted_system_prompt, # Use the formatted prompt.
            msg_history=msg_history,
            init_canvas_str=init_canvas_str,
            seed_mode=self.seed_mode,
            gen_mode=self.gen_mode,
            **add_args
        )

        all_llm_output += f"</answer>"
        return all_llm_output


    def parse_model_to_svg(self, model_rep_sketch, description="object"):
        import re
        import xml.etree.ElementTree as ET
        
        try:
            stroke_info_list = self._extract_stroke_info(model_rep_sketch)
            print("================ stroke_info_list ================")
            print(stroke_info_list)
            print("==============================================")
            semantic_labels = [info['id'] for info in stroke_info_list if info['id']]

            print("================ MODEL OUTPUT ================")
            print(model_rep_sketch)
            print("==============================================")

            strokes_list = []
            t_values = []

            for info in stroke_info_list:
                points = info.get("points", [])
                stroke_t_values = info.get("t_values", [])

                if not points:
                    continue

                normalized_points = [
                    f"x{max(1, min(int(x), self.res))}y{max(1, min(int(y), self.res))}"
                    for x, y in points
                ]

                if len(stroke_t_values) != len(normalized_points):
                    if len(normalized_points) == 1:
                        stroke_t_values = [0.0]
                    else:
                        stroke_t_values = [
                            i / (len(normalized_points) - 1)
                            for i in range(len(normalized_points))
                        ]

                strokes_list.append(normalized_points)
                t_values.append(stroke_t_values)

            if not strokes_list:
                print("[Warning] No valid strokes were available for SVG generation.")
                return (
                    f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'width="{self.grid_size}" height="{self.grid_size}" '
                    f'viewBox="0 0 {self.grid_size} {self.grid_size}"></svg>'
                )

            all_control_points = utils.get_control_points_t(
                strokes_list,
                t_values,
                self.cells_to_pixels_map,
            )

            raw_perfect_svg = utils.format_svg_t(
                all_control_points,
                dim=self.grid_size,
                stroke_width=self.stroke_width,
            )

            ET.register_namespace('', "http://www.w3.org/2000/svg")
            try:
                root = ET.fromstring(raw_perfect_svg)
            except Exception as e:
                print(f"⚠️ XML parsing failed: {e}")
                return raw_perfect_svg

            all_paths = root.findall('.//{http://www.w3.org/2000/svg}path')
            
            if len(all_paths) == 0:
                print(f"⚠️ No paths were generated; returning the original SVG")
                return raw_perfect_svg

            clean_description = description.replace(' ', '_').replace('-', '_')
            
            if len(semantic_labels) == len(all_paths):
                matched_labels = semantic_labels
            else:
                matched_labels = self._match_labels_to_paths(
                    all_paths, 
                    stroke_info_list, 
                    semantic_labels, 
                    clean_description
                )
            
            while len(matched_labels) < len(all_paths):
                idx = len(matched_labels) + 1
                matched_labels.append(f"{clean_description}_part_{idx}")
            
            if len(matched_labels) > len(all_paths):
                matched_labels = matched_labels[:len(all_paths)]

            new_global_g = ET.Element('{http://www.w3.org/2000/svg}g', {
                'id': str(clean_description),
                'stroke': 'black',
                'stroke-width': str(self.stroke_width),
                'fill': 'none',
                'stroke-linecap': 'round'
            })

            for idx, path_elem in enumerate(all_paths):
                stroke_class = matched_labels[idx].strip() if idx < len(matched_labels) else f"{clean_description}_part_{idx + 1}"
                
                stroke_class = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', stroke_class)
                stroke_class = re.sub(r'_+', '_', stroke_class)
                stroke_class = stroke_class.strip('_')
                
                path_elem.set('data-class', stroke_class)
                
                for attr in ['stroke', 'stroke-width', 'fill']:
                    if attr in path_elem.attrib:
                        del path_elem.attrib[attr]
                
                new_global_g.append(path_elem)

            grid_dim = self.grid_size
            if isinstance(grid_dim, (tuple, list)):
                width, height = grid_dim[0], grid_dim[1]
            else:
                width, height = grid_dim, grid_dim
                
            root.clear() 
            root.tag = '{http://www.w3.org/2000/svg}svg'
            root.set('width', str(width))
            root.set('height', str(height))
            root.set('viewBox', f"0 0 {width} {height}")
            root.append(new_global_g)

            final_svg_str = ET.tostring(root, encoding='utf-8').decode('utf-8')
            
            if '<path' not in final_svg_str:
                print(f"⚠️ Final SVG contains no paths; returning the original SVG")
                return raw_perfect_svg
            
            return final_svg_str
            
        except Exception as e:
            print(f"❌ parse_model_to_svg failed: {e}")
            import traceback
            traceback.print_exc()
            return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{self.grid_size}" height="{self.grid_size}" viewBox="0 0 {self.grid_size} {self.grid_size}">
        <rect width="100%" height="100%" fill="white"/>
        <text x="50%" y="50%" text-anchor="middle" font-size="20" fill="red">Error: {description}</text>
    </svg>'''

    def _extract_stroke_info(self, model_rep_sketch):
        """
        Extract stroke information from LLM JSON output.

        Supported formats:
        1. {"strokes": [...]}
        2. {"answer": {"strokes": [...]}}
        3. {"thinking": "...", "answer": {"strokes": [...]}}
        4. {"strokes": {"s1": {...}, "s2": {...}}}
        5. JSON wrapped by <answer>...</answer> or Markdown fences.
        """

        import json
        import re

        stroke_info_list = []

        try:
            clean_output = model_rep_sketch.strip()

            clean_output = re.sub(
                r"```(?:json)?",
                "",
                clean_output,
                flags=re.IGNORECASE,
            )
            clean_output = clean_output.replace("```", "")

            clean_output = re.sub(
                r"</?answer>",
                "",
                clean_output,
                flags=re.IGNORECASE,
            ).strip()

            first_brace = clean_output.find("{")
            last_brace = clean_output.rfind("}")

            if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
                print("[Warning] No JSON object found in model output.")
                return []

            clean_output = clean_output[first_brace:last_brace + 1]

            data = json.loads(clean_output)

        except Exception as e:
            print(f"JSON decode error: {e}")
            return []

        if not isinstance(data, dict):
            print(f"Unexpected JSON type: {type(data).__name__}")
            return []

        # Unwrap nested answer objects when present.
        if "answer" in data and isinstance(data["answer"], dict):
            data = data["answer"]

        if not isinstance(data, dict):
            print(f"Unexpected answer type: {type(data).__name__}")
            return []

        strokes = data.get("strokes", [])

        if isinstance(strokes, dict):
            normalized_strokes = []

            for stroke_key, stroke_value in strokes.items():
                if not isinstance(stroke_value, dict):
                    print(
                        f"Skip invalid stroke '{stroke_key}': "
                        f"expected dict, got {type(stroke_value).__name__}"
                    )
                    continue

                stroke_data = dict(stroke_value)

                if not stroke_data.get("id"):
                    stroke_data["id"] = stroke_key

                normalized_strokes.append(stroke_data)

            strokes = normalized_strokes

        elif not isinstance(strokes, list):
            print(f"Unexpected strokes type: {type(strokes).__name__}")
            return []

        for index, stroke in enumerate(strokes):
            if not isinstance(stroke, dict):
                print(
                    f"Skip invalid stroke #{index + 1}: "
                    f"expected dict, got {type(stroke).__name__}"
                )
                continue

            points = stroke.get("points", [])
            t_values = stroke.get("t_values", [])

            if not isinstance(points, list):
                print(
                    f"Skip invalid stroke #{index + 1}: "
                    "'points' must be a list"
                )
                continue

            if not isinstance(t_values, list):
                print(
                    f"Skip invalid stroke #{index + 1}: "
                    "'t_values' must be a list"
                )
                continue

            info = {
                "points": [],
                "t_values": [],
                "id": stroke.get("id", ""),
            }

            for point in points:
                if not isinstance(point, str):
                    continue

                match = re.fullmatch(
                    r"x(\d+)y(\d+)",
                    point.strip(),
                )

                if match:
                    x, y = match.groups()
                    info["points"].append((int(x), int(y)))

            try:
                info["t_values"] = [
                    float(t)
                    for t in t_values
                ]
            except (TypeError, ValueError):
                print(
                    f"Skip invalid stroke #{index + 1}: "
                    "invalid t_values"
                )
                continue

            if not info["points"]:
                print(
                    f"Skip invalid stroke #{index + 1}: "
                    "no valid points"
                )
                continue

            if len(info["points"]) != len(info["t_values"]):
                print(
                    f"Skip invalid stroke #{index + 1}: "
                    "points/t_values length mismatch "
                    f"({len(info['points'])} vs {len(info['t_values'])})"
                )
                continue

            stroke_info_list.append(info)

        return stroke_info_list

    def _match_labels_to_paths(self, all_paths, stroke_info_list, semantic_labels, default_prefix):
        import re
        
        if len(semantic_labels) == len(all_paths):
            return semantic_labels
        
        if len(semantic_labels) < len(all_paths):
            matched = []
            
            for path_idx, path_elem in enumerate(all_paths):
                d_attr = path_elem.get('d', '')
                
                path_points = re.findall(r'([\d.]+)\s+([\d.]+)', d_attr)
                if not path_points:
                    matched.append(f"{default_prefix}_part_{path_idx + 1}")
                    continue
                
                path_coords = []
                for x, y in path_points[:5]:  
                    try:
                        # Assume a 780x780 SVG and a 64x64 grid.
                        grid_x = int(float(x) / 780 * 64)
                        grid_y = int(float(y) / 780 * 64)
                        path_coords.append((grid_x, grid_y))
                    except:
                        pass
                
                best_match_idx = -1
                best_match_score = -1
                
                for idx, stroke_info in enumerate(stroke_info_list):
                    if idx >= len(semantic_labels):
                        break
                        
                    stroke_points = stroke_info.get('points', [])
                    if not stroke_points:
                        continue
                    
                    if path_coords and stroke_points:
                        start_match = self._point_distance(path_coords[0], stroke_points[0]) if stroke_points else 100
                        end_match = self._point_distance(path_coords[-1], stroke_points[-1]) if stroke_points else 100
                        score = 100 - (start_match + end_match) / 2
                        if score > best_match_score:
                            best_match_score = score
                            best_match_idx = idx
                
                if best_match_idx >= 0 and best_match_idx < len(semantic_labels):
                    matched.append(semantic_labels[best_match_idx])
                else:
                    matched.append(f"{default_prefix}_part_{path_idx + 1}")
            
            return matched
        
        return semantic_labels[:len(all_paths)]


    def _point_distance(self, p1, p2):
        if not p1 or not p2:
            return 100
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


    def generate_object_from_anchors(
        self,
        anchors_json
    ):

        object_id = anchors_json["object_id"]

        svg_path = self.anchors_to_svg(
            anchors_json,
            object_id
        )

        png_path = f"{self.path2save}/{object_id}.png"

        self.render_svg_to_png(
            svg_path,
            png_path
        )

        overlay_path = f"{self.path2save}/{object_id}_canvas.png"

        self.overlay_object(
            png_path,
            overlay_path
        )

        return svg_path

    def extract_svg_inner(self, svg_str):
        import re

        svg_str = re.sub(r"<\?xml.*?\?>", "", svg_str)

        inner = re.search(r"<svg[^>]*>(.*)</svg>", svg_str, re.S)

        if inner:
            return inner.group(1).strip()

        return svg_str

    def compose_scene(self, object_svgs, layout, output_name="final_scene"):

        def load_svg(svg_path):
            with open(svg_path, "r", encoding="utf-8") as f:
                return self.extract_svg_inner(f.read())

        def apply_transform(svg_inner, x, y, scale, rotation):
            cx = x * 512
            cy = y * 512
            return f"""
            <g transform="translate({cx},{cy}) rotate({rotation}) scale({scale})">
                <g transform="translate(-256,-256)">
                    {svg_inner}
                </g>
            </g>
            """

        objects_to_render = []
        for obj in object_svgs:
            obj_id = obj["object_id"]
            if obj_id not in layout:
                continue
            t = layout[obj_id]
            objects_to_render.append({
                "svg_path": obj["svg_path"],
                "layout": t,
                "z_index": t.get("z_index", 0) 
            })

        objects_to_render.sort(key=lambda item: item["z_index"], reverse=True)

        composited = []
        for item in objects_to_render:
            t = item["layout"]
            svg_inner = load_svg(item["svg_path"])
            composited.append(apply_transform(svg_inner, t["x"], t["y"], t["scale"], t.get("rotation", 0.0)))

        final_svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
            {''.join(composited)}
        </svg>
        """

        out_path = f"{self.path2save}/{output_name}.svg"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_svg)

        return out_path

    def call_global_layout_planner(self, scene_prompt, scene_objects):
        msg = {
            "scene_concept": scene_prompt,
            "objects": [
                {
                    "object_id": obj["object_id"],
                    "description": obj["description"]
                }
                for obj in scene_objects
            ]
        }

        prompt = f"""
    You are a professional illustrator and scene composition designer.

    Scene:
    "{scene_prompt}"

    Your task is NOT to predict coordinates.

    Your task is to create a high-level scene composition plan.

    For each object determine:

    1. anchor
    (one of)

    top_left
    top_center
    top_right

    center_left
    center
    center_right

    bottom_left
    bottom_center
    bottom_right

    2. relation

    left_of(object)
    right_of(object)

    above(object)
    below(object)

    under(object)
    on(object)

    inside(object)

    in_front_of(object)
    behind(object)

    3. size

    small
    medium
    large
    dominant

    4. depth

    foreground
    midground
    background

    Rules:

    - Only 1 or 2 major objects should receive anchors.
    - Remaining objects should be positioned using relations.
    - Preserve semantic correctness.
    - Preserve visual balance.
    - Think like an artist creating a thumbnail sketch.
    - Avoid assigning anchors to every object.

    Return JSON only.

    Example:

    {{
    "tree": {{
        "anchor":"center_left",
        "size":"large",
        "depth":"midground"
    }},

    "cow": {{
        "relation":"under(tree)",
        "size":"medium",
        "depth":"foreground"
    }}
    }}
    """

        response = self.call_llm(
            system_message=prompt,
            other_msg=[
                {
                    "role": "user",
                    "content": json.dumps(msg, indent=2)
                }
            ],
            additional_args={}
        )

        clean_response = re.sub(
            r"<think>.*?</think>",
            "",
            response,
            flags=re.DOTALL
        ).strip()

        clean_response = (
            clean_response
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        match = re.search(
            r"(\{[\s\S]*\})",
            clean_response
        )

        if not match:
            raise ValueError("No JSON found.")

        return json.loads(match.group(1))

    def resolve_composition_graph(self, composition_plan, raw_scene_plan=None):
        if isinstance(composition_plan, list):
            composition_plan = {
                item["object_id"]: {
                    k: v for k, v in item.items()
                    if k != "object_id"
                }
                for item in composition_plan
            }

        raw_depth_lookup = {}
        if raw_scene_plan:
            if isinstance(raw_scene_plan, list):
                raw_depth_lookup = {item["object_id"]: int(item.get("depth", 1)) for item in raw_scene_plan}
          
            elif isinstance(raw_scene_plan, dict):
                raw_depth_lookup = {k: int(v.get("depth", 1)) for k, v in raw_scene_plan.items()}

        ANCHOR_POS = {
            "top_left": (0.2, 0.2), "top_center": (0.5, 0.2), "top_right": (0.8, 0.2),
            "center_left": (0.25, 0.5), "center": (0.5, 0.5), "center_right": (0.75, 0.5),
            "bottom_left": (0.25, 0.75), "bottom_center": (0.5, 0.75), "bottom_right": (0.75, 0.75)
        }

        SIZE_MAP = {
            "small": 0.25, "medium": 0.45, "large": 0.7, "dominant": 1.0
        }

        RELATION_OFFSET = {
            "left_of": (-0.20, 0), "right_of": (0.20, 0),
            "above": (0, -0.15), "below": (0, 0.15), "under": (0, 0.15),
            "on": (0, -0.12), "behind": (-0.05, -0.10), "in_front_of": (0.05, 0.10),
            "inside": (0, 0)
        }

        resolved = {}

        for obj_id, info in composition_plan.items():
            if "anchor" not in info:
                continue

            x, y = ANCHOR_POS.get(info["anchor"], (0.5, 0.5))
            
            z_index = raw_depth_lookup.get(obj_id, 1)

            resolved[obj_id] = {
                "x": x,
                "y": y,
                "scale": SIZE_MAP.get(info.get("size", "medium"), 0.45),
                "rotation": 0,
                "z_index": z_index  
            }

        unresolved = True
        while unresolved:
            unresolved = False
            for obj_id, info in composition_plan.items():
                if obj_id in resolved:
                    continue

                relation_str = info.get("relation")
                if relation_str is None:
                    continue

                match = re.match(r"([a-zA-Z_]+)\((.*?)\)", relation_str)
                if not match:
                    continue

                relation_type = match.group(1)
                parent_obj = match.group(2)

                if parent_obj not in resolved:
                    unresolved = True
                    continue

                parent = resolved[parent_obj]
                dx, dy = RELATION_OFFSET.get(relation_type, (0, 0))

                z_index = raw_depth_lookup.get(obj_id, parent["z_index"] + 1)

                resolved[obj_id] = {
                    "x": parent["x"] + dx,
                    "y": parent["y"] + dy,
                    "scale": SIZE_MAP.get(info.get("size", "medium"), 0.45),
                    "rotation": 0,
                    "z_index": z_index  
                }

        for obj_id in resolved:
            scale = resolved[obj_id]["scale"]
            margin = scale / 2
            resolved[obj_id]["x"] = max(margin, min(1 - margin, resolved[obj_id]["x"]))
            resolved[obj_id]["y"] = max(margin, min(1 - margin, resolved[obj_id]["y"]))

        return resolved

    def render_svg_to_base64(self, svg_path):

        png_bytes = cairosvg.svg2png(url=svg_path)

        img = Image.open(io.BytesIO(png_bytes))

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        return base64.b64encode(buf.getvalue()).decode()

    def critique_layout(self, layout, scene_svg_path):

        base64_img = self.render_svg_to_base64(scene_svg_path)

        prompt = f"""
    Current Layout:
    {json.dumps(layout, indent=2)}
    """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"please respond with JSON format.{LAYOUT_CRITIC_PROMPT}"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_img}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.2
        )

        text = response.choices[0].message.content

        response_text = text 

        clean_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

        clean_text = clean_text.replace("```json", "").replace("```", "").strip()

        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", clean_text)

        if match:
            json_content = match.group(1)
        else:
            print("====== Unparseable raw VLM response ======")
            print(response_text)
            print("====================================")
            raise ValueError("VLM critique response does not contain a valid JSON object/array.")

        return json.loads(json_content)

    def revise_layout(self, old_layout, feedback):

        prompt = LAYOUT_REVISER_PROMPT.format(
            layout=json.dumps(old_layout, indent=2),
            feedback=json.dumps(feedback, indent=2)
        )

        response = self.call_llm(
            system_message="You are an expert scene layout reviser.",
            other_msg=[{
                "role": "user",
                "content": prompt
            }],
            additional_args={}
        )

        print(response)

        clean_response = re.sub(
            r"<think>.*?</think>",
            "",
            response,
            flags=re.DOTALL
        ).strip()

        clean_response = clean_response.replace("```json", "").replace("```", "").strip()

        match = re.search(
            r"(\[[\s\S]*\]|\{[\s\S]*\})",
            clean_response
        )

        if not match:
            raise ValueError("No JSON found in response")

        content = match.group(1)

        print("CONTENT:")
        print(content)


        return json.loads(content)  

    def generate_refined_layout(
        self,
        object_svgs,
        max_iterations=3,
        target_score=80
    ):

        print("=" * 60)
        print("GLOBAL LAYOUT REFINEMENT")
        print("=" * 60)

        layout_raw = self.call_global_layout_planner(object_svgs)
        print("LAYOUT RAW:")
        print(layout_raw)
        print(type(layout_raw))
        layout = json.loads(layout_raw)

        best_layout = layout
        best_score = -1
        best_scene_svg_path = None  

        for iteration in range(max_iterations):
            iter_num = iteration + 1
            print(f"\nIteration {iter_num}/{max_iterations}")

            current_scene_name = f"scene_iter_{iter_num}"
            scene_svg_path = self.compose_scene(
                object_svgs,
                layout,
                output_name=current_scene_name  # Pass the current iteration name.
            )

            critique = self.critique_layout(
                layout,
                scene_svg_path
            )

            score = critique.get("score", 0)
            print("score:", score)

            if score > best_score:
                best_score = score
                best_layout = layout
                best_scene_svg_path = scene_svg_path

            if critique.get("is_satisfactory", False):
                print("Layout satisfactory.")
                break

            if score >= target_score:
                print("Target score reached.")
                break

            print("Revising layout...")

            layout = self.revise_layout(
                layout,
                critique
            )

        print("Best layout score:", best_score)

        if best_scene_svg_path is None:
            best_scene_svg_path = scene_svg_path

        return best_layout, best_scene_svg_path

    def call_vlm_for_layout_selection(self, global_concept: str, candidate_images_paths: list, constraints_context: str) -> str:
        def encode_image_to_base64_url(image_path):
            ext = os.path.splitext(image_path)[1].lower().replace('.', '')
            if ext not in ['png', 'jpg', 'jpeg', 'webp']:
                ext = 'png'
            with open(image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            return f"data:image/{ext};base64,{encoded_string}"

        user_content = [
            {
                "type": "text",
                "text": (
                    "You are an expert art director and spatial reasoning vision model.\n"
                    f"The user wants to arrange multiple SVG sketch assets to compose a complete scene based on this concept: '{global_concept}'.\n\n"
                    "We ran 3 independent optimization trials to find the best layout. "
                    "Below are the 3 final canvas previews generated by these trials (Indexed as 0, 1, and 2 in order).\n"
                    "Please analyze them carefully and select the single BEST trial based on the following criteria:\n"
                    "1. **Semantic Correctness**: Do the relative positions (left, right, above, below) match human common sense and the expected constraints?\n"
                    "2. **No Critical Occlusion**: Important foreground objects should NOT fully cover or heavily block background objects unless intended.\n"
                    "3. **Aesthetic Proportion**: Objects should not shrink into tiny dots or explode to take over the whole canvas unpleasantly. The composition should look balanced and satisfying.\n\n"
                    f"For your reference, here are the geometry constraints that the layout optimization was supposed to follow:\n```json\n{constraints_context}\n```\n\n"
                    "CRITICAL: You must return a strict JSON object containing 'best_trial_index' (integer 0, 1, or 2) and a brief 'reason' explaining your choice. "
                    "Do not include any markdown code blocks outside of the JSON. Output raw JSON only."
                )
            }
        ]

        for idx, img_path in enumerate(candidate_images_paths):
            try:
                base64_url = encode_image_to_base64_url(img_path)
                
                user_content.append({
                    "type": "text",
                    "text": f"--- CANDIDATE IMAGE INDEX: {idx} ---"
                })
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_url
                    }
                })
            except Exception as e:
                print(f"Failed to encode candidate image {img_path}: {e}")

        try:
            
            response = self.client.chat.completions.create(
                model=self.model,  
                messages=[
                    {
                        "role": "user",
                        "content": f"please respond with JSON format.{user_content}"
                    }
                ],
                response_format={"type": "json_object"}, 
                temperature=0.2,                         
                max_tokens=300
            )
            
            vlm_output = response.choices[0].message.content.strip()
            return vlm_output

        except Exception as api_err:
            print(f"VLM candidate-review request failed: {api_err}")
            fallback_json = {"best_trial_index": 0, "reason": "Fallback due to API error."}
            return json.dumps(fallback_json)

    def render_temporary_canvas(self, temp_layout_history, output_dir, step_name):
        import xml.etree.ElementTree as ET
        import re
        from PIL import Image
        import cairosvg
        
        ET.register_namespace('', "http://www.w3.org/2000/svg")
        
        svg_root = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'width': '512',
            'height': '512',
            'viewBox': '0 0 512 512'
        })
        
        ET.SubElement(svg_root, 'rect', {
            'width': '512',
            'height': '512',
            'fill': '#FFFFFF'
        })

        for obj_idx, obj in enumerate(temp_layout_history):
            obj_svg_path = obj.get("bg_svg_path")
            if not obj_svg_path or not os.path.exists(obj_svg_path):
                print(f"Object {obj.get('object_id', 'unknown')} SVG file was not found: {obj_svg_path}")
                continue
            
            try:
                with open(obj_svg_path, 'r', encoding='utf-8') as f:
                    svg_content = f.read()
                
                svg_content = self._clean_svg_content(svg_content)
                
                from xml.etree.ElementTree import fromstring, tostring
                child_root = fromstring(svg_content)
                
                g_element = ET.Element('g')
                
                x = obj.get("x", 256.0)
                y = obj.get("y", 256.0)
                scale = obj.get("scale", 1.0)
                rotation = obj.get("rotation", 0.0)
                raw_w = float(obj.get("raw_w", 100.0))
                raw_h = float(obj.get("raw_h", 100.0))
                raw_cx = raw_w / 2.0
                raw_cy = raw_h / 2.0
                total_scale = scale
                
                transform_parts = []
                transform_parts.append(f"translate({x}, {y})")  # Apply last: translate to the target position.
                if rotation != 0:
                    transform_parts.append(f"rotate({rotation})")  # Implementation note.
                if total_scale != 1.0:
                    transform_parts.append(f"scale({total_scale})")  # Implementation note.
                transform_parts.append(f"translate({-raw_cx}, {-raw_cy})")  # Implementation note.
                
                transform_str = " ".join(transform_parts)
                g_element.set('transform', transform_str)
                
                print(f"  📍 {obj.get('object_id')}: target ({x:.1f}, {y:.1f}), original center ({raw_cx:.1f}, {raw_cy:.1f}), scale {total_scale:.2f}")
                
                final_center_x = x
                final_center_y = y
                final_width = raw_w * total_scale
                final_height = raw_h * total_scale
                final_left = final_center_x - final_width / 2
                final_top = final_center_y - final_height / 2
                
                for child in child_root:
                    import copy
                    new_child = copy.deepcopy(child)
                    g_element.append(new_child)
                
                svg_root.append(g_element)
                
            except Exception as e:
                print(f"❌ Failed to parse object {obj.get('object_id', 'unknown')} SVG: {e}")
                import traceback
                traceback.print_exc()

        svg_string = '<?xml version="1.0" encoding="utf-8"?>\n'
        svg_string += ET.tostring(svg_root, encoding='unicode')
        
        svg_string = self._clean_svg_string(svg_string)
        
        preview_canvas_svg = os.path.join(output_dir, f"{step_name}_preview.svg")
        with open(preview_canvas_svg, 'w', encoding='utf-8') as f:
            f.write(svg_string)
        
        preview_canvas_png = os.path.join(output_dir, f"{step_name}_preview.png")
        
        try:
            cairosvg.svg2png(url=preview_canvas_svg, write_to=preview_canvas_png, dpi=96)
        except Exception as e:
            print(f"❌ CairoSVG PNG rendering failed: {e}")
            
            try:
                fallback_img = Image.new('RGB', (512, 512), color='white')
                from PIL import ImageDraw
                draw = ImageDraw.Draw(fallback_img)
                draw.text((10, 10), f"SVG Render Failed: {str(e)[:50]}", fill='black')
                draw.text((10, 30), f"Step: {step_name}", fill='black')
                fallback_img.save(preview_canvas_png)
                print(f"⚠️ Generated placeholder PNG using fallback renderer: {preview_canvas_png}")
            except:
                print(f"❌ Fallback rendering also failed")
        
        return preview_canvas_png, preview_canvas_svg

    
    def rebuild_official_canvas(self, final_layout_history, output_png_path, output_svg_path, canvas_size=780, target_stroke_width=2.0):
        
        ET.register_namespace('', "http://www.w3.org/2000/svg")
        
        svg_root = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'width': str(canvas_size),
            'height': str(canvas_size),
            'viewBox': f"0 0 {canvas_size} {canvas_size}"
        })
        
        for obj in final_layout_history:
            obj_svg_path = obj.get("bg_svg_path")
            if not obj_svg_path or not os.path.exists(obj_svg_path):
                print(f"⚠️ Object {obj.get('object_id', 'unknown')} SVG file was not found: {obj_svg_path}")
                continue
            
            try:
                with open(obj_svg_path, 'r', encoding='utf-8') as f:
                    svg_content = f.read()
                
                svg_content = self._clean_svg_content(svg_content)
                scale = obj.get("scale", 1.0)
                abs_scale = max(abs(scale), 1e-5)
                compensated_stroke_width = target_stroke_width / abs_scale

                
                from xml.etree.ElementTree import fromstring
                child_root = fromstring(svg_content)

                raw_cx = obj.get("raw_w", 100) * scale / 2.0
                raw_cy = obj.get("raw_h", 100) * scale / 2.0
                
                for node in child_root.iter():
                    if node.tag.endswith('style') and node.text:
                        node.text = re.sub(
                            r'stroke-width\s*:\s*[^;}\s]+', 
                            f'stroke-width: {compensated_stroke_width:.2f}', 
                            node.text
                        )

                    if 'stroke-width' in node.attrib:
                        node.set('stroke-width', f"{compensated_stroke_width:.2f}")
                    
                    if 'style' in node.attrib:
                        style_str = node.get('style', '')
                        if 'stroke-width' in style_str:
                            new_style = re.sub(r'stroke-width\s*:\s*[^;]+', f'stroke-width: {compensated_stroke_width:.2f}', style_str)
                            node.set('style', new_style)
                    if node.tag.endswith('path'):
                        if node.get('class') != 'white-bg' and node.get('fill') != 'white':
                            node.set('stroke-width', f"{compensated_stroke_width:.2f}")
                
                g_element = ET.Element('g')
                
                default_center = canvas_size / 2.0
                x = obj.get("x", default_center)
                y = obj.get("y", default_center)
                rotation = obj.get("rotation", 0.0)
                
                transform_parts = []
                transform_parts.append(f"translate({x}, {y})")
                if rotation != 0:
                    transform_parts.append(f"rotate({rotation})")
                if abs_scale != 1.0:
                    transform_parts.append(f"scale({abs_scale})")
                transform_parts.append(f"translate({-raw_cx}, {-raw_cy})")
                
                transform_str = " ".join(transform_parts)
                g_element.set('transform', transform_str)
                
                for child in child_root:
                    new_child = copy.deepcopy(child)
                    g_element.append(new_child)
                
                svg_root.append(g_element)
                
            except Exception as e:
                print(f"❌ Failed to parse object while finalizing composition: {obj.get('object_id', 'unknown')} SVG: {e}")
        
        svg_string = '<?xml version="1.0" encoding="utf-8"?>\n'
        svg_string += ET.tostring(svg_root, encoding='unicode')
        svg_string = self._clean_svg_string(svg_string)
        
        try:
            with open(output_svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_string)
        except Exception as e:
            print(f"❌ Failed to save final SVG: {e}")
            return None, None
        
        try:
            cairosvg.svg2png(url=output_svg_path, write_to=output_png_path, dpi=96)
        except Exception as e:
            try:
                cairosvg.svg2png(bytestring=svg_string.encode('utf-8'), write_to=output_png_path, dpi=96)
            except Exception as e2:
                fallback_img = Image.new('RGB', (canvas_size, canvas_size), color='white')
                fallback_img.save(output_png_path)
        
        return output_png_path, output_svg_path


    def _clean_svg_content(self, svg_content):
        """Clean SVG content by removing declarations that may cause conflicts."""
        svg_content = re.sub(r'<\?xml[^?]*\?>', '', svg_content)
        svg_content = re.sub(r'<!DOCTYPE[^>]*>', '', svg_content)
        svg_content = re.sub(r'xmlns="http://www\.w3\.org/2000/svg"', '', svg_content)
        svg_content = re.sub(r'class=""', '', svg_content)
        svg_content = re.sub(r'style=""', '', svg_content)
        
        return svg_content.strip()

    def _clean_svg_string(self, svg_string):
        def fix_duplicate_attrs(match):
            tag = match.group(0)
            parts = tag.split(' ')
            seen = set()
            unique_parts = []
            for part in parts:
                if 'xmlns' in part and 'xmlns' in seen:
                    continue
                if 'xmlns' in part:
                    seen.add('xmlns')
                unique_parts.append(part)
            return ' '.join(unique_parts)
        
        svg_string = re.sub(r'<[^>]+>', fix_duplicate_attrs, svg_string)
        
        return svg_string
        
    def _encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def call_composition_reviewer(self, target_object, preview_canvas_png, temp_layout_history):
        base64_image = self._encode_image_to_base64(preview_canvas_png)
        
        canvas_size = 512
        
        layout_info = []
        for obj in temp_layout_history:
            layout_info.append({
                "object_id": obj["object_id"],
                "x": obj.get("x", 256),
                "y": obj.get("y", 256),
                "scale": obj.get("scale", 1.0),
                "rotation": obj.get("rotation", 0)
            })
        
        system_prompt = """You are a precise layout adjustment expert. Your task is to output NUMERICAL adjustment values that can be directly applied to object positions and scales.

    CRITICAL RULES:
    1. You MUST output numerical values (dx, dy in pixels, dscale as multiplier)
    2. Do NOT output semantic instructions like "center it" or "make it larger"
    3. Be specific: "dx: 30, dy: -20" instead of "move right and up"
    4. Base all adjustments on the 512x512 canvas coordinate system
    5. dscale: 1.0 = no change, >1.0 = enlarge, <1.0 = shrink

    Output format MUST be valid JSON only."""
        
        user_prompt = f"""
    **Canvas Size**: {canvas_size}x{canvas_size} pixels (0,0 is top-left, 512,512 is bottom-right)

    **Current Layout**:
    {json.dumps(layout_info, indent=2)}

    **Target Scene Description**:
    {target_object.get('incremental_prompt', target_object.get('description', ''))}

    **Evaluation & Adjustment Instructions**:

    1. **Position Evaluation**:
    - First object should be centered: target (256, 256)
    - Calculate exact dx, dy needed to reach ideal position
    - Example: If object at (100, 100), need dx: 156, dy: 156 to center

    2. **Scale Evaluation**:
    - First object: scale should be 0.8-1.2 depending on object type
    - Bowl: scale 0.8-1.0
    - Dog: scale 0.6-0.8 (when next to bowl)
    - Calculate dscale = target_scale / current_scale

    3. **Boundary Constraints**:
    - Keep objects within 30-482px range (not touching edges)
    - x and y must stay between 30 and 482 after adjustment

    **Output JSON** (adjustments in PIXELS and SCALE MULTIPLIERS):
    {{
    "verdict": "Brief numerical analysis",
    "satisfied": false,
    "adjustments": [
        {{
        "object_id": "bowl",
        "dx": 156,
        "dy": 156,
        "dscale": 1.2,
        "drotation": 0,
        "reason": "Moving from corner to center"
        }}
    ]
    }}

    If satisfied, set "satisfied": true and adjustments can be empty array.
    """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": f"please respond with JSON format.{system_prompt}"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.2,
            )
            
            result_text = response.choices[0].message.content.strip()
            refinement_plan = json.loads(result_text)
            
            if "adjustments" not in refinement_plan:
                refinement_plan["adjustments"] = []
            
            return refinement_plan
            
        except Exception as e:
            print(f"❌ Composition critique failed: {e}")
            return {
                "verdict": "Using fallback",
                "satisfied": True,
                "adjustments": []
            }

    def query_multimodal_model(self, system_prompt, user_prompt, image_base64=None):
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        user_content = []
        
        user_content.append({
            "type": "text",
            "text": user_prompt
        })
        
        if image_base64:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}",
                    "detail": "high" 
                }
            })
            
        messages.append({
            "role": "user",
            "content": f"please respond with JSON format.{user_content}"
        })
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                
                response = self.client.chat.completions.create(
                    model=self.model,  
                    messages=messages,
                    temperature=0.0,  
                    max_tokens=1000
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"⚠️ VLM request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2) 
                else:
                    print("❌ Maximum retry count reached; the VLM did not return valid constraints.")
                    return "semantic_loss += torch.tensor(0.0)"
                    
    def _format_objects_info(self, scene_objects):
        if not scene_objects:
            return "No objects in scene yet."
        
        info_str = ""
        for obj in scene_objects:
            obj_id = obj.get('object_id', 'unknown')
            description = obj.get('description', '')
            raw_w = obj.get('raw_w', 'N/A')
            raw_h = obj.get('raw_h', 'N/A')
            depth = obj.get('depth', 1)
            
            info_str += f"""
Object ID: {obj_id}
  Description: {description}
  Original Size: {raw_w} x {raw_h}
  Depth Level: {depth}
"""
        return info_str

    def call_vlm_for_layout_code(self, concept, objects, similar_experiences=None, 
                                    available_operations=None, additional_context="",
                                    current_canvas_base64=None, object_svg_map=None):
        import base64
        import json

        objects_info = []
        for obj in objects:
            obj_id = obj.get("object_id") if isinstance(obj, dict) else getattr(obj, "object_id", str(obj))
            obj_desc = obj.get("description") if isinstance(obj, dict) else getattr(obj, "description", "")
            obj_depth = obj.get("depth") if isinstance(obj, dict) else getattr(obj, "depth", 1)
            obj_complexity = obj.get("visual_complexity", 1.0) if isinstance(obj, dict) else getattr(obj, "visual_complexity", 1.0)
            
            raw_w = obj.get("raw_w") if isinstance(obj, dict) else getattr(obj, "raw_w", None)
            raw_h = obj.get("raw_h") if isinstance(obj, dict) else getattr(obj, "raw_h", None)
            
            size_info = f" (size: {raw_w}x{raw_h})" if raw_w and raw_h else " (size determined after generation)"
            
            objects_info.append(
                f"  - ID: '{obj_id}', depth: {obj_depth}"
            )
        
        objects_context = "\n".join(objects_info)
        
        svg_mapping_context = ""
        if object_svg_map:
            svg_mapping_context = "\n=== Object-to-SVG Mapping ===\n"
            for obj_id, svg_info in object_svg_map.items():
                svg_mapping_context += f"- {obj_id}: {svg_info.get('description', '')}\n"
                if svg_info.get('bbox'):
                    bbox = svg_info['bbox']
                    svg_mapping_context += f"  Bounding box: x={bbox.get('x', 0):.1f}, y={bbox.get('y', 0):.1f}, w={bbox.get('w', 0):.1f}, h={bbox.get('h', 0):.1f}\n"
           
                if svg_info.get('grid_position'):
                    grid_col, grid_row = svg_info['grid_position']
                    svg_mapping_context += f"  Current grid position: ({grid_col}, {grid_row})\n"
            svg_mapping_context += "\nNote: Each object ID maps to an independent SVG file. Adjust its transform to change position and scale.\n"
        
        ops_desc = ", ".join(available_operations) if available_operations else "move, scale, rotate, delete, layered, modify"
        
        experience_context = ""
        if similar_experiences and len(similar_experiences) > 0:
            experience_context = "\n=== Relevant Successful Experiences (Reference Only) ===\n"
            for i, exp in enumerate(similar_experiences[:3]):
                experience_context += f"\nExperience {i+1}:\n"
                if "operation_sequence" in exp:
                    experience_context += f"  Operation sequence: {exp['operation_sequence']}\n"
                if "key_parameters" in exp:
                    experience_context += f"  Key parameters: {json.dumps(exp['key_parameters'], indent=2)}\n"
            experience_context += "\nUse these experiences as references and adapt them to the current scene objects.\n"

        system_prompt = (
            "You are an expert Python code generator for sketch composition and layout arrangement. "
            "Your task is to generate executable Python code that arranges multiple sketch objects on a canvas.\n\n"
            
            "=== GRID COORDINATE SYSTEM ===\n"
            f"- Grid size: {self.res}x{self.res} cells (indexed 0 to {self.res-1})\n"
            f"- Cell size: {self.cell_size} pixels\n"
            f"- Total canvas: {self.grid_size}x{self.grid_size} pixels\n"
            "- Grid (0, 0) is top-left corner\n"
            f"- Grid ({self.res-1}, {self.res-1}) is bottom-right corner\n"
            "- IMPORTANT: You must use GRID COORDINATES in move() calls, not pixel coordinates!\n"
            "- Grid coordinates are shown as grid lines on the provided canvas image\n\n"
            
            "=== COORDINATE CONVERSION ===\n"
            f"- move(x, y) expects grid coordinates (0 to {self.res-1})\n"
            f"- The center of the canvas is at grid ({self.res//2}, {self.res//2})\n"
            f"- Grid coordinate (gx, gy) maps to pixel center at:\n"
            f"  px = gx * {self.cell_size} + {self.cell_size/2}\n"
            f"  py = gy * {self.cell_size} + {self.cell_size/2}\n"
            "- DO NOT use pixel coordinates (0-512) in move() calls!\n\n"
            
            "=== IMPORTANT: You can SEE the current canvas ===\n"
            "You will be provided with an image of the current canvas showing all sketches in their current positions.\n"
            "The image has a grid overlay to help you identify positions.\n"
            "This allows you to:\n"
            "1. See which objects are overlapping\n"
            "2. See which objects are out of bounds\n"
            "3. Understand the current spatial relationships\n"
            "4. Make informed decisions about where to move objects\n\n"
            
            "=== AVAILABLE FUNCTIONS ===\n"
            f"You have access to these functions: {ops_desc}\n\n"
            
            "Function signatures:\n"
            f"- move(object_id: str, x: float, y: float) -> None\n"
            f"    Moves the object to grid position (x, y). x and y are grid coordinates (0 to {self.res-1}).\n"
            f"    Example: move('table', {self.res//2}, {self.res//2}) moves to center of canvas\n"
            f"    Example: move('chair', {self.res//2 + 1}, {self.res//2}) moves to right of center\n\n"
            
            "- scale(object_id: str, factor: float) -> None\n"
            "    Scales the object by the given factor (0.3 - 1.5 recommended).\n\n"
            
            "- rotate(object_id: str, degrees: float) -> None\n"
            "    Rotates the object by the given degrees (clockwise).\n\n"
            
            "- delete(object_id: str) -> None\n"
            "    Removes the object from the scene.\n\n"
            
            "- layered(object_id: str, layer: str) -> None\n"
            "    Sets the depth layer. Valid layers: 'bg'(0), 'mid'(1), 'fg'(2), 'front'(3).\n\n"
            
            "- modify(object_id: str, new_description: str = None, regenerate: bool = True) -> None\n"
            "    Regenerates the raw sketch. Optionally provide a new description.\n\n"
            
            "=== HOW TO ANALYZE THE CURRENT CANVAS ===\n"
            "Look at the provided image with grid overlay and identify:\n"
            "1. Which objects are overlapping? -> Use move() to separate them\n"
            "2. Which objects are outside the canvas? -> Use move() to bring them inside (keep within 0 to {self.res-1})\n"
            "3. Which objects are too large/small? -> Use scale() to adjust\n"
            "4. What is the overall composition? -> Make adjustments for better balance\n"
            "5. Count grid cells to determine precise positions\n\n"
            
            "=== LAYOUT PRINCIPLES ===\n"
            "1. Background objects should be placed first (lower depth number)\n"
            "2. Foreground objects should be placed last (higher depth number)\n"
            "3. Avoid overlapping objects as much as possible\n"
            "4. Consider the visual balance and aesthetics\n"
            "5. Use reasonable scale factors based on real-world proportions\n"
            "6. Use comments to explain your reasoning\n"
            f"7. Keep objects at least 1 grid cell away from edges (positions 1 to {self.res-2})\n\n"
            
            "=== OUTPUT FORMAT ===\n"
            "You MUST output ONLY executable Python code. No Markdown blocks, no backticks, no explanations outside the code.\n"
            "The code should be clean, well-commented, and ready to execute.\n\n"
            
            "=== IMPORTANT RULES ===\n"
            "1. DO NOT import any modules (they are already available in the execution environment)\n"
            "2. DO NOT use print() or any I/O operations (unless for debugging)\n"
            "3. DO NOT define functions or classes (use the available functions directly)\n"
            "4. All object IDs must match exactly the IDs provided in the scene objects\n"
            "5. The code should be robust and handle edge cases gracefully\n"
            "6. If you detect potential overlaps, include adjustments to avoid them\n"
            "7. Analyze the current canvas image carefully before generating code\n"
            f"8. ALWAYS use grid coordinates (0-{self.res-1}) for move() calls, NEVER pixel coordinates (0-512)!\n\n"
            
            "=== SAMPLE CODE STRUCTURE (showing how to use each function) ===\n"
            f"# Grid size: {self.res}x{self.res}, center at ({self.res//2}, {self.res//2})\n\n"

            "# move() - places an object at a specific grid position\n"
            f"move('object_id', 15, 32)  # Move to x15y32\n\n"

            "# scale() - changes the size of an object\n"
            f"scale('object_id', 0.7)  # Scale to 70% of original size\n\n"

            "# rotate() - rotates an object\n"
            f"rotate('object_id', 15)  # Rotate 15 degrees clockwise\n\n"

            "# layered() - adjusts the depth/layer of an object (z-order)\n"
            f"layered('object_id', 'bg')   # Send to background\n"
            f"layered('object_id', 'mid')  # Middle layer\n"
            f"layered('object_id', 'fg')   # Bring to foreground\n"
            f"layered('object_id', 'front')# Very front\n\n"

            "# delete() - removes an object from the scene\n"
            f"delete('object_id')  # Remove this object\n\n"

            "# modify() - regenerates a sketch with a new description\n"
            f"modify('object_id', 'a more detailed description')  # Regenerate\n\n"

            "# Note: You can combine multiple operations on the same object\n"
            f"# Example: move('table', 9, 30)\n"
            f"#          scale('table', 0.8)\n"
            f"#          layered('table', 'bg')\n\n"

            "# IMPORTANT: layering is about z-order (which objects appear in front of others)\n"
            "# - Objects with 'bg' layer appear behind all others\n"
            "# - Objects with 'front' layer appear in front of all others\n"
            "# - Use layering to create proper depth relationships between objects\n"
        )

        user_prompt = f"""

    Available Scene Objects:
    {objects_context}

    {svg_mapping_context}

    {experience_context}

    {additional_context}

    === INSTRUCTIONS ===
    Look at the current canvas image carefully. The image shows grid lines to help you identify positions.
    Each object is labeled with its ID in the image.

    1. Identify which objects need adjustment (overlapping, out of bounds, etc.)
    2. Generate Python code that uses GRID COORDINATES to move/scale/rotate objects
    3. Aim for a balanced, aesthetic final composition

    REMEMBER:
    - Use grid coordinates (0 to {self.res-1}) for x and y in move()
    - The center of canvas is at grid ({self.res//2}, {self.res//2})
    - DO NOT use pixel coordinates (0-512)!

    Generate ONLY the Python code, no other text:
    """
        
        try:
            if current_canvas_base64:
                json_str = self.query_multimodal_model_with_image_and_text(
                    system_prompt, 
                    user_prompt, 
                    current_canvas_base64
                )
            else:
                json_str = self.query_model_text_only(system_prompt, user_prompt)
        except AttributeError:
            if current_canvas_base64:
                json_str = self.query_multimodal_model(system_prompt, user_prompt, current_canvas_base64)
            else:
                from PIL import Image
                import io
                blank_img = Image.new('RGB', (512, 512), color='white')
                buffered = io.BytesIO()
                blank_img.save(buffered, format="PNG")
                encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                json_str = self.query_multimodal_model(system_prompt, user_prompt, encoded_image)
        
        code_response = self._clean_code_response(json_str)
        
        return code_response


    def query_multimodal_model_with_image_and_text(self, system_prompt, user_prompt, image_base64=None):
        if image_base64 is None:
            raise ValueError("image_base64 is required")
        images = [image_base64] if isinstance(image_base64, str) else list(image_base64)
        
        content = [
            {
                "type": "text",
                "text": user_prompt
            }
        ]
        
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img}"
                }
            })
        
        messages = [
            {
                "role": "system", 
                "content": f"please respond with JSON format.{system_prompt}"
            },
            {
                "role": "user",
                "content": content
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model, 
            messages=messages,
            response_format={"type": "json_object"},  
            temperature=0.3,                         
            max_tokens=2048
        )
        
        return response.choices[0].message.content.strip()

    def _clean_code_response(self, response: str) -> str:
        import re
        pattern = r'```(?:python)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            cleaned = matches[0].strip()
        else:
            cleaned = response.strip()
        
        cleaned = cleaned.strip()
        
        if cleaned.startswith("python\n"):
            cleaned = cleaned[7:]
        
        return cleaned


    def call_vlm_for_code_repair(self, repair_prompt: str, failed_code: str, diagnosis: Dict) -> str:
        system_prompt = (
            "You are an expert Python debugger and code repair specialist. "
            "You are given code that failed during execution, along with a diagnosis of the failure. "
            "Your task is to fix the code so that it works correctly.\n\n"
            
            "=== AVAILABLE FUNCTIONS ===\n"
            "- move(object_id: str, x: float, y: float) -> None\n"
            "- scale(object_id: str, factor: float) -> None\n"
            "- rotate(object_id: str, degrees: float) -> None\n"
            "- delete(object_id: str) -> None\n"
            "- layered(object_id: str, layer: str) -> None\n"
            "- modify(object_id: str, new_description: str = None, regenerate: bool = True) -> None\n\n"
            
            "=== REPAIR GUIDELINES ===\n"
            "1. Identify the exact step where the failure occurred\n"
            "2. Rewrite only the problematic parts, keep working parts as-is\n"
            "3. Add defensive checks (e.g., check if objects exist before operating on them)\n"
            "4. If the failure is about overlaps, adjust positions or scales\n"
            "5. If the failure is about boundaries, move objects inside the canvas\n"
            "6. If the failure is about depth, adjust layered() calls\n"
            "7. If the failure is about missing objects, add them back\n\n"
            
            "=== OUTPUT FORMAT ===\n"
            "Output ONLY the repaired Python code. No Markdown blocks, no backticks, no explanations.\n"
        )
        
        user_prompt = f"""
    === FAILURE DIAGNOSIS ===
    Error Type: {diagnosis.get('error_type', 'unknown')}
    Affected Objects: {diagnosis.get('affected_objects', [])}
    Recommended Fix: {diagnosis.get('recommended_fix', 'No specific fix recommended')}
    Diagnosis Details: {json.dumps(diagnosis, indent=2)}

    === FAILED CODE ===
    {failed_code}


    === REPAIR INSTRUCTIONS ===
    {repair_prompt}

    Please generate the repaired code. Focus on fixing the specific issues identified in the diagnosis.
    """
        
        try:
            repaired_code = self.query_model_text_only(system_prompt, user_prompt)
        except AttributeError:
            from PIL import Image
            import io
            import base64
            blank_img = Image.new('RGB', (512, 512), color='white')
            buffered = io.BytesIO()
            blank_img.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            repaired_code = self.query_multimodal_model(system_prompt, user_prompt, encoded_image)
        
        repaired_code = self._clean_code_response(repaired_code)
        
        return repaired_code

    def call_vlm_for_constraints_code(self, global_concept, all_scene_objects, current_canvas_path):
        import base64
        import json

        with open(current_canvas_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        system_prompt = (
            "You are an expert visual sketch artist and spatial layout coordinator. "
            "You are given an initial canvas (PNG) where all generated sketch elements are temporarily placed in a clean, "
            "non-overlapping grid layout (Asset Inventory) so you can clearly inspect their actual appearance, aspect ratios, and visual details.\n\n"
            
            "Your job is to look at this initial grid canvas, comprehend the true geometry of each asset, "
            "and output a strict JSON object containing geometric constraints that will 'pull' these elements from their temporary grid "
            "into a realistic, visually harmonious final composition.\n\n"

            "=== THE CORE LAYOUT MECHANISM: POLAR MAGNETS ===\n"
            "- The initial grid positions (x, y) ARE TEMPORARY. You must use the constraints to rearrange them.\n"
            "- To establish a spatial relationship between Object A and Object B, you MUST output EXACTLY ONE pair of `angular_position` (defines the direction) "
            "and `keep_close` (defines the distance range). Think of this as turning Object B into a magnet anchor for Object A.\n\n"

            "=== SIZE CHAINING RULE (VISUAL DIMENSION RATIO) ===\n"
            "- The backend calculates size ratio based on the **MAX DIMENSION (longest side: max(width, height))** of the object, NOT the 2D area.\n"
            "- You MUST link all objects together in a continuous relative size comparison tree (`size_compare`) so no object arbitrarily shrinks or blows up.\n"
            "- REAL-WORLD SCALE SANITY CHECK: Keep ratios realistic based on the LONGEST SIDE. For example, if a crane is 300px tall and a container is 100px long, the ratio is 3.0. "
            "Do NOT output massive ratios (e.g., > 4.0) for background assets like 'ground' or 'sky' to prevent them from choking the canvas; a ratio of 1.5 - 2.5 is usually enough to signify scale differences.\n\n"

            "=== OUTPUT FORMAT ===\n"
            "You MUST output a single valid JSON object with exactly ONE key. Do not include any Markdown blocks, backticks, or conversational text outside the JSON:\n"
            "{\n"
            '  "constraints": [ ... list of geometric constraints ... ]\n'
            "}\n\n"

            "=== AVAILABLE CONSTRAINT TYPES ===\n"
            "1. Angular Positioning (`angular_position`):\n"
            "   - PURPOSE: Pulls the subject center toward a specific directional axis relative to the object's geometric center.\n"
            "   - 'base_direction': The reference anchor direction based on standard 2D canvas coordinates (0° points Right, 90° points Down due to Canvas Y-axis pointing down).\n"
            "     - 'E'  (East / Straight Right, 0° - e.g., 3 o'clock direction)\n"
            "     - 'SE' (Southeast / Diagonal Bottom-Right, 45°)\n"
            "     - 'S'  (South / Straight Down, 90° - e.g., 6 o'clock direction)\n"
            "     - 'SW' (Southwest / Diagonal Bottom-Left, 135°)\n"
            "     - 'W'  (West / Straight Left, 180° - e.g., 9 o'clock direction)\n"
            "     - 'NW' (Northwest / Diagonal Top-Left, -135°)\n"
            "     - 'N'  (North / Straight Up, -90° - e.g., 12 o'clock direction)\n"
            "     - 'NE' (Northeast / Diagonal Top-Right, -45°)\n"
            "   - 'offset_angle': Optional float (in degrees, from -22.5 to +22.5) to fine-tune the orientation vector. Positive means clockwise, negative means counter-clockwise.\n"
            "   - Parameters: {\"type\": \"angular_position\", \"subject\": \"id1\", \"object\": \"id2\", \"base_direction\": \"E\"|\"SE\"|\"S\"|\"SW\"|\"W\"|\"NW\"|\"N\"|\"NE\", \"offset_angle\": 0.0}\n\n"
            
            "2. Proximity Control (`keep_close`):\n"
            "   - Purpose: Controls the distance (radius) between the subject and the object centers. MUST be paired with every `angular_position` to fully lock a relationship.\n"
            "   - Parameters: {\"type\": \"keep_close\", \"subject\": \"id1\", \"object\": \"id2\", \"min_dist\": 50. \"max_dist\": 250}\n\n"
            
            "3. Relative Scale Chaining (`size_compare`):\n"
            "   - Purpose: Constrains the relative size of the LONGEST SIDE between two elements.\n"
            "   - Parameters: {\"type\": \"size_compare\", \"subject\": \"id1\", \"object\": \"id2\", \"relation\": \"larger_than\"|\"smaller_than\", \"ratio\": 2.0}\n\n"

            "=== CRITICAL CONSTRAINT PRINCIPLE ===\n"
            "- NEVER output redundant or conflicting spatial constraints for the same pair of objects.\n"
            "- Trust your eyes: Inspect the provided grid image, identify which asset is which ID, and plan how they should merge into the ultimate scene.\n\n"

            "=== SAMPLE CORRECT OUTPUT ===\n"
            "{\n"
            '  "constraints": [\n'
            '    {"type": "angular_position", "subject": "object A", "object": "object B", "base_direction": "SE", "offset_angle": 10.0},\n'
            '    {"type": "keep_close", "subject": "object A", "object": "object B", "min_dist": 60, "max_dist": 200},\n'
            '    {"type": "size_compare", "subject": "object B", "object": "object A", "relation": "larger_than", "ratio": 3.0},\n'
            "  ]\n"
            "}"
        )
        
        formatted_objects_info = []
        for obj in all_scene_objects:
            obj_id = obj.get("object_id") if isinstance(obj, dict) else getattr(obj, "object_id", str(obj))
            obj_depth = obj.get("depth") if isinstance(obj, dict) else getattr(obj, "depth", 0)
            formatted_objects_info.append(f"- ID: '{obj_id}', Layer Depth: {obj_depth}")
            
        objects_context_str = "\n".join(formatted_objects_info)

        user_prompt = f"""
        Global Concept: "{global_concept}"
        Available Scene Objects with Layer Info:
        {objects_context_str}

        Strictly follow the depth-matching rules. Output the JSON constraints for ALL objects to ensure a coherent composition.
        """

        json_str = self.query_multimodal_model(system_prompt, user_prompt, encoded_image)
        return json_str

    def init_optimizable_layout(self, current_step_objects, layout_history, global_layout_plan=None):
        optim_params = {}
        history_map = {item["object_id"]: item for item in layout_history}
        
        if isinstance(global_layout_plan, str):
            try:
                global_plan = json.loads(global_layout_plan)
            except Exception:
                global_plan = {}
        else:
            global_plan = global_layout_plan if global_layout_plan else {}
        
        for obj in current_step_objects:
            obj_id = obj["object_id"]
            raw_w = float(obj.get("raw_w", 100.0))
            raw_h = float(obj.get("raw_h", 100.0))
            
            if obj_id in history_map:
                hist = history_map[obj_id]
                optim_params[obj_id] = {
                    "x": torch.tensor(float(hist["x"]), requires_grad=True),
                    "y": torch.tensor(float(hist["y"]), requires_grad=True),
                    "scale": torch.tensor(float(hist["scale"]), requires_grad=True),
                    "raw_w": raw_w,
                    "raw_h": raw_h,
                    "frozen": True  
                }
            else:
                if obj_id in global_plan:
                    plan = global_plan[obj_id]
                    
                    init_x = float(plan.get("x", 0.5)) * 512.0
                    init_y = float(plan.get("y", 0.5)) * 512.0
                    
                    init_scale = float(plan.get("scale", 0.8))
                    init_scale = max(0.35, min(2.0, init_scale))
                    
                else:
                    init_x = 256.0
                    init_y = 256.0
                    init_scale = 0.8
                    print(f"⚠️ New object {obj_id}: not found in global plan; using default initialization -> pos=({init_x},{init_y}), scale={init_scale}")
                
                optim_params[obj_id] = {
                    "x": torch.tensor(init_x, requires_grad=True, dtype=torch.float32),
                    "y": torch.tensor(init_y, requires_grad=True, dtype=torch.float32),
                    "scale": torch.tensor(init_scale, requires_grad=True, dtype=torch.float32),
                    "raw_w": raw_w,
                    "raw_h": raw_h,
                    "frozen": False 
                }
        
        
        return optim_params

    def render_scene_manifest_to_png(self, scene_objects, output_canvas_path, canvas_size=512):
        svg_root = ET.Element("svg", {
            "width": str(canvas_size),
            "height": str(canvas_size),
            "viewBox": f"0 0 {canvas_size} {canvas_size}"
        })
        
        ET.SubElement(svg_root, "rect", {
            "width": "100%",
            "height": "100%",
            "fill": "#FFFFFF"
        })

        for obj in scene_objects:
            svg_file_path = obj.get("svg_path")
            if not svg_file_path or not os.path.exists(svg_file_path):
                print(f"⚠️ Asset {obj.get('object_id')} SVG path does not exist; skipping rendering.")
                continue
                
            try:
                tree = ET.parse(svg_file_path)
                root_sub = tree.getroot()
                
                group = ET.Element("g", {"id": f"group_{obj['object_id']}"})
                
                for child in list(root_sub):
                    group.append(child)
                    
                local_center_x = obj["raw_w"] / 2.0
                local_center_y = obj["raw_h"] / 2.0
                
                trans_x = obj["x"] - local_center_x * obj["scale"]
                trans_y = obj["y"] - local_center_y * obj["scale"]
                
                transform_str = f"translate({trans_x}, {trans_y}) scale({obj['scale']})"
                group.set("transform", transform_str)
                
                svg_root.append(group)
                
            except Exception as e:
                print(f"❌ Failed to parse asset {obj.get('object_id')}: {e}")

        temp_svg_path = output_canvas_path.replace(".png", "_manifest.svg")
        new_tree = ET.ElementTree(svg_root)
        
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        with open(temp_svg_path, "wb") as f:
            new_tree.write(f, encoding="utf-8", xml_declaration=True)

        try:
            import cairosvg
            cairosvg.svg2png(url=temp_svg_path, write_to=output_canvas_path)
        except ImportError:
            try:
                from svglib.svglib import svg2rlg
                from reportlab.graphics import renderPM
                drawing = svg2rlg(temp_svg_path)
                renderPM.drawToFile(drawing, output_canvas_path, fmt="PNG")
            except ImportError:
                print("❌ Error: neither cairosvg nor svglib is available; high-quality rasterization cannot be completed.")
                Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255)).save(output_canvas_path, "PNG")

