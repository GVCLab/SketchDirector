import xml.etree.ElementTree as ET
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import json

def grid_point_to_pixel(point_str, cells_to_pixels_map):

    match = re.match(r"x(\d+)y(\d+)", point_str)

    if match is None:
        raise ValueError(f"Invalid point: {point_str}")

    x = int(match.group(1))
    y = int(match.group(2))

    key = f"x{x}y{y}"

    if key not in cells_to_pixels_map:
        raise KeyError(
            f"{key} not found. Example keys: "
            f"{list(cells_to_pixels_map.keys())[:5]}"
        )

    px = cells_to_pixels_map[key][0]
    py = cells_to_pixels_map[key][1]

    return px, py

def extract_svg(text):

    if text is None:
        return None

    code_block_pattern = r"```(?:svg|xml)?\s*(<svg[\s\S]*?</svg>)\s*```"
    match = re.search(code_block_pattern, text, re.IGNORECASE)

    if match:
        return match.group(1).strip()

    svg_pattern = r"(<svg[\s\S]*?</svg>)"
    match = re.search(svg_pattern, text, re.IGNORECASE)

    if match:
        return match.group(1).strip()

    open_tag_pattern = r"(<svg[\s\S]*)"
    match = re.search(open_tag_pattern, text, re.IGNORECASE)

    if match:
        svg_partial = match.group(1)

        if "</svg>" not in svg_partial:
            svg_partial += "\n</svg>"

        return svg_partial.strip()

    raise ValueError("❌ No valid SVG found in LLM response:\n" + text)
    
def visualize_skeleton_on_image(bg_image, skeleton_data, res=50, save_path=None):
    
    canvas = bg_image.resize((512, 512)).convert("RGB")
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)
    
    skeleton_color = (255, 0, 0)  
    box_color = (0, 255, 255)     
    
    edges = skeleton_data.get('edges', [])
    nodes_def = skeleton_data.get('nodes_definition', {})
    
    if 'frames' in skeleton_data:
        frame = skeleton_data['frames'][0]
    elif 'placement' in skeleton_data:
        frame = skeleton_data['placement']
    else:
        print("Invaild")
        return canvas

    nodes = frame.get('nodes', {})

    for node_id, pos in nodes.items():
        node_info = nodes_def.get(node_id, {})
        if isinstance(node_info, dict) and 'area_box' in node_info:
            box_dim = node_info['area_box'] 
            if len(box_dim) == 2:
                bw, bh = box_dim[0], box_dim[1]
                
                grid_x1, grid_y1 = pos[0] - bw/2, pos[1] - bh/2
                grid_x2, grid_y2 = pos[0] + bw/2, pos[1] + bh/2
                
                bx1 = (grid_x1 / res) * width
                by1 = (grid_y1 / res) * height
                bx2 = (grid_x2 / res) * width
                by2 = (grid_y2 / res) * height
                
                draw.rectangle([bx1, by1, bx2, by2], outline=box_color, width=2)
              
                draw.text((bx1, by1 - 10), f"box:{node_id.split('_')[-1]}", fill=box_color)

    for edge in edges:
        start_id, end_id = edge[0], edge[1]
        if start_id in nodes and end_id in nodes:
            p1_x = (nodes[start_id][0] / res) * width
            p1_y = (nodes[start_id][1] / res) * height
            p2_x = (nodes[end_id][0] / res) * width
            p2_y = (nodes[end_id][1] / res) * height
            
            draw.line([p1_x, p1_y, p2_x, p2_y], fill=skeleton_color, width=3)

    for node_id, pos in nodes.items():
        px_x = (pos[0] / res) * width
        px_y = (pos[1] / res) * height
        
        r = 6
        draw.ellipse([px_x-r, px_y-r, px_x+r, px_y+r], fill=skeleton_color, outline="white", width=1)
        
        label = node_id.split('_')[-1]
        draw.text((px_x + 7, px_y - 7), label, fill="white")

    if save_path:
        canvas.save(save_path)
        print(f"Saved in: {save_path}")
    
    return canvas

def overlay_grid_on_image(image, res=64, target_size=(512, 512), grid_color=(200, 200, 200, 100), line_width=1):
   
    img = image.resize(target_size, Image.LANCZOS).convert("RGBA")
    width, height = img.size
    
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    x_step = width / res
    y_step = height / res
    
    for i in range(res + 1):
        x = i * x_step
        draw.line([(x, 0), (x, height)], fill=grid_color, width=line_width)
        
    for j in range(res + 1):
        y = height - (j * y_step) 
        draw.line([(0, y), (width, y)], fill=grid_color, width=line_width)

    try:
        font = ImageFont.load_default()
    except:
        font = None

    for i in range(0, res + 1, 5):
      
        curr_x = i * x_step
        curr_y = height - (i * y_step)
        
        draw.text((curr_x + 2, height - 15), f"{i}", fill=(255, 0, 0, 200), font=font)
       
        draw.text((5, curr_y - 12), f"{i}", fill=(255, 0, 0, 200), font=font)

    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")


def create_grid_image(res=50, cell_size=12, header_size=12):
   
    rows = res
    cols = res
    
    img_width = (cols + 1) * cell_size
    img_height = (rows + 1) * cell_size
    
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", header_size*0.85)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw the headers
    for j in range(cols):
        text = str(j +1)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (j + 1) * cell_size + (cell_size - text_width) / 2
        text_y = img_height - cell_size # - (cell_size - text_height) / 2
        draw.text((text_x, text_y), text, fill="black", font=font)
    
    for i in range(rows):
        text = str(rows - i)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (cell_size - text_width) / 2
        text_y = i * cell_size + (cell_size - text_height) / 2 - 0.2*text_height
        draw.text((text_x, text_y), text, fill="black", font=font)
    
    i = 1
    draw.line([(i * cell_size, 0), (i * cell_size, img_height)], fill="black")
    draw.line([(0, img_height -  cell_size), (img_width, img_height -  cell_size)], fill="black")
    
    positions={}
    for i in range(rows)[::-1]:
        for j in range(cols):
            if j == 0:
                draw.rectangle([(j + 0) * cell_size, (i + 0) * cell_size, (j + 1) * cell_size, (i + 1) * cell_size], outline="black")
            if i == rows - 1:
                draw.rectangle([(j + 0) * cell_size, (i + 1) * cell_size, (j + 1) * cell_size, (i + 2) * cell_size], outline="black")
    
            text = f"x{j + 1}y{i + 1}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (j + 1) * cell_size + (cell_size - text_width) / 2
            text_y = (i + 1) * cell_size + (cell_size - text_height) / 2
            
            center_y = int(img_height - cell_size - (i * cell_size) - cell_size / 2)
            center_x = int(j * cell_size + cell_size / 2 + cell_size)
            positions[text] = (center_x, center_y)
    return img, positions


def cells_to_pixels(res=50, cell_size=12, header_size=12):
    rows = res
    cols = res
    
    img_width = (cols + 1) * cell_size
    img_height = (rows + 1) * cell_size

    positions={}
    # Draw the grid
    for i in range(rows)[::-1]:
        for j in range(cols):
            text = f"x{j + 1}y{i + 1}"
            
            center_y = int(img_height - cell_size - (i * cell_size) - cell_size / 2)
            center_x = int(j * cell_size + cell_size / 2 + cell_size)
            positions[text] = (center_x, center_y)

    return positions


def image_to_str(image: Image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    image = base64.b64encode(buffer.read()).decode('utf-8')
    return image



def bezier_point(P, t):
    """Calculate a point on the Bézier curve for a given t."""
    return (1-t)**3 * P[0] + 3*(1-t)**2 * t * P[1] + 3*(1-t) * t**2 * P[2] + t**3 * P[3]


import torch
import random

def generate_bezier_from_sampled_points(
    sampled_points,
    num_segments=1,
    control_points_per_seg=4,  # cubic Bézier = 4
    radius=0.05,
    canvas_width=512,
    canvas_height=512,
    device="cuda"
):
    all_paths = []

    for sp in sampled_points:
        points = []

        p0 = (float(sp[0]), float(sp[1]))
        points.append(p0)

        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + (control_points_per_seg - 2)

        for j in range(num_segments):
            for k in range(control_points_per_seg - 1):
                dx = radius * (random.random() - 0.5)
                dy = radius * (random.random() - 0.5)

                p1 = (p0[0] + dx, p0[1] + dy)
                points.append(p1)
                p0 = p1  

        points = torch.tensor(points, dtype=torch.float32).to(device)

        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height

        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=torch.tensor(1.0),
            is_closed=False
        )

        all_paths.append(path)

    return all_paths

def estimate_bezier_control_points_helper(sampled_points, t_values):
    n = len(sampled_points)
    
    if n == 1:
        P0 = np.array(sampled_points[0])
        P1 = np.array(sampled_points[0]).astype(np.float64) + 0.0001
        return np.array([P0, P1])
        
    if n == 2:
        P0 = np.array(sampled_points[0])
        P1 = np.array(sampled_points[1])
        return np.array([P0, P1])

    if n > len(t_values):
        t_values = np.linespace(0,1,n)
    
    elif n == 3:
        A = np.zeros((n, 3))
        for i in range(n):
            t = t_values[i]
            A[i, 0] = (1-t)**2
            A[i, 1] = 2*(1-t)*t
            A[i, 2] = t**2
        
        B = np.array(sampled_points).reshape(-1, 2)  # Assuming 2D points
        
        P = np.linalg.lstsq(A, B, rcond=None)[0]
        return P

    A = np.zeros((n, 4))
    for i in range(n):
        t = t_values[i]
        A[i, 0] = (1-t)**3
        A[i, 1] = 3*(1-t)**2 * t
        A[i, 2] = 3*(1-t) * t**2
        A[i, 3] = t**3
    
    B = np.array(sampled_points).reshape(-1, 2)  # Assuming 2D points
    
    P = np.linalg.lstsq(A, B, rcond=None)[0]
    return P

    
def estimate_bezier_control_points( sampled_points, t_values):
    if len(sampled_points) != len(t_values):
        t_values = np.linspace(0,1, len(sampled_points))
    P = estimate_bezier_control_points_helper(sampled_points, t_values)

    if len(sampled_points) > 4:
        errors = []
        for i, t in enumerate(t_values):
            B_t = bezier_point(P, t)
            error = np.linalg.norm(B_t - sampled_points[i])
            errors.append(error)
        error = np.mean(errors)
        
        if error > 5 and len(sampled_points) >= 7:
            mid = len(sampled_points) // 2
            left_sampled_points = sampled_points[:mid+1]
            right_sampled_points = sampled_points[mid:]
            left_t_values = np.array(t_values[:mid+1])
            right_t_values = np.array(t_values[mid:])

            if len(left_sampled_points) == 3: 
                left_sampled_points.append(right_sampled_points[0])
                left_t_values.append(right_t_values[0])
                
            left_t_values = (left_t_values - left_t_values[0]) / (left_t_values[-1] - left_t_values[0])
            right_t_values = (right_t_values - right_t_values[0]) / (right_t_values[-1] - right_t_values[0])

            P_left = estimate_bezier_control_points_helper(left_sampled_points, left_t_values)
            P_right = estimate_bezier_control_points_helper(right_sampled_points, right_t_values)
            P_right[0] = P_left[-1] 
            return [P_left, P_right]
    return [P]


    
def get_control_points_t(strokes_all, t_values_all, cells_to_pixels_map):
    net_points = []      
    for j in range(len(strokes_all)):
        sampled_cells = strokes_all[j]
        t_values = t_values_all[j]
        sampled_points = []
        for cell in sampled_cells:
            y,x = cells_to_pixels_map[cell]
            sampled_points.append([y,x])
        points_lst = estimate_bezier_control_points(sampled_points, t_values)
        net_points.append(points_lst)
    return net_points

def get_control_points(strokes_all, cells_to_pixels_map):
    net_points = []      
    for j in range(len(strokes_all)):
        sampled_cells = strokes_all[j] 
        sampled_points = []
        
        for cell_item in sampled_cells:
            print(f"Processing cell: {cell_item}")
            y, x = cells_to_pixels_map[cell_item]
            sampled_points.append([y, x])
        
        if sampled_points:
            points_lst = estimate_bezier_control_points(sampled_points)
            net_points.append(points_lst)
            
    return net_points


def get_control_points_single_stroke(strokes_all, t_values_all, cells_to_pixels_map):
    sampled_points = []
    for cell in strokes_all:
        y,x = cells_to_pixels_map[cell]
        sampled_points.append([y,x])
    points_lst = estimate_bezier_control_points(sampled_points, t_values_all)
    return points_lst


def create_svg_path_data(control_points):
  
    path_data = 'M ' + np.array2string(np.array(control_points[0]), formatter={'float_kind':lambda x: "%.2f" % x}, separator=' ')[1:-1]    
    
    if len(control_points) == 1:
        path_data += ' '
   
    elif len(control_points) == 2:
        path_data += ' L '
 
    elif len(control_points) == 3:
        path_data += ' Q '
 
    elif len(control_points) == 4:
        path_data += ' C '
    
    for point in control_points[1:]:
        path_data += str(point[0]) + " " + str(point[1]) + " "
    
    return path_data


def format_svg_t(all_control_points, dim, stroke_width):
    svg_width, svg_height = dim
    sketch_text_svg = f"""<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">\n"""        
    for i, path in enumerate(all_control_points):
        gropu_text = f"""<g id="s{i + 1}" stroke="black" stroke-width="{stroke_width}" fill="none" stroke-linecap="round">\n"""
        for sub_path_cp in path:  #sometimes 1 or 2 
            path_data = create_svg_path_data(sub_path_cp)
            gropu_text += f"""<path d="{path_data}"/>\n"""
        gropu_text += "</g>\n"
        sketch_text_svg += gropu_text
    sketch_text_svg += "</svg>"
    return sketch_text_svg

def format_svg(all_control_points, dim, stroke_width=2):
    svg_width, svg_height = dim
   
    spacing = stroke_width * 2 
    
    sketch_text_svg = f"""<svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" xmlns="http://www.w3.org/2000/svg">\n"""        
    
    for i, path in enumerate(all_control_points):
      
        group_text = f"""<g id="s{i + 1}" stroke="black" stroke-width="{stroke_width}" 
        fill="none" stroke-linecap="round" stroke-dasharray="0 {spacing}">\n"""
        
        for sub_path_cp in path:
           
            path_data = create_svg_path_data(sub_path_cp)
            group_text += f"""<path d="{path_data}"/>\n"""
            
        group_text += "</g>\n"
        sketch_text_svg += group_text
        
    sketch_text_svg += "</svg>"
    return sketch_text_svg


def format_svg_single_stroke(group, dim, stroke_width, stroke_counter, stroke_color="black"):
    sketch_text_svg = ""      
    gropu_text = f"""<g id="s{stroke_counter}" stroke="{stroke_color}" stroke-width="{stroke_width}" fill="none" stroke-linecap="round">\n"""
    for sub_path_cp in group: 
        path_data = create_svg_path_data(sub_path_cp)
        gropu_text += f"""<path d="{path_data}"/>\n"""
    gropu_text += "</g>\n"
    sketch_text_svg += gropu_text
    return sketch_text_svg

def parse_xml_string_t(llm_output,res):
    try:
        clean_output = llm_output.strip()
        clean_output = clean_output.replace("</answer>", "")

        if clean_output.startswith("```"):
            clean_output = re.sub(
                r"```(?:json)?",
                "",
                clean_output
            )
            clean_output = clean_output.replace("```","")

        clean_output = clean_output[
            :clean_output.rfind("}")+1
        ]
        data=json.loads(clean_output)


    except Exception as e:
        print("JSON parse error:",e)
        return "[]","[]"

    if isinstance(data,dict) and "answer" in data:
        data=data["answer"]
    if not isinstance(data,dict):
        return "[]","[]"
    strokes=data.get("strokes",[])


    strokes_list="[\n"
    t_values_list="[\n"
    for stroke in strokes:
        if not isinstance(stroke,dict):
            continue
        points = stroke.get(
            "points",
            []
        )
        t_values = stroke.get(
            "t_values",
            []
        )
        strokes_list += f"{points},\n"
        t_values_list += f"{t_values},\n"
    strokes_list += "]"
    t_values_list += "]"

    strokes_list = re.sub(
        r'\d+',
        lambda x:str(
            max(
                min(int(x.group()),res),
                1
            )
        ),
        strokes_list
    )

    return strokes_list,t_values_list

def parse_xml_string(llm_output, res):

    strokes_start_marker = "<strokes>"
    strokes_end_marker = "</strokes>"

    start_index = llm_output.find(strokes_start_marker)
    if start_index != -1:
        end_index = llm_output.find(strokes_end_marker, start_index)
    else:
        return None  

    if end_index == -1:
        return None  

    strokes_str = llm_output[start_index:end_index + len(strokes_end_marker)].strip()#[:-1]
    xml_str = f"<wrap>{strokes_str}</wrap>"
    root = ET.fromstring(xml_str)
    
    final_strokes = []
    
    for stroke in root.find('strokes'):
        points_text = stroke.find('points').text
        points_list = re.findall(r'x\d+y\d+', points_text)
            
        processed_points = []
        for p in points_list:
            new_p = re.sub(r'\d+', lambda x: str(max(1, min(int(x.group()), res))), p)
            processed_points.append(new_p)
            
        final_strokes.append(processed_points)
    return final_strokes

def parse_description(llm_output):
    desc_start_marker = "<description>"
    desc_end_marker = "</description>"

    # Find start
    start_index = llm_output.find(desc_start_marker)
    if start_index == -1:
        return None

    # Find end
    end_index = llm_output.find(desc_end_marker, start_index)
    if end_index == -1:
        return None

    desc_str = llm_output[start_index:end_index + len(desc_end_marker)].strip()

    xml_str = f"<wrap>{desc_str}</wrap>"

    try:
        root = ET.fromstring(xml_str)
        description = root.find('description').text

        if description is not None:
            return description.strip()
        else:
            return ""

    except ET.ParseError:
        return None

def parse_xml_string_single_stroke(llm_output, res, stroke_counter):
    strokes_start_marker = f"<s{stroke_counter}>"
    strokes_end_marker = f"</s{stroke_counter}>"

    start_index = llm_output.find(strokes_start_marker)
    if start_index != -1:
        end_index = llm_output.find(strokes_end_marker, start_index)
    else:
        return None  

    if end_index == -1:
        return None 

    strokes_str = llm_output[start_index:end_index + len(strokes_end_marker)].strip()#[:-1]
    xml_str = f"<wrap>{strokes_str}</wrap>"
    root = ET.fromstring(xml_str)
    
    stroke = root.find(f"s{stroke_counter}")
    points_text = stroke.find('points').text

    t_values_text = stroke.find('t_values').text

    strokes_list = f"[{points_text}]"
    t_values_list = f"[{t_values_text}]"
    
    strokes_list = re.sub(r'\d+', lambda x: str(min(int(x.group()), res)), strokes_list)
    strokes_list = re.sub(r'\d+', lambda x: str(max(int(x.group()), 1)), strokes_list)
    
    return strokes_list, t_values_list


def get_cur_stroke_text(stroke_counter, llm_output):
    start_marker = f"<s{stroke_counter}>"
    end_marker = f"</s{stroke_counter}>"

    start_index = llm_output.find(start_marker)
    if start_index != -1:
        # start_index += len(strokes_start_marker)  # Move past the marker
        end_index = llm_output.find(end_marker, start_index)
    else:
        return ""  

    if end_index == -1:
        return ""  

    strokes_str = llm_output[start_index:end_index + len(end_marker)].strip()#[:-1]
    return strokes_str
