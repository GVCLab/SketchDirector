import os
import re
import xml.etree.ElementTree as ET
import cairosvg
from PIL import Image
import io
import numpy as np
from bs4 import BeautifulSoup
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from svgpathtools import (
    Arc,
    CubicBezier,
    Line,
    Path,
    QuadraticBezier,
    parse_path,
    svg2paths,
    wsvg,
)
def resample_svg_to_point_particles(input_svg_path, output_svg_path, num_target_strokes, radius=0.5):
    try:
        tree = ET.parse(input_svg_path)
        root = tree.getroot()
        viewbox_str = root.attrib.get('viewBox', '0 0 780 780')
        width_str = root.attrib.get('width', '780')
        height_str = root.attrib.get('height', '780')
    except Exception as e:
        print(f"[Warning] Failed to read SVG attributes; using default 780 space: {e}")
        viewbox_str, width_str, height_str = '0 0 780 780', '780', '780'


    paths, _ = svg2paths(input_svg_path)
    valid_paths = [p for p in paths if p.length() > 1e-5]

    if len(valid_paths) == 0:
        valid_paths = [Path(CubicBezier(100+100j, 200+150j, 300+250j, 400+400j))]

    path_lengths = [p.length() for p in valid_paths]
    total_length = sum(path_lengths)


    global_points = []

    for path_idx, path in enumerate(valid_paths):
        samples_count = max(50, int(path_lengths[path_idx] * 2))
        for t in np.linspace(0, 1, samples_count):
            p = path.point(t)
            global_points.append(np.array([p.real, p.imag]))
    global_points = np.array(global_points)


    diffs = np.diff(global_points, axis=0)
    lens = np.sqrt((diffs**2).sum(axis=1))
    cum_lens = np.concatenate(([0], np.cumsum(lens)))
    global_total_len = cum_lens[-1] if cum_lens[-1] > 1e-5 else 1.0


    target_lengths = np.linspace(0, global_total_len, num_target_strokes)

    svg_paths_list = []
    for l in target_lengths:
        idx = np.searchsorted(cum_lens, l)
        if idx == 0:
            cx, cy = global_points[0]
        elif idx >= len(cum_lens):
            cx, cy = global_points[-1]
        else:
            ratio = (l - cum_lens[idx-1]) / (cum_lens[idx] - cum_lens[idx-1] + 1e-8)
            cx, cy = global_points[idx-1] + ratio * (global_points[idx] - global_points[idx-1])


        p0 = (cx - radius, cy)
        p1 = (cx - radius * 0.33, cy)
        p2 = (cx + radius * 0.33, cy)
        p3 = (cx + radius, cy)


        particle_stroke = Path(CubicBezier(
            complex(p0[0], p0[1]),
            complex(p1[0], p1[1]),
            complex(p2[0], p2[1]),
            complex(p3[0], p3[1])
        ))
        svg_paths_list.append(particle_stroke)


    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_str}" height="{height_str}" viewBox="{viewbox_str}">\n')


        for idx, single_path in enumerate(svg_paths_list):
            d_string = single_path.d()
            f.write(f'  <path id="particle_{idx}" d="{d_string}" fill="none" stroke="black" stroke-width="7.0" stroke-linecap="round" stroke-linejoin="round" />\n')

        f.write('</svg>')


def resample_svg_to_fixed_cubic_beziers(input_svg_path, output_svg_path, num_target_strokes):
    with open(input_svg_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()

    soup = BeautifulSoup(svg_content, 'xml')


    all_paths = soup.find_all('path')

    if not all_paths:
        with open(output_svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        return


    path_info_list = []
    total_segments = 0

    for path_elem in all_paths:
        d_attr = path_elem.get('d')
        if not d_attr:
            continue


        curves = parse_path_to_curves(d_attr)
        if not curves:
            continue


        data_class = path_elem.get('data-class', '')

        path_info_list.append({
            'element': path_elem,
            'curves': curves,
            'num_curves': len(curves),
            'data_class': data_class,
            'original_d': d_attr
        })
        total_segments += len(curves)

    if total_segments == 0:
        with open(output_svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        return


    target_segments_per_path = []
    remaining = num_target_strokes

    for i, info in enumerate(path_info_list):
        if i == len(path_info_list) - 1:

            target = max(1, remaining)
        else:

            proportion = info['num_curves'] / total_segments
            target = max(1, int(round(num_target_strokes * proportion)))

            target = min(target, remaining - (len(path_info_list) - i - 1))

        target_segments_per_path.append(target)
        remaining -= target


    while sum(target_segments_per_path) < num_target_strokes:
        target_segments_per_path[-1] += 1
    while sum(target_segments_per_path) > num_target_strokes:
        if target_segments_per_path[-1] > 1:
            target_segments_per_path[-1] -= 1
        else:
            break


    new_paths = []

    for idx, info in enumerate(path_info_list):
        curves = info['curves']
        data_class = info['data_class']
        target_num = target_segments_per_path[idx]
        original_elem = info['element']


        all_segments = []

        if target_num == len(curves):

            all_segments = curves
        else:

            segments_per_curve = []
            remaining_segments = target_num

            for i, curve in enumerate(curves):
                if i == len(curves) - 1:
                    segs = remaining_segments
                else:

                    curve_len = estimate_curve_length(curve)
                    total_len = sum(estimate_curve_length(c) for c in curves)
                    if total_len > 0:
                        segs = max(1, int(round(target_num * curve_len / total_len)))
                    else:
                        segs = max(1, target_num // len(curves))
                    segs = min(segs, remaining_segments - (len(curves) - i - 1))

                segments_per_curve.append(segs)
                remaining_segments -= segs


            for curve, segs in zip(curves, segments_per_curve):
                if segs == 1:
                    all_segments.append(curve)
                else:
                    split_curves = split_curve_into_segments(curve, segs)
                    all_segments.extend(split_curves)


        for seg in all_segments:
            new_path = soup.new_tag('path')

            d_str = curve_to_d_string(seg)
            new_path['d'] = d_str
            new_path['data-class'] = data_class


            for attr_name, attr_value in original_elem.attrs.items():
                if attr_name not in ['d', 'data-class']:
                    new_path[attr_name] = attr_value

            new_paths.append(new_path)


    for info in path_info_list:
        info['element'].decompose()


    svg_root = soup.find('svg')
    if svg_root is None:
        svg_root = soup


    g_elem = svg_root.find('g')
    if g_elem is None:
        g_elem = soup.new_tag('g')

        for attr_name, attr_value in svg_root.attrs.items():
            if attr_name != 'xmlns':
                g_elem[attr_name] = attr_value

        svg_root.clear()
        svg_root.append(g_elem)

    g_elem.extend(new_paths)


    with open(output_svg_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))


def parse_path_to_curves(d_string):
    curves = []
    d_string = d_string.strip()
    commands = re.findall(r'([MLCQ])\s*([\d\s.,-]+)', d_string)

    current_point = None

    for cmd, params_str in commands:

        params = re.findall(r'[-+]?\d*\.?\d+', params_str)
        if not params:
            continue

        coords = [float(p) for p in params]

        if cmd == 'M':

            if len(coords) >= 2:
                current_point = (coords[0], coords[1])
        elif cmd == 'L':

            if current_point and len(coords) >= 2:
                x1, y1 = current_point
                x2, y2 = coords[0], coords[1]
                curves.append({
                    'type': 'line',
                    'start': (x1, y1),
                    'end': (x2, y2)
                })
                current_point = (x2, y2)
        elif cmd == 'C':

            if current_point and len(coords) >= 6:
                x1, y1 = current_point
                x2, y2 = coords[0], coords[1]
                x3, y3 = coords[2], coords[3]
                x4, y4 = coords[4], coords[5]
                curves.append({
                    'type': 'cubic',
                    'start': (x1, y1),
                    'cp1': (x2, y2),
                    'cp2': (x3, y3),
                    'end': (x4, y4)
                })
                current_point = (x4, y4)
        elif cmd == 'Q':

            if current_point and len(coords) >= 4:
                x1, y1 = current_point
                x2, y2 = coords[0], coords[1]
                x3, y3 = coords[2], coords[3]

                cp1 = (x1 + 2*(x2 - x1)/3, y1 + 2*(y2 - y1)/3)
                cp2 = (x3 + 2*(x2 - x3)/3, y3 + 2*(y2 - y3)/3)
                curves.append({
                    'type': 'cubic',
                    'start': (x1, y1),
                    'cp1': cp1,
                    'cp2': cp2,
                    'end': (x3, y3)
                })
                current_point = (x3, y3)

    return curves


def split_curve_into_segments(curve, num_segments):
    if num_segments <= 1:
        return [curve]

    if curve['type'] == 'line':
        return split_line_segments(curve, num_segments)
    elif curve['type'] == 'cubic':
        return split_cubic_segments(curve, num_segments)
    else:
        return [curve]


def split_line_segments(line, num_segments):
    segments = []
    x1, y1 = line['start']
    x2, y2 = line['end']

    for i in range(num_segments):
        t0 = i / num_segments
        t1 = (i + 1) / num_segments

        start_x = x1 + t0 * (x2 - x1)
        start_y = y1 + t0 * (y2 - y1)
        end_x = x1 + t1 * (x2 - x1)
        end_y = y1 + t1 * (y2 - y1)

        segments.append({
            'type': 'line',
            'start': (start_x, start_y),
            'end': (end_x, end_y)
        })

    return segments


def split_cubic_segments(cubic, num_segments):
    segments = []


    p0 = np.array(cubic['start'])
    p1 = np.array(cubic['cp1'])
    p2 = np.array(cubic['cp2'])
    p3 = np.array(cubic['end'])

    for i in range(num_segments):
        t0 = i / num_segments
        t1 = (i + 1) / num_segments


        if t0 == 0:
            left, right = de_casteljau_split(p0, p1, p2, p3, t1)
            seg_control = left
        else:

            left_t1, right_t1 = de_casteljau_split(p0, p1, p2, p3, t1)

            t_relative = t0 / t1 if t1 > 0 else 0
            if t_relative > 0 and t_relative < 1:
                left_t0, right_t0 = de_casteljau_split(
                    left_t1[0], left_t1[1], left_t1[2], left_t1[3], t_relative
                )
                seg_control = right_t0
            else:
                seg_control = left_t1

        segments.append({
            'type': 'cubic',
            'start': tuple(seg_control[0]),
            'cp1': tuple(seg_control[1]),
            'cp2': tuple(seg_control[2]),
            'end': tuple(seg_control[3])
        })

    return segments


def de_casteljau_split(p0, p1, p2, p3, t):
    q0 = p0 + t * (p1 - p0)
    q1 = p1 + t * (p2 - p1)
    q2 = p2 + t * (p3 - p2)


    r0 = q0 + t * (q1 - q0)
    r1 = q1 + t * (q2 - q1)


    s0 = r0 + t * (r1 - r0)


    left = (p0, q0, r0, s0)


    right = (s0, r1, q2, p3)

    return left, right


def estimate_curve_length(curve):
    if curve['type'] == 'line':
        x1, y1 = curve['start']
        x2, y2 = curve['end']
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)
    elif curve['type'] == 'cubic':

        p0 = np.array(curve['start'])
        p1 = np.array(curve['cp1'])
        p2 = np.array(curve['cp2'])
        p3 = np.array(curve['end'])

        total_len = 0
        prev = p0
        for i in range(1, 21):
            t = i / 20

            point = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
            total_len += np.linalg.norm(point - prev)
            prev = point

        return total_len
    else:
        return 1.0


def curve_to_d_string(curve):
    if curve['type'] == 'line':
        x1, y1 = curve['start']
        x2, y2 = curve['end']
        return f"M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f}"
    elif curve['type'] == 'cubic':
        x1, y1 = curve['start']
        cx1, cy1 = curve['cp1']
        cx2, cy2 = curve['cp2']
        x2, y2 = curve['end']
        return f"M {x1:.2f} {y1:.2f} C {cx1:.2f} {cy1:.2f} {cx2:.2f} {cy2:.2f} {x2:.2f} {y2:.2f}"
    else:
        return ""


def get_bbox(svg_path):
    try:

        tree = ET.parse(svg_path)
        root = tree.getroot()


        namespaces = {'svg': 'http://www.w3.org/2000/svg'}


        paths = root.findall('.//{http://www.w3.org/2000/svg}path')
        if not paths:

            paths = root.findall('.//path')

        if not paths:
            print(f"[Warning] No path elements found in {svg_path}; using the default size.")
            return {"w": 100.0, "h": 100.0}

        xmin_total, xmax_total = float('inf'), float('-inf')
        ymin_total, ymax_total = float('inf'), float('-inf')


        for path_node in paths:
            d_string = path_node.attrib.get('d', '')
            if not d_string.strip():
                continue


            path_obj = parse_path(d_string)
            if len(path_obj) == 0:
                continue


            xmin, xmax, ymin, ymax = path_obj.bbox()


            if xmin < xmin_total: xmin_total = xmin
            if xmax > xmax_total: xmax_total = xmax
            if ymin < ymin_total: ymin_total = ymin
            if ymax > ymax_total: ymax_total = ymax


        if xmin_total == float('inf'):
            return {"w": 100.0, "h": 100.0}


        raw_w = xmax_total - xmin_total
        raw_h = ymax_total - ymin_total

        return {
            "w": float(raw_w),
            "h": float(raw_h),
        }

    except Exception as e:
        print(f"[Warning] Failed to parse SVG bounding box: {e}; falling back to viewBox.")

        try:
            viewbox = root.attrib.get('viewBox', '').split()
            if len(viewbox) == 4:
                return {"w": float(viewbox[2]), "h": float(viewbox[3])}
        except:
            pass
        return {"w": 100.0, "h": 100.0}

def path_to_shapely_polygon(path, samples_per_curve=20, padding=2):
    polygons = []
    for subpath in path.continuous_subpaths():
        points = []
        for segment in subpath:
            if isinstance(segment, Line):
                points.append((segment.start.real, segment.start.imag))
                points.append((segment.end.real, segment.end.imag))
            elif isinstance(segment, (CubicBezier, QuadraticBezier, Arc)):
                points.append((segment.start.real, segment.start.imag))
                for t in np.linspace(0, 1, samples_per_curve):
                    p = segment.point(t)
                    points.append((p.real, p.imag))
                points.append((segment.end.real, segment.end.imag))
        if len(points) >= 2:
            if len(points) == 2 or points[0] != points[-1]:
                poly = LineString(points).buffer(padding)
            else:
                poly = Polygon(points).buffer(padding)
            polygons.append(poly)
    return unary_union(polygons) if polygons else None


def inject_pure_vector_white_box(svg_path: str, output_svg_path: str, padding: float = 15.0):
    if not os.path.exists(svg_path):
        print(f"[Error] Input file does not exist: {svg_path}")
        return


    ET.register_namespace('', "http://www.w3.org/2000/svg")
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[Error] Failed to parse SVG XML: {e}")
        return

    ns = {'svg': 'http://www.w3.org/2000/svg'}


    all_paths_data = []
    all_geoms = []


    for path_elem in root.findall('.//svg:path', ns):
        d_attr = path_elem.get('d')
        if not d_attr:
            continue

        try:
            p_obj = parse_path(d_attr)
            geom = path_to_shapely_polygon(p_obj, samples_per_curve=20, padding=2)

            if geom and not geom.is_empty:
                all_geoms.append(geom)

                all_paths_data.append({
                    'attrs': dict(path_elem.attrib),
                    'geom': geom,
                    'd': d_attr,
                    'element': path_elem
                })
        except Exception as e:
            print(f"[Warning] Skipping an unparseable path geometry: {e}")

    if not all_geoms:
        print("[Error] No valid path geometry found in the input SVG.")
        return


    merged_all = unary_union(all_geoms)
    fill_distance = 30
    solid_fill = merged_all.buffer(fill_distance).buffer(-fill_distance)
    final_white_bg = solid_fill.buffer(padding)

    if final_white_bg.geom_type == "MultiPolygon":
        final_white_bg = max(final_white_bg.geoms, key=lambda x: x.area)
    if final_white_bg.geom_type == "Polygon" and len(final_white_bg.interiors) > 0:
        final_white_bg = Polygon(final_white_bg.exterior)

    def geom_to_svg_path(geom, precision=2):
        if geom.is_empty: return ""
        coords = list(geom.exterior.coords)
        if len(coords) < 3: return ""
        points = [f"{x:.{precision}f},{y:.{precision}f}" for x, y in coords]
        return f"M {points[0]} " + " ".join([f"L {p}" for p in points[1:]]) + " Z"

    outline_path_d = geom_to_svg_path(final_white_bg)


    cleaned_paths_data = []
    skipped_count = 0
    for d in all_paths_data:
        if final_white_bg.intersects(d['geom']):
            cleaned_paths_data.append(d)
        else:
            skipped_count += 1


    all_groups = {}

    def collect_group_ids(element, parent_path=""):
        """Collect IDs from nested SVG groups."""
        for child in element:
            if child.tag.endswith('g'):
                g_id = child.get('id')
                if g_id:
                    full_path = f"{parent_path}/{g_id}" if parent_path else g_id
                    all_groups[full_path] = {
                        'id': g_id,
                        'element': child,
                        'parent_path': parent_path,
                        'paths': child.findall('.//svg:path', ns)
                    }

                collect_group_ids(child, f"{parent_path}/{g_id}" if g_id else parent_path)

    collect_group_ids(root)


    if not all_groups:

        all_g_elements = root.findall('.//svg:g', ns)
        for g_elem in all_g_elements:
            g_id = g_elem.get('id')
            if g_id:
                all_groups[g_id] = {
                    'id': g_id,
                    'element': g_elem,
                    'parent_path': '',
                    'paths': g_elem.findall('.//svg:path', ns)
                }


    if not all_groups:
        print("[Warning] No SVG group ID found; using the default object ID.")
        g_id = "default_object"
    else:


        best_g = max(all_groups.values(), key=lambda x: len(x['paths']))
        g_id = best_g['id']


        all_g_ids = [info['id'] for info in all_groups.values()]
        if len(all_g_ids) > 1:
            pass


    minx, miny, maxx, maxy = final_white_bg.bounds


    new_width = maxx - minx
    new_height = maxy - miny


    margin = 10
    new_width += 2 * margin
    new_height += 2 * margin


    offset_x = -(minx - margin)
    offset_y = -(miny - margin)

    def translate_path_d(d_string, dx, dy):
        try:

            paths = parse_path(d_string)
            translated_paths = paths.translated(complex(dx, dy))
            return translated_paths.d()
        except:


            return d_string


    new_svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{new_width:.1f}" height="{new_height:.1f}">',
        '  <defs>',
        '    <style>',
        '      .white-box-bg { fill: white; stroke: none; }',
        '      .stroke-group { stroke-linecap: round; stroke-linejoin: round; }',
        '    </style>',
        '  </defs>',
        '  '
    ]


    translated_bg_d = translate_path_d(outline_path_d, offset_x, offset_y)
    new_svg_lines.append(f'  <path class="white-box-bg" d="{translated_bg_d}"/>')
    new_svg_lines.append('  ')
    new_svg_lines.append(f'  <g id="{g_id}" class="stroke-group" stroke="black" stroke-width="7.0" fill="none">')


    for d in cleaned_paths_data:
        attrs = d['attrs']

        attrs.pop('d', None)


        if 'fill' not in attrs: attrs['fill'] = 'none'
        if 'stroke' not in attrs: attrs['stroke'] = 'black'


        translated_d = translate_path_d(d["d"], offset_x, offset_y)


        attr_str = " ".join([f'{k}="{v}"' for k, v in attrs.items()])

        new_svg_lines.append(f'    <path {attr_str} d="{translated_d}" />')

    new_svg_lines.append('  </g>')
    new_svg_lines.append('</svg>')


    output_dir = os.path.dirname(output_svg_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_svg_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(new_svg_lines))


def clean_svg_namespace(svg_path, temp_suffix='_clean_temp'):
    ET.register_namespace('', "http://www.w3.org/2000/svg")

    tree = ET.parse(svg_path)
    root = tree.getroot()

    def remove_ns(element):
        if element.tag.endswith('}svg') or element.tag.startswith('svg:'):
            element.tag = 'svg'
        elif '}' in element.tag:
            element.tag = element.tag.split('}')[-1]
        elif element.tag.startswith('svg:'):
            element.tag = element.tag.replace('svg:', '')

        for child in element:
            remove_ns(child)

    remove_ns(root)

    temp_clean_path = svg_path.replace('.svg', f'{temp_suffix}.svg')
    tree.write(temp_clean_path, encoding='utf-8', xml_declaration=True)
    return temp_clean_path


def filter_svg_paths_by_length(svg_path, length_threshold=15.0):
    clean_svg_file = clean_svg_namespace(svg_path)
    loaded_paths, loaded_attrs = svg2paths(clean_svg_file)


    if os.path.exists(clean_svg_file):
        os.remove(clean_svg_file)

    filtered_path_objs = []
    filtered_attrs = []
    deleted_points = []
    kept_points = []


    all_lengths = [seg.length() for p_obj in loaded_paths for seg in p_obj]
    if all_lengths:
        stats = {
            'total_segments': len(all_lengths),
            'max_length': max(all_lengths),
            'min_length': min(all_lengths),
            'mean_length': np.mean(all_lengths)
        }
    else:
        print("[Warning] No paths found in the SVG file.")
        stats = {'total_segments': 0, 'max_length': 0, 'min_length': 0, 'mean_length': 0}


    for p_obj, attr in zip(loaded_paths, loaded_attrs):
        new_subsegments = []
        for segment in p_obj:
            if segment.length() < length_threshold:
                deleted_points.append([segment.start.real, segment.start.imag])
            else:
                new_subsegments.append(segment)
                kept_points.append([segment.start.real, segment.start.imag])

        if new_subsegments:
            filtered_path_objs.append(Path(*new_subsegments))
            filtered_attrs.append(attr)


    if not filtered_path_objs and loaded_paths:
        print(f"[Warning] Length threshold {length_threshold} removed all paths; keeping the original paths.")
        filtered_path_objs = loaded_paths
        filtered_attrs = loaded_attrs
        kept_points = [[seg.start.real, seg.start.imag] for p_obj in loaded_paths for seg in p_obj]
        deleted_points = []

    return filtered_path_objs, filtered_attrs, kept_points, deleted_points, stats


def extract_control_points(path_objs):
    points_list = []
    for p_obj in path_objs:
        for segment in p_obj:
            if isinstance(segment, CubicBezier):
                for pt in [segment.start, segment.control1, segment.control2, segment.end]:
                    points_list.append([pt.real, pt.imag])
            elif isinstance(segment, Line):
                for pt in [segment.start, segment.end]:
                    points_list.append([pt.real, pt.imag])
            else:

                for pt in [segment.start, segment.end]:
                    points_list.append([pt.real, pt.imag])

    return np.array(points_list) if points_list else np.array([])


def extract_points_array_from_svg(svg_path):
    clean_svg_file = clean_svg_namespace(svg_path)
    loaded_paths, _ = svg2paths(clean_svg_file)


    if os.path.exists(clean_svg_file):
        os.remove(clean_svg_file)

    return extract_control_points(loaded_paths)


def save_filtered_svg_and_points(filtered_path_objs, filtered_attrs, output_dir, base_name):
    svg_path = os.path.join(output_dir, f"{base_name}_filtered.svg")
    points_path = os.path.join(output_dir, f"{base_name}_filtered_points.txt")


    wsvg(filtered_path_objs, attributes=filtered_attrs, filename=svg_path)


    points_array = extract_control_points(filtered_path_objs)
    if len(points_array) > 0:
        np.savetxt(points_path, points_array, fmt='%f', delimiter=',')
    else:
        open(points_path, 'w').close()

    return svg_path, points_path, points_array


def create_debug_visualization(original_svg_path, kept_points, deleted_points, output_path=None):
    if output_path is None:
        output_path = original_svg_path.replace('_filtered.svg', '_debug_dots.svg')


    tree = ET.parse(original_svg_path)
    root = tree.getroot()


    g_labels = ET.Element('g', id='debug_markers')


    for pt in kept_points:
        circle = ET.Element('circle', cx=str(pt[0]), cy=str(pt[1]), r='2', fill='black')
        g_labels.append(circle)


    for pt in deleted_points:
        circle = ET.Element('circle', cx=str(pt[0]), cy=str(pt[1]), r='4',
                           fill='red', stroke='white', stroke_width='0.5')
        g_labels.append(circle)

    root.append(g_labels)
    tree.write(output_path)

    return output_path


def filter_short_curves(svg_path, length_threshold=15.0, output_dir=None, debug=True):
    import xml.etree.ElementTree as ET
    from bs4 import BeautifulSoup


    if output_dir is None:
        output_dir = os.path.dirname(svg_path)

    base_name = os.path.basename(svg_path).replace('.svg', '')


    with open(svg_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()

    soup = BeautifulSoup(svg_content, 'xml')


    svg_root = soup.find('svg')
    orig_viewbox = svg_root.get('viewBox', '0 0 512 512') if svg_root else '0 0 512 512'
    orig_width = svg_root.get('width', '512') if svg_root else '512'
    orig_height = svg_root.get('height', '512') if svg_root else '512'


    all_paths_with_context = []

    def collect_paths_with_context(element, parent_g=None, depth=0):
        """Collect paths together with their parent-group context."""
        for child in element.children:
            if child.name == 'g':

                g_info = {
                    'element': child,
                    'id': child.get('id', ''),
                    'attrs': dict(child.attrs),
                    'parent': parent_g,
                    'depth': depth
                }

                collect_paths_with_context(child, g_info, depth + 1)
            elif child.name == 'path':

                d_attr = child.get('d', '')
                if d_attr:
                    path_info = {
                        'element': child,
                        'd': d_attr,
                        'attrs': dict(child.attrs),
                        'parent_g': parent_g,
                        'depth': depth
                    }
                    all_paths_with_context.append(path_info)


    collect_paths_with_context(svg_root)


    kept_paths = []
    deleted_paths = []
    kept_points = []
    deleted_points = []
    stats = {'total_segments': len(all_paths_with_context)}

    for path_info in all_paths_with_context:
        try:
            d_attr = path_info['d']
            parsed_path = parse_path(d_attr)


            total_length = 0
            points = []
            for segment in parsed_path:
                seg_length = segment.length()
                total_length += seg_length

                for t in np.linspace(0, 1, 10):
                    point = segment.point(t)
                    points.append([point.real, point.imag])

            if total_length >= length_threshold:
                kept_paths.append(path_info)
                kept_points.extend(points)
            else:
                deleted_paths.append(path_info)
                deleted_points.extend(points)
        except Exception as e:
            print(f"[Warning] Failed to parse SVG path: {e}")

            kept_paths.append(path_info)

    stats['kept'] = len(kept_paths)
    stats['deleted'] = len(deleted_paths)


    new_svg = BeautifulSoup(features='xml')
    new_svg_root = new_svg.new_tag('svg',
                                   xmlns="http://www.w3.org/2000/svg",
                                   width=orig_width,
                                   height=orig_height,
                                   viewBox=orig_viewbox)
    new_svg.append(new_svg_root)


    def build_hierarchy(paths, root_element):
        top_level_groups = {}
        paths_by_parent = {}

        for path_info in paths:
            parent = path_info.get('parent_g')
            if parent is None:

                paths_by_parent[None] = paths_by_parent.get(None, []) + [path_info]
            else:
                parent_id = parent.get('id', '')
                if parent_id not in paths_by_parent:
                    paths_by_parent[parent_id] = []
                paths_by_parent[parent_id].append(path_info)

                if parent.get('depth', 0) == 0:
                    top_level_groups[parent_id] = parent


        for g_id, g_info in top_level_groups.items():

            new_g = new_svg.new_tag('g')
            for attr_name, attr_value in g_info['attrs'].items():
                new_g[attr_name] = attr_value


            if g_id in paths_by_parent:
                for path_info in paths_by_parent[g_id]:
                    new_path = new_svg.new_tag('path')
                    for attr_name, attr_value in path_info['attrs'].items():
                        new_path[attr_name] = attr_value
                    new_g.append(new_path)

            root_element.append(new_g)


        if None in paths_by_parent:
            for path_info in paths_by_parent[None]:
                new_path = new_svg.new_tag('path')
                for attr_name, attr_value in path_info['attrs'].items():
                    new_path[attr_name] = attr_value
                root_element.append(new_path)


    build_hierarchy(kept_paths, new_svg_root)


    filtered_svg_path = os.path.join(output_dir, f"{base_name}_filtered.svg")
    with open(filtered_svg_path, 'w', encoding='utf-8') as f:
        f.write(str(new_svg))


    points_array = np.array(kept_points) if kept_points else np.array([])
    filtered_points_path = os.path.join(output_dir, f"{base_name}_filtered_points.npy")
    if len(points_array) > 0:
        np.save(filtered_points_path, points_array)


    debug_svg_path = None
    if debug and filtered_svg_path:
        debug_svg_path = create_debug_visualization(
            filtered_svg_path, kept_points, deleted_points
        )

    if filtered_points_path:
        pass

    return {
        'filtered_svg': filtered_svg_path,
        'filtered_points': filtered_points_path,
        'points_var': points_array,
        'debug_visualization': debug_svg_path,
        'kept_points': kept_points,
        'deleted_points': deleted_points,
        'statistics': stats
    }

def filter_short_curves_simple(svg_path, length_threshold=15.0, output_dir=None):
    results = filter_short_curves(svg_path, length_threshold, output_dir, debug=False)
    return results['filtered_svg'], results['points_var']

def extract_white_box_mask(
    bg_svg_path,
    output_path,
    canvas_size=None
):

    tree = ET.parse(bg_svg_path)
    root = tree.getroot()


    for elem in list(root):

        tag = elem.tag.split("}")[-1]


        if tag == "defs":
            continue


        if tag == "path":
            cls = elem.attrib.get("class","")

            if cls == "white-box-bg":
                continue


        root.remove(elem)


    svg_bytes = ET.tostring(
        root,
        encoding="utf-8"
    )


    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes,
        output_width=canvas_size,
        output_height=canvas_size
    )


    img = Image.open(
        io.BytesIO(png_bytes)
    ).convert("RGBA")


    alpha = img.getchannel("A")


    mask = alpha.point(
        lambda x:255 if x>10 else 0
    )


    mask.save(output_path)

    return output_path
