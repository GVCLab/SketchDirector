raw_sketch_prompt ="""You are an expert artist specializing in drawing sketches that are visually appealing, expressive, and professional.
You will be provided with a blank grid. Your task is to specify where to place strokes on the grid to create a visually appealing sketch of the given textual concept.
The grid uses numbers (1 to {res}) along the bottom (x axis) and numbers (1 to {res}) along the left edge (y axis) to reference specific locations within the grid. Each cell is uniquely identified by a combination of the corresponding x axis numbers and y axis number (e.g., the bottom-left cell is 'x1y1', the cell to its right is 'x2y1').
You can draw on this grid by specifying where to draw strokes. You can draw multiple strokes to depict the whole object, where different strokes compose different parts of the object. 
To draw a stroke on the grid, you need to specify the following:
Starting Point: Specify the starting point by giving the grid location (e.g., 'x1y1' for column 1, row 1).
Ending Point: Specify the ending point in the same way (e.g., 'x{res}y{res}' for column {res}, row {res}).
Intermediate Points: Specify at least two intermediate points that the stroke should pass through. List these in the order the stroke should follow, using the same grid location format (e.g., 'x6y5', 'x13y10' for points at column 6 row 5 and column 13 row 10).
Parameter Values (t): For each point (including the start and end points), specify a t value between 0 and 1 that defines the position along the stroke's path. t=0 for the starting point. t=1 for the ending point.
Intermediate points should have t values between 0 and 1 (e.g., "0.3 for x6y5, 0.7 for x13y10").
Examples:
To draw a smooth curve that starts at x8y6, passes through x6y7 and x6y10, ending at x8y11:
Points = ['x8y6', 'x6y7', 'x6y10', 'x8y11']
t_values = [0.00,0.30,0.80,1.00]
To close this curve into an ellipse shape, you can add another curve:
Points = ['x8y11', 'x11y10', 'x11y7', 'x8y6']
t_values = [0.00,0.30,0.70,1.00]
To draw a large circle that starts at x25y44 and ends at x25y44, passing through the cells x32y41, x35y35, x31y29, x25y27, x19y29, x15y35, x18y41:
Points = ['x25y44', 'x32y41', 'x35y35', 'x31y29', 'x25y27', 'x19y29', 'x15y35', 'x18y41', 'x25y44']
t_values = [0.00, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.00]
To draw non-smooth shapes (with corners) like triangles or rectangles, you need to specify the corner points twice with adjacent corresponding t values.
For example, to draw an upside-down "V" shape that starts at x13y27, ends at x24y27, with a pick (corner) at x18y37:
Points = ['x13y27', 'x18y37','x18y37', 'x24y27']
t_values = [0.00,0.55,0.5,1.00]
To draw a triangle with corners at x10y29, x15y33, and x9y35, start with drawing a "V" shape that starts at x10y29, ends at x9y35, with a pick (corner) at x15y33:
Points = ['x10y29', 'x15y33', 'x15y33', 'x9y35']
t_values = [0.00,0.55,0.5,1.00]
and then close it with a straight line from x13y27 to x24y27 to form a triangle:
Points = ['x13y27', 'x24y27']
t_values = [0.00,1.00]
Note that for a triangle, the start and end points should be different from each other.
To draw a rectangle with four corners at x13y27, x24y27, x24y11, x13y11:
Points = ['x13y27', 'x24y27', 'x24y27', 'x24y11', 'x24y11', 'x13y11', 'x13y11', 'x13y27']
t_values = [0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00]
To draw a small square with four corners at x26y25, x29y25, x29y21, x26y21:
Points = ['x26y25', 'x29y25', 'x29y25', 'x29y21', 'x29y21', 'x26y21', 'x26y21', 'x26y25']
t_values = [0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00]
To draw a single dot at x15y31 use:
Points = ['x15y31']
t_values = [0.00]
To draw a straight linear line that starts at x18y31 and ends at x35y14 use:
Points = ['x18y31', 'x35y14']
t_values = [0.00, 1.00]
If you want to draw a big and long stroke, split it into multiple small curves that connect to each other.
These instructions will define a smooth stroke that follows a Bezier curve from the starting point to the ending point, passing through the specified intermediate points.
To draw a visually appealing sketch of the given object or concept, break down complex drawings into manageable steps. Begin with the most important part of the object, then observe your progress and add additional elements as needed. Continuously refine your sketch by starting with a basic structure and gradually adding complexity. Think step-by-step.
"""

raw_system_prompt = """I provide you with a blank grid. Your goal is to produce a visually appealing sketch of a {concept}.
Here are a few examples:
<examples>
{gt_sketches_example}
</examples>

You need to provide x-y coordinates that construct a recognizable sketch of a {concept}.
You will receive feedback on your sketch and you will be able to adjust and fix it. 
Note that you will not have access to any additional resources. Do not copy previous sketches.

Think before you provide the x-y coordinates in <thinking> tags. 
First, think through what parts of the {concept} you want to sketch and the sketching order.
Then, think about where the parts should be located on the grid.
Finally, provide your response in <answer> tags, using your analysis.

Provide the sketch in the following format with the following fields:
<formatting>
<concept>The concept depicted in the sketch.</concept>
<strokes>This element holds a collection of individual stroke elements that define the sketch. 
Each stroke is uniquely identified by its own tag (e.g., <s1>, <s2>, etc.).
Within each stroke element, there are three key pieces of information: 
<points>A list of x-y coordinates defining the curve. These points define the path the stroke follows.</points>
<t_values>A series of numerical timing values that correspond to the points. These values define the progression of the stroke over time, ranging from 0 to 1, indicating the order or speed at which the stroke is drawn.</t_values>
<id>The semantic part category that this stroke belongs to. Use only the object part name, without describing shape, position, viewpoint, size, or appearance.</id>
</strokes>
</formatting>
"""

gt_sketches_example = """
<example>
To draw a house, start by drawing the front of the house:
<concept>House</concept>
<strokes>
    <s1>
        <points>'x13y27', 'x24y27', 'x24y27', 'x24y11', 'x24y11', 'x13y11', 'x13y11', 'x13y27'</points>
        <t_values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t_values>
        <id>house base</id>
    </s1>
    <s2>
        <points>'x13y27', 'x18y37','x18y37', 'x24y27'</points>
        <t_values>0.00,0.55,0.5,1.00</t_values>
        <id>roof</id>
    </s2>
</strokes>

Next we add the house's right section:
<concept>House</concept>
<strokes>
    <s1>
        <points>'x13y27', 'x24y27', 'x24y27', 'x24y11', 'x24y11', 'x13y11', 'x13y11', 'x13y27'</points>
        <t_values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t_values>
        <id>house base</id>
    </s1>
    <s2>
        <points>'x13y27', 'x18y37','x18y37', 'x24y27'</points>
        <t_values>0.00,0.55,0.5,1.00</t_values>
        <id>roof</id>
    </s2>
    <s3>
        <points>'x24y27', 'x36y28', 'x36y28', 'x36y21', 'x36y21', 'x36y12', 'x36y12', 'x24y11'</points>
        <t_values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t_values>
        <id>house base</id>
    </s3>
    <s4>
        <points>'x18y37', 'x30y38', 'x30y38', 'x36y28'</points>
        <t_values>0.00,0.55,0.5,1.00</t_values>
        <id>roof</id>
    </s4>
</strokes>

Now that we have the general structure of the house, we can add details to it, like windows and a door:
<concept>House</concept>
<strokes>
    <s1>
        <points>'x13y27', 'x24y27', 'x24y27', 'x24y11', 'x24y11', 'x13y11', 'x13y11', 'x13y27'</points>
        <t_values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t_values>
        <id>house base</id>
    </s1>
    <s2>
        <points>'x13y27', 'x18y37','x18y37', 'x24y27'</points>
        <t_values>0.00,0.55,0.5,1.00</t_values>
        <id>roof</id>
    </s2>
    <s3>
        <points>'x24y27', 'x36y28', 'x36y28', 'x36y21', 'x36y21', 'x36y12', 'x36y12', 'x24y11'</points>
        <t_values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t_values>
        <id>house base</id>
    </s3>
    <s4>
        <points>'x18y37', 'x30y38', 'x30y38', 'x36y28'</points>
        <t_values>0.00,0.55,0.5,1.00</t_values>
        <id>roof</id>
    </s4>
    <s5>
        <points>'x26y25', 'x29y25', 'x29y25', 'x29y21', 'x29y21', 'x26y21', 'x26y21', 'x26y25'</points>
        <t_values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t_values>
        <id>window</id>
    </s5>
    <s6>
        <points>'x31y25', 'x34y25', 'x34y25', 'x34y21', 'x34y21', 'x31y21', 'x31y21','x31y25'</points>
        <t_values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t_values>
        <id>window</id>
    </s6>
    <s7>
        <points>'x17y11', 'x17y18', 'x17y18', 'x21y18', 'x21y18', 'x21y11', 'x21y11', 'x17y11'</points>
        <t_values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t_values>
        <id>door</id>
    </s7>
</strokes>

and here is the complete example:
<concept>House</concept>
<strokes>
    <s1>
        <points>'x13y27', 'x24y27', 'x24y27', 'x24y11', 'x24y11', 'x13y11', 'x13y11', 'x13y27'</points>
        <t_values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t_values>
        <id>house base</id>
    </s1>
    <s2>
        <points>'x24y27', 'x36y28', 'x36y28', 'x36y21', 'x36y21', 'x36y12', 'x36y12', 'x24y11'</points>
        <t_values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t_values>
        <id>house base</id>
    </s2>
    <s3>
        <points>'x13y27', 'x18y37','x18y37', 'x24y27'</points>
        <t_values>0.00,0.55,0.5,1.00</t_values>
        <id>roof</id>
    </s3>
    <s4>
        <points>'x18y37', 'x30y38', 'x30y38', 'x36y28'</points>
        <t_values>0.00,0.55,0.5,1.00</t_values>
        <id>roof</id>
    </s4>
    <s5>
        <points>'x26y25', 'x29y25', 'x29y25', 'x29y21', 'x29y21', 'x26y21', 'x26y21', 'x26y25'</points>
        <t_values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t_values>
        <id>window</id>
    </s5>
    <s6>
        <points>'x31y25', 'x34y25', 'x34y25', 'x34y21', 'x34y21', 'x31y21', 'x31y21','x31y25'</points>
        <t_values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t_values>
        <id>window</id>
    </s6>
    <s7>
        <points>'x17y11', 'x17y18', 'x17y18', 'x21y18', 'x21y18', 'x21y11', 'x21y11', 'x17y11'</points>
        <t_values>0.00,0.3,0.25,0.5,0.5,0.75,0.75,1.00</t_values>
        <id>door</id>
    </s7>
</strokes>
</example>
"""

scene_planner_prompt =""" You are a scene decomposition planning expert specialized in progressive, layered video/image composition.

You will receive:
1. An image
2. A target scene description

Your task is to determine which specific, important, and distinct semantic objects should be inserted into the image.
The image may or may not contain any scenes.
Your task is to insert the new objects missing from the image based on the scene description.
Do not regenerate, replace, or reinterpret existing scene elements.

Important Notes:
- You do not need to draw objects.
- You do not need to generate SVGs.
- You only need to plan the scene composition.
----------------------------------------
Responsibilities:

Step 1:
Parse the target scene description. Decompose the target scene into **discrete, concrete, drawable, and important semantic objects**.
CRITICAL: Only extract core assets, furniture, props, characters, and necessary spatial floor anchors. Do NOT extract global room structures like walls or ceilings.

Step 2:
Sort all extracted objects strictly by **Depth and Rendering Topology Order (From Bottom-most to Top-most)**.
- The absolute background or foundational support structures (e.g., floor, grassland, table) MUST come FIRST.
- The foreground or overlapping objects MUST come LATER.
- CRITICAL OCCLUSION RULE: Multiple independent objects can share the EXACT SAME integer "depth" value if they reside on the same conceptual layer or surface. However, their order inside this JSON array MUST strictly obey the "Painter's Algorithm" (Occlusion Order): If Object B visually overlaps, sits on top of, or obscures Object A, Object A MUST appear BEFORE Object B in the sequence. Never sort them arbitrarily.

Step 3:
For each object in the sorted sequence, construct an **"incremental_prompt"**. 
- The first object's incremental_prompt is just its own contextual description.
- For subsequent objects, the incremental_prompt MUST accumulatedly include all previous (farther) objects plus the current object to describe the growing total scene.
- CRITICAL EQUAL WEIGHING: To prevent the vector optimizer from neglecting previously generated layers, the incremental_prompt MUST treat all current and past elements with equal narrative weight. Use structured conjunctions (e.g., "featuring BOTH [past elements] AND [current element]") to explicitly re-emphasize and re-validate every single active object and their explicit spatial interactions.

Step 4:
Determine:
- object complexity: lower the object complexity value, the lower the level of detail.

Complexity Rule:
- 0.5: stroke_budget = 16, detail_level = "low"
- 1.0: stroke_budget = 32, detail_level = "low"
- 2.0: stroke_budget = 64, detail_level = "medium"
- 3.0: stroke_budget = 128, detail_level = "high"

----------------------------------------
Object Rules (STRICT):

Each object must satisfy ALL of the following:
- Concrete physical entity / Major asset or standalone furniture/prop/terrain
- Single instance (no clusters, no groups of the SAME item, unless it's a unified background element like a grassland patch)
- Independently drawable
- Clearly locatable in 2D/3D space

**Object name must be a singular noun**:
- ✅ "zebra", "grassland"
- ❌ "zebras" (If there are two, output two separate "zebra" objects with different IDs, e.g., zebra_1, zebra_2)

----------------------------------------
CRITICAL: Object Merging vs. Spatial Anchor Rules (DETERMINE BY CONTEXT)

You must carefully evaluate whether a secondary element is a "minor interacting item" (should be merged) or a "spatial anchor/ground" (should be kept as a separate object).

1. **Strong Interaction / Containment Rules (Merge Minor Items)**:
   - If an object is actively consuming or holding a minor item, merge it.
   - Example: "A sheep eating grass" -> Extract ONLY `object_id: "sheep"`, describe its pose as "grazing". Do NOT output a separate "grass" asset because the grass here is just a small mouthful of food.
   - Example: "A vase filled with flowers" -> Extract ONLY `object_id: "vase"`.

2. **Spatial Anchor / Ground Rules (KEEP as Separate Object)**:
   - If an element serves as the **physical ground, base, or environment template** that characters walk or stand on, you MUST extract it as a standalone spatial anchor object. In depth order, grounds or large templates usually appear earlier than the characters standing on them.
   - Example: "Two zebras walking in the grass" -> The grass here means a "grassland / grassy ground". You MUST extract `object_id: "grassland"` and two separate `object_id: "zebra"` objects.

----------------------------------------
CRITICAL: Description Rules (For Isolated Black-Line Sketching):
- The "description" field must ONLY describe the single object's **geometric shapes, contours, outline profiles, pose/action, and structural styles**.
- All adjectives and modifying phrases must function strictly as **attributive modifiers (定语)** directly describing the structure of the noun.
- **DO NOT include any colors, materials, textures, or lighting** (e.g., ❌ green, brown, wooden, plastic, glossy).
- **DO NOT include any external spatial relationships with OTHER independent objects** in the description (e.g., ❌ "on the table").
- ✅ "a patch of ground covered with simple vector grass tufts" (Good description for a grassland anchor)
- ✅ "a four-legged square dining table in perspective view" (Good description for a table structure)

----------------------------------------
CRITICAL: Incremental Prompt Rules (For Contextual Canvas Alignment):
- The "incremental_prompt" field must describe the **cumulative scene state up to this object**, with high visual emphasis on ALL elements present so far.
- It MUST explicitly re-state and emphasize all previous objects alongside the new one to prevent the model from treating past objects as negligible background noise.
- It **SHOULD** include strong dual-entity or multi-entity phrasing, precise spatial relationships, interactions and styles.

----------------------------------------
Valid object examples:
- bookshelf, sofa, table
- zebra (individual character)
- grassland / ground (as a discrete spatial ground asset)
- hill

Invalid object examples (DO NOT OUTPUT these):
❌ wall, ceiling (Violates room structure rule)
❌ zebras (Plural name forbidden; split into zebra_1 and zebra_2)

----------------------------------------
Decomposition and Sorting constraints:
- Given "Two zebras walking in the grass", you should extract 3 objects sorted from Far to Near:
  1. `object_id: "grassland"` (The background/ground anchor)
     incremental_prompt: "a minimalist black line art of a wide grassland"
  2. `object_id: "zebra_1"` (First character on the grass)
     incremental_prompt: "a minimalist black line art of a zebra walking on the grassland"
  3. `object_id: "zebra_2"` (Second character on the grass)
     incremental_prompt: "a minimalist black line art of two zebras walking on the grassland"

----------------------------------------
Output Format:

ONLY output valid JSON. The root structure must be a JSON array, **strictly sorted from the farthest background object to the closest foreground object**.

Example:
[
    {
        "object_id": "window",
        "description": "a rectangular multi-pane window structure with clean linear frames",
        "incremental_prompt": "a line sketch of a rectangular window on a blank background",
        "complexity": 1.0,
        "depth": 0
    },
    {
        "object_id": "table",
        "description": "a four-legged square dining table in perspective view",
        "incremental_prompt": "a line sketch of a four-legged table placed under a rectangular window",
        "complexity": 1.5,
        "depth": 1
    },
    {
        "object_id": "cup",
        "description": "a small cylindrical ceramic cup with a side handle",
        "incremental_prompt": "a line sketch featuring BOTH a four-legged table under a window AND a small cylindrical cup placed on it",
        "complexity": 0.5,
        "depth": 2
    },
    {
        "object_id": "plate",
        "description": "a flat circular dinner plate with a slightly raised rim",
        "incremental_prompt": "a line sketch featuring a four-legged table under a window, with BOTH a cylindrical cup AND a flat circular dinner plate sitting together on the table",
        "complexity": 0.5,
        "depth": 2
    }
]

Do NOT output explanations. Do NOT output markdown. Do NOT output thinking.
"""


SINGLE_LAYOUT_PROMPT = """
You are a scene layout planner specialized in progressive, autoregressive object placement.

The final canvas size is 512x512 pixels. 
Each object's base canvas is also 512x512 pixels before scaling.
When `scale` is applied, the actual rendering width and height of the object become (512 * scale).

CRITICAL COORDINATE SYSTEM:
- The coordinates `x` and `y` represent the NORMALIZED CENTER POSITION (0.0 to 1.0) of the object on the final canvas.
- (0.0, 0.0) is Top-Left, (1.0, 1.0) is Bottom-Right, and (0.5, 0.5) is the absolute center.

----------------------------------------
INPUT CONTEXT:
You will receive:
1. An image: The current background canvas containing already drawn far-ground elements.
2. A list of existing objects already placed on this canvas, including their precise normalized center coordinates (x, y) and scales.
3. The description of the new object to be inserted.
4. The incremental context prompt describing the overall semantic relationships (e.g., character standing on a table).

----------------------------------------
CRITICAL BOUNDARY CONSTRAINTS (NO CLIP/OVERFLOW):
- The new object MUST be FULLY contained within the canvas. NO part of the object should overflow outside the boundaries.
- To prevent clipping, the normalized center position (x, y) must satisfy the following safety margin based on its scale:
  * (scale / 2) <= x <= 1 - (scale / 2)
  * (scale / 2) <= y <= 1 - (scale / 2)
- For example:
  * If `scale` is 1.0, its center (x, y) MUST be exactly (0.5, 0.5) to stay completely within the canvas.
  * If `scale` is 0.5, then x and y must be strictly between 0.25 and 0.75.
  * If `scale` is 1.2, it is impossible to stay within boundaries without clipping, so avoid overly large scales unless it's a structural background anchor like a grassland plane.

----------------------------------------
SEMANTIC & SPATIAL RELATIONSHIPS:
- Carefully analyze the existing objects' positions.
- If the new object needs to be placed ON TOP of or INSIDE an existing object (e.g., a cat on a table), use the historical (x, y, scale) of that table to calculate the correct corresponding `y` coordinate (usually slightly above the table's center, depending on the scale) to ensure semantic consistency.

----------------------------------------
Output Format:
You MUST only reply with a valid JSON object for this SINGLE new object containing exactly four keys: "x", "y", "scale", and "rotation".

{
    "x": float,          // 0.0 ~ 1.0 normalized center X
    "y": float,          // 0.0 ~ 1.0 normalized center Y
    "scale": float,      // 0.1 ~ 1.5 scaling factor
    "rotation": float    // degrees (optional, default 0.0)
}

CRITICAL: Do NOT output markdown code blocks (e.g., do NOT wrap in ```json). Do NOT output thinking steps. Do NOT output explanations. Only valid JSON.
"""

LAYOUT_CRITIC_PROMPT = """
You are an expert visual composition critic.

You will evaluate a composed SVG scene.

Evaluate:
1. Object overlap
2. Visual balance
3. Semantic correctness
4. Scale consistency
5. Spatial relationships
6. Foreground/background hierarchy
7. Canvas utilization

ONLY output valid JSON.
Return JSON:

{
  "score": 72,
  "is_satisfactory": false,
  "issues": [
    {
      "object_id": "cat",
      "problem": "Cat overlaps bicycle",
      "suggestion": "Move cat left and slightly down"
    }
  ]
}
"""

LAYOUT_REVISER_PROMPT = """
Revise the scene layout according to critic feedback.

Original layout:
{layout}

Critic feedback:
{feedback}

ONLY output valid JSON.
"""

anchor_planner_prompt = """
You are an expert semantic anchor planner for compositional vector graphics generation.

You are given:
1. a canvas
2. one target object

Your task is to generate semantic anchor points for the object.

You are NOT generating SVG.
You are NOT drawing continuous contours.
You are ONLY generating semantic grounding anchors for downstream vector optimization.

--------------------------------------------------
Existing Scene Constraint:

The image already contains valid scene content.

You must ONLY generate anchors for the target object.

Do NOT:
- generate background structure
- generate unrelated anchors

--------------------------------------------------
Coordinate System:

The image is mapped to a 2D grid from 1 to {res} on both x and y axes.

Each location is represented as:

'x<number>y<number>'

Examples:
- x1y1 = bottom-left
- x{res}y{res} = top-right

--------------------------------------------------
Anchor Planning Goal:

Each anchor MUST represent a **physically distinct and visually meaningful part** of the object.

Examples of valid part names for different objects:
- Table: desktop, left_leg, right_leg, back_leg, drawer_front
- Chair: seat, backrest, left_armrest, right_armrest, front_left_leg
- Cat: head, torso, tail, left_ear, right_ear
- Bicycle: front_wheel, rear_wheel, frame_top_tube, handlebar, seat

Anchors are NOT:
- geometric features (corners, edges, centerlines, directions)
- sub-part regions (tabletop_corners, legs_top, legs_feet)
- abstract attributes (orientation, spread, direction)

--------------------------------------------------
Anchor Rules:

- Fully outline the object
- One anchor ID = one physical part
- Do NOT split a single part into multiple geometric anchors (e.g., "legs_top" and "legs_feet" is illegal)

--------------------------------------------------
Layout Constraint:

The generated anchors MUST:
- remain inside the predicted object region
- follow the predicted scale
- follow the visual effect
- follow the predicted orientation
- remain spatially compact

--------------------------------------------------
Output Rules:

- ONLY output valid JSON
- Do NOT output explanations
- Do NOT output markdown
- Do NOT output thinking

--------------------------------------------------
Required Output Format:
ONLY output valid JSON.

Example for a table:
{
    "object_id": "table",
    "anchors": [
        {
            <id> desktop</id>
            <points> "x25y20", "x26y20", "x27y20", "x28y20", "x29y20", "x30y20", "x31y20", "x32y20", "x33y20", "x34y20", "x35y20"</points>
        },
        {
            <id> leg</id>
            <points> "x26y20", "x26y18", "x26y16", "x26y14", "x26y12"</points>
        },
        {
            <id> leg</id>
            <points> "x34y20", "x34y18", "x34y16", "x34y14", "x34y12"</points>
        }
    ]
}
"""

system_prompt = """
You are an expert dot-matrix sketch artist.

Your goal is NOT to draw strokes.
Your goal IS to generate a dense dot-matrix representation of a recognizable object silhouette.

The output must look visually recognizable EVEN IF YOU ONLY SEE THE POINTS (no lines drawn).

The grid uses coordinates from 1 to {res} on both x and y axes.

---

CRITICAL OBJECTIVE:
You must generate a HIGH-DENSITY SET OF POINTS that visually reconstruct the SHAPE of the object.

If someone plots ONLY your points, they should immediately recognize the object (e.g., cat, table, house).

---

ABSOLUTE REQUIREMENTS:

1. DENSE COVERAGE (MOST IMPORTANT)
- You MUST NOT output sparse outlines.
- You must fill BOTH:
  - object boundary (outline)
  - interior structural density (light filling when needed for recognition)
- Every important region of the object must contain MANY points.

2. SHAPE-FIRST PRIORITY
- Prioritize recognizability over mathematical curves.
- It is better to overdraw than underdraw.

3. LOCAL CONTINUITY IS OPTIONAL
- You are NOT required to form smooth paths.
- Points may jump if needed.
- Focus on VISUAL DENSITY, not stroke continuity.

4. DOT-MATRIX STYLE
- Imagine the object is drawn using LED pixels or printer dots.
- You are placing thousands of dots to “light up” the shape.

5. MULTI-POINT SAMPLING PER REGION
For every part of the object:
- head/body/legs/edges/etc.
- you MUST sample MANY points across the region, not just edges.

---

GOOD EXAMPLE BEHAVIOR:
- A cat = dense head circle + dense ears triangles + dense body ellipse + tail curve filled with points
- A table = dense rectangle surface + 4 legs filled with vertical dot columns

---

BAD BEHAVIOR:
- Only outline corners
- Single contour line
- Sparse trajectories
- Thin wireframe drawings

---

OUTPUT STYLE:
Return structured strokes, BUT strokes are ONLY for grouping.
Inside each stroke, you MUST provide MANY dense points.

Each stroke should represent ONE semantic region, not a path.

Examples of stroke meaning:
- "cat head region dots"
- "table surface region dots"
- "left ear dense dots"

---

IMPORTANT NOTE:
You are NOT drawing lines.
You are generating a DOT FIELD that encodes shape.

Think like:
- pixel art (but irregular sampling)
- point cloud silhouette
- ink stamping density map

"""

sketch_first_prompt = """I provide you with a blank grid.

Your goal is to generate a DOT-MATRIX SKETCH of a {concept}.

The sketch must be recognizable purely from the density of points.

NOTES:
- Do NOT draw lines or sparse contours.
- Do NOT focus on strokes as paths.
- Focus on dense point placement that forms a visible silhouette.

Even if all points are shown without connection, the object must be clearly identifiable.

Here are a few examples:
<examples>
{gt_sketches_str}
</examples>
---
---

You may use unlimited number of points.

Think before you output:
1. Identify main parts of the {concept}
2. Decide how each part looks as a dense dot region
3. Fill those regions with many coordinate points
4. Ensure overall silhouette is immediately recognizable

Return your answer in structured format.

<formatting>
<concept>{concept}</concept>
<strokes>
Each stroke is a semantic region filled with dense points.
</strokes>
</formatting>
"""

gt_sketches_str = """
<example>
Here is an example of drawing a house using DENSE DOT-MATRIX POINTS that form a clearly recognizable silhouette.

IMPORTANT: This is NOT a line drawing. This is a POINT CLOUD representation.

<concept>House</concept>
<strokes>

    <s1>
        <points>
        'x10y10','x11y10','x12y10','x13y10','x14y10','x15y10',
        'x10y11','x11y11','x12y11','x13y11','x14y11','x15y11',
        'x10y12','x11y12','x12y12','x13y12','x14y12','x15y12',
        'x10y13','x11y13','x12y13','x13y13','x14y13','x15y13',
        'x10y14','x11y14','x12y14','x13y14','x14y14','x15y14'
        </points>
        <id>house base</id>
    </s1>

    <s2>
        <points>
        'x10y14','x11y15','x12y16','x13y15','x14y14',
        'x11y14','x12y15','x13y14'
        </points>
        <id>roof</id>
    </s2>

    <s3>
        <points>
        'x12y10','x12y11','x12y12','x13y10','x13y11','x13y12'
        </points>
        <id>door</id>
    </s3>

</strokes>
</example>
"""