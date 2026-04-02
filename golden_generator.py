import pandas as pd
import random
import os
import numpy as np

# Seed
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)
# ==========================================
# Part 1: Core Configuration and System Prompt
# ==========================================

# The ultimate System Prompt (Must match exactly with purge_dataset)
SYSTEM_PROMPT = """You are an expert SVG code generator. Your task is to generate clean, strictly valid, and standalone SVG code based on the user's text description.

You MUST adhere to the following strict rules:
1. STRICT OUTPUT: Output ONLY the raw SVG code. No markdown formatting (no ```xml), no HTML wrappers, and no conversational text.
2. THE CANVAS RULE: Always use exactly <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200' width='256' height='256'>.
3. THE ADAPTIVE KISS PRINCIPLE: For basic shapes, you MUST use primitives (<rect>, <circle>, <ellipse>, <line>, <polygon>, <polyline>). For complex/natural objects, use optimized <path> elements. ANTI-HALLUCINATION: NEVER invent non-existent tags like <triangle>, <square>, <star>, <curve>, <arc>, <background>, or <layer>. Use valid SVG alternatives (e.g., <polygon>, <rect>, <path>, <g>).
4. DRAWING ORDER: Render elements from back to front.
5. STYLE & COLOR: Use direct presentation attributes ONLY (e.g., fill='black').
6. SECURITY & VALIDITY: Ensure all tags are properly closed. Use single quotes for all attributes."""

# Standardized SVG head tag
SVG_HEAD = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200' width='256' height='256'>"

# ==========================================
# Part 2: Dynamic Color Engine
# ==========================================

# Extended library of standard CSS web colors
EXTENDED_CSS_COLORS = [
    "aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige", "bisque", "black", "blanchedalmond", "blue", 
    "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse", "chocolate", "coral", "cornflowerblue", "cornsilk", 
    "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray", "darkgreen", "darkkhaki", "darkmagenta", 
    "darkolivegreen", "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen", "darkslateblue", "darkslategray", 
    "darkturquoise", "darkviolet", "deeppink", "deepskyblue", "dimgray", "dodgerblue", "firebrick", "floralwhite", 
    "forestgreen", "fuchsia", "gainsboro", "ghostwhite", "gold", "goldenrod", "gray", "green", "greenyellow", "honeydew", 
    "hotpink", "indianred", "indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon", 
    "lightblue", "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgray", "lightgreen", "lightpink", "lightsalmon", 
    "lightseagreen", "lightskyblue", "lightslategray", "lightsteelblue", "lightyellow", "lime", "limegreen", "linen", 
    "magenta", "maroon", "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen", 
    "mediumslateblue", "mediumspringgreen", "mediumturquoise", "mediumvioletred", "midnightblue", "mintcream", "mistyrose", 
    "moccasin", "navajowhite", "navy", "oldlace", "olive", "olivedrab", "orange", "orangered", "orchid", "palegoldenrod", 
    "palegreen", "paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", 
    "purple", "rebeccapurple", "red", "rosybrown", "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen", 
    "seashell", "sienna", "silver", "skyblue", "slateblue", "slategray", "snow", "springgreen", "steelblue", "tan", 
    "teal", "thistle", "tomato", "turquoise", "violet", "wheat", "white", "whitesmoke", "yellow", "yellowgreen"
]

def generate_dynamic_color():
    """
    Randomly generates a color in one of three formats: CSS name, Hex code, or RGB code.
    This trains the model to understand and apply various color string formats.
    """
    # Distribution: 50% CSS names, 30% Hex codes, 20% RGB values
    color_format = random.choices(['css', 'hex', 'rgb'], weights=[0.5, 0.3, 0.2])[0]
    
    if color_format == 'css':
        return random.choice(EXTENDED_CSS_COLORS)
    elif color_format == 'hex':
        return f"#{random.randint(0, 0xFFFFFF):06X}"
    else: # rgb
        return f"rgb({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)})"

# ==========================================
# Part 3: Compositional Grammar Engine
# ==========================================

class DynamicPromptEngine:
    """
    Compositional Grammar Engine: Deconstructs sentences into multiple slots 
    and randomly combines them to generate tens of thousands of unique sentence 
    structures, entirely eliminating prompt overfitting.
    """
    def __init__(self):
        self.prefixes = ["A ", "Draw a ", "Create a ", "Generate a ", "Show me a ", "Make a ", "I want a ", "Can you draw a ", ""]
        self.sizes = ["", "large ", "small ", "medium-sized ", "tiny ", "massive "]
        self.styles = ["", "simple ", "solid ", "basic ", "perfect ", "flat ", "clean ", "geometric "]
        self.positions = ["", " in the center.", " in the middle.", " placed centrally.", " on the canvas."]

    def generate(self, shape_type, color, color2=None):
        prefix = random.choice(self.prefixes)
        size = random.choice(self.sizes)
        style = random.choice(self.styles)
        pos = random.choice(self.positions)
        
        # 清理多余空格并确保首字母大写、句号结尾
        def format_sentence(text):
            text = " ".join(text.split())
            if not text.endswith('.'): text += '.'
            if text: text = text[0].upper() + text[1:]
            return text

        if shape_type == 'circle':
            nouns = ["circle", "circular shape", "round shape", "disk"]
            return format_sentence(f"{prefix}{size}{style}{color} {random.choice(nouns)}{pos}")
            
        elif shape_type == 'rect':
            nouns = ["square", "box", "square shape", "regular quadrilateral"]
            return format_sentence(f"{prefix}{size}{style}{color} {random.choice(nouns)}{pos}")
            
        elif shape_type == 'triangle':
            nouns = ["triangle", "triangular shape", "three-sided polygon"]
            verbs = ["pointing upwards", "facing up", "oriented towards the top"]
            return format_sentence(f"{prefix}{style}{color} {random.choice(nouns)} {random.choice(verbs)}{pos}")
            
        elif shape_type == 'containment':
            outer = ["square", "box", "frame"]
            inner = ["circle", "dot", "round shape"]
            templates = [
                f"{prefix}{size}{color} {random.choice(outer)} containing a {color2} {random.choice(inner)}.",
                f"{prefix}{color2} {random.choice(inner)} centered inside a {color} {random.choice(outer)}.",
                f"{prefix}{color} {random.choice(outer)} with a {style}{color2} {random.choice(inner)} in it."
            ]
            return format_sentence(random.choice(templates))
            
        elif shape_type == 'side_by_side':
            templates = [
                f"{prefix}two shapes side by side: a {color} circle on the left and a {color2} triangle on the right.",
                f"{prefix}{color} circle and a {color2} triangle placed horizontally.",
                f"{prefix}left is a {color} circle, right is a {color2} triangle.",
                f"{prefix}{color2} triangle positioned to the right of a {color} circle."
            ]
            return format_sentence(random.choice(templates))
            
        elif shape_type == 'repetition':
            templates = [
                f"{prefix}horizontal row of three {color} circles.",
                f"{prefix}three {color} circles aligned horizontally.",
                f"{prefix}pattern of three consecutive {color} circular shapes.",
                f"{prefix}line of three {color} circles."
            ]
            return format_sentence(random.choice(templates))
            
        elif shape_type == 'overlap':
            templates = [
                f"{prefix}two overlapping squares. A {color} square partially covering a {color2} square.",
                f"{prefix}{color} square placed on top of a {color2} square.",
                f"{prefix}overlapping layout with a {color2} box behind a {color} box.",
                f"{prefix}{color} square intersecting with a {color2} square."
            ]
            return format_sentence(random.choice(templates))
            
        elif shape_type == 'rectangle':
            nouns = ["rectangle", "rectangular shape", "wide box"]
            modifiers = ["wide ", "horizontal ", "elongated ", ""]
            return format_sentence(f"{prefix}{random.choice(modifiers)}{size}{color} {random.choice(nouns)}{pos}")
            
        elif shape_type == 'ellipse':
            nouns = ["ellipse", "oval shape", "flattened circle", "elliptical shape"]
            modifiers = ["horizontal ", "wide ", ""]
            return format_sentence(f"{prefix}{random.choice(modifiers)}{size}{color} {random.choice(nouns)}{pos}")
            
        elif shape_type == 'polygon_pentagon':
            nouns = ["pentagon", "five-sided polygon", "regular pentagon"]
            return format_sentence(f"{prefix}{size}{style}{color} {random.choice(nouns)}{pos}")
            
        elif shape_type == 'semicircle':
            nouns = ["semicircle", "half-circle", "dome shape"]
            modifiers = ["with the flat side on the bottom", "cut horizontally", "facing up"]
            return format_sentence(f"{prefix}{size}{style}{color} {random.choice(nouns)} {random.choice(modifiers)}.")
            
        else:
            return format_sentence(f"{prefix}{color} {shape_type}.")

# ==========================================
# Part 4: Golden Dataset Generator
# ==========================================

def generate_golden_dataset(num_samples=2500):
    print(f"Generating {num_samples} golden anchor samples...")
    data = []
    prompt_engine = DynamicPromptEngine()
    
    shape_pool = [
        'circle', 'rect', 'triangle', 
        'containment', 'side_by_side', 'repetition', 'overlap',
        'rectangle', 'ellipse', 'polygon_pentagon', 'semicircle'
    ]
    
    for i in range(num_samples):
        shape_type = random.choice(shape_pool)
        
        # Generate primary and secondary colors dynamically
        color = generate_dynamic_color()
        color2 = generate_dynamic_color()
        
        # Ensure colors are visually distinct for multi-shape configurations
        while color == color2:
            color2 = generate_dynamic_color()
            
        # Use the engine to dynamically generate the natural language Prompt
        user_prompt = prompt_engine.generate(shape_type, color, color2)
        
        # Pure math and vector generation logic
        if shape_type == 'circle':
            r = random.randint(30, 80)
            svg = f"{SVG_HEAD}<circle cx='100' cy='100' r='{r}' fill='{color}'/></svg>"
            
        elif shape_type == 'rect':
            w = random.randint(60, 120)
            svg = f"{SVG_HEAD}<rect x='{100-w//2}' y='{100-w//2}' width='{w}' height='{w}' fill='{color}'/></svg>"
            
        elif shape_type == 'triangle':
            svg = f"{SVG_HEAD}<polygon points='100,40 40,160 160,160' fill='{color}'/></svg>"
            
        elif shape_type == 'containment':
            svg = f"{SVG_HEAD}<rect x='40' y='40' width='120' height='120' fill='{color}'/><circle cx='100' cy='100' r='40' fill='{color2}'/></svg>"
            
        elif shape_type == 'side_by_side':
            svg = f"{SVG_HEAD}<circle cx='60' cy='100' r='40' fill='{color}'/><polygon points='140,60 100,140 180,140' fill='{color2}'/></svg>"
            
        elif shape_type == 'repetition':
            svg = f"{SVG_HEAD}<circle cx='50' cy='100' r='25' fill='{color}'/><circle cx='100' cy='100' r='25' fill='{color}'/><circle cx='150' cy='100' r='25' fill='{color}'/></svg>"
            
        elif shape_type == 'overlap':
            svg = f"{SVG_HEAD}<rect x='60' y='60' width='80' height='80' fill='{color2}'/><rect x='90' y='90' width='80' height='80' fill='{color}'/></svg>"
            
        elif shape_type == 'rectangle':
            w = random.randint(100, 160)
            h = random.randint(40, 80)
            svg = f"{SVG_HEAD}<rect x='{100-w//2}' y='{100-h//2}' width='{w}' height='{h}' fill='{color}'/></svg>"
            
        elif shape_type == 'ellipse':
            rx = random.randint(60, 90)
            ry = random.randint(30, 50)
            svg = f"{SVG_HEAD}<ellipse cx='100' cy='100' rx='{rx}' ry='{ry}' fill='{color}'/></svg>"
            
        elif shape_type == 'polygon_pentagon':
            svg = f"{SVG_HEAD}<polygon points='100,30 165,75 140,150 60,150 35,75' fill='{color}'/></svg>"
            
        elif shape_type == 'semicircle':
            svg = f"{SVG_HEAD}<path d='M 50,120 A 50,50 0 0,1 150,120 Z' fill='{color}'/></svg>"

        # Format ChatML structure and append the end-of-sequence token
        svg_with_eos = svg + "<|im_end|>"
        full_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        data.append({
            'id': f'golden_anchor_{i}',
            'prompt': full_prompt,
            'svg': svg_with_eos
        })
        
    return pd.DataFrame(data)

# ==========================================
# Part 5: Merge, Shuffle and Save
# ==========================================

def merge_and_shuffle(purged_csv_path, output_csv_path, num_golden=2500):
    if not os.path.exists(purged_csv_path):
        print(f"Error: Cannot find purged file '{purged_csv_path}'. Please run the purge script first!")
        return
        
    print(f"Loading purged official data: {purged_csv_path}")
    df_purged = pd.read_csv(purged_csv_path)
    
    # Generate the pristine golden dataset
    df_golden = generate_golden_dataset(num_samples=num_golden)
    
    print("Merging purged data with golden samples...")
    df_combined = pd.concat([df_purged, df_golden], ignore_index=True)
    
    print("Performing global shuffle...")
    # Shuffle the dataset to ensure batch distribution during training is even
    df_final = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the finalized training dataset to disk
    df_final.to_csv(output_csv_path, index=False)
    print(f"\nSuccess! V5 training dataset created. Saved to: {output_csv_path}")

# ==========================================
# Main Execution Entry
# ==========================================
if __name__ == "__main__":
    # Ensure you have run purge_dataset.py to generate 'purged_train_v5.csv' before running this
    merge_and_shuffle("purged_train_v5.csv", "final_train_v5.csv", num_golden=2500)