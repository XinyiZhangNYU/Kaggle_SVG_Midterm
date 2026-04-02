import pandas as pd
import xml.etree.ElementTree as ET
import re
import random
import numpy as np
import os


# Seed
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)

# ==========================================
# Part 1: Core Configuration and Constants
# ==========================================

ALLOWED_TAGS = {
    'svg', 'g', 'path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 
    'polygon', 'defs', 'use', 'symbol', 'clipPath', 'mask', 
    'linearGradient', 'radialGradient', 'stop', 'text', 'tspan', 
    'title', 'desc', 'style', 'pattern', 'marker', 'filter'
}

SYSTEM_PROMPT = """You are an expert SVG code generator. Your task is to generate clean, strictly valid, and standalone SVG code based on the user's text description.

You MUST adhere to the following strict rules:
1. STRICT OUTPUT: Output ONLY the raw SVG code. No markdown formatting (no ```xml), no HTML wrappers, and no conversational text.
2. THE CANVAS RULE: Always use exactly <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200' width='256' height='256'>.
3. THE ADAPTIVE KISS PRINCIPLE: For basic shapes, you MUST use primitives (<rect>, <circle>, <ellipse>, <line>, <polygon>, <polyline>). For complex/natural objects, use optimized <path> elements. ANTI-HALLUCINATION: NEVER invent non-existent tags like <triangle>, <square>, <star>, <curve>, <arc>, <background>, or <layer>. Use valid SVG alternatives (e.g., <polygon>, <rect>, <path>, <g>).
4. DRAWING ORDER: Render elements from back to front.
5. STYLE & COLOR: Use direct presentation attributes ONLY (e.g., fill='black').
6. SECURITY & VALIDITY: Ensure all tags are properly closed. Use single quotes for all attributes."""

# ==========================================
# Part 2: Data Purging & Quality Functions
# ==========================================

def is_safe_xml(svg_string):
    """Strict tag whitelist + deep attribute safety scan"""
    try:
        root = ET.fromstring(svg_string)
        root_tag = root.tag.split('}')[-1]
        if root_tag.lower() != 'svg': return False
        for elem in root.iter():
            tag = elem.tag.split('}')[-1]
            if tag not in ALLOWED_TAGS: return False
            for attr_name, attr_value in elem.attrib.items():
                attr_lower = attr_name.lower()
                val_lower = attr_value.lower()
                if attr_lower.startswith('on'): return False
                if 'http://' in val_lower or 'https://' in val_lower:
                    if not attr_lower.startswith('xmlns'): return False
    except ET.ParseError:
        return False 
    return True

def compress_and_format_svg(svg_str):
    """Extreme compression (smart float rounding) and root tag standardization"""
    svg_str = str(svg_str).strip()
    svg_str = svg_str.replace('\n', ' ').replace('\r', ' ').replace('"', "'")
    svg_str = svg_str.replace('```xml', '').replace('```', '')
    
    # Smart float compression
    # Preserves precision up to 2 decimals, strips trailing zeros/dots to save Tokens
    def optimize_match(match):
        val = float(match.group(0))
        formatted = f"{val:.2f}"
        return formatted.rstrip('0').rstrip('.')
    
    # Match any number with a decimal point
    svg_str = re.sub(r'\d*\.\d+', optimize_match, svg_str)
    
    svg_str = re.sub(r'\s+', ' ', svg_str)
    
    standard_svg_head = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200' width='256' height='256'>"
    svg_str = re.sub(r'<svg[^>]*>', standard_svg_head, svg_str, count=1, flags=re.IGNORECASE)
    svg_str = svg_str.replace(' />', '/>')
    return svg_str.strip()

def fails_semantic_check(prompt, svg_str):
    """Semantic and tag matching check"""
    p_lower = str(prompt).lower()
    svg_lower = str(svg_str).lower()
    if any(word in p_lower for word in ['circle', 'round', 'face']):
        if '<circle' not in svg_lower and '<ellipse' not in svg_lower: return True
    if any(word in p_lower for word in ['square', 'rectangle', 'box']):
        if '<rect' not in svg_lower: return True
    return False

def assess_high_quality_path(row):
    """
    Smart quality evaluator: checks command count, coordinate boundaries, 
    and prompt entropy. Returns True for high-quality data.
    """
    prompt = str(row['prompt']).strip()
    svg_str = str(row['svg'])
    
    # 1. Semantic Information Entropy Check (Block repetitive nonsense)
    words = prompt.split()
    if len(words) < 3: return False  
    unique_words = set([w.lower() for w in words])
    if len(unique_words) / len(words) < 0.5:
        return False
        
    # 2. Check Path complexity and coordinate boundaries
    path_d_matches = re.findall(r"<path[^>]*d='([^']*)'", svg_str, re.IGNORECASE)
    
    if not path_d_matches:
        return False 
        
    total_commands = 0
    for d_str in path_d_matches:
        commands = re.findall(r'[MmLlCcZzHhVvSsQqTtAa]', d_str)
        total_commands += len(commands)
        
        # 3. Visual Bounding Box Check (-20 to 220 allowed)
        coords = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+', d_str)]
        if coords:
            if max(coords) > 220 or min(coords) < -20:
                return False

    # 4. Golden Instruction Range (20 to 80 commands)
    if total_commands < 20 or total_commands > 80:
        return False
        
    return True

# ==========================================
# Part 3: Pipeline Execution
# ==========================================

def purge_dataset(input_file, output_file, target_samples=5000):
    print(f"🚀 Starting V5 Advanced Purging Pipeline for: {input_file} ...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Error: Cannot find {input_file}")
        return
        
    initial_len = len(df)
    
    # Basic cleaning and crash prevention
    mask_viewbox = df['svg'].astype(str).str.contains(r'viewBox=["\']0\.0 0\.0 200\.0 200\.0["\']|viewBox=["\']0 0 200 200["\']')
    df = df[mask_viewbox].copy()
    
    df['svg'] = df['svg'].apply(compress_and_format_svg)
    df = df[df['svg'].str.len() <= 1200].copy()
    df = df[df['svg'].apply(is_safe_xml)].copy()
    
    mask_semantic = df.apply(lambda row: not fails_semantic_check(row['prompt'], row['svg']), axis=1)
    df = df[mask_semantic].copy()
    print(f"🔪 Step 1-6 (Format, Safety, Basic Semantic): {len(df)} remaining")
    
    # 3D Smart Quality Filtering (Extract high-quality Moderate Paths)
    print("🧠 Executing 3D Quality Heuristics (Entropy, Bounds, Density)...")
    mask_quality = df.apply(assess_high_quality_path, axis=1)
    df = df[mask_quality].copy()
    print(f"🔪 Step 7 (High-Quality Paths Extraction): {len(df)} extremely clean semantic paths found!")
    
    # Randomly truncate to target amount
    if len(df) > target_samples:
        print(f"🎲 Downsampling to target {target_samples} samples for optimal balance...")
        df = df.sample(n=target_samples, random_state=42).reset_index(drop=True)
    
    # Ensure <|im_end|> token exists at the end of the SVG sequence
    print("Verifying <|im_end|> tokens...")
    mask_missing_eos = ~df['svg'].astype(str).str.endswith('<|im_end|>')
    df.loc[mask_missing_eos, 'svg'] = df.loc[mask_missing_eos, 'svg'].astype(str) + "<|im_end|>"
    
    # Imprint System Prompt and ChatML formatting
    print("Injecting System Prompt into training data...")
    df['prompt'] = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n" + df['prompt'] + "<|im_end|>\n<|im_start|>assistant\n"
    
    df.to_csv(output_file, index=False)
    print(f"🎉 Purging complete! Distilled {len(df)} golden semantic samples from {initial_len} raw rows.")
    print(f"💾 Saved to: {output_file}")

# Example usage:
if __name__ == "__main__":
    purge_dataset("train.csv", "purged_train_v5.csv", target_samples=5000)