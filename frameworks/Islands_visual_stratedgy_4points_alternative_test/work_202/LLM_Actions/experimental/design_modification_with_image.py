"""
Experimental: Multimodal design modification using image input.
Gives the LLM visual geometric awareness of the airfoil shape.
"""
import os
import json
import subprocess
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from typing import List, Dict, Any

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
import sys
sys.path.insert(0, PARENT_DIR)
from llm_design_actions import run_action

# Load API key
dotenv_path = os.path.join(PARENT_DIR, '.env')
load_dotenv(dotenv_path)
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
if api_key:
    genai.configure(api_key=api_key)

SCHEMAS_DIR = os.path.join(PARENT_DIR, 'schemas')

def load_schema(name: str) -> str:
    with open(os.path.join(SCHEMAS_DIR, f"{name}.json"), 'r') as f:
        return f.read()

def load_image(image_path: str):
    """Load image for Gemini multimodal input."""
    return Image.open(image_path)

def get_gemini_multimodal_response(prompt: str, images: List[Any], temperature: float = 1.0) -> Dict:
    """Send prompt + images to Gemini and parse JSON response."""
    try:
        model = genai.GenerativeModel('gemini-3-pro-preview')
        content = [prompt] + images
        config = genai.types.GenerationConfig(temperature=temperature)
        response = model.generate_content(content, generation_config=config)
        text = response.text
        start, end = text.find('{'), text.rfind('}') + 1
        if start == -1:
            print(f"No JSON in response: {text[:200]}")
            return {}
        return json.loads(text[start:end])
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        return {}
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {}

def modify_with_image(
    base_csv: str,
    geometry_image: str,
    output_dir: str,
    name: str = "modified",
    context: List[Dict] = None,
    temperature: float = 1.0
) -> str:
    """
    Modify airfoil design using both CSV data and geometry image.
    
    Args:
        base_csv: Path to parent CSV file
        geometry_image: Path to geometry PNG image
        output_dir: Output directory for results
        name: Name prefix for output files
        context: Optional list of inspiration designs
        temperature: LLM temperature for creativity
    
    Returns:
        Path to generated CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load parent CSV content
    with open(base_csv, 'r') as f:
        parent_content = f.read()
    
    # Load geometry image
    image = load_image(geometry_image)
    
    # Build context string
    ctx_str = ""
    if context:
        ctx_str = "Inspiration designs:\n"
        for i, item in enumerate(context):
            ctx_str += f"  {i+1}. Reward={item.get('reward')}, Rank={item.get('ranking')}\n"
    
    schema = load_schema('modify_direct')
    
    prompt = f"""You are an expert aerodynamics engineer optimizing airfoil designs.

{ctx_str}

CURRENT AIRFOIL DESIGN:
The attached image shows the current airfoil geometry. Analyze its shape carefully.

CSV DATA ({base_csv}):
```
{parent_content}
```

CSV FORMAT:
- Line 1: n_cp (control points) n_sp (sampling points)
- Lines 2 to n_cp+1: radius values per control point (0=curved, 1=sharp)
- Lines n_cp+2 to 2*n_cp+1: edgy values per control point (0=smooth, 1=angular)
- Remaining lines: x y coordinates of each control point

VISUAL ANALYSIS INSTRUCTIONS:
1. Look at the image - identify the leading edge, trailing edge, and overall profile
2. Notice any asymmetries, bulges, or sharp corners
3. Consider how moving control points would reshape the airfoil
4. Think about aerodynamic improvements (smoother curves, better leading edge, etc.)

MODIFICATION TASK:
Based on BOTH the image and CSV data, propose modifications to improve aerodynamic performance.

REQUIREMENTS:
- Provide NEW values (different from the original) for each point you modify
- Consider the visual geometry when choosing which points to move
- Each value array: [x, y, radius, edgy]

OUTPUT FORMAT (JSON only):
{{
  "pt_idx": [0, 1, 2, 3],
  "values": [
    [new_x0, new_y0, new_radius0, new_edgy0],
    [new_x1, new_y1, new_radius1, new_edgy1],
    [new_x2, new_y2, new_radius2, new_edgy2],
    [new_x3, new_y3, new_radius3, new_edgy3]
  ]
}}

Schema reference:
{schema}"""

    # Call Gemini with image
    params = get_gemini_multimodal_response(prompt, [image], temperature=temperature)
    print(f"DEBUG: Gemini response: {json.dumps(params, indent=2)}")
    
    if not params or 'pt_idx' not in params or 'values' not in params:
        print(f"Error: Invalid response. Got: {params}")
        return None
    
    # Remove schema metadata if present
    for field in ['$schema', 'title', 'description', 'type', 'properties', 'required']:
        params.pop(field, None)
    
    # Execute modification
    params['base_csv'] = base_csv
    params['out_dir'] = output_dir
    params['name'] = name
    
    csv_path = run_action('modify_direct', **params)
    
    # Generate visualization using fenics_env
    test_mod_script = os.path.join(PARENT_DIR, 'test_modification.py')
    fenics_python = '/root/miniconda3/envs/fenics_env/bin/python'
    cmd = [fenics_python, test_mod_script, csv_path, '-o', output_dir]
    subprocess.run(cmd, capture_output=True, text=True)
    
    return csv_path

def batch_modify_with_images(
    base_csv: str,
    geometry_image: str,
    output_dir: str,
    n_modifications: int = 5,
    temperature: float = 1.0
) -> List[str]:
    """Run multiple modifications with image input."""
    results = []
    for i in range(n_modifications):
        print(f"\n--- Modification {i+1}/{n_modifications} ---")
        csv_path = modify_with_image(
            base_csv=base_csv,
            geometry_image=geometry_image,
            output_dir=output_dir,
            name=f"imgmod{i}",
            temperature=temperature
        )
        if csv_path:
            results.append(csv_path)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Modify airfoil using image input")
    parser.add_argument("csv", help="Path to parent CSV file")
    parser.add_argument("image", help="Path to geometry image PNG")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument("-n", "--name", default="modified", help="Output name prefix")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="LLM temperature")
    parser.add_argument("--batch", type=int, default=1, help="Number of modifications to generate")
    args = parser.parse_args()
    
    if args.batch > 1:
        batch_modify_with_images(args.csv, args.image, args.output, args.batch, args.temperature)
    else:
        modify_with_image(args.csv, args.image, args.output, args.name, temperature=args.temperature)
