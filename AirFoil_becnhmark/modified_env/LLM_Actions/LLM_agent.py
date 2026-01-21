import os
import json
import subprocess
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Any

# Custom imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_design_actions import run_action

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: Google API Key not found.")

SCHEMAS_DIR = os.path.join(os.path.dirname(__file__), 'schemas')

def load_schema(name: str) -> str:
    path = os.path.join(SCHEMAS_DIR, f"{name}.json")
    with open(path, 'r') as f: return f.read()

def get_gemini_response(prompt: str, images: List[Any] = None, temperature: float = 1.0) -> Dict:
    try:
        model = genai.GenerativeModel('gemini-3-pro-preview')
        content = [prompt] + (list(images) if images else [])
        generation_config = genai.types.GenerationConfig(temperature=temperature)
        response = model.generate_content(content, generation_config=generation_config)
        text = response.text
        start, end = text.find('{'), text.rfind('}') + 1
        if start == -1:
            print(f"No JSON found in response: {text[:200]}")
            return {}
        json_str = text[start:end]
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Raw response: {text[:500]}")
        return {}
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {}

def format_context(context: List[Dict]) -> str:
    ctx_str = """Design Context:

PARAMETER DEFINITIONS:
- radius: Local radius of curve around control points, in [0,1] (maximal sharpness for radius = 1)
- edgy: Controls the smoothness of the curve, in [0,1] (maximal smoothness for edgy = 0)
- n_cp (n_pts): Number of random points joined together by Bezier curves. Hard set to 4 for this task.
- n_sp (n_sampling_pts): Number of sampled points on each Bezier curve joining two control points
- control_pts: (x, y) coordinates of control points that define the airfoil shape

Example of (un-optimized) baseline airfoil (reset/4/shape_0.csv):
- n_cp: 4, n_sp: 10
- radius: [0.5, 0.5, 0.5, 0.5]
- edgy: [0.5, 0.5, 0.5, 0.5]
- control_pts: [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]

Inspirations:
"""
    for i, item in enumerate(context):
        ctx_str += f"  {i+1}. Vector={item.get('vector')}, Reward={item.get('reward')}, Rank={item.get('ranking')}\n"
    return ctx_str

def run_llm_action(action: str, context: List[Dict], output_dir: str, base_csv: str = None, name: str = "llm_design", temperature: float = 1.5):
    """Orchestrates: LLM Params -> Design Action -> Visualization/Mesh"""
    os.makedirs(output_dir, exist_ok=True)
    schema = load_schema(action)
    ctx_text = format_context(context)
    images = [img for item in context for img in item.get('images', [])]
    
    # Creative prompt for generate actions
    if action in ['generate', 'generate_direct']:
        prompt = f"""{ctx_text}

TASK: Design a completely NEW and INNOVATIVE airfoil geometry. Be creative and create completly new design for iptimized perfoamce

DESIGN GUIDELINES:
- DO NOT simply copy the inspiration designs
- Vary sampling resolution (n_sp for Bezier curve smoothness)
- Control point coordinates: determines the shape of the airfouil **high importance**
- Radius: Controls Bezier curvature strength
- Edgy: Controls corner sharpness (0=smooth, 1=sharp)
- Consider both symmetric and asymmetric configurations
- Be creative but ensure the airfoil shape can be meshed successfully and is optimized.

Output JSON strictly following this schema:
{schema}"""
    elif action in ['modify', 'modify_direct']:
        # Read parent CSV content to show LLM what it's modifying
        parent_content = ""
        if base_csv and os.path.exists(base_csv):
            with open(base_csv, 'r') as f:
                parent_content = f.read()
        
        prompt = f"""{ctx_text}

TASK: MODIFY the existing airfoil design to improve aerodynamic performance. 

CURRENT DESIGN TO MODIFY ({base_csv}):
```
{parent_content}
```

CSV FORMAT: Line 1 = n_cp n_sp. Lines 2-(n_cp+1) = radius per point. Lines (n_cp+2)-(2*n_cp+1) = edgy per point. Remaining lines = x y coordinates per point.

INSTRUCTIONS:
1. YOU MUST CHANGE AT LEAST SOME VALUES - do NOT return the same values as above.
2. Modify control point positions (x, y) to create a better airfoil shape.
3. Adjust radius (curve sharpness) and edgy (smoothness) values.
4. Make meaningful changes, not tiny adjustments.

REQUIRED JSON FORMAT (only pt_idx and values needed):
{{
  "pt_idx": [0, 1, 2, 3],
  "values": [
    [NEW_X, NEW_Y, NEW_RADIUS, NEW_EDGY],
    [NEW_X, NEW_Y, NEW_RADIUS, NEW_EDGY],
    [NEW_X, NEW_Y, NEW_RADIUS, NEW_EDGY],
    [NEW_X, NEW_Y, NEW_RADIUS, NEW_EDGY]
  ]
}}

Return ONLY the JSON with DIFFERENT values than the original. Schema:
{schema}"""
    else:
        prompt = f"{ctx_text}\nOutput JSON for '{action}' strictly following:\n{schema}"
        if base_csv: prompt += f"\nBase CSV to modify: {base_csv}"
    
    # 1. Get Params from LLM with higher temperature for creativity
    params = get_gemini_response(prompt, images, temperature=temperature)
    print(f"DEBUG: Gemini raw params: {json.dumps(params, indent=2)}")
    if not params:
        print("Error: LLM returned empty params")
        return None
    
    # Remove all schema metadata fields that LLM might include
    schema_fields = ['$schema', 'title', 'description', 'type', 'properties', 'required']
    for field in schema_fields:
        params.pop(field, None)
    
    # Validate required fields
    if action in ['generate', 'generate_direct']:
        if 'n_cp' not in params or 'n_sp' not in params or 'params' not in params:
            print(f"Error: Missing required fields. Got: {list(params.keys())}")
            print(f"Params content: {params}")
            return None
    elif action in ['modify', 'modify_direct']:
        if 'pt_idx' not in params or 'values' not in params:
            print(f"Error: Missing required fields for modify. Got: {list(params.keys())}")
            print(f"Params content: {params}")
            return None
    
    # 2. Execute Design Action (CSV generation)
    params['out_dir'] = output_dir
    params['name'] = name
    if base_csv: params['base_csv'] = base_csv
    
    csv_path = run_action(action, **params)
    
    # 3. Visualization and Meshing using fenics_env
    test_mod_script = os.path.join(os.path.dirname(__file__), 'test_modification.py')
    fenics_python = '/root/miniconda3/envs/fenics_env/bin/python'
    
    cmd = [fenics_python, test_mod_script, csv_path, '-o', output_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Meshing error: {result.stderr}")
    
    return csv_path

if __name__ == "__main__":
    print("LLM Agent ready.")