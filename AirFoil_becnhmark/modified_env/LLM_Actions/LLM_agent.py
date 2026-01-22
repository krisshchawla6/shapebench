import os
import json
import subprocess
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any

# Custom imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_design_actions import run_action
from prompts import (
    format_context,
    GENERATE_SYSTEM, GENERATE_DIRECT_SYSTEM,
    MODIFY_SYSTEM, MODIFY_DIRECT_SYSTEM
)
from prompts.generate import get_generate_prompt
from prompts.generate_direct import get_generate_direct_prompt
from prompts.modify import get_modify_prompt
from prompts.modify_direct import get_modify_direct_prompt

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: Google API Key not found.")


def get_gemini_response(system_prompt: str, user_prompt: str, images: List[Any] = None, temperature: float = 1.0) -> Dict:
    """Get response from Gemini with system and user prompts."""
    try:
        model = genai.GenerativeModel(
            'gemini-2.0-flash',
            system_instruction=system_prompt
        )
        content = [user_prompt] + (list(images) if images else [])
        generation_config = genai.types.GenerationConfig(temperature=temperature)
        response = model.generate_content(content, generation_config=generation_config)
        text = response.text
        
        # Extract JSON from response
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


def run_llm_action(action: str, context: List[Dict], output_dir: str, base_csv: str = None, name: str = "llm_design", temperature: float = 1.0, skip_vis: bool = False):
    """Orchestrates: LLM Params -> Design Action -> Visualization/Mesh"""
    os.makedirs(output_dir, exist_ok=True)
    ctx_text = format_context(context)
    images = [img for item in context for img in item.get('images', [])]
    
    # Build prompts based on action type
    if action == 'generate':
        system_prompt = GENERATE_SYSTEM
        user_prompt = get_generate_prompt(ctx_text)
    elif action == 'generate_direct':
        system_prompt = GENERATE_DIRECT_SYSTEM
        user_prompt = get_generate_direct_prompt(ctx_text)
    elif action == 'modify':
        system_prompt = MODIFY_SYSTEM
        base_content = ""
        if base_csv and os.path.exists(base_csv):
            with open(base_csv, 'r') as f:
                base_content = f.read().strip()
        user_prompt = get_modify_prompt(ctx_text, base_csv or "", base_content)
    elif action == 'modify_direct':
        system_prompt = MODIFY_DIRECT_SYSTEM
        base_content = ""
        if base_csv and os.path.exists(base_csv):
            with open(base_csv, 'r') as f:
                base_content = f.read().strip()
        user_prompt = get_modify_direct_prompt(ctx_text, base_csv or "", base_content)
    else:
        print(f"Unknown action: {action}")
        return None
    
    # 1. Get Params from LLM
    params = get_gemini_response(system_prompt, user_prompt, images, temperature=temperature)
    print(f"DEBUG: Gemini raw params: {json.dumps(params, indent=2)}")
    
    if not params:
        print("Error: LLM returned empty params")
        return None
    
    # Remove schema metadata fields that LLM might accidentally include
    schema_fields = ['$schema', 'title', 'description', 'type', 'properties', 'required']
    for field in schema_fields:
        params.pop(field, None)
    
    # Validate required fields
    if action in ['generate', 'generate_direct']:
        if 'n_cp' not in params or 'params' not in params:
            print(f"Error: Missing required fields. Got: {list(params.keys())}")
            return None
        if 'n_sp' not in params:
            params['n_sp'] = 10  # Default value
    elif action in ['modify', 'modify_direct']:
        if 'pt_idx' not in params or 'values' not in params:
            print(f"Error: Missing required fields for modify. Got: {list(params.keys())}")
            return None
    
    # 2. Execute Design Action (CSV generation)
    params['out_dir'] = output_dir
    params['name'] = name
    if base_csv and action in ['modify', 'modify_direct']:
        params['base_csv'] = base_csv
    
    csv_path = run_action(action, **params)
    
    # 3. Visualization (optional)
    if not skip_vis:
        test_mod_script = os.path.join(os.path.dirname(__file__), 'test_modification.py')
        cmd = [sys.executable, test_mod_script, csv_path, '-o', output_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Visualization error: {result.stderr}")
    
    return csv_path


if __name__ == "__main__":
    print("LLM Agent ready.")
