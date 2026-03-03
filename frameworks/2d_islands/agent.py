import os
import re
import json
import shutil
import subprocess
import sys
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple

from .design_actions import run_action
from .prompts.generate import get_generate_prompt, get_generate_system, sample_strategy, GENERATE_STRATEGY_NAMES
from .prompts.generate_direct import get_generate_direct_prompt
from .prompts.modify import get_modify_prompt
from .prompts.modify_direct import get_modify_direct_prompt
from .prompts import gaussain as gaussain_prompts
from .prompts import (
    GENERATE_DIRECT_SYSTEM,
    MODIFY_SYSTEM,
    MODIFY_DIRECT_SYSTEM,
)

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: Google API Key not found.")


def extract_structured_response(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    analysis = None
    rationale = None
    json_str = None

    analysis_match = re.search(r'<ANALYSIS>(.*?)</ANALYSIS>', text, re.DOTALL)
    if analysis_match:
        analysis = analysis_match.group(1).strip()

    rationale_match = re.search(r'<DESIGN_RATIONALE>(.*?)</DESIGN_RATIONALE>', text, re.DOTALL)
    if rationale_match:
        rationale = rationale_match.group(1).strip()

    design_match = re.search(r'<DESIGN>(.*?)</DESIGN>', text, re.DOTALL)
    if design_match:
        design_content = design_match.group(1).strip()
        start, end = design_content.find('{'), design_content.rfind('}') + 1
        if start != -1 and end > start:
            json_str = design_content[start:end]

    if not analysis:
        analysis_match = re.search(r'##\s*Analysis\s*\n(.*?)(?=##|\{|$)', text, re.DOTALL | re.IGNORECASE)
        if analysis_match:
            analysis = analysis_match.group(1).strip()

    if not rationale:
        rationale_match = re.search(r'##\s*(?:Design\s+)?Reasoning\s*\n(.*?)(?=##|\{|$)', text, re.DOTALL | re.IGNORECASE)
        if rationale_match:
            rationale = rationale_match.group(1).strip()

    if json_str is None:
        params_match = re.search(r'##\s*Design\s+Parameters\s*\n.*?(\{.*?\})', text, re.DOTALL | re.IGNORECASE)
        if params_match:
            json_str = params_match.group(1).strip()
        else:
            start, end = text.find('{'), text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]

    return analysis, rationale, json_str


def get_gemini_response(
    system_prompt: str,
    user_prompt: str,
    images: List[Any] = None,
    temperature: float = 1.0,
    return_full: bool = False,
    return_raw: bool = False
):
    try:
        model = genai.GenerativeModel(
            'gemini-3-pro-preview',
            system_instruction=system_prompt
        )
        content = [user_prompt] + (list(images) if images else [])
        generation_config = genai.types.GenerationConfig(temperature=temperature)
        response = model.generate_content(content, generation_config=generation_config)
        text = response.text

        analysis, rationale, json_str = extract_structured_response(text)

        if json_str is None:
            print(f"No JSON found in response: {text[:500]}")
            if return_full:
                return {}, analysis, rationale
            return {}

        params = json.loads(json_str)

        if return_full and return_raw:
            return params, analysis, rationale, text
        if return_full:
            return params, analysis, rationale
        return params

    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        if return_full and return_raw:
            return {}, None, None, text if 'text' in dir() else ''
        if return_full:
            return {}, None, None
        return {}
    except Exception as e:
        print(f"Gemini Error: {e}")
        if return_full and return_raw:
            return {}, None, None, text if 'text' in dir() else ''
        if return_full:
            return {}, None, None
        return {}


def run_llm_action(
    action: str,
    context: List[Dict],
    output_dir: str,
    base_csv: str = None,
    name: str = "llm_design",
    temperature: float = 1.0,
    skip_vis: bool = False,
    debug_dir: str = None,
    strategy_idx: int = None,
    random_strategy: bool = True
):
    os.makedirs(output_dir, exist_ok=True)
    # format_context is expected to be set on the context items already by the framework
    # We need to get it from the environment -- the framework's run.py injects it
    from .prompts.base import format_response_instructions  # noqa: F811
    ctx_text = _format_context_from_items(context)
    images = [img for item in context for img in item.get('images', [])]

    strategy_name = None

    if action == 'generate':
        if strategy_idx is None and random_strategy:
            strategy_idx, strategy_name = sample_strategy()
        elif strategy_idx is not None:
            strategy_idx = strategy_idx % len(GENERATE_STRATEGY_NAMES)
            strategy_name = GENERATE_STRATEGY_NAMES[strategy_idx]
        system_prompt = get_generate_system(strategy_idx)
        user_prompt = get_generate_prompt(ctx_text, strategy_idx)

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

    elif action in ['gaussain', 'gaussian']:
        if strategy_idx is None and random_strategy:
            strategy_idx, strategy_name = gaussain_prompts.sample_strategy()
        elif strategy_idx is not None:
            strategy_idx = strategy_idx % len(gaussain_prompts.GENERATE_STRATEGY_NAMES)
            strategy_name = gaussain_prompts.GENERATE_STRATEGY_NAMES[strategy_idx]
        system_prompt = gaussain_prompts.get_generate_system(strategy_idx)
        user_prompt = gaussain_prompts.get_generate_prompt(ctx_text, strategy_idx)
    else:
        print(f"Unknown action: {action}")
        return None

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, 'context.txt'), 'w', encoding='utf-8') as f:
            f.write(ctx_text)
        with open(os.path.join(debug_dir, 'system_prompt.txt'), 'w', encoding='utf-8') as f:
            f.write(system_prompt)
        with open(os.path.join(debug_dir, 'user_prompt.txt'), 'w', encoding='utf-8') as f:
            f.write(user_prompt)
        if strategy_name:
            with open(os.path.join(debug_dir, 'strategy.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Strategy: {strategy_name} (idx={strategy_idx})")
        for idx, img_path in enumerate(images):
            if isinstance(img_path, str) and os.path.exists(img_path):
                try:
                    shutil.copy2(img_path, os.path.join(debug_dir, f"image_{idx}_{os.path.basename(img_path)}"))
                except Exception as e:
                    print(f"Warning: Could not copy image {img_path}: {e}")

    params, analysis, rationale = get_gemini_response(
        system_prompt, user_prompt, images,
        temperature=temperature, return_full=True
    )

    if debug_dir:
        if analysis:
            with open(os.path.join(debug_dir, 'llm_analysis.txt'), 'w', encoding='utf-8') as f:
                f.write(analysis)
        if rationale:
            with open(os.path.join(debug_dir, 'llm_rationale.txt'), 'w', encoding='utf-8') as f:
                f.write(rationale)
        with open(os.path.join(debug_dir, 'llm_params.json'), 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2)

    if not params:
        print("Error: LLM returned empty params")
        return None

    for field in ['$schema', 'title', 'description', 'type', 'properties', 'required']:
        params.pop(field, None)

    if action in ['generate', 'generate_direct', 'gaussain', 'gaussian']:
        if 'n_cp' not in params or 'params' not in params:
            print(f"Error: Missing required fields. Got: {list(params.keys())}")
            return None
        if 'n_sp' not in params:
            params['n_sp'] = 10
    elif action in ['modify', 'modify_direct']:
        if 'pt_idx' not in params or 'values' not in params:
            print(f"Error: Missing required fields for modify. Got: {list(params.keys())}")
            return None

    params['out_dir'] = output_dir
    params['name'] = name
    if base_csv and action in ['modify', 'modify_direct']:
        params['base_csv'] = base_csv

    csv_path = run_action(action, **params)

    if not skip_vis:
        test_mod_script = os.path.join(os.path.dirname(__file__), 'test_modification.py')
        if os.path.exists(test_mod_script):
            cmd = [sys.executable, test_mod_script, csv_path, '-o', output_dir]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Visualization error: {result.stderr}")

    return csv_path


# The framework injects the env's format_context via _env_format_context.
_env_format_context = None


def set_env_format_context(fn):
    global _env_format_context
    _env_format_context = fn


def _format_context_from_items(context):
    """Use the env-provided format_context if available, else fall back."""
    if _env_format_context is not None:
        return _env_format_context(context)
    # Fallback: basic text formatting
    if not context:
        return "No previous designs available."
    lines = []
    for i, item in enumerate(context):
        lines.append(f"Design {i+1}: reward={item.get('reward', 0):.4f}")
    return '\n'.join(lines)
