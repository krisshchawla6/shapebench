import os
import re
import json
import shutil
from typing import List, Dict, Any, Optional, Tuple

import google.generativeai as genai
from dotenv import load_dotenv

from .design_actions import run_action_neuralfoil, N_CST
from .prompts import gaussian_neuralfoil as gaussian_prompts

_THIS_DIR = os.path.dirname(__file__)
_DOTENV_PATHS = [
    os.path.join(_THIS_DIR, '.env'),
    os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)), 'frameworks', '.env'),
]
for _p in _DOTENV_PATHS:
    if os.path.exists(_p):
        load_dotenv(_p, override=True)
        break

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: Google API Key not found.")

_env_format_context = None
_llm_backend = None
_image_analyzer = None


def set_env_format_context(fn):
    global _env_format_context
    _env_format_context = fn


def set_llm_backend(backend, image_analyzer=None):
    global _llm_backend, _image_analyzer
    _llm_backend = backend
    _image_analyzer = image_analyzer


def extract_structured_response(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    analysis = rationale = json_str = None

    m = re.search(r'<ANALYSIS>(.*?)</ANALYSIS>', text, re.DOTALL)
    if m:
        analysis = m.group(1).strip()
    m = re.search(r'<DESIGN_RATIONALE>(.*?)</DESIGN_RATIONALE>', text, re.DOTALL)
    if m:
        rationale = m.group(1).strip()
    m = re.search(r'<DESIGN>(.*?)</DESIGN>', text, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        s, e = inner.find('{'), inner.rfind('}') + 1
        if s != -1 and e > s:
            json_str = inner[s:e]

    if not analysis:
        m = re.search(r'##\s*Analysis\s*\n(.*?)(?=##|\{|$)', text, re.DOTALL | re.IGNORECASE)
        if m:
            analysis = m.group(1).strip()
    if not rationale:
        m = re.search(r'##\s*(?:Design\s+)?R(?:easoning|ationale)\s*\n(.*?)(?=##|\{|$)',
                      text, re.DOTALL | re.IGNORECASE)
        if m:
            rationale = m.group(1).strip()
    if json_str is None:
        m = re.search(r'##\s*Design\s+Parameters\s*\n.*?(\{.*?\})', text, re.DOTALL | re.IGNORECASE)
        if m:
            json_str = m.group(1).strip()
        else:
            s, e = text.find('{'), text.rfind('}') + 1
            if s != -1 and e > s:
                json_str = text[s:e]

    return analysis, rationale, json_str


def get_gemini_response(system_prompt, user_prompt, images=None,
                        temperature=1.0, return_full=False):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash',
                                      system_instruction=system_prompt)
        content = [user_prompt]
        if images:
            try:
                from PIL import Image
                for img_path in images:
                    if isinstance(img_path, str) and os.path.exists(img_path):
                        content.append(Image.open(img_path))
                    else:
                        # Fallback: keep textual item if it's not a local file path.
                        content.append(img_path)
            except Exception:
                # If PIL loading fails for any reason, keep original behavior.
                content.extend(list(images))
        cfg = genai.types.GenerationConfig(temperature=temperature)
        response = model.generate_content(content, generation_config=cfg)
        text = response.text
        analysis, rationale, json_str = extract_structured_response(text)
        if json_str is None:
            print(f"No JSON found in response: {text[:500]}")
            return ({}, analysis, rationale) if return_full else {}
        params = json.loads(json_str)
        return (params, analysis, rationale) if return_full else params
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        return ({}, None, None) if return_full else {}
    except Exception as e:
        print(f"Gemini Error: {e}")
        return ({}, None, None) if return_full else {}


REQUIRED_KEYS = ["upper_weights", "lower_weights", "leading_edge_weight", "TE_thickness"]


def _validate_params(params: dict) -> Optional[str]:
    """Return an error string if params are invalid, else None."""
    # For fixed-TE problems (e.g., multipoint_hpa), prompt may omit TE_thickness.
    # Defaulting here keeps validation strict while avoiding avoidable failed iterations.
    if "TE_thickness" not in params:
        params["TE_thickness"] = 0.0

    for key in REQUIRED_KEYS:
        if key not in params:
            return f"Missing required field '{key}'. Got: {list(params.keys())}"

    for arr_key in ("upper_weights", "lower_weights"):
        val = params[arr_key]
        if not isinstance(val, list):
            return f"'{arr_key}' must be a list, got {type(val).__name__}"
        if len(val) != N_CST:
            return f"'{arr_key}' must have {N_CST} elements, got {len(val)}"
        if not all(isinstance(v, (int, float)) for v in val):
            return f"'{arr_key}' must contain only numbers"

    for scalar_key in ("leading_edge_weight", "TE_thickness"):
        val = params[scalar_key]
        if not isinstance(val, (int, float)):
            return f"'{scalar_key}' must be a number, got {type(val).__name__}"

    return None


def run_llm_action_neuralfoil(
    action, context, output_dir, name="design",
    temperature=1.0, debug_dir=None, strategy_idx=None, random_strategy=True,
    scratchpad: str = "",
) -> Optional[str]:
    os.makedirs(output_dir, exist_ok=True)
    ctx_text = _env_format_context(context, scratchpad=scratchpad)
    images = [img for item in context for img in item.get('images', [])]

    strategy_name = None
    if action in ('gaussain', 'gaussian'):
        if strategy_idx is None and random_strategy:
            strategy_idx, strategy_name = gaussian_prompts.sample_strategy()
        elif strategy_idx is not None:
            strategy_idx = strategy_idx % len(gaussian_prompts.GENERATE_STRATEGY_NAMES)
            strategy_name = gaussian_prompts.GENERATE_STRATEGY_NAMES[strategy_idx]
        system_prompt = gaussian_prompts.get_generate_system(strategy_idx)
        user_prompt = gaussian_prompts.get_generate_prompt(ctx_text, strategy_idx)
    else:
        print(f"Unknown action: {action}")
        return None

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, 'context.txt'), 'w') as f:
            f.write(ctx_text)
        with open(os.path.join(debug_dir, 'system_prompt.txt'), 'w') as f:
            f.write(system_prompt)
        with open(os.path.join(debug_dir, 'user_prompt.txt'), 'w') as f:
            f.write(user_prompt)
        if strategy_name:
            with open(os.path.join(debug_dir, 'strategy.txt'), 'w') as f:
                f.write(f"Strategy: {strategy_name} (idx={strategy_idx})")
        for idx, img_path in enumerate(images):
            if isinstance(img_path, str) and os.path.exists(img_path):
                try:
                    shutil.copy2(img_path,
                                 os.path.join(debug_dir, f"image_{idx}_{os.path.basename(img_path)}"))
                except Exception:
                    pass

    if _llm_backend is not None:
        augmented_prompt = user_prompt
        if images and _image_analyzer:
            img_paths = [p for p in images if isinstance(p, str)]
            if img_paths:
                img_text = _image_analyzer(img_paths)
                if img_text:
                    augmented_prompt = f"[Airfoil Analysis]\n{img_text}\n\n{user_prompt}"
        try:
            text, _tokens, _logprobs = _llm_backend(system_prompt, augmented_prompt, temperature)
            analysis, rationale, json_str = extract_structured_response(text)
            params = json.loads(json_str) if json_str else {}
        except Exception as e:
            print(f"LLM backend error: {e}")
            params, analysis, rationale = {}, None, None
    else:
        params, analysis, rationale = get_gemini_response(
            system_prompt, user_prompt, images,
            temperature=temperature, return_full=True)

    if debug_dir:
        if analysis:
            with open(os.path.join(debug_dir, 'llm_analysis.txt'), 'w') as f:
                f.write(analysis)
        if rationale:
            with open(os.path.join(debug_dir, 'llm_rationale.txt'), 'w') as f:
                f.write(rationale)
        with open(os.path.join(debug_dir, 'llm_params.json'), 'w') as f:
            json.dump(params, f, indent=2)

    if not params:
        print("Error: LLM returned empty params")
        return None

    for field in ['$schema', 'title', 'description', 'type', 'properties', 'required']:
        params.pop(field, None)

    err = _validate_params(params)
    if err:
        print(f"Error: {err}")
        return None

    params.setdefault('name', name)
    return run_action_neuralfoil(action, params=params, out_dir=output_dir, name=name)
