import os
import re
import json
import shutil
from typing import List, Dict, Any, Optional, Tuple

import google.generativeai as genai
from dotenv import load_dotenv

from .design_actions import run_action_drivaer
from .prompts import gaussian_drivaer as gaussian_prompts

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path, override=True)

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
        m = re.search(r'##\s*(?:Design\s+)?Reasoning\s*\n(.*?)(?=##|\{|$)', text, re.DOTALL | re.IGNORECASE)
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
        content = [user_prompt] + (list(images) if images else [])
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


REQUIRED_KEYS = [
    "car_size", "car_width", "car_len", "ramp_angle",
    "front_bumper_length", "wind_screen_x", "wind_screen_z",
    "side_mirrors_x", "side_mirrors_z", "rear_window_x",
    "rear_window_z", "trunklid_angle", "trunklid_x", "trunklid_z",
    "diffusor_angle", "car_green_house_angle",
    "car_front_hood_angle", "car_air_intake_angle",
    "tires_diameter", "tires_width",
]


def run_llm_action_drivaer(
    action, context, output_dir, name="design",
    temperature=1.0, debug_dir=None, strategy_idx=None, random_strategy=True,
) -> Optional[str]:
    os.makedirs(output_dir, exist_ok=True)
    ctx_text = _env_format_context(context)
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
                    shutil.copy2(img_path, os.path.join(debug_dir, f"image_{idx}_{os.path.basename(img_path)}"))
                except Exception:
                    pass

    if _llm_backend is not None:
        augmented_prompt = user_prompt
        if images and _image_analyzer:
            img_paths = [p for p in images if isinstance(p, str)]
            if img_paths:
                img_text = _image_analyzer(img_paths)
                if img_text:
                    augmented_prompt = f"[Flow Field Analysis]\n{img_text}\n\n{user_prompt}"
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

    for key in REQUIRED_KEYS:
        if key not in params:
            print(f"Error: Missing required field '{key}'. Got: {list(params.keys())}")
            return None

    params.setdefault('name', name)
    return run_action_drivaer(action, params=params, out_dir=output_dir, name=name)
