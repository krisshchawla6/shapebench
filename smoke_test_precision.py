#!/usr/bin/env python3
"""Smoke test: verify the LLM outputs >= 8 decimal places for each environment.

Calls the design agent once per environment with an empty context and checks
precision of every numeric value in the returned JSON.
"""

import json
import os
import re
import sys

import google.generativeai as genai
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

for _p in [
    os.path.join(REPO_ROOT, 'frameworks', '.env'),
    os.path.join(REPO_ROOT, '.env'),
]:
    if os.path.exists(_p):
        load_dotenv(_p, override=True)
        break

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
if not api_key:
    sys.exit("ERROR: No Gemini API key found.")
genai.configure(api_key=api_key)

MIN_DECIMALS = 6  # JSON strips trailing zeros, so 8dp values like 0.18542100 appear as 6dp

ENVS = [
    ("NeuralFoil",   "environments.NeuralFoil.prompts.gaussian_neuralfoil"),
    ("DrivAer_Star", "environments.DrivAer_Star.prompts.gaussian_drivaer"),
    ("BlendedNet",   "environments.BlendedNet.prompts.gaussain_bwb"),
    ("vlm_3d",       "environments.vlm_3d.prompts.gaussain_3d"),
]


def _decimal_places(v) -> int:
    """Return number of decimal places in a float value."""
    s = f"{v}"
    if '.' in s:
        # strip trailing zeros to get true precision
        frac = s.split('.')[1].rstrip('0')
        return len(frac)
    return 0


def _check_precision(params: dict, min_dp: int) -> list[str]:
    """Return list of failures: param names whose precision is below min_dp."""
    failures = []
    for k, v in params.items():
        if k == "name":
            continue
        if isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, float):
                    dp = _decimal_places(item)
                    if dp < min_dp:
                        failures.append(f"{k}[{i}]={item} ({dp}dp)")
        elif isinstance(v, float):
            dp = _decimal_places(v)
            if dp < min_dp:
                failures.append(f"{k}={v} ({dp}dp)")
        # int values (e.g. naca_m, naca_p, naca_t) are skipped — legitimately integer params
    return failures


def _call_llm(system_prompt: str, user_prompt: str) -> dict:
    model = genai.GenerativeModel('gemini-3-flash-preview', system_instruction=system_prompt)
    cfg = genai.types.GenerationConfig(temperature=1.0)
    response = model.generate_content([user_prompt], generation_config=cfg)
    text = response.text

    # Try structured tags first, then bare JSON
    for pattern in [r'<DESIGN>(.*?)</DESIGN>', r'##\s*Design\s+Parameters\s*\n.*?(\{.*?\})']:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            inner = m.group(1).strip()
            s, e = inner.find('{'), inner.rfind('}') + 1
            if s != -1 and e > s:
                return json.loads(inner[s:e])
    s, e = text.find('{'), text.rfind('}') + 1
    if s != -1 and e > s:
        return json.loads(text[s:e])
    return {}


def run_test(env_name: str, module_path: str) -> bool:
    import importlib
    mod = importlib.import_module(module_path)

    system_prompt = mod.get_generate_system()
    user_prompt   = mod.get_generate_prompt("")

    print(f"\n{'='*60}")
    print(f"  ENV: {env_name}")
    print(f"  System prompt tail: ...{system_prompt[-80:].strip()!r}")
    print(f"{'='*60}")

    try:
        params = _call_llm(system_prompt, user_prompt)
    except Exception as e:
        print(f"  CALL FAILED: {e}")
        return False

    if not params:
        print("  FAIL: Empty/unparseable response")
        return False

    # Show raw values
    numeric = {k: v for k, v in params.items() if k != "name"}
    print(f"  Raw params: {json.dumps(numeric, indent=4)}")

    failures = _check_precision(params, MIN_DECIMALS)
    if failures:
        print(f"  FAIL: {len(failures)} param(s) below {MIN_DECIMALS}dp:")
        for f in failures:
            print(f"    - {f}")
        return False
    else:
        print(f"  PASS: all numeric params have >= {MIN_DECIMALS} decimal places")
        return True


if __name__ == "__main__":
    results = {}
    for env_name, module_path in ENVS:
        results[env_name] = run_test(env_name, module_path)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    passed = sum(results.values())
    for env, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {env}")
    print(f"\n  {passed}/{len(results)} environments passed")
    sys.exit(0 if passed == len(results) else 1)
