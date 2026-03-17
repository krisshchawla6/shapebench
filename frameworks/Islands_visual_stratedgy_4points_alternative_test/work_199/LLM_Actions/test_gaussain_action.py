#!/usr/bin/env python3
"""
Test the Gaussian-mean action prompt for diversity and precision.
"""

import os
import sys
import re
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from LLM_agent import get_gemini_response, extract_structured_response
from llm_design_actions import gaussian_sampling
from prompts import format_context
from prompts import gaussain as gaussain_prompts
from diversity import compute_population_diversity


def _ensure_api_key() -> bool:
    return bool(
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GEMINI_KEY")
    )


def _extract_float_precision(json_str: str):
    floats = re.findall(r"-?\d+\.\d+", json_str)
    if not floats:
        return []
    return [len(val.split(".")[1]) for val in floats]


def _extract_params_from_raw(raw_text: str):
    analysis, rationale, json_str = extract_structured_response(raw_text)
    if not json_str:
        return None, None
    try:
        params = json.loads(json_str)
    except json.JSONDecodeError:
        return None, json_str
    return params, json_str


def test_precision(samples: int, temperature: float, min_decimal_places: int, min_ratio: float):
    context = format_context([])
    system_prompt = gaussain_prompts.get_generate_system()
    user_prompt = gaussain_prompts.get_generate_prompt(context)

    precision_pass = True
    stats = []

    for i in range(samples):
        params, _, _, raw_text = get_gemini_response(
            system_prompt,
            user_prompt,
            temperature=temperature,
            return_full=True,
            return_raw=True,
        )
        parsed, json_str = _extract_params_from_raw(raw_text)
        if not parsed:
            print(f"[precision] Sample {i}: failed to parse JSON response.")
            precision_pass = False
            continue

        precisions = _extract_float_precision(json_str)
        if not precisions:
            print(f"[precision] Sample {i}: no float values detected.")
            precision_pass = False
            continue

        ratio = sum(p >= min_decimal_places for p in precisions) / len(precisions)
        stats.append((len(precisions), ratio))
        print(f"[precision] Sample {i}: {ratio*100:.1f}% floats have >= {min_decimal_places} decimals")
        if ratio < min_ratio:
            precision_pass = False

    return precision_pass, stats


def test_diversity(samples: int, temperature: float):
    context = format_context([])
    system_prompt = gaussain_prompts.get_generate_system()
    user_prompt = gaussain_prompts.get_generate_prompt(context)

    vectors = []
    for i in range(samples):
        params, _, _, raw_text = get_gemini_response(
            system_prompt,
            user_prompt,
            temperature=temperature,
            return_full=True,
            return_raw=True,
        )
        parsed, _ = _extract_params_from_raw(raw_text)
        if not parsed or "params" not in parsed:
            print(f"[diversity] Sample {i}: missing params in response.")
            continue

        mean = np.array(parsed["params"], dtype=float).flatten()
        sampled = gaussian_sampling(mean)
        vectors.append({"vector": sampled.tolist()})

    metrics = compute_population_diversity(vectors)
    print(
        "[diversity] mean_pairwise_distance={:.4f} min_pairwise_distance={:.4f} "
        "population_diversity={:.4f}".format(
            metrics["mean_pairwise_distance"],
            metrics["min_pairwise_distance"],
            metrics["population_diversity"],
        )
    )

    diversity_pass = metrics["population_diversity"] > 0.05 and metrics["min_pairwise_distance"] > 0.01
    return diversity_pass, metrics


def main():
    parser = argparse.ArgumentParser(description="Test gaussain prompt precision and diversity.")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="LLM sampling temperature")
    parser.add_argument("--min-decimals", type=int, default=4, help="Minimum decimal places for precision")
    parser.add_argument("--min-ratio", type=float, default=0.7, help="Minimum ratio of precise floats")
    args = parser.parse_args()

    if not _ensure_api_key():
        print("Missing Gemini API key. Set GOOGLE_API_KEY, GEMINI_API_KEY, or GEMINI_KEY.")
        sys.exit(1)

    precision_ok, _ = test_precision(
        samples=args.samples,
        temperature=args.temperature,
        min_decimal_places=args.min_decimals,
        min_ratio=args.min_ratio,
    )
    diversity_ok, _ = test_diversity(
        samples=args.samples,
        temperature=args.temperature,
    )

    if precision_ok and diversity_ok:
        print("PASS: Precision and diversity checks succeeded.")
        sys.exit(0)
    print("FAIL: One or more checks failed.")
    sys.exit(2)


if __name__ == "__main__":
    main()
