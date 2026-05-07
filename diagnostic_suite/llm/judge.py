from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from diagnostic_suite.config import DEFAULT_CONFIG
from diagnostic_suite.types import DiagnosticInput, EvidenceBundle, LLMDiagnosticReport
from diagnostic_suite.llm.output_parser import fallback_llm_report, parse_llm_report
from diagnostic_suite.llm.prompt_system import build_system_prompt
from diagnostic_suite.llm.prompt_user import build_user_prompt


BackendFn = Callable[..., str]


@dataclass
class JudgeConfig:
    model_name: str = DEFAULT_CONFIG.thresholds.llm_model_name
    temperature: float = DEFAULT_CONFIG.thresholds.llm_temperature
    max_output_tokens: int = DEFAULT_CONFIG.thresholds.llm_max_output_tokens
    prompt_version: str = DEFAULT_CONFIG.thresholds.prompt_version


def _google_backend_factory(cfg: JudgeConfig) -> BackendFn:
    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GEMINI_KEY")
    )
    if not api_key:
        raise RuntimeError("No Gemini API key found in environment.")

    import google.generativeai as genai

    genai.configure(api_key=api_key)

    def _run(system_prompt: str, user_prompt: str, image_paths: Optional[List[str]] = None) -> str:
        model = genai.GenerativeModel(cfg.model_name, system_instruction=system_prompt)
        generation_config = genai.types.GenerationConfig(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
        )
        content: List[Any] = [user_prompt]
        for image_path in image_paths or []:
            if not image_path or not os.path.exists(image_path):
                continue
            try:
                # Send actual image bytes to Gemini so the judge can reason about geometry realism.
                content.append(genai.upload_file(image_path))
            except Exception:
                # Keep request robust: skip unreadable images instead of failing entire diagnosis.
                continue
        response = model.generate_content(content, generation_config=generation_config)
        return (response.text or "").strip()

    return _run


def run_llm_judge(
    diag_input: DiagnosticInput,
    evidence_bundle: EvidenceBundle,
    backend: Optional[BackendFn] = None,
    config: Optional[JudgeConfig] = None,
) -> LLMDiagnosticReport:
    """Run primary LLM diagnostic judge.

    The backend can be injected for deterministic testing. If not supplied,
    a Gemini backend is created from environment variables.
    """
    cfg = config or JudgeConfig()
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(diag_input, evidence_bundle)
    image_paths = [p for p in (diag_input.images or []) if isinstance(p, str)]

    runner = backend
    if runner is None:
        try:
            runner = _google_backend_factory(cfg)
        except Exception as exc:
            return fallback_llm_report(
                reason=f"Unable to initialize LLM backend: {exc}",
                parser_warnings=[f"prompt_version={cfg.prompt_version}", f"model_name={cfg.model_name}"],
            )

    try:
        try:
            raw = runner(system_prompt, user_prompt, image_paths)
        except TypeError:
            # Backward-compatible path for older injected backends that only accept two args.
            raw = runner(system_prompt, user_prompt)
    except Exception as exc:
        return fallback_llm_report(
            reason=f"LLM call failed: {exc}",
            parser_warnings=[f"prompt_version={cfg.prompt_version}", f"model_name={cfg.model_name}"],
        )

    report = parse_llm_report(raw_text=raw, model_name=cfg.model_name)
    if report.prompt_version is None:
        report.prompt_version = cfg.prompt_version
    return report

