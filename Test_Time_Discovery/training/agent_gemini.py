"""Gemini image analysis agent for TTT two-model pipeline.

The frozen Gemini model analyzes simulation images (Cp, Cfx fields)
and produces a text description of the flow field. This text is then
passed to the trainable model as additional context.

No tinker dependency. Uses Google Generative AI SDK.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_GEMINI_POOL = ThreadPoolExecutor(max_workers=2)

ANALYSIS_SYSTEM_PROMPT = ("You are an expert aerodynamicist analyzing CFD simulation results for a blended-wing-body aircraft.\n\nGiven the pressure coefficient (Cp) and skin friction coefficient (Cfx) field images, provide a concise technical analysis covering:\n1. Pressure distribution quality (leading edge suction peak, trailing edge recovery)\n2. Flow separation indicators (regions of adverse pressure gradient, Cfx reversal)\n3. Lift generation efficiency (extent of low-pressure region on upper surface)\n4. Drag sources (pressure drag from separation, skin friction distribution)\n5. Design improvement suggestions based on the flow field patterns\n\nBe quantitative where possible and focus on actionable observations.")

ANALYSIS_USER_PROMPT = "Analyze these aerodynamic field images for a blended-wing-body design.\n\nCurrent design reward (L/D): {reward:.4f}\n\nProvide a concise technical analysis of the flow field that will help guide the next design iteration."


def configure_gemini(api_key: Optional[str] = None) -> None:
    import google.generativeai as genai
    key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
    if key:
        genai.configure(api_key=key)
    else:
        logger.warning("No Google API key found for Gemini image analysis")


def analyze_images_sync(
    image_paths: List[str],
    reward: float = 0.0,
    model_name: str = "gemini-2.5-flash",
    system_prompt: str = ANALYSIS_SYSTEM_PROMPT,
    temperature: float = 0.3,
) -> str:
    import google.generativeai as genai

    valid_paths = [p for p in image_paths if isinstance(p, str) and os.path.exists(p)]
    if not valid_paths:
        return ""

    try:
        model = genai.GenerativeModel(model_name, system_instruction=system_prompt)

        images = []
        for path in valid_paths:
            img = _load_image(path)
            if img is not None:
                images.append(img)

        if not images:
            return ""

        user_text = ANALYSIS_USER_PROMPT.format(reward=reward)
        content = [user_text] + images

        config = genai.types.GenerationConfig(temperature=temperature)
        response = model.generate_content(content, generation_config=config)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        return ""


async def analyze_images_async(
    image_paths: List[str],
    reward: float = 0.0,
    model_name: str = "gemini-2.5-flash",
    system_prompt: str = ANALYSIS_SYSTEM_PROMPT,
    temperature: float = 0.3,
) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _GEMINI_POOL,
        lambda: analyze_images_sync(
            image_paths, reward, model_name, system_prompt, temperature
        ),
    )


def analyze_parent_images(
    parent_images: List[str],
    parent_reward: float,
    model_name: str = "gemini-2.5-flash",
) -> str:
    if not parent_images:
        return ""
    return analyze_images_sync(parent_images, reward=parent_reward, model_name=model_name)


def _load_image(path: str) -> Any:
    try:
        from PIL import Image
        return Image.open(path)
    except ImportError:
        import google.generativeai as genai
        return genai.upload_file(path)
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
        return None
