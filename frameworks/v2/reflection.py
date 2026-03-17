"""ReflectionAgent: orchestrates the post-design reflection + scratchpad update cycle.

After each design is generated, the framework calls:
    scratchpad = reflection_agent.run_cycle(scratchpad, reflection_inputs, iteration_nb, debug_dir)

reflection_inputs is provided by environment.get_reflection_inputs(design_path, case_dir) and
contains: intended_params, actual_action, designer_analysis, designer_reasoning, geometry_image_path.

Per-environment prompt files live at:
    frameworks/v2/prompts/<env_name>/reflection.py

To add reflection support for a new environment, create that file with:
    REFLECTION_SYSTEM, SCRATCHPAD_UPDATE_SYSTEM,
    build_reflection_prompt(), build_scratchpad_update_prompt()
"""

import os
import re
import importlib

import google.generativeai as genai
from dotenv import load_dotenv

_FRAMEWORKS_DIR = os.path.dirname(os.path.abspath(__file__))
for _dotenv_path in [
    os.path.join(_FRAMEWORKS_DIR, '.env'),
    os.path.join(os.path.dirname(_FRAMEWORKS_DIR), 'frameworks', '.env'),
]:
    if os.path.exists(_dotenv_path):
        load_dotenv(_dotenv_path)
        break

_api_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            or os.getenv("GEMINI_KEY"))
if _api_key:
    genai.configure(api_key=_api_key)


def _call_gemini(system_prompt, user_prompt, images=None, temperature=0.7):
    model = genai.GenerativeModel('gemini-3-flash-preview', system_instruction=system_prompt)
    content = [user_prompt] + (list(images) if images else [])
    config = genai.types.GenerationConfig(temperature=temperature)
    response = model.generate_content(content, generation_config=config)
    return response.text


def _extract_scratchpad(text):
    match = re.search(r'```\n?(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class ReflectionAgent:
    """Loads env-specific reflection prompts and runs the reflection + scratchpad cycle."""

    def __init__(self, env_name: str):
        self._env_name = env_name
        self._prompts = None
        try:
            module_path = f"frameworks.v2.prompts.{env_name}.reflection"
            self._prompts = importlib.import_module(module_path)
        except ImportError:
            print(f"v2: no reflection prompts found for env '{env_name}' "
                  f"(looked for frameworks/v2/prompts/{env_name}/reflection.py) "
                  f"— reflection step will be skipped.")

    def has_prompts(self) -> bool:
        return self._prompts is not None

    def run_cycle(self, scratchpad: str, reflection_inputs: dict,
                  iteration_nb: int, debug_dir=None) -> str:
        """Run reflection → scratchpad update. Returns updated scratchpad string."""
        if not self.has_prompts():
            return scratchpad

        intended_params = reflection_inputs.get('intended_params', {})
        actual_action = reflection_inputs.get('actual_action', [])
        designer_analysis = reflection_inputs.get('designer_analysis', '')
        designer_reasoning = reflection_inputs.get('designer_reasoning', '')
        geometry_image_path = reflection_inputs.get('geometry_image_path')

        # --- Reflection call ---
        reflection_prompt = self._prompts.build_reflection_prompt(
            intended_params, actual_action, designer_analysis, designer_reasoning
        )
        images = []
        if geometry_image_path and os.path.exists(geometry_image_path):
            from PIL import Image
            images.append(Image.open(geometry_image_path))

        reflection_text = _call_gemini(
            self._prompts.REFLECTION_SYSTEM, reflection_prompt, images
        )

        # --- Scratchpad update call ---
        update_prompt = self._prompts.build_scratchpad_update_prompt(
            scratchpad, reflection_text, intended_params, iteration_nb + 1
        )
        update_response = _call_gemini(
            self._prompts.SCRATCHPAD_UPDATE_SYSTEM, update_prompt
        )
        updated_scratchpad = _extract_scratchpad(update_response)

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, 'reflection.txt'), 'w') as f:
                f.write(reflection_text)
            with open(os.path.join(debug_dir, 'reflection_prompt.txt'), 'w') as f:
                f.write(reflection_prompt)
            with open(os.path.join(debug_dir, 'scratchpad_update_prompt.txt'), 'w') as f:
                f.write(update_prompt)
            with open(os.path.join(debug_dir, 'scratchpad_update_raw.txt'), 'w') as f:
                f.write(update_response)

        return updated_scratchpad
