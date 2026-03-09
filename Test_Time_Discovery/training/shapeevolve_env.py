"""ShapeEvolveEnv -- discover Environment adapter for ShapeEvolve's BaseEnvironment.

This is the central integration point. It implements discover's Environment
interface so that discover's training loop (PUCT, rollouts, advantages,
gradient steps) can drive ShapeEvolve's design optimization.

discover calls: env_type(renderer, initial_state=state, sampler=sampler, config=config)
discover's loop: initial_observation() -> policy generates tokens -> step(tokens, step_idx)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, ClassVar, Dict, List, Optional

import tinker

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "discover"))

from ttt_discover.rl.types import Action, StepResult
from ttt_discover.tinker_utils import logtree, renderers
from ttt_discover.tinker_utils.dataset_builder import Environment, VerifyResult, SAFE_GRADE_EXECUTOR

from environments.base import BaseEnvironment

from .shapeevolve_state import ShapeEvolveState, params_to_construction

logger = logging.getLogger(__name__)


class ShapeEvolveEnv(Environment):
    """Adapter: wraps a ShapeEvolve BaseEnvironment as a discover Environment.

    Call ShapeEvolveEnv.configure(base_env, ...) once before training starts
    to set the shared BaseEnvironment instance, framework prompts, and Gemini config.

    All environment-specific keys (required design params, gaussian sampling
    function name, etc.) are derived dynamically from the framework modules
    passed in configure() -- nothing is hardcoded to a specific environment.
    """

    env_name: ClassVar[str] = "ShapeEvolve"
    state_type = ShapeEvolveState

    _base_env: ClassVar[Optional[BaseEnvironment]] = None
    _gemini_model: ClassVar[str] = "gemini-2.5-flash"
    _use_image_analysis: ClassVar[bool] = True
    _problem_description: ClassVar[str] = ""
    _required_keys: ClassVar[List[str]] = []
    _fw_agent: ClassVar[Optional[Any]] = None
    _fw_prompts: ClassVar[Optional[Any]] = None
    _fw_design_actions: ClassVar[Optional[Any]] = None
    _fw_sampling: ClassVar[Optional[Any]] = None
    _use_gaussian_sampling: ClassVar[bool] = True
    _sampler_type: ClassVar[str] = "puct"
    _gaussian_fn = None

    @classmethod
    def configure(
        cls,
        base_env: BaseEnvironment,
        gemini_model: str = "gemini-2.5-flash",
        use_image_analysis: bool = True,
        problem_description: str = "",
        required_keys: Optional[List[str]] = None,
        fw_agent: Optional[Any] = None,
        fw_prompts: Optional[Any] = None,
        fw_design_actions: Optional[Any] = None,
        fw_sampling: Optional[Any] = None,
        use_gaussian_sampling: bool = True,
        sampler_type: str = "puct",
    ):
        """Configure the shared BaseEnvironment and framework modules."""
        cls._base_env = base_env
        cls._gemini_model = gemini_model
        cls._use_image_analysis = use_image_analysis
        cls._problem_description = problem_description
        cls._fw_agent = fw_agent
        cls._fw_prompts = fw_prompts
        cls._fw_design_actions = fw_design_actions
        cls._fw_sampling = fw_sampling
        cls._use_gaussian_sampling = use_gaussian_sampling
        cls._sampler_type = sampler_type

        if required_keys is not None:
            cls._required_keys = required_keys
        elif fw_agent and hasattr(fw_agent, "REQUIRED_KEYS"):
            cls._required_keys = fw_agent.REQUIRED_KEYS
        else:
            cls._required_keys = []

        gfn = _find_gaussian_fn(fw_design_actions)
        if gfn:
            cls._gaussian_fn = staticmethod(gfn)
        else:
            cls._gaussian_fn = None

    @classmethod
    def create_initial_state(cls, problem_type: str) -> ShapeEvolveState:
        """Create a seed state for PUCT archive initialization."""
        return ShapeEvolveState(
            timestep=-1,
            construction=[],
            code="",
            value=0.0,
            observation="",
        )

    def __init__(self, renderer: renderers.Renderer, initial_state: ShapeEvolveState, **kwargs):
        if self._base_env is None:
            raise RuntimeError(
                "ShapeEvolveEnv._base_env not set. Call ShapeEvolveEnv.configure(base_env) before training."
            )
        super().__init__(renderer, initial_state=initial_state, **kwargs)
        self._prompt_blocks = None

    def get_question(self) -> str:
        """Build the user prompt from parent state + framework prompt templates."""
        params = self.initial_state.design_params
        context_list = []

        if self._fw_prompts and hasattr(self._fw_prompts, "format_context"):
            format_context = self._fw_prompts.format_context
        else:
            format_context = None

        if self._fw_prompts and hasattr(self._fw_prompts, "get_generate_prompt"):
            context_str = self.initial_state.to_prompt()
            strategy_idx = self.initial_state.timestep % 3 if self.initial_state.timestep >= 0 else 0
            parts = self._fw_prompts.get_generate_prompt(context_str, strategy_idx)
            return parts if isinstance(parts, str) else "\n".join(parts)

        state_context = self.initial_state.to_prompt()
        if state_context and state_context != "No previous design available.":
            context_list.append(state_context)

        if self._problem_description:
            context_list.insert(0, self._problem_description)

        if format_context:
            return format_context(context_list)

        if self._fw_prompts and hasattr(self._fw_prompts, "format_response_instructions"):
            context_list.append(self._fw_prompts.format_response_instructions())

        if self._required_keys:
            k = ", ".join(self._required_keys)
            example = "{" + ", ".join(f'\"{k}\": ...' for k in self._required_keys) + "}"
            context_list.append(f"Return a JSON object with keys: {k}\nExample: {example}")

        return "\n\n".join(context_list) if context_list else "No previous design available."

    def get_reference_answer(self) -> str:
        return "N/A (design optimization, no fixed reference)"

    def check_format(self, parsed_text: str) -> bool:
        """Check that the response contains valid JSON with required keys."""
        params = _parse_design_params(parsed_text, self._fw_agent)
        if params is None:
            return False
        if self._required_keys:
            return all(k in params for k in self._required_keys)
        return True

    def _get_code_languages(self) -> list[str]:
        return ["json"]

    def _should_keep_code_separators(self) -> bool:
        return True

    def step(self, action: Action, step_idx: int) -> StepResult:
        """Process LLM output: parse JSON design -> simulate -> update PUCT archive."""
        t_step_start = time.time()
        message = action.content if hasattr(action, "content") else str(action)

        parse_success = False
        response = message

        correct_format = self.check_format(response)
        if not correct_format:
            field = _parse_design_params(response, self._fw_agent)
            if field is not None:
                correct_format = True

        outs = None
        metrics = {}

        if correct_format:
            params = _parse_design_params(response, self._fw_agent)
            if params:
                parse_success = True
                outs = self._run_simulation(params, step_idx)

        if outs is None or not parse_success:
            _write_iteration_debug(
                self.log_path, step_idx, self.initial_state, response, None, outs,
                time.time() - t_step_start, self._sampler_type,
            )

            if hasattr(self.sampler, "record_failed_rollout"):
                self.sampler.record_failed_rollout(self.initial_state)

            return StepResult(
                observation="",
                reward=0.0,
                done=True,
                info={
                    "Problem: ": self._problem_description,
                    "Response: ": response[:500],
                    "Format: INVALID -- no valid design JSON found": "",
                },
                correct=False,
                format_correct=False,
                message="Invalid JSON",
            )

        step_result = StepResult(
            observation=outs.feedback or "",
            reward=outs.reward,
            done=True,
            info={
                "Format: OK, Reward: ": f"{outs.reward:.4f}",
                ", Raw Score: ": f"{outs.reward:.4f}",
                ", Msg: ": outs.feedback or "",
            },
            correct=outs.reward > 0,
            format_correct=True,
            message=response,
        )

        try:
            next_state = self._create_next_state(step_idx, response, outs)
            step_result.next_state = next_state
        except Exception as e:
            logger.warning(f"Failed to create next state: {e}")

        _write_iteration_debug(
            self.log_path, step_idx, self.initial_state, response, params, outs,
            time.time() - t_step_start, self._sampler_type,
        )

        return step_result

    def _create_next_state(self, step_idx: int, parsed_code: str, outs: VerifyResult) -> ShapeEvolveState:
        """Create the child ShapeEvolveState from simulation results."""
        construction = params_to_construction(
            _parse_design_params(parsed_code, self._fw_agent) or {}
        )

        gemini_text = ""
        if self._use_image_analysis:
            image_paths = outs.info.get("images", []) if outs.info else []
            design_path = outs.info.get("design_path", "") if outs.info else ""
            if image_paths:
                try:
                    from .agent_gemini import analyze_images_sync
                    gemini_text = analyze_images_sync(
                        image_paths, reward=outs.reward, model_name=self._gemini_model
                    )
                except Exception as e:
                    logger.warning(f"Gemini analysis failed: {e}")

        return ShapeEvolveState(
            timestep=step_idx,
            construction=construction,
            code=parsed_code,
            value=outs.reward,
            parent_values=[self.initial_state.value] if self.initial_state.value is not None else [],
            parents=[self.initial_state.to_dict()] if self.initial_state.construction else [],
            observation=outs.feedback or "",
            design_path=outs.info.get("design_path", "") if outs.info else "",
            image_paths=outs.info.get("images", []) if outs.info else [],
            gemini_analysis=gemini_text,
        )

    def _run_simulation(self, params: Dict[str, Any], step_idx: int) -> VerifyResult:
        """Write design JSON, run BaseEnvironment.simulate() in a thread pool."""
        try:
            loop = asyncio.get_event_loop()
            result = loop.run_in_executor(
                SAFE_GRADE_EXECUTOR,
                partial(self._simulate_sync, params, step_idx),
            )
            result = asyncio.get_event_loop().run_until_complete(result)
            return result
        except asyncio.TimeoutError:
            return VerifyResult(
                reward=0.0,
                feedback=f"Simulation timed out (limit={self._base_env.timeout}s)",
                correct=False,
                info={"error": "Simulation timeout"},
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Simulation error: {e}\n{tb}")
            return VerifyResult(
                reward=0.0,
                feedback=str(e),
                correct=False,
                info={"error": str(e)},
            )

    def _simulate_sync(self, params: Dict[str, Any], step_idx: int) -> VerifyResult:
        """Synchronous simulation: optionally gaussian-sample, then write JSON -> simulate."""
        if self._use_gaussian_sampling and self._gaussian_fn:
            params = self._gaussian_fn(params)

        run_id = f"step_{step_idx}_{uuid.uuid4().hex[:8]}"
        case_dir = os.path.join(self.log_path, "simulations", run_id)
        os.makedirs(case_dir, exist_ok=True)

        design_path = os.path.join(case_dir, "design.json")
        with open(design_path, "w") as f:
            json.dump(params, f, indent=2, default=str)

        reward, results, feedback, images = self._base_env.simulate(params, case_dir)

        return VerifyResult(
            reward=reward,
            feedback=feedback or "",
            correct=reward > 0,
            info={
                "design_path": design_path,
                "images": images or [],
                "feedback": feedback or "",
                "metrics": results or {},
            },
        )


def _find_gaussian_fn(fw_design_actions) -> Optional[callable]:
    """Auto-discover the gaussian sampling function from a framework's design_actions module."""
    if fw_design_actions is None:
        return None
    for name in dir(fw_design_actions):
        if "gaussian_sampling" in name:
            fn = getattr(fw_design_actions, name, None)
            if callable(fn):
                logger.info(f"Using gaussian function: {name}")
                return fn
    return None


def _write_iteration_debug(
    log_path: str,
    step_idx: int,
    parent_state,
    response: str,
    params: Optional[Dict],
    outs,
    elapsed_s: float,
    sampler_type: str,
):
    """Write per-rollout debug: a detailed JSON in debug/step_X/ and a one-liner in tinker_progress.jsonl."""
    try:
        debug_dir = os.path.join(log_path, "debug", f"step_{step_idx}")
        os.makedirs(debug_dir, exist_ok=True)

        run_id = f"rollout_{uuid.uuid4().hex[:8]}"
        debug_path = os.path.join(debug_dir, f"{run_id}.json")

        tinker_api_set = os.getenv("TINKER_API_KEY")
        tinker_base_url = os.getenv("TINKER_BASE_URL")

        debug = {
            "step_idx": step_idx,
            "sampler_type": sampler_type or "default",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": elapsed_s,
            "parent_state": parent_state.to_dict() if hasattr(parent_state, "to_dict") else str(parent_state),
            "response": response[:2000] if response else "",
            "params": params,
            "reward": outs.reward if outs else None,
            "feedback": outs.feedback if outs else None,
            "design_path": outs.info.get("design_path", "") if outs and outs.info else "",
            "images": outs.info.get("images", []) if outs and outs.info else [],
        }

        with open(debug_path, "w") as f:
            json.dump(debug, f, indent=2, default=str)

        progress_line = json.dumps({
            "timestep": step_idx,
            "reward": outs.reward if outs else 0.0,
            "elapsed_s": elapsed_s,
            "time": time.strftime("%H:%M:%S"),
            "sampler": sampler_type,
        }, default=str)

        progress_path = os.path.join(log_path, "tinker_progress.jsonl")
        with open(progress_path, "a") as f:
            f.write(progress_line + "\n")

    except Exception as e:
        logger.warning(f"Debug log write failed: {e}")


def _parse_design_params(text: str, fw_agent=None) -> Optional[Dict[str, Any]]:
    """Extract design parameters from LLM response text."""
    if fw_agent and hasattr(fw_agent, "extract_structured_response"):
        try:
            _analysis, _rationale, params = fw_agent.extract_structured_response(text)
            if params:
                return params
        except Exception:
            pass

    try:
        json_str = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_str:
            return json.loads(json_str.group())
    except Exception:
        pass

    for s in text.split("```"):
        s = s.strip()
        if s.startswith("{"):
            try:
                return json.loads(s)
            except Exception:
                pass

    return None
