"""
tinker_backend.py — Synchronous wrapper around Tinker's async sampling API.

Provides `generate_design(system_prompt, user_prompt, temperature)` as a
drop-in replacement for Gemini in framework agents, while capturing
(ModelInput, TokensWithLogprobs) for RL trajectory construction.
"""

import asyncio
import threading
import os
import logging

import tinker
from ttt_discover.tinker_utils.completers import TwoPhaseTokenCompleter, TokensWithLogprobs
from ttt_discover.tinker_utils.misc_utils import Tokenizer

logger = logging.getLogger(__name__)

_loop = None
_loop_thread = None
_sampling_client = None
_tokenizer = None
_policy = None

_last_model_input = None
_last_tokens_with_logprobs = None
_lock = threading.Lock()

_phase1_max_tokens = 26000
_temperature = 1.0


def _ensure_loop():
    global _loop, _loop_thread
    if _loop is not None:
        return
    _loop = asyncio.new_event_loop()
    _loop_thread = threading.Thread(target=_loop.run_forever, daemon=True)
    _loop_thread.start()


def configure(
    sampling_client: tinker.SamplingClient,
    tokenizer: Tokenizer,
    phase1_max_tokens: int = 26000,
    temperature: float = 1.0,
):
    global _sampling_client, _tokenizer, _policy, _phase1_max_tokens, _temperature
    _ensure_loop()
    _sampling_client = sampling_client
    _tokenizer = tokenizer
    _phase1_max_tokens = phase1_max_tokens
    _temperature = temperature
    _policy = TwoPhaseTokenCompleter(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        phase1_max_tokens=phase1_max_tokens,
        temperature=temperature,
    )


def update_sampling_client(sampling_client: tinker.SamplingClient):
    global _sampling_client, _policy
    _sampling_client = sampling_client
    _policy = TwoPhaseTokenCompleter(
        sampling_client=sampling_client,
        tokenizer=_tokenizer,
        phase1_max_tokens=_phase1_max_tokens,
        temperature=_temperature,
    )


def _build_model_input(system_prompt: str, user_prompt: str) -> tinker.ModelInput:
    renderer_name = os.environ.get("TTT_RENDERER", "gpt_oss_no_sysprompt")
    if renderer_name == "gpt_oss_no_sysprompt":
        text = (
            f"<|start|>system<|message|>{system_prompt}<|end|>"
            f"<|start|>user<|message|>{user_prompt}<|end|>"
            f"<|start|>assistant<|channel|>analysis<|message|>"
        )
    else:
        text = f"{system_prompt}\n\n{user_prompt}"
    tokens = _tokenizer.encode(text, add_special_tokens=False)
    return tinker.ModelInput(chunks=[tinker.types.EncodedTextChunk(tokens=tokens)])


async def _sample_async(model_input, stop, temperature):
    policy = TwoPhaseTokenCompleter(
        sampling_client=_sampling_client,
        tokenizer=_tokenizer,
        phase1_max_tokens=_phase1_max_tokens,
        temperature=temperature,
    )
    return await policy(model_input, stop)


def generate_design(system_prompt: str, user_prompt: str, temperature: float = 1.0):
    """Synchronous entry point. Returns (text, tokens, logprobs).

    Captures (ModelInput, TokensWithLogprobs) internally for RL trajectory.
    """
    global _last_model_input, _last_tokens_with_logprobs
    _ensure_loop()

    model_input = _build_model_input(system_prompt, user_prompt)
    stop = ["<|end|>"]

    future = asyncio.run_coroutine_threadsafe(
        _sample_async(model_input, stop, temperature),
        _loop,
    )
    twl: TokensWithLogprobs = future.result(timeout=600)

    text = _tokenizer.decode(twl.tokens)

    with _lock:
        _last_model_input = model_input
        _last_tokens_with_logprobs = twl

    n_tokens = len(twl.tokens)
    logger.info(f"Tinker generated {n_tokens} tokens")
    return text, twl.tokens, twl.logprobs


def pop_last_trajectory_data():
    """Return and clear the last (ModelInput, TokensWithLogprobs) pair."""
    global _last_model_input, _last_tokens_with_logprobs
    with _lock:
        ob, ac = _last_model_input, _last_tokens_with_logprobs
        _last_model_input = None
        _last_tokens_with_logprobs = None
    return ob, ac
