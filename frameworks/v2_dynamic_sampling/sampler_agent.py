"""Dynamic sampling agent: LLM generates Python code for offspring sampling.

Two-level evolution:
  Level 1 (designs)  — existing island/LLM loop, unchanged
  Level 2 (code)     — SamplerCodeDatabase tracks generated code blocks
                       ranked by best offspring reward; LLM mutates/creates them

No Gaussian fallback. If all retries fail, offspring slots for that iteration
are simply skipped (batch shrinks). The traceback is injected into the next
prompt so the LLM can self-correct.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import traceback as tb

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
_DB_MAX_SIZE     = 20
_TOP_CODES_K     = 3
_SANDBOX_TIMEOUT = 40   # seconds for exec + fn call combined
_STAGNATION_THRESHOLD = 1e-5

SYSTEM_PROMPT = (
    "You implement one step of a numerical optimization algorithm for aerodynamic "
    "shape optimization. Each call you write:\n\n"
    "    def generate_samples(center, n, bounds, rng, history):\n\n"
    "Inputs:\n"
    "  center  — dict of parameter arrays/scalars (the designer agent's proposed mean).\n"
    "            A reference point — you are free to move far away from it.\n"
    "  n       — int. Exact number of candidate designs to return.\n"
    "  bounds  — dict {key: (lo, hi)}. Hard limits.\n"
    "  rng     — numpy.random.Generator\n"
    "  history — {\n"
    "      'best_designs':     [{'params': {...}, 'reward': float}, ...],  "
    "# top-K sorted best-first\n"
    "      'reward_trajectory': [float, ...],                              "
    "# best-so-far per iter, oldest->newest\n"
    "      'param_trajectory':  [{'params': {...}, 'reward': float}, ...]  "
    "# best design per recent iter, oldest->newest\n"
    "    }\n\n"
    "Pre-built helpers already in scope (do not redefine):\n"
    "  params_to_vec(p)   — dict → flat np.ndarray  (key order matches center)\n"
    "  vec_to_params(v)   — flat np.ndarray → dict  (auto-clipped to bounds)\n"
    "  n_dims             — int, total parameter dimensions\n"
    "  param_keys         — list[str], numerical parameter names in vector order\n\n"
    "Output: list of exactly n dicts with the same keys as center.\n"
    "Available: np (numpy), math. No other imports.\n"
    "Output only the function inside ```python ... ```. No other text."
)


# ── Database ─────────────────────────────────────────────────────────────────

class SamplerCodeDatabase:
    """Persistent JSON database of generated sampling code blocks."""

    def __init__(self, path: str):
        self.path = path
        self._entries: dict[str, dict] = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    for e in json.load(f):
                        self._entries[e['id']] = e
            except Exception as exc:
                print(f"  SamplerCodeDatabase: failed to load {path}: {exc}")

    def upsert(self, code_id: str, code: str, iteration: int):
        if code_id not in self._entries:
            self._entries[code_id] = {
                'id': code_id,
                'code': code,
                'best_reward': float('-inf'),
                'uses': 0,
                'created_iter': iteration,
                'last_used_iter': iteration,
            }
        else:
            self._entries[code_id]['last_used_iter'] = iteration
        self._entries[code_id]['uses'] += 1

    def record_result(self, code_id: str, llm_best_reward: float,
                      beat_gaussian: bool | None, iteration: int):
        if code_id not in self._entries:
            return
        e = self._entries[code_id]
        e['best_reward'] = max(e['best_reward'], float(llm_best_reward))
        if beat_gaussian is not None:
            e['beat_gaussian'] = e.get('beat_gaussian', 0) + int(beat_gaussian)
        self._save()

    def top(self, k: int) -> list[dict]:
        entries = [e for e in self._entries.values()
                   if e['best_reward'] > float('-inf')]
        entries.sort(key=lambda e: e['best_reward'], reverse=True)
        return entries[:k]

    def _save(self):
        entries = sorted(self._entries.values(),
                         key=lambda e: e['best_reward'], reverse=True)
        entries = entries[:_DB_MAX_SIZE]
        self._entries = {e['id']: e for e in entries}
        try:
            with open(self.path, 'w') as f:
                json.dump(entries, f, indent=2)
        except Exception as exc:
            print(f"  SamplerCodeDatabase: save failed: {exc}")


# ── Strategy selection ────────────────────────────────────────────────────────

def _choose_strategy(top_codes: list, iteration: int) -> str:
    """Alternate between NOVEL and MODIFY each iteration.

    NOVEL  — write a completely new optimization algorithm.
    MODIFY — take the best existing code and improve it.

    Falls back to NOVEL if no existing codes to modify.
    """
    if not top_codes:
        return 'novel'
    return 'novel' if iteration % 2 == 0 else 'modify'


# ── Agent ────────────────────────────────────────────────────────────────────

class SamplerAgent:
    """LLM-driven sampling agent. Generates one Python function per iteration."""

    def __init__(self, output_dir: str, model: str = 'gemini-2.5-flash',
                 max_retries: int = 3, hybrid: bool = False):
        self.model_name   = model
        self.max_retries  = max_retries
        self.hybrid       = hybrid
        self.code_db      = SamplerCodeDatabase(
            os.path.join(output_dir, 'sampler_code_db.json'))
        self._last_code_id: str | None = None
        self._configure_llm()

    def _configure_llm(self):
        try:
            import google.generativeai as genai
            key = (os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
                   or os.getenv('GEMINI_KEY'))
            if key:
                genai.configure(api_key=key)
        except Exception as exc:
            print(f"  SamplerAgent: genai config failed: {exc}")

    def _call_llm(self, user_prompt: str) -> str:
        import google.generativeai as genai
        model = genai.GenerativeModel(self.model_name,
                                      system_instruction=SYSTEM_PROMPT)
        cfg   = genai.types.GenerationConfig(temperature=0.9)
        resp  = model.generate_content(user_prompt, generation_config=cfg)
        return resp.text.strip()

    def generate_batch(
        self,
        center_params: dict,
        n: int,
        bounds: dict,
        top_design_contexts: list,
        reward_trajectory: list,
        param_trajectory: list,
        iteration: int,
        debug_dir: str | None = None,
    ) -> list[dict]:
        if n <= 0:
            return []

        top_codes   = self.code_db.top(_TOP_CODES_K)
        strategy    = _choose_strategy(top_codes, iteration)
        user_prompt = _build_user_prompt(
            center_params, n, bounds, top_design_contexts,
            reward_trajectory, param_trajectory, top_codes, iteration, self.hybrid,
            strategy=strategy)

        prev_code:  str | None = None
        prev_error: str | None = None

        for attempt in range(self.max_retries):
            up = user_prompt
            if prev_code is not None and prev_error is not None:
                up += (
                    f"\n\n=== PREVIOUS ATTEMPT (FAILED) ===\n{prev_code}"
                    f"\n\n=== ERROR ===\n{prev_error}"
                    f"\n\nFix the function. Same output format required."
                )

            try:
                response_text = self._call_llm(up)
            except Exception as exc:
                prev_error = f"LLM call failed: {exc}"
                prev_code  = ""
                print(f"  SamplerAgent attempt {attempt+1}/{self.max_retries}: LLM error: {exc}")
                if debug_dir:
                    _write_debug_attempt(debug_dir, iteration, attempt,
                                         up, raw_response=None, code=None,
                                         error=prev_error, succeeded=False)
                continue

            code = _extract_code(response_text)
            if code is None:
                prev_error = "No ```python ... ``` block found in response."
                prev_code  = response_text[:500]
                print(f"  SamplerAgent attempt {attempt+1}/{self.max_retries}: no code block")
                if debug_dir:
                    _write_debug_attempt(debug_dir, iteration, attempt,
                                         up, raw_response=response_text, code=None,
                                         error=prev_error, succeeded=False)
                continue

            history = {
                'best_designs':      top_design_contexts,
                'reward_trajectory': reward_trajectory,
                'param_trajectory':  param_trajectory,
            }
            rng = np.random.default_rng(iteration * 1000 + attempt)

            try:
                result = _exec_sandbox(code, center_params, n, bounds, rng, history)
            except Exception:
                prev_error = tb.format_exc()
                prev_code  = code
                print(f"  SamplerAgent attempt {attempt+1}/{self.max_retries}: "
                      f"exec failed:\n{prev_error[-400:]}")
                if debug_dir:
                    _write_debug_attempt(debug_dir, iteration, attempt,
                                         up, raw_response=response_text, code=code,
                                         error=prev_error, succeeded=False)
                continue

            code_id = hashlib.md5(code.encode()).hexdigest()[:8]
            self.code_db.upsert(code_id, code, iteration)
            self._last_code_id = code_id

            if debug_dir:
                _write_debug_attempt(debug_dir, iteration, attempt,
                                     up, raw_response=response_text, code=code,
                                     error=None, succeeded=True)

            print(f"  SamplerAgent: {len(result)} offspring "
                  f"(iter={iteration} attempt={attempt+1})")
            return result

        print(f"  SamplerAgent: all {self.max_retries} retries failed "
              f"for iteration {iteration} — offspring slots skipped")
        self._last_code_id = None
        return []

    def update_performance(
        self,
        llm_offspring_rewards: list[float],
        gaussian_best_reward: float | None,
        iteration: int,
    ):
        if self._last_code_id is None or not llm_offspring_rewards:
            return
        llm_best = float(max(llm_offspring_rewards))
        if gaussian_best_reward is not None:
            beat_gaussian: bool | None = llm_best > float(gaussian_best_reward)
        else:
            beat_gaussian = None
        self.code_db.record_result(self._last_code_id, llm_best, beat_gaussian, iteration)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_code(text: str) -> str | None:
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if 'def generate_samples' in text:
        return text.strip()
    return None


def _build_sandbox_helpers(center: dict, bounds: dict):
    """Build pre-defined helper functions injected into the sandbox namespace."""
    keys = [k for k, v in center.items()
            if not isinstance(v, str)]

    # Build slices into the flat vector for each key
    slices: dict[str, tuple[int, int]] = {}
    idx = 0
    for k in keys:
        v = center[k]
        n = np.asarray(v).size
        slices[k] = (idx, idx + n)
        idx += n
    n_dims = idx

    def params_to_vec(p: dict) -> np.ndarray:
        parts = []
        for k in keys:
            parts.append(np.asarray(p[k]).flatten())
        return np.concatenate(parts)

    def vec_to_params(v: np.ndarray, ref: dict | None = None) -> dict:
        if ref is None:
            ref = center
        out = dict(ref)
        for k in keys:
            s, e = slices[k]
            raw = v[s:e]
            if k in bounds:
                lo, hi = bounds[k]
                raw = np.clip(raw, lo, hi)
            orig = center[k]
            if isinstance(orig, list):
                out[k] = raw.tolist()
            elif isinstance(orig, np.ndarray):
                out[k] = raw.reshape(orig.shape)
            elif raw.size == 1:
                out[k] = float(raw[0])
            else:
                out[k] = raw
        return out

    return {
        'params_to_vec': params_to_vec,
        'vec_to_params': vec_to_params,
        'n_dims': n_dims,
        'param_keys': keys,
    }


def _exec_sandbox(code: str, center: dict, n: int, bounds: dict,
                  rng: np.random.Generator, history: dict) -> list[dict]:
    helpers = _build_sandbox_helpers(center, bounds)
    namespace: dict = {'np': np, 'math': math, **helpers}

    class _Timeout(Exception):
        pass

    def _handler(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.alarm(_SANDBOX_TIMEOUT)
        exec(code, namespace)          # noqa: S102
        fn = namespace.get('generate_samples')
        if fn is None:
            raise ValueError("Code did not define generate_samples()")
        result = fn(center, n, bounds, rng, history)
        signal.alarm(0)
    except _Timeout:
        raise TimeoutError(f"generate_samples exceeded {_SANDBOX_TIMEOUT}s")
    finally:
        signal.signal(signal.SIGALRM, old)
        signal.alarm(0)

    if not isinstance(result, list):
        raise ValueError(f"generate_samples returned {type(result).__name__}, expected list")
    if len(result) != n:
        raise ValueError(f"generate_samples returned {len(result)} items, expected {n}")
    for i, d in enumerate(result):
        if not isinstance(d, dict):
            raise ValueError(f"result[{i}] is {type(d).__name__}, expected dict")

    for d in result:
        for key, (lo, hi) in bounds.items():
            if key in d:
                d[key] = np.clip(d[key], lo, hi)

    return result


def _stagnation_iters(reward_trajectory: list) -> int:
    """Count trailing iterations with no meaningful reward improvement."""
    n = 0
    for i in range(len(reward_trajectory) - 1, 0, -1):
        if abs(reward_trajectory[i] - reward_trajectory[i - 1]) < _STAGNATION_THRESHOLD:
            n += 1
        else:
            break
    return n


def _build_user_prompt(
    center: dict, n: int, bounds: dict,
    top_designs: list, reward_trajectory: list, param_trajectory: list,
    top_codes: list, iteration: int, hybrid: bool,
    strategy: str = "novel",
) -> str:
    lines: list[str] = [f"Current iteration: {iteration}", ""]

    stag = _stagnation_iters(reward_trajectory)

    # ── Strategy task block ───────────────────────────────────────────────────
    if strategy == 'novel':
        lines += [
            "=== YOUR TASK: GENERATE NOVEL ALGORITHM ===",
            "Implement a completely new optimization approach.",
            "Do not replicate the code already in the database.",
            "Use the trajectory data, top designs, and bounds however you see fit.",
            "",
        ]
    else:  # modify
        lines += [
            "=== YOUR TASK: MODIFY EXISTING ALGORITHM ===",
            "Take the rank=1 code shown below and improve it meaningfully.",
            "Do NOT return the same code — make a substantive improvement.",
            "",
        ]

    # ── Optimization state ────────────────────────────────────────────────────
    if reward_trajectory:
        if stag >= 3:
            lines += [
                f"=== OPTIMIZATION STATE: STAGNATED ({stag} iterations flat) ===",
                "Perturbation-based methods will NOT improve from here.",
                "The param_trajectory gives paired (x_i, f_i) data.",
                "To estimate a gradient direction: stack delta_x (param differences)",
                "and delta_f (reward differences), then solve:",
                "  grad = np.linalg.lstsq(delta_x, delta_f, rcond=None)[0]",
                "Step along -grad (maximizing reward) with varied step sizes.",
                "Alternatively, try a large random restart far from the current best.",
                "",
            ]
        elif stag == 0 and len(reward_trajectory) >= 2:
            delta = reward_trajectory[-1] - reward_trajectory[-2]
            lines += [
                f"=== OPTIMIZATION STATE: IMPROVING (last delta={delta:+.5f}) ===",
                "",
            ]

    # ── Numerical context ─────────────────────────────────────────────────────
    lines.append("=== PARAMETER SPACE ===")
    for k, v in center.items():
        b = bounds.get(k, 'unknown')
        lines.append(f"  {k}: center={v}  bounds={b}")
    lines.append("")

    lines.append("=== REWARD TRAJECTORY (best-so-far per iteration, oldest -> newest) ===")
    lines.append(" ".join(f"{r:.4f}" for r in reward_trajectory)
                 if reward_trajectory else "(none yet)")
    lines.append("")

    if param_trajectory:
        lines.append("=== PARAMETER TRAJECTORY "
                     "(best design at each recent iteration, oldest -> newest) ===")
        traj_data = [{'iteration': idx, 'reward': e['reward'], 'params': e['params']}
                     for idx, e in enumerate(param_trajectory)]
        lines.append(json.dumps(traj_data, indent=2, default=_json_default))
        lines.append("")

    if top_designs:
        lines.append(f"=== TOP {len(top_designs)} DESIGNS (sorted best -> worst) ===")
        designs_data = [{'rank': idx + 1, 'reward': d['reward'], 'params': d['params']}
                        for idx, d in enumerate(top_designs)]
        lines.append(json.dumps(designs_data, indent=2, default=_json_default))
        lines.append("")

    # ── Code database — framing depends on strategy ───────────────────────────
    if top_codes:
        if strategy == 'novel':
            lines.append("=== ALGORITHMS ALREADY TRIED — DO NOT REPLICATE ===")
            for rank, ce in enumerate(top_codes, 1):
                hdr = (f"--- rank={rank}  best_reward={ce['best_reward']:.4f}"
                       f"  uses={ce['uses']}")
                if hybrid and 'beat_gaussian' in ce:
                    hdr += f"  beat_gaussian={ce['beat_gaussian']}/{ce['uses']}"
                hdr += " ---"
                lines.append(hdr)
                lines.append(ce['code'])
                lines.append("---")
        else:  # modify
            lines.append("=== BEST ALGORITHM — IMPROVE THIS ===")
            ce = top_codes[0]
            hdr = (f"--- rank=1  best_reward={ce['best_reward']:.4f}"
                   f"  uses={ce['uses']}")
            if hybrid and 'beat_gaussian' in ce:
                hdr += f"  beat_gaussian={ce['beat_gaussian']}/{ce['uses']}"
            hdr += " ---"
            lines.append(hdr)
            lines.append(ce['code'])
            lines.append("---")
            if len(top_codes) > 1:
                lines.append("")
                lines.append("=== OTHER TRIED ALGORITHMS (for context only) ===")
                for rank, ce in enumerate(top_codes[1:], 2):
                    hdr = (f"--- rank={rank}  best_reward={ce['best_reward']:.4f}"
                           f"  uses={ce['uses']}")
                    if hybrid and 'beat_gaussian' in ce:
                        hdr += f"  beat_gaussian={ce['beat_gaussian']}/{ce['uses']}"
                    hdr += " ---"
                    lines.append(hdr)
                    lines.append(ce['code'])
                    lines.append("---")
        lines.append("")

    lines.append(f"Generate generate_samples to produce {n} candidates.")
    return "\n".join(lines)


def _write_debug_attempt(debug_dir: str, iteration: int, attempt: int,
                          prompt: str, raw_response: str | None,
                          code: str | None, error: str | None, succeeded: bool):
    """Write all sampler artefacts for one LLM attempt to the debug directory.

    Files written (always, regardless of success/failure):
      sampler_iter{N}_a{A}_prompt.txt    — full prompt sent to the LLM
      sampler_iter{N}_a{A}_response.txt  — raw LLM response (includes any reasoning)
      sampler_iter{N}_a{A}.py            — extracted code block (if any)
      sampler_iter{N}_a{A}_error.txt     — traceback / error message (if failed)

    The code_db.json is always written by SamplerCodeDatabase._save() independently.
    """
    os.makedirs(debug_dir, exist_ok=True)
    stem = os.path.join(debug_dir, f'sampler_iter{iteration}_a{attempt}')
    suffix = '' if succeeded else '_FAILED'

    with open(f'{stem}{suffix}_prompt.txt', 'w') as f:
        f.write(f"=== SYSTEM ===\n{SYSTEM_PROMPT}\n\n=== USER ===\n{prompt}")

    if raw_response is not None:
        with open(f'{stem}{suffix}_response.txt', 'w') as f:
            f.write(raw_response)

    if code is not None:
        with open(f'{stem}{suffix}.py', 'w') as f:
            f.write(code)

    if error is not None:
        with open(f'{stem}{suffix}_error.txt', 'w') as f:
            f.write(error)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)
