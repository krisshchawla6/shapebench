"""Dynamic optimizer agent: LLM generates a Python mini-optimizer per iteration.

Two-level evolution:
  Level 1 (designs)  — existing island/LLM designer loop, unchanged
  Level 2 (code)     — SamplerCodeDatabase tracks generated optimizer code blocks
                       ranked by best offspring reward; LLM mutates/creates them

The LLM writes an `optimize(center, evaluate, budget, bounds, rng, history)`
function that calls `evaluate(params_dict) -> float` up to `budget` times.
This enables sequential, adaptive strategies: gradient descent, line search,
CMA-ES, finite-difference adjoints, etc.

No fallback. If all retries fail, those offspring slots are skipped.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import traceback as tb

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
_DB_MAX_SIZE          = 20
_TOP_CODES_K          = 3
_STAGNATION_THRESHOLD = 1e-5

SYSTEM_PROMPT = (
    "You implement one iteration of a numerical optimization algorithm.\n\n"
    "Write exactly:\n\n"
    "    def optimize(center, evaluate, budget, bounds, rng, history):\n\n"
    "Parameters:\n"
    "  center   — dict of parameter arrays/scalars. The designer LLM's proposed params for this\n"
    "             iteration (i.e. the s0 design). Use this as your starting point for optimization.\n"
    "  evaluate — callable(params_dict) -> float. Runs the real simulator (expensive).\n"
    "             Raises an exception when the budget is exhausted — let it propagate, do not catch it.\n"
    "  budget   — int. Total evaluate() calls available this iteration.\n"
    "  bounds   — dict {key: (lo, hi)}. Hard parameter limits.\n"
    "  rng      — numpy.random.Generator.\n"
    "  history  — {\n"
    "      'best_designs':      [{'params': {...}, 'reward': float}, ...],  # top-K, best-first\n"
    "      'reward_trajectory': [float, ...],                               # best-so-far per iter\n"
    "      'param_trajectory':  [{'params': {...}, 'reward': float}, ...]   # best design per iter\n"
    "    }\n"
    "    WARNING: all history lists may be empty early in the run. Always guard before indexing,\n"
    "    e.g. use `center` as fallback: x0 = params_to_vec(history['best_designs'][0]['params'] if history['best_designs'] else center)\n\n"
    "Pre-built helpers already in scope (do not redefine):\n"
    "  params_to_vec(p)   — dict → flat np.ndarray  (key order matches center)\n"
    "  vec_to_params(v)   — flat np.ndarray → dict  (auto-clipped to bounds)\n"
    "  n_dims             — int, total parameter dimensions\n"
    "  param_keys         — list[str], parameter names in vector order\n\n"
    "Contract:\n"
    "  - Return: None. All designs are captured via evaluate() calls.\n"
    "  - Each evaluate() call is a real simulation — spend the budget wisely.\n"
    "  - A full finite-difference gradient costs exactly n_dims evaluate() calls.\n\n"
    "FORBIDDEN — the following will make your code worthless:\n"
    "  - Hard-coded numerical target values for any parameter (e.g. x[3] = 0.42, weight = 0.4300).\n"
    "    ALL target/goal values must be computed from `history` or `center` — never typed as literals.\n"
    "  - The ONLY valid constraint enforcement is vec_to_params() which clips to bounds.\n"
    "  - The reward function already handles all physical constraints — do not re-implement them.\n\n"
    "Available: np (numpy), math. No other imports.\n"
    "Output only the function inside ```python ... ```. No other text."
)


# ── Budget sentinel ───────────────────────────────────────────────────────────

class _BudgetExhausted(BaseException):
    """Raised when the oracle budget is fully consumed.

    Inherits from BaseException (not Exception) so that `except Exception: pass`
    in LLM-generated code does NOT accidentally swallow it.
    """


# ── Database ─────────────────────────────────────────────────────────────────

class SamplerCodeDatabase:
    """Persistent JSON database of generated optimizer code blocks."""

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

    def record_result(self, code_id: str, best_reward: float,
                      beat_gaussian: bool | None, iteration: int):
        if code_id not in self._entries:
            return
        e = self._entries[code_id]
        e['best_reward'] = max(e['best_reward'], float(best_reward))
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
    """Alternate NOVEL / MODIFY each iteration."""
    if not top_codes:
        return 'novel'
    return 'novel' if iteration % 2 == 0 else 'modify'


# ── Agent ────────────────────────────────────────────────────────────────────

class SamplerAgent:
    """LLM-driven optimizer agent. Generates one Python optimizer per iteration."""

    def __init__(self, output_dir: str, model: str = 'gemini-3-flash-preview',
                 max_retries: int = 3, hybrid: bool = False):
        self.model_name  = model
        self.max_retries = max_retries
        self.hybrid      = hybrid
        self.code_db     = SamplerCodeDatabase(
            os.path.join(output_dir, 'sampler_code_db.json'))
        self._last_code_id: str | None = None
        self._configure_llm()

    def _configure_llm(self):
        if self.model_name.startswith('gemini'):
            try:
                import google.generativeai as genai
                key = (os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
                       or os.getenv('GEMINI_KEY'))
                if key:
                    genai.configure(api_key=key)
            except Exception as exc:
                print(f"  SamplerAgent: genai config failed: {exc}")
        elif self.model_name.startswith('claude'):
            try:
                import anthropic  # noqa: F401 — just verify it's installed
            except ImportError:
                print("  SamplerAgent: 'anthropic' package not installed. "
                      "Run: pip install anthropic")
        else:
            # OpenAI or OpenAI-compatible (Ollama, vLLM, OpenRouter, etc.)
            try:
                import openai  # noqa: F401
            except ImportError:
                print("  SamplerAgent: 'openai' package not installed. "
                      "Run: pip install openai")

    def _call_llm(self, user_prompt: str) -> str:
        if self.model_name.startswith('gemini'):
            import google.generativeai as genai
            model = genai.GenerativeModel(self.model_name,
                                          system_instruction=SYSTEM_PROMPT)
            cfg  = genai.types.GenerationConfig(temperature=0.9)
            resp = model.generate_content(user_prompt, generation_config=cfg)
            return resp.text.strip()

        elif self.model_name.startswith('claude'):
            import anthropic
            key    = os.getenv('ANTHROPIC_API_KEY')
            client = anthropic.Anthropic(api_key=key)
            msg    = client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=0.9,
                system=SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': user_prompt}],
            )
            return msg.content[0].text.strip()

        else:
            # OpenAI or OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter)
            import openai
            base_url = os.getenv('OPENAI_BASE_URL')  # set for Ollama/vLLM/OpenRouter
            api_key  = os.getenv('OPENAI_API_KEY', 'ollama')
            client   = openai.OpenAI(api_key=api_key,
                                     **({"base_url": base_url} if base_url else {}))
            resp = client.chat.completions.create(
                model=self.model_name,
                temperature=0.9,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user',   'content': user_prompt},
                ],
            )
            return resp.choices[0].message.content.strip()

    def run_optimizer(
        self,
        center_params: dict,
        evaluate_fn,
        budget: int,
        bounds: dict,
        top_design_contexts: list,
        reward_trajectory: list,
        param_trajectory: list,
        iteration: int,
        debug_dir: str | None = None,
    ) -> bool:
        """Generate and execute an LLM optimizer that calls evaluate_fn up to budget times.

        Returns True if code executed successfully (at least once), False if all retries failed.
        """
        if budget <= 0:
            return False

        n_dims = sum(np.asarray(v).size for v in center_params.values()
                     if not isinstance(v, str))

        top_codes   = self.code_db.top(_TOP_CODES_K)
        strategy    = _choose_strategy(top_codes, iteration)
        user_prompt = _build_user_prompt(
            center_params, budget, n_dims, bounds, top_design_contexts,
            reward_trajectory, param_trajectory, top_codes, iteration,
            self.hybrid, strategy=strategy)

        prev_code:  str | None = None
        prev_error: str | None = None

        for attempt in range(self.max_retries):
            up = user_prompt
            if prev_code is not None and prev_error is not None:
                up += (
                    f"\n\n=== PREVIOUS ATTEMPT (FAILED) ===\n{prev_code}"
                    f"\n\n=== ERROR ===\n{prev_error}"
                    f"\n\nFix the function. Same contract applies."
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
                _exec_sandbox(code, center_params, evaluate_fn, budget, bounds, rng, history)
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

            print(f"  SamplerAgent: optimizer ran (iter={iteration} attempt={attempt+1})")
            return True

        print(f"  SamplerAgent: all {self.max_retries} retries failed "
              f"for iteration {iteration} — offspring slots skipped")
        self._last_code_id = None
        return False

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
    if 'def optimize' in text:
        return text.strip()
    return None


def _build_sandbox_helpers(center: dict, bounds: dict):
    """Build pre-defined helper functions injected into the sandbox namespace."""
    keys = [k for k, v in center.items() if not isinstance(v, str)]

    slices: dict[str, tuple[int, int]] = {}
    flat_keys: list[str] = []
    idx = 0
    for k in keys:
        n = np.asarray(center[k]).size
        slices[k] = (idx, idx + n)
        idx += n
        if n == 1:
            flat_keys.append(k)
        else:
            flat_keys.extend(f'{k}[{j}]' for j in range(n))
    n_dims = idx

    def params_to_vec(p: dict) -> np.ndarray:
        return np.concatenate([np.asarray(p[k]).flatten() for k in keys])

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
        'n_dims':        n_dims,
        'param_keys':    flat_keys,
    }


def _exec_sandbox(code: str, center: dict, evaluate_fn, budget: int,
                  bounds: dict, rng: np.random.Generator, history: dict):
    """Execute LLM optimizer code in a restricted namespace.

    The oracle passed to the LLM is wrapped with a budget counter.
    When budget is exhausted _BudgetExhausted (a BaseException subclass) is raised,
    propagating out of the LLM's loop without being caught by `except Exception`.
    """
    helpers = _build_sandbox_helpers(center, bounds)

    calls = [0]

    def _budgeted_evaluate(params_dict: dict) -> float:
        if calls[0] >= budget:
            raise _BudgetExhausted(f"Budget of {budget} evaluate() calls exhausted.")
        calls[0] += 1
        return evaluate_fn(params_dict)

    # Ensure history lists are never empty so LLM code can safely index into them.
    # best_designs and param_trajectory fall back to a center-based entry; reward_trajectory
    # falls back to [0.0] so trajectory arithmetic always has at least one element.
    safe_history = dict(history)
    if not safe_history.get('best_designs'):
        safe_history['best_designs'] = [{'params': center, 'reward': float('-inf')}]
    if not safe_history.get('param_trajectory'):
        safe_history['param_trajectory'] = [{'params': center, 'reward': float('-inf')}]
    if not safe_history.get('reward_trajectory'):
        safe_history['reward_trajectory'] = [0.0]

    namespace: dict = {'np': np, 'math': math, 'evaluate': _budgeted_evaluate, **helpers}

    exec(code, namespace)  # noqa: S102
    fn = namespace.get('optimize')
    if fn is None:
        raise ValueError("Code did not define optimize()")

    try:
        fn(center, _budgeted_evaluate, budget, bounds, rng, safe_history)
    except _BudgetExhausted:
        pass  # normal termination — budget consumed


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
    center: dict, budget: int, n_dims: int, bounds: dict,
    top_designs: list, reward_trajectory: list, param_trajectory: list,
    top_codes: list, iteration: int, hybrid: bool,
    strategy: str = "novel",
) -> str:
    lines: list[str] = [f"Current iteration: {iteration}", ""]

    stag = _stagnation_iters(reward_trajectory)

    # ── Strategy task block ───────────────────────────────────────────────────
    if strategy == 'novel':
        lines += [
            "=== YOUR TASK: GENERATE NOVEL OPTIMIZER ===",
            "Implement a completely new optimization approach.",
            "Do not replicate the code already in the database.",
            "",
        ]
    else:
        lines += [
            "=== YOUR TASK: IMPROVE EXISTING OPTIMIZER ===",
            "Take the rank=1 code shown below and improve it meaningfully.",
            "Do NOT return the same code — make a substantive improvement.",
            "",
        ]

    # ── Optimization state ────────────────────────────────────────────────────
    if reward_trajectory:
        if stag >= 3:
            lines += [
                f"=== OPTIMIZATION STATE: STAGNATED ({stag} iterations flat) ===",
                "Random perturbation will NOT escape this plateau.",
                "Recommended: use evaluate() to compute an exact finite-difference gradient.",
                "Example (costs n_dims calls, leaves budget-n_dims for line search):",
                "  best = history['best_designs'][0]['params'] if history['best_designs'] else center",
                "  x0 = params_to_vec(best)",
                "  f0 = evaluate(vec_to_params(x0))",
                "  grad = np.zeros(n_dims)",
                "  for i in range(n_dims):",
                "      xp = x0.copy(); xp[i] += 1e-4",
                "      grad[i] = (evaluate(vec_to_params(xp)) - f0) / 1e-4",
                "  # Then line-search along +grad (maximising reward):",
                "  for step in np.logspace(-1, -4, budget - n_dims - 1):",
                "      evaluate(vec_to_params(x0 + step * grad))",
                "",
            ]
        elif stag == 0 and len(reward_trajectory) >= 2:
            delta = reward_trajectory[-1] - reward_trajectory[-2]
            lines += [
                f"=== OPTIMIZATION STATE: IMPROVING (last delta={delta:+.5f}) ===",
                "",
            ]

    # ── Budget / dimensionality context ──────────────────────────────────────
    lines += [
        f"=== BUDGET ===",
        f"  budget={budget}  n_dims={n_dims}",
        f"  Full FD gradient costs {n_dims} calls → {budget - n_dims} remaining for search.",
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

    # ── Code database ─────────────────────────────────────────────────────────
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
        else:
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

    lines.append(
        f"Implement optimize with budget={budget} evaluate() calls.  "
        f"n_dims={n_dims} → a full FD gradient costs {n_dims} calls."
    )
    return "\n".join(lines)


def _write_debug_attempt(debug_dir: str, iteration: int, attempt: int,
                          prompt: str, raw_response: str | None,
                          code: str | None, error: str | None, succeeded: bool):
    os.makedirs(debug_dir, exist_ok=True)
    stem   = os.path.join(debug_dir, f'sampler_iter{iteration}_a{attempt}')
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
