# ShapeEvolve Architecture

## Overview

ShapeEvolve is a benchmark harness for aerodynamic shape optimization. It
separates three concerns cleanly:

1. **Environment** — wraps a physics simulator and reward function behind a
   standard interface (`BaseEnvironment`).
2. **Framework** — an optimization strategy (LLM-based, classical, or hybrid)
   that calls the environment through that interface.
3. **Benchmark runner** — loads an environment + reward + framework and drives
   the optimization loop, writing standard output files.

---

## System Flow

```
run_benchmark.py  ──────────────────────────────────────┐
  --framework openevolve_adapter                         │
  --environment SuperWing                                │  loads via importlib
  --reward ld_ratio                                      │
  --output-dir /path/to/out                              │
                                                         ▼
                                            frameworks/<name>/run.py
                                              add_args(parser)
                                              run(env, args, output_dir)
                                                         │
                          ┌──────────────────────────────┤
                          │                              │
                    LLM frameworks                Classical frameworks
              (islands, v2, v3,             (GA/PSO, BO, CMA-ES, L-BFGS-B)
           openevolve_adapter)                           │
                          │                              │
                          │ env.run_llm_action()         │ env.write_design()
                          │ env.simulate()               │ env.simulate()
                          │ env.build_context_entry()    │ env.get_param_bounds()
                          │                              │
                          └──────────────────────────────┘
                                                         │
                                              environments/<name>/
                                                environment.py
                                                         │
                                         ┌───────────────┴───────────┐
                                    solver / surrogate          rewards/<name>.py
                                    (neural net, CFD, etc.)     evaluate(run_sim,
                                                                 design_path,
                                                                 case_dir)
                                                                 → (reward, results)
```

**Output files** (written by all frameworks to `output_dir/`):

| File | Contents |
|------|----------|
| `results.csv` | `iteration, design, reward, best_reward[, extra_cols]` |
| `results.json` | Full database snapshot with metrics, images, feedback per design |
| `design_<i>/` | Per-iteration case directories with design files and sim outputs |
| `lineage.json` / `lineage_tree.png` | Optional design genealogy (LLM frameworks) |

---

## BaseEnvironment Contract

All frameworks interact with the environment exclusively through
`environments/base.py:BaseEnvironment`.

| Method | Inputs | Outputs | Used by |
|--------|--------|---------|---------|
| `simulate(design_path, case_dir)` | JSON design path, output dir | `(reward: float, results: dict)` | All frameworks |
| `write_design(x, output_dir, name)` | numpy array, paths | `design_path: str` | Classical frameworks |
| `get_param_bounds()` | — | `(lb: ndarray, ub: ndarray)` | Classical + OpenEvolve |
| `run_llm_action(action, context, output_dir, name, ...)` | action str, context list, paths | `design_path: str \| None` | LLM frameworks (not OpenEvolve adapter) |
| `build_context_entry(db_entry)` | DB row | context dict | LLM frameworks (not OpenEvolve adapter) |
| `get_prompt_blocks()` | — | dict of prompt strings/callables | LLM frameworks (not OpenEvolve adapter) |
| `sample_gaussian(mean_params, output_dir, name, std_scale)` | param dict, paths, float | `design_path: str` | `v2_batch` |
| `get_reflection_inputs(design_path, case_dir)` | paths | dict or `None` | `v2`, `v2_batch` |
| `get_named_param_bounds()` | — | dict of `name → (lo, hi)` | `v2_dynamic_sampling`, `v3_dynamic_optimizer` |

`results` dict from `simulate` must contain at minimum:
```python
{
    "metrics": {"cl": 0.51, "cd": 0.028, ...},  # scalar values
    "images":  ["/path/to/pressure.png", ...],   # optional
    "feedback": "..."                             # optional qualitative text
}
```

---

## Design Format

All environments represent a design as a **JSON file on disk**. The keys and
value ranges are environment-specific:

- **SuperWing**: 38 named parameters (Kulfan CST upper/lower surface
  coefficients, twist, sweep, dihedral, chord distribution) + `mach` + `aoa`.
  Defined in `environments/SuperWing/design_actions.py:PARAM_NAMES` and
  `BOUNDS`.
- Other environments follow the same pattern — JSON dict of named floats.

---

## Framework Reference

### LLM Frameworks (use `env.run_llm_action`)

These frameworks do **not** call an LLM directly. All LLM logic is delegated
to the environment's `run_llm_action()` / `agent.py`.

#### `islands` — base island-model LLM evolution

```
CLI args: --action gaussain --iterations N --inspirations K
          --initialize_n_sample I --pw_alpha 3.0
          --num_islands M --migration_interval 10 --migration_rate 0.1
          --baseline <path> --debug
```

**Loop:**
1. Sample parent + K inspirations from database (power-law rank bias).
2. `env.run_llm_action(action, context, output_dir, name)` → design JSON path.
3. `env.simulate(design_path, case_dir)` → `(reward, results)`.
4. Update database; write `results.csv`.

**LLM I/O (inside `agent.py` / `environment.py`):**
- Input: history of prior designs formatted by `prompt_blocks.py:format_context()`
- Prompt structure (SuperWing): `<ANALYSIS>` / `<DESIGN_RATIONALE>` / `<DESIGN>` XML tags
- Output: JSON design file

**No secondary LLM. No scratchpad. No reflection.**

---

#### `v2` — islands + Gemini reflection + scratchpad

Everything from `islands`, plus:

- `--reflection_interval`, `--scratchpad_len`
- After each successful design, `ReflectionAgent` (`frameworks/v2/reflection.py`)
  makes two additional Gemini calls → updates `scratchpad.txt`.
- Scratchpad string is passed through: `env.run_llm_action(..., scratchpad=scratchpad)`.
- Env-specific reflection prompt templates in `frameworks/v2/prompts/<env_name>/reflection.py`.

---

#### `v2_batch` — v2 + N Gaussian children per LLM center

- LLM proposes a center design; `llm_params.json` written alongside it.
- N additional designs sampled via `env.sample_gaussian(llm_mean, ..., std_scale)`.
- Reflection runs on the center only.
- Std scale follows a scheduler: `fixed` | `geometric` | `adaptive`.

---

#### `v2_dynamic_sampling` — v2 + LLM-generated Python sampler

- A second LLM call (`SamplerAgent`, `frameworks/v2_dynamic_sampling/sampler_agent.py`)
  writes a Python function:
  ```python
  def generate_samples(center, n, bounds, rng, history): ...
  ```
- The function is executed in a sandbox; offspring generated from its output.
- Sampler code and history persisted in `sampler_code_db.json`.

---

#### `v3_dynamic_optimizer` — v2 + LLM-generated optimizer with oracle access

- Second LLM writes a Python function:
  ```python
  def optimize(center, evaluate, budget, bounds, rng, history): ...
  ```
  where `evaluate(params_dict) → float` is a live call to the real surrogate.
- The LLM-written optimizer is given a real-simulation budget per iteration.
- Optional `--hybrid` flag adds Gaussian batch on top.

---

### Classical Frameworks (no LLM)

| Framework | Algorithm | Key args |
|-----------|-----------|----------|
| `GA` / `GA_parallel` | Particle Swarm Optimization | `--n_particles` |
| `BO` | Gaussian Process + Expected Improvement (skopt) | `--n_initial`, `--n_calls` |
| `BO_torch` / `BO_torch_approx` | BoTorch GP + acquisition | `--n_initial`, `--n_calls` |
| `cmaes` | CMA-ES (with optional resume) | `--sigma0`, `--popsize` |
| `lbfgsb` | L-BFGS-B gradient-free closure | `--maxiter` |

All use `env.write_design(x, ...)` + `env.simulate(design_path, case_dir)` only.

---

### `openevolve_adapter` — OpenEvolve as a competing baseline

**Fundamentally different approach:** OpenEvolve's LLM sees fitness scores and
MAP-Elites feature coordinates, then proposes a new **parameter JSON** as a
full rewrite. No ShapeEvolve LLM stack (`agent.py`, `prompt_blocks.py`,
`gaussian_superwing.py`) is used at all.

```
CLI args: --iterations N --population_size 50 --num_islands 3
          --parallel_evaluations 2 --llm_model gemini-2.0-flash
          --feature_dimensions <dim1> [dim2 ...]
          --checkpoint_interval 10
```

**What `run.py` does:**
1. `env.get_param_bounds()` → generate midpoint initial design via `env.write_design()`.
2. Serialize env class + reward class + kwargs into `run_config.json`.
3. Write `openevolve_config.yaml` (`diff_based_evolution: false`, `file_suffix: .json`,
   `language: json`, custom `template_dir`).
4. Set `OE_RUN_CONFIG` + `OE_OUTPUT_DIR` env vars (inherited by worker processes).
5. Call `openevolve.run_evolution(initial_design, evaluator.py, config, ...)`.
6. Translate OpenEvolve checkpoints → `results.csv` / `results.json`.

**What `evaluator.py` does (runs in OpenEvolve's worker processes):**
1. Reads `run_config.json` and instantiates env + reward (cached per worker).
2. `env.simulate(program_path, case_dir)` — `program_path` IS the design JSON.
3. Returns `{"combined_score": reward, "cl": ..., "cd": ..., ...}`.

**Prompt strategy:**
- `system_message.txt`: "optimize parameters to maximize FITNESS SCORE" + one
  paragraph of domain context. No analysis/rationale/strategy instructions.
- `full_rewrite_user.txt`: OpenEvolve's standard structure (fitness,
  MAP-Elites feature coords, evolution history, current params), with
  "program/code" replaced by "design/parameter set".

**What OpenEvolve adds over `islands`:**
- MAP-Elites: designs placed in a grid by feature dimensions (e.g. `cl`, `cd`),
  encouraging the LLM to explore diverse aerodynamic regimes.
- OpenEvolve's own island model + migration (independent of ShapeEvolve's).
- No parent/inspiration history formatted by ShapeEvolve — OpenEvolve manages
  its own evolution history display.

**What is absent vs. `islands` / `v2` / `v3`:**
- No `<ANALYSIS>` / `<DESIGN_RATIONALE>` / `<DESIGN>` XML tags.
- No `gaussian_superwing.py` strategy sampling.
- No scratchpad, no reflection LLM call.
- No image injection into prompts.

---

## SuperWing Environment Detail

**Simulator:** `environments/SuperWing/solver.py` — loads `ATsurf_M`
(`WingPDETransformer`, Hugging Face `yunplus/AeroTransformer`), a neural
surrogate for transonic wing aerodynamics. Returns `CL`, `CD`, `CM`, `LD`.

**Reward modules** (`environments/SuperWing/rewards/`):

| Module | Class | Objective |
|--------|-------|-----------|
| `ld_ratio` | `LDRatioReward` | Maximize CL/CD at fixed Mach + AoA |
| `range_optimization` | `RangeOptimizationReward` | Breguet range objective; AoA from bisection |
| `weighted_cl_range_optimization` | `WeightedClRangeOptimizationReward` | Weighted multi-CL range |
| `min_cd_cl055` | `MinCdCl055Reward` | Min CD + CL penalty |
| `min_cd_alternative_variation` | `MinCdAlternativeVariationReward` | Min CD with CL floor |
| `multipoint_avg_cd` | `MultipointAvgCdReward` | Mean CD over multiple CL targets |
| `multipoint_mach_range_optimization` | `MultipointMachRangeOptimizationReward` | Range averaged over Mach list |
| `min_cd_avf_altenrtaive` | `MinCdAvgAlternativeReward` | Multipoint min CD per Mach |

**LLM integration** (`environments/SuperWing/agent.py`):
- Backends: Gemini (default `gemini-3-flash-preview`), Claude, OpenAI-compatible.
- System prompt: `prompts/gaussian_superwing.py:get_generate_system()`
- User prompt: design history via `prompt_blocks.py:format_context()`
- Response: XML tags `<ANALYSIS>`, `<DESIGN_RATIONALE>`, `<DESIGN>` (JSON params)
- Optional strategy sampling: `gaussian_superwing.py:sample_strategy()`

---

## Benchmark Cases

`environments/SuperWing/rewards/permutation_samples.json` — 30 benchmark cases,
each with:
```json
{
  "case_id": "sw_001",
  "problem_class": "l_d_optimization",
  "reward": "ld_ratio",
  "params": {"mach": 0.86, "aoa": 4.85},
  "note": "..."
}
```

Run all cases for a framework:
```bash
python run_benchmark_samples.py --framework openevolve_adapter --iterations 100
```

Run a single case:
```bash
python run_benchmark.py \
  --framework openevolve_adapter \
  --environment SuperWing \
  --reward ld_ratio \
  --mach 0.86 --aoa 4.85 \
  --iterations 100 \
  --output-dir results/oe_sw001
```

---

## `Islands_visual_stratedgy_4points_alternative_test`

This is a **self-contained research workspace**, not a standard framework
plugin. It is not registered with the benchmark harness.

- Has its own `environment.py`, `LLM_Actions/LLM_agent.py`, `shapes_utils.py`.
- LLM call in `LLM_agent.py` directly uses Gemini (`gemini-3-flash-preview`).
- LLM output: CSV geometry actions (not JSON param dicts).
- Not wired to `BaseEnvironment.run_llm_action`.
- Multiple `work_*` subdirectories are iterative snapshots of the same experiment.
