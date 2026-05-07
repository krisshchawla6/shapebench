"""
Scan environment results dirs and extract best_result.json for every blank
method slot in best_rewards that has actual run data (not just -5.0 penalty).

Reports:
  - run count, total_evals, non_penalty_evals, best_reward
  - whether result was written or skipped (all-failed)

Output: writes best_result.json into each method folder.
"""

import json, csv
from pathlib import Path

BASE = Path('/scratch/ShapeEvolve')
BEST = BASE / 'best_rewards'
PENALTY = -5.0
PENALTY_TOL = 0.005

# ── helpers ────────────────────────────────────────────────────────────────────

def best_from_csv(csv_path, reward_cols=('best_reward', 'gbest_reward', 'reward')):
    """Return (n_rows, n_non_penalty, best_reward) from a CSV file."""
    try:
        rows = list(csv.DictReader(open(csv_path)))
    except Exception:
        return 0, 0, None
    if not rows:
        return 0, 0, None
    # pick the best available column
    rcol = next((c for c in reward_cols if c in rows[0]), None)
    if rcol is None:
        return 0, 0, None
    vals = []
    for r in rows:
        try:
            vals.append(float(r[rcol]))
        except (ValueError, TypeError):
            pass
    raw_vals = []
    raw_col = 'reward'
    if raw_col in rows[0]:
        for r in rows:
            try:
                raw_vals.append(float(r[raw_col]))
            except (ValueError, TypeError):
                pass
    n_non_penalty = sum(1 for v in raw_vals if v > PENALTY + PENALTY_TOL)
    best = max(vals) if vals else None
    return len(rows), n_non_penalty, best


def best_from_json(json_path, score_keys=('best_score', 'reward', 'best_reward')):
    """Return best_reward from a results.json / best_result.json."""
    try:
        d = json.loads(Path(json_path).read_text())
        for k in score_keys:
            if k in d and d[k] is not None:
                return float(d[k])
    except Exception:
        pass
    return None


def write_result(method_dir, best_reward, source_dir, extra=None):
    """Write best_result.json into method_dir (only if better than existing)."""
    out = Path(method_dir) / 'best_result.json'
    existing = None
    if out.exists():
        try:
            existing = json.loads(out.read_text()).get('reward')
        except Exception:
            pass
    if existing is not None and existing >= best_reward:
        return False  # already have better or equal
    payload = {'reward': round(best_reward, 6), 'source_dir': str(source_dir)}
    if extra:
        payload.update(extra)
    out.write_text(json.dumps(payload, indent=2))
    return True


# ── env-specific scanners ─────────────────────────────────────────────────────

def scan_flat_csv(results_root, task, method_prefix):
    """Scan flat run_<method_prefix>_<task>_* dirs for best CSV result."""
    rr = Path(results_root)
    task_l = task.lower()
    prefix_l = method_prefix.lower()
    best, total_rows, n_non_p, best_dir, run_count = None, 0, 0, None, 0
    for d in sorted(rr.iterdir()):
        if not d.is_dir():
            continue
        n = d.name.lower()
        if prefix_l not in n or task_l not in n:
            continue
        csv_p = d / 'results.csv'
        if not csv_p.exists():
            continue
        rows, non_p, b = best_from_csv(csv_p)
        run_count += 1
        total_rows += rows
        n_non_p += non_p
        if b is not None and (best is None or b > best):
            best = b
            best_dir = d
    return run_count, total_rows, n_non_p, best, best_dir


def scan_flat_json(results_root, task, method_prefix):
    """Scan flat run_<method_prefix>_<task>_* dirs for best results.json."""
    rr = Path(results_root)
    task_l = task.lower()
    prefix_l = method_prefix.lower()
    best, best_dir, run_count = None, None, 0
    for d in sorted(rr.iterdir()):
        if not d.is_dir():
            continue
        n = d.name.lower()
        if prefix_l not in n or task_l not in n:
            continue
        rj = d / 'results.json'
        if rj.exists():
            b = best_from_json(rj)
        else:
            csv_p = d / 'results.csv'
            if csv_p.exists():
                _, _, b = best_from_csv(csv_p)
            else:
                continue
        run_count += 1
        if b is not None and (best is None or b > best):
            best = b
            best_dir = d
    return run_count, 0, 0, best, best_dir


def scan_nested(results_root, method_subdir, task):
    """Scan results_root/method_subdir/<task>_s* dirs (SuperWing / CCA style)."""
    md = Path(results_root) / method_subdir
    if not md.is_dir():
        return 0, 0, 0, None, None
    task_l = task.lower()
    best, total_rows, n_non_p, best_dir, run_count = None, 0, 0, None, 0
    for d in sorted(md.iterdir()):
        if not d.is_dir():
            continue
        if task_l not in d.name.lower():
            continue
        run_count += 1
        # try results.json first (openevolve/shinka)
        rj = d / 'results.json'
        if rj.exists():
            b = best_from_json(rj)
            if b is not None and (best is None or b > best):
                best = b
                best_dir = d
            continue
        # try results.csv
        csv_p = d / 'results.csv'
        if csv_p.exists():
            rows, non_p, b = best_from_csv(csv_p)
            total_rows += rows
            n_non_p += non_p
            if b is not None and (best is None or b > best):
                best = b
                best_dir = d
    return run_count, total_rows, n_non_p, best, best_dir


# ── per-environment extraction configs ────────────────────────────────────────

# Each entry: (env_name, task_name, method_key, scanner_fn, *scanner_args)
# scanner_fn returns (run_count, total_rows, n_non_p, best_reward, best_dir)

def get_jobs():
    jobs = []

    # ── BlendedNet ─────────────────────────────────────────────────────────
    BN_RES = BASE / 'environments/BlendedNet/results'
    BN_TASKS = [
        'multipoint_mach_cd', 'multipoint_mach_range', 'range_optimization',
        'shapebench_5', 'shapebench_5_max_LD', 'shapebench_5_max_LD_total_drag',
        'shapebench_5_max_LD_warmstart_cornerA', 'shapebench_5_total_drag',
        'shapebench_5_total_drag_constrained', 'shapebench_case5',
        'shapebench_case6', 'static_margin_constrained',
    ]
    BN_METHODS = {
        'GA':                 ('run_ga_', 'csv'),
        'cmaes':              ('run_cmaes_', 'csv'),
        'lbfgsb':             ('run_lbfgsb_', 'csv'),
        'v3_dynamic_optimizer': ('run_v3_flash2_5_', 'csv'),
        'openevolve':         ('run_openevolve_adapter_', 'json'),
        'shinka':             ('run_shinka_adapter_', 'json'),
    }
    for task in BN_TASKS:
        for mkey, (prefix, fmt) in BN_METHODS.items():
            mdir = BEST / 'BlendedNet' / task / mkey
            # skip if best_result already has real data
            br = mdir / 'best_result.json'
            if br.exists():
                try:
                    v = json.loads(br.read_text()).get('reward', PENALTY)
                    if v > PENALTY + PENALTY_TOL:
                        continue
                except Exception:
                    pass
            fn = scan_flat_csv if fmt == 'csv' else scan_flat_json
            jobs.append((mdir, 'BlendedNet', task, mkey, fn, BN_RES, task, prefix))

    # ── Ceras ──────────────────────────────────────────────────────────────
    CE_RES = BASE / 'environments/CERAS/results'
    CE_METHODS = {
        'GA':                 ('run_gap_fuel_mass', 'csv'),
        'cmaes':              ('run_cmaes_fuel_mass', 'csv'),
        'lbfgsb':             ('run_lbfgsb_fuel_mass', 'csv'),
        'v3_dynamic_optimizer': ('run_v3_fuel_mass', 'csv'),
        'openevolve':         ('run_openevolve_fuel_mass', 'json'),
        'shinka':             ('run_shinka_fuel_mass', 'json'),
        'BO_torch':           ('run_bo_fuel_mass', 'csv'),
    }
    for mkey, (prefix, fmt) in CE_METHODS.items():
        mdir = BEST / 'Ceras/fuel_mass' / mkey
        br = mdir / 'best_result.json'
        if br.exists():
            try:
                v = json.loads(br.read_text()).get('reward', PENALTY)
                if v > PENALTY + PENALTY_TOL or v < -100:  # fuel mass is large negative
                    continue
            except Exception:
                pass
        fn = scan_flat_csv if fmt == 'csv' else scan_flat_json
        jobs.append((mdir, 'Ceras', 'fuel_mass', mkey, fn, CE_RES, 'fuel_mass', prefix))

    # ── DrivaerStar ────────────────────────────────────────────────────────
    DS_RES = BASE / 'environments/DrivAer_Star/results'
    DS_TASKS = [
        'cd_cl_constrained_vtk_E', 'cd_cl_constrained_vtk_F', 'cd_cl_constrained_vtk_N',
        'cd_only', 'cd_only_super_tight_bounds_vtk_E', 'cd_only_tight_bounds_vtk_E',
        'cd_only_tightened_bounds_vtk_E', 'cd_only_vtk_E', 'cd_only_vtk_F', 'cd_only_vtk_N',
        'downforce_efficiency_vtk_E', 'downforce_efficiency_vtk_F', 'downforce_efficiency_vtk_N',
    ]
    DS_METHODS = {
        'GA':       ('run_ga_', 'csv'),
        'cmaes':    ('run_cmaes_', 'csv'),
        'lbfgsb':   ('run_lbfgsb_', 'csv'),
        'openevolve': ('run_openevolve_adapter_', 'json'),
        'shinka':   ('run_shinka_adapter_', 'json'),
    }
    for task in DS_TASKS:
        for mkey, (prefix, fmt) in DS_METHODS.items():
            mdir = BEST / 'DrivaerStar' / task / mkey
            if not mdir.exists():
                continue
            br = mdir / 'best_result.json'
            if br.exists():
                continue
            fn = scan_flat_csv if fmt == 'csv' else scan_flat_json
            jobs.append((mdir, 'DrivaerStar', task, mkey, fn, DS_RES, task, prefix))

    # ── NeuralFoil ─────────────────────────────────────────────────────────
    NF_RES = BASE / 'environments/NeuralFoil/results'
    NF_TASKS = ['ld_ratio_constrained_m02_re1e7_normalized', 'reward_exact_notebook']
    NF_METHODS = {
        'GA':       ('run_ga_', 'csv'),
        'cmaes':    ('run_cmaes_', 'csv'),
        'lbfgsb':   ('run_lbfgsb_', 'csv'),
        'openevolve': ('run_openevolve_adapter_', 'json'),
        'shinka':   ('run_shinka_adapter_', 'json'),
    }
    for task in NF_TASKS:
        for mkey, (prefix, fmt) in NF_METHODS.items():
            mdir = BEST / 'NeuralFoil' / task / mkey
            if not mdir.exists():
                continue
            br = mdir / 'best_result.json'
            if br.exists():
                continue
            fn = scan_flat_csv if fmt == 'csv' else scan_flat_json
            jobs.append((mdir, 'NeuralFoil', task, mkey, fn, NF_RES, task, prefix))

    # ── SuperSonic_Transport_aircraft (Mixed_integer_yiren) ─────────────────
    SS_RES = BASE / 'environments/Mixed_integer_yiren/results'
    SS_TASKS_DIR = BEST / 'SuperSonic_Transport_aircraft'
    SS_TASKS = [t.name for t in SS_TASKS_DIR.iterdir() if t.is_dir()]
    SS_METHODS = {
        'GA':                 ('run_ga_', 'csv'),
        'BO_torch':           ('run_bo_torch_approx_', 'csv'),
        'lbfgsb':             ('run_lbfgsb_', 'csv'),
        'v3_dynamic_optimizer': ('run_v3_dynamic_optimizer_', 'csv'),
        'openevolve':         ('run_openevolve_adapter_', 'json'),
        'shinka':             ('run_shinka_adapter_', 'json'),
    }
    for task in SS_TASKS:
        for mkey, (prefix, fmt) in SS_METHODS.items():
            mdir = BEST / 'SuperSonic_Transport_aircraft' / task / mkey
            if not mdir.exists():
                continue
            br = mdir / 'best_result.json'
            if br.exists():
                continue
            fn = scan_flat_csv if fmt == 'csv' else scan_flat_json
            jobs.append((mdir, 'SuperSonic', task, mkey, fn, SS_RES, task, prefix))

    # ── Superwing ──────────────────────────────────────────────────────────
    SW_RES = BASE / 'environments/SuperWing/results'
    SW_TASKS_DIR = BEST / 'Superwing'
    SW_TASKS = [t.name for t in SW_TASKS_DIR.iterdir() if t.is_dir()]
    SW_METHODS = {
        'GA':                 ('GA_parallel', 'nested_csv'),
        'BO_torch':           ('BO_torch',    'nested_csv'),
        'lbfgsb':             ('lbfgsb',      'nested_csv'),
        'v3_dynamic_optimizer': ('v3_dynamic_optimizer', 'nested_csv'),
        'openevolve':         ('openevolve_adapter', 'nested_json'),
        'shinka':             ('shinka_adapter',     'nested_json'),
    }
    for task in SW_TASKS:
        for mkey, (subdir, fmt) in SW_METHODS.items():
            mdir = BEST / 'Superwing' / task / mkey
            if not mdir.exists():
                continue
            br = mdir / 'best_result.json'
            if br.exists():
                continue
            jobs.append((mdir, 'Superwing', task, mkey, scan_nested, SW_RES, subdir, task))

    # ── VortexNet ──────────────────────────────────────────────────────────
    VN_RES = BASE / 'environments/vortexnet/results'
    VN_TASKS = [t.name for t in (BEST/'VortexNet').iterdir() if t.is_dir()]
    VN_METHODS = {
        'openevolve': ('run_openevolve_adapter_vortexnet_', 'json'),
        'shinka':     ('run_shinka_adapter_vortexnet_',     'json'),
    }
    for task in VN_TASKS:
        for mkey, (prefix, fmt) in VN_METHODS.items():
            mdir = BEST / 'VortexNet' / task / mkey
            if not mdir.exists():
                continue
            br = mdir / 'best_result.json'
            if br.exists():
                continue
            fn = scan_flat_csv if fmt == 'csv' else scan_flat_json
            jobs.append((mdir, 'VortexNet', task, mkey, fn, VN_RES, task, prefix))

    return jobs


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    jobs = get_jobs()
    print(f'Total jobs to process: {len(jobs)}\n')

    written = 0
    all_failed = 0
    no_data = 0

    for entry in jobs:
        mdir = entry[0]
        env, task, mkey = entry[1], entry[2], entry[3]
        scanner = entry[4]
        args = entry[5:]

        mdir = Path(mdir)
        mdir.mkdir(parents=True, exist_ok=True)

        run_count, total_rows, n_non_p, best, best_dir = scanner(*args)

        label = f'{env}/{task}/{mkey}'

        if run_count == 0:
            print(f'NO_DATA  {label}')
            no_data += 1
            continue

        if best is None or best <= PENALTY + PENALTY_TOL:
            print(f'ALL_FAIL {label}  runs={run_count} evals={total_rows} non_penalty={n_non_p} best={best}')
            all_failed += 1
            continue

        ok = write_result(mdir, best, best_dir)
        status = 'WRITTEN ' if ok else 'SKIPPED '
        print(f'{status} {label}  runs={run_count} evals={total_rows} non_penalty={n_non_p} best={best:.4f}')
        if ok:
            written += 1

    print(f'\nDone. written={written}  all_failed={all_failed}  no_data={no_data}')
