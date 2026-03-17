#!/usr/bin/env python3
"""
plot_tinker_analysis.py — Analyse Tinker LLM training dynamics for TTT runs.

Reads tinker_progress.jsonl, train.log, and checkpoints.jsonl to assess:
  - Is the LLM improving over gradient updates?
  - How does mean/failure rate change epoch-by-epoch?
  - How does token count (reasoning depth) evolve?

Output: <results_dir>/tinker_analysis.png

Usage:
    python analysis/plot_tinker_analysis.py <results_dir> [--label "Run name"]
    python analysis/plot_tinker_analysis.py environments/BlendedNet/results/ttt_1000
    python analysis/plot_tinker_analysis.py environments/vlm_3d_2pt/results/ttt_1000
"""

import os, sys, json, argparse, re
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

STYLE = {
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset':  'cm',
    'font.size':         10,
    'axes.labelsize':    11,
    'axes.titlesize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   8.5,
    'axes.linewidth':    0.6,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'figure.dpi':        200,
}

FAIL_THRESH = -9.0   # rewards <= this are counted as failures


def load_tinker_progress(path):
    """
    Load tinker_progress.jsonl.  Multiple runs may be stacked (restarts).
    Returns only the LAST complete run (iter 0..N-1 once).
    """
    all_rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                all_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Detect run boundaries: when iteration goes back to 0 after > 0
    run_starts = [0]
    for i in range(1, len(all_rows)):
        if all_rows[i]['iteration'] == 0 and all_rows[i - 1]['iteration'] > 0:
            run_starts.append(i)

    # Use last run's data
    last_start = run_starts[-1]
    rows = all_rows[last_start:]

    # Deduplicate: last row per iteration
    by_iter = {}
    for r in rows:
        by_iter[r['iteration']] = r
    rows = [by_iter[i] for i in sorted(by_iter)]

    return rows, len(run_starts)


def epoch_stats(rows):
    """Aggregate per-epoch: mean reward, failure rate, mean tokens, best reward."""
    by_epoch = {}
    for r in rows:
        ep = r.get('epoch', 0)
        by_epoch.setdefault(ep, []).append(r)

    epochs, mean_r, fail_r, mean_tok, best_r = [], [], [], [], []
    for ep in sorted(by_epoch):
        rs = by_epoch[ep]
        rewards  = np.array([r['reward'] for r in rs])
        tokens   = np.array([r.get('n_tokens', 0) for r in rs])
        valid    = rewards[rewards > FAIL_THRESH]
        epochs.append(ep)
        mean_r.append(float(np.mean(valid)) if len(valid) > 0 else np.nan)
        fail_r.append(float(np.mean(rewards <= FAIL_THRESH)) * 100)
        mean_tok.append(float(np.mean(tokens)))
        best_r.append(float(np.max(rewards)))

    return (np.array(epochs), np.array(mean_r), np.array(fail_r),
            np.array(mean_tok), np.array(best_r))


def load_train_log(path):
    """
    Parse train.log lines: epoch=N data_items=M groups=K best_reward=R
    Returns per-line tuples sorted by file order (= chronological within last run).
    Detects last run by looking for epoch counter resetting.
    """
    entries = []
    with open(path) as f:
        for line in f:
            m = re.search(
                r'epoch=(\d+)\s+data_items=(\d+)\s+groups=(\d+)\s+best_reward=([\d.]+)',
                line)
            if m:
                entries.append({
                    'epoch':      int(m.group(1)),
                    'data_items': int(m.group(2)),
                    'groups':     int(m.group(3)),
                    'best_reward': float(m.group(4)),
                })

    # Detect last run (epoch counter resets to 0)
    run_starts = [0]
    for i in range(1, len(entries)):
        if entries[i]['epoch'] < entries[i - 1]['epoch']:
            run_starts.append(i)
    last = run_starts[-1]
    entries = entries[last:]

    # One entry per epoch (keep last seen for that epoch)
    by_epoch = {}
    for e in entries:
        by_epoch[e['epoch']] = e
    entries = [by_epoch[ep] for ep in sorted(by_epoch)]
    return entries


def load_checkpoints(path):
    batches = []
    with open(path) as f:
        for line in f:
            try:
                b = json.loads(line.strip())
                batches.append(b.get('batch', 0))
            except:
                pass
    return sorted(set(batches))


def rolling(arr, w=5):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode='valid')


def plot_analysis(results_dir, label='TTT Run'):
    prog_path  = os.path.join(results_dir, 'tinker_progress.jsonl')
    train_path = os.path.join(results_dir, 'train.log')
    ckpt_path  = os.path.join(results_dir, 'checkpoints.jsonl')

    if not os.path.exists(prog_path):
        print(f"tinker_progress.jsonl not found in {results_dir}")
        return

    rows, n_runs = load_tinker_progress(prog_path)
    print(f"  Loaded {len(rows)} rollouts from last run ({n_runs} total restarts)")

    epochs, mean_r, fail_r, mean_tok, best_r = epoch_stats(rows)

    train_entries = load_train_log(train_path) if os.path.exists(train_path) else []
    checkpoints   = load_checkpoints(ckpt_path) if os.path.exists(ckpt_path) else []

    n_updates = len(checkpoints)
    print(f"  {len(epochs)} epochs, {n_updates} checkpoint batches in last run")

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'{label} — Tinker LLM Training Dynamics', fontsize=13,
                 fontweight='bold', y=1.01)

    # ── Panel 1: Mean reward per epoch (excl. failures) ─────────────────────
    ax = axes[0, 0]
    valid_mask = ~np.isnan(mean_r)
    ax.plot(epochs[valid_mask], mean_r[valid_mask],
            color='#2471a3', lw=1.5, alpha=0.5, zorder=2, label='Per-epoch mean')
    if valid_mask.sum() >= 5:
        rm = rolling(mean_r[valid_mask], w=5)
        ax.plot(epochs[valid_mask][4:], rm,
                color='#c0392b', lw=2.2, zorder=3, label='Rolling mean (w=5)')
    ax.set_xlabel('Epoch (gradient update)')
    ax.set_ylabel('Mean reward (failures excluded)')
    ax.set_title('Mean Reward per Epoch')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    # ── Panel 2: Failure rate per epoch ──────────────────────────────────────
    ax = axes[0, 1]
    ax.bar(epochs, fail_r, color='#e74c3c', alpha=0.55, width=0.8, zorder=2)
    if len(fail_r) >= 5:
        rm_f = rolling(fail_r, w=5)
        ax.plot(epochs[4:], rm_f, color='#922b21', lw=2.0, zorder=3,
                label='Rolling mean (w=5)')
        ax.legend(fontsize=8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Failure rate (%)')
    ax.set_title('Rollout Failure Rate per Epoch')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.15, axis='y')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    # ── Panel 3: Best reward at each gradient step (from train.log) ──────────
    ax = axes[1, 0]
    if train_entries:
        ep_tr  = [e['epoch']      for e in train_entries]
        br_tr  = [e['best_reward'] for e in train_entries]
        ax.plot(ep_tr, br_tr, color='#27ae60', lw=2.0, marker='o',
                markersize=3, zorder=3, label='Best reward in batch')
        ax.fill_between(ep_tr, br_tr, alpha=0.08, color='#27ae60')
        # Annotate final best
        ax.annotate(f'{br_tr[-1]:.2f}', xy=(ep_tr[-1], br_tr[-1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='#27ae60')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Best reward in training batch')
    ax.set_title('Best Reward at Each Gradient Update Step')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    # ── Panel 4: Mean token count per epoch ──────────────────────────────────
    ax = axes[1, 1]
    ax.plot(epochs, mean_tok, color='#8e44ad', lw=1.5, alpha=0.5, zorder=2,
            label='Mean tokens')
    if len(mean_tok) >= 5:
        rm_t = rolling(mean_tok, w=5)
        ax.plot(epochs[4:], rm_t, color='#6c3483', lw=2.2, zorder=3,
                label='Rolling mean (w=5)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean tokens per rollout')
    ax.set_title('Token Count per Rollout (Reasoning Depth)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    # Summary text
    valid_rewards = np.array([r['reward'] for r in rows])
    valid_only = valid_rewards[valid_rewards > FAIL_THRESH]
    summary_lines = [
        f"Rollouts: {len(rows)}   |   Epochs: {len(epochs)}   |   Restarts: {n_runs - 1}",
        f"Best reward: {float(valid_rewards.max()):.3f}   |   Mean (excl. failures): {float(np.mean(valid_only)):.3f}",
        f"Failure rate overall: {100 * np.mean(valid_rewards <= FAIL_THRESH):.1f}%   |   Checkpoints saved: {n_updates}",
        f"Reward trend: {'▲ improving' if len(mean_r[valid_mask]) > 1 and mean_r[valid_mask][-1] > mean_r[valid_mask][0] else '▼ declining / flat'}",
    ]
    fig.text(0.01, -0.03, '\n'.join(summary_lines), fontsize=8.5,
             fontfamily='monospace', color='#333',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', alpha=0.8))

    fig.tight_layout()
    out = os.path.join(results_dir, 'tinker_analysis.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', help='Path to TTT results directory')
    parser.add_argument('--label', type=str, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(repo, results_dir)

    label = args.label or os.path.basename(os.path.dirname(results_dir)) + ' ' + os.path.basename(results_dir)
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    plot_analysis(results_dir, label=label)
