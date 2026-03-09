#!/usr/bin/env python3
"""
animate_llm_agent_3d.py  -  Cinematic animation of the LLM-driven
delta-wing evolutionary optimiser.

Single cohesive dark dashboard per iteration with two build-up phases:
  Phase A  "Reasoning"  -- strategy badge + reasoning quote + chart
  Phase B  "Result"     -- Cp image + parameters + reward appear

Usage:
    python animate_llm_agent_3d.py <results_dir> [every_n]
"""

import os, sys, re, json, textwrap
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# -- Colour palette (dark dashboard) ----------------------------------
BG      = '#0d1117'
CARD    = '#161b22'
BORDER  = '#30363d'
TEXT    = '#e6edf3'
DIM     = '#7d8590'
ACCENT  = '#58a6ff'
GREEN   = '#3fb950'
RED     = '#f85149'
ORANGE  = '#d29922'

STRAT_COL = {
    'exploit':     '#58a6ff',
    'diversify':   '#3fb950',
    'hybrid':      '#bc8cff',
    'novel':       '#d29922',
    'no_strategy': '#7d8590',
}
STRAT_LABEL = {
    'exploit':     'EXPLOITATION',
    'diversify':   'EXPLORATION',
    'hybrid':      'HYBRID',
    'novel':       'NOVEL DESIGN',
    'no_strategy': 'BASELINE',
}

STYLE = {
    'font.family':      'sans-serif',
    'font.sans-serif':  ['DejaVu Sans', 'Helvetica', 'Arial'],
    'mathtext.fontset':  'dejavusans',
    'font.size':         9,
    'text.color':        TEXT,
    'axes.facecolor':    CARD,
    'axes.edgecolor':    BORDER,
    'axes.labelcolor':   DIM,
    'xtick.color':       DIM,
    'ytick.color':       DIM,
    'figure.facecolor':  BG,
    'figure.dpi':        150,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
}


# -- Data helpers ------------------------------------------------------
def load_results_csv(results_dir):
    csv_path = os.path.join(results_dir, 'results.csv')
    data = {
        'iteration': [], 'design': [], 'reward': [], 'best_reward': [],
        'CL': [], 'CDi': [], 'L_D': [],
    }
    with open(csv_path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            data['iteration'].append(int(parts[0]))
            data['design'].append(parts[1])
            data['reward'].append(float(parts[2]))
            data['best_reward'].append(float(parts[3]))
            data['CL'].append(float(parts[4]))
            data['CDi'].append(float(parts[5]))
            data['L_D'].append(float(parts[7]))
    for k in data:
        if k != 'design':
            data[k] = np.array(data[k])
    return data


def _read(path, default=''):
    return open(path).read().strip() if os.path.exists(path) else default


def load_ctx(results_dir, idx):
    base = os.path.join(results_dir, f'design_{idx}', 'context')
    ctx = {}
    raw = _read(os.path.join(base, 'strategy.txt'))
    m = re.search(r'Strategy:\s*(\w+)', raw)
    ctx['strategy'] = m.group(1) if m else 'no_strategy'
    ctx['rationale'] = _read(os.path.join(base, 'llm_rationale.txt'))
    pp = os.path.join(base, 'llm_params.json')
    ctx['params'] = json.load(open(pp)) if os.path.exists(pp) else {}
    return ctx


def crop_hifi(path):
    img = Image.open(path)
    w, h = img.size
    return img.crop((int(w * 0.335), int(h * 0.14),
                     int(w * 0.645), int(h * 0.96)))


def extract_quote(rationale):
    text = rationale.replace('\n', ' ')
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'`[^`]+`', '', text)
    sentences = re.split(r'(?<=[.!])\s+', text)
    keywords = [
        'sweep', 'twist', 'camber', 'drag', 'lift', 'reward',
        'explore', 'exploit', 'vortex', 'thickness', 'dihedral',
        'airfoil', 'washout', 'design', 'strategy', 'chord',
    ]
    for s in sentences:
        s = s.strip().lstrip('*-\u2022 0123456789.')
        if 20 < len(s) < 160 and any(kw in s.lower() for kw in keywords):
            return s[:120] + ('...' if len(s) > 120 else '')
    for s in sentences:
        s = s.strip().lstrip('*-\u2022 0123456789.')
        if len(s) > 25:
            return s[:120] + ('...' if len(s) > 120 else '')
    return 'Generating novel design...'


# -- Drawing helpers ---------------------------------------------------
def rounded_box(ax, x, y, w, h, color=CARD, border=BORDER,
                alpha=1.0, radius=0.008):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f'round,pad={radius}',
        facecolor=color, edgecolor=border, linewidth=0.8, alpha=alpha,
        transform=ax.transData, clip_on=False)
    ax.add_patch(box)


def draw_progress_bar(fig, progress, color):
    ax = fig.add_axes([0.0, 0.978, 1.0, 0.022])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor='#21262d',
                                     transform=ax.transAxes))
    ax.add_patch(mpatches.Rectangle((0, 0), progress, 1,
                                     facecolor=color, alpha=0.65,
                                     transform=ax.transAxes))
    ax.text(progress, 0.5, f' {int(progress*100)}%',
            fontsize=7, color=TEXT, va='center', clip_on=True)


def draw_reward_chart(fig, rect, data, up_to, n_total):
    ax = fig.add_axes(rect)
    k = up_to + 1
    rewards = data['reward']
    best_r  = data['best_reward']
    p5  = np.percentile(rewards, 5)
    p95 = np.percentile(rewards, 95)
    mg  = max((p95 - p5) * 0.15, 2.0)
    ylo, yhi = p5 - mg, p95 + mg

    for j in range(k):
        c = GREEN if rewards[j] >= 0 else RED
        a = 0.25 if j < k - 1 else 1.0
        ax.scatter(j, np.clip(rewards[j], ylo, yhi),
                   c=c, s=14, alpha=a, edgecolors='none', zorder=2)

    ax.plot(data['iteration'][:k], np.clip(best_r[:k], ylo, yhi),
            color=ACCENT, linewidth=1.3, alpha=0.9, zorder=3)
    ax.scatter([up_to], [np.clip(rewards[up_to], ylo, yhi)],
               s=70, facecolors='none', edgecolors=ORANGE,
               linewidths=2, zorder=5)
    ax.axhline(0, color=DIM, linewidth=0.4, linestyle='--', alpha=0.3)
    ax.set_xlim(-1, n_total)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel('Iteration', fontsize=8)
    ax.set_ylabel('Reward', fontsize=8)
    ax.tick_params(labelsize=7)


# -- Phase A: "Reasoning" ---------------------------------------------
def render_phase_a(results_dir, data, idx, n, ctx, quote):
    strat = ctx['strategy']
    sc = STRAT_COL.get(strat, DIM)
    sl = STRAT_LABEL.get(strat, strat.upper())

    fig = plt.figure(figsize=(14, 7), facecolor=BG)
    draw_progress_bar(fig, (idx + 0.3) / n, sc)

    ax = fig.add_axes([0, 0, 1, 0.975])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.set_facecolor(BG)

    # -- Header --
    ax.text(0.04, 0.94, f'Iteration {idx}',
            fontsize=24, fontweight='bold', color=TEXT, va='top')
    ax.text(0.04, 0.885, f'of {n - 1}', fontsize=11, color=DIM, va='top')

    # Strategy badge
    rounded_box(ax, 0.22, 0.905, 0.20, 0.045, color=sc + '18', border=sc)
    ax.text(0.32, 0.927, sl, fontsize=11, fontweight='bold',
            color=sc, va='center', ha='center')

    # Best reward
    best = data['best_reward'][max(idx - 1, 0)]
    ax.text(0.96, 0.94, f'Best {best:.1f}',
            fontsize=14, color=ACCENT, va='top', ha='right', fontweight='bold')
    ax.text(0.96, 0.89, f'L/D {best + 5.45:.1f}',
            fontsize=10, color=DIM, va='top', ha='right')

    # -- Cp placeholder --
    rounded_box(ax, 0.035, 0.30, 0.53, 0.55,
                color='#0d1117', border=BORDER)
    # Pulsing dots animation
    ax.text(0.30, 0.57, 'Reasoning', fontsize=20, color=DIM,
            alpha=0.35, ha='center', va='center', fontstyle='italic')
    for i, dot_x in enumerate([0.27, 0.30, 0.33]):
        ax.scatter(dot_x, 0.52, s=40, c=sc, alpha=0.15 + 0.2 * i,
                   edgecolors='none', zorder=10)

    # -- Quote --
    quote_y = 0.23
    ax.plot([0.04, 0.56], [quote_y + 0.015, quote_y + 0.015],
            color=sc, linewidth=2.5, alpha=0.5, clip_on=False)
    wrapped_q = textwrap.fill(f'\u201c{quote}\u201d', width=78)
    ax.text(0.04, quote_y - 0.01, wrapped_q, fontsize=9.5,
            color=TEXT, alpha=0.8, va='top', fontstyle='italic',
            linespacing=1.4)

    # -- Parameters placeholder --
    px = 0.60
    ax.text(px, 0.83, 'Design Parameters', fontsize=12,
            color=DIM, alpha=0.25, va='top', fontweight='bold')
    labels = ['LE Sweep', 'Root Chord', 'Twist (root)',
              'Twist (tip)', 'Dihedral', 'Airfoil']
    yy = 0.77
    for lb in labels:
        ax.text(px + 0.01, yy, lb, fontsize=9.5, color=DIM,
                alpha=0.15, va='top')
        ax.text(0.93, yy, '---', fontsize=9.5, color=DIM,
                alpha=0.15, va='top', ha='right')
        yy -= 0.055

    # -- Reward chart --
    draw_reward_chart(fig, [0.60, 0.07, 0.36, 0.28],
                      data, max(idx - 1, 0), n)
    return fig


# -- Phase B: "Result" -------------------------------------------------
def render_phase_b(results_dir, data, idx, n, ctx, quote):
    strat = ctx['strategy']
    sc = STRAT_COL.get(strat, DIM)
    sl = STRAT_LABEL.get(strat, strat.upper())
    p = ctx.get('params', {})
    reward = data['reward'][idx]
    ld = data['L_D'][idx]

    fig = plt.figure(figsize=(14, 7), facecolor=BG)
    draw_progress_bar(fig, (idx + 0.8) / n, sc)

    ax = fig.add_axes([0, 0, 1, 0.975])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.set_facecolor(BG)

    # -- Header --
    ax.text(0.04, 0.94, f'Iteration {idx}',
            fontsize=24, fontweight='bold', color=TEXT, va='top')
    ax.text(0.04, 0.885, f'of {n - 1}', fontsize=11, color=DIM, va='top')

    rounded_box(ax, 0.22, 0.905, 0.20, 0.045, color=sc + '18', border=sc)
    ax.text(0.32, 0.927, sl, fontsize=11, fontweight='bold',
            color=sc, va='center', ha='center')

    # -- Cp image --
    geom_png = os.path.join(results_dir, f'design_{idx}', 'geometry.png')
    ax_img = fig.add_axes([0.04, 0.30, 0.52, 0.55])
    ax_img.axis('off')
    ax_img.set_facecolor('#0d1117')
    if os.path.exists(geom_png):
        hifi = crop_hifi(geom_png)
        ax_img.imshow(hifi)
    else:
        ax_img.text(0.5, 0.5, 'No image', ha='center', va='center',
                    color=DIM, fontsize=12, transform=ax_img.transAxes)
    ax.text(0.30, 0.86, r'VortexNet Corrected $C_p$',
            fontsize=10, color=DIM, ha='center', va='top')

    # -- Quote (dimmer now) --
    quote_y = 0.23
    ax.plot([0.04, 0.56], [quote_y + 0.015, quote_y + 0.015],
            color=sc, linewidth=2.5, alpha=0.3, clip_on=False)
    wrapped_q = textwrap.fill(f'\u201c{quote}\u201d', width=78)
    ax.text(0.04, quote_y - 0.01, wrapped_q, fontsize=9.5,
            color=TEXT, alpha=0.45, va='top', fontstyle='italic',
            linespacing=1.4)

    # -- Parameters --
    px = 0.60
    ax.text(px, 0.83, 'Design Parameters', fontsize=12,
            color=TEXT, va='top', fontweight='bold')

    name = p.get('name', data['design'][idx] if idx < len(data['design']) else '')
    if name:
        ax.text(px + 0.01, 0.79, name, fontsize=8.5, color=ACCENT,
                va='top', fontstyle='italic')

    rows = [
        ('LE Sweep',      f"{p.get('le_sweep', '?')}\u00b0"),
        ('Root Chord',    f"{p.get('root_chord_in', '?')} in"),
        ('Twist (root)',  f"{p.get('twist_root', '?')}\u00b0"),
        ('Twist (tip)',   f"{p.get('twist_tip', '?')}\u00b0"),
        ('Dihedral',      f"{p.get('dihedral', '?')}\u00b0"),
        ('Airfoil',
         f"NACA {p.get('naca_m',0)}{p.get('naca_p',0)}{p.get('naca_t',12):02d}"),
    ]
    yy = 0.74
    for label, val in rows:
        ax.text(px + 0.01, yy, label, fontsize=9.5, color=DIM, va='top')
        ax.text(0.93, yy, val, fontsize=9.5, color=TEXT, va='top',
                ha='right', fontweight='bold')
        yy -= 0.053

    # -- Reward verdict --
    rv_c = GREEN if reward >= 0 else RED
    is_best = (idx == 0) or (reward > data['best_reward'][max(idx - 1, 0)])

    vy = 0.39
    rounded_box(ax, px - 0.005, vy - 0.005, 0.345, 0.065,
                color=rv_c + '12', border=rv_c)
    ax.text(px + 0.01, vy + 0.042, f'Reward  {reward:.1f}',
            fontsize=15, fontweight='bold', color=rv_c, va='top')
    if is_best and reward >= 0:
        ax.text(0.935, vy + 0.042, 'NEW BEST', fontsize=9,
                fontweight='bold', color=ORANGE, va='top', ha='right')
    ax.text(px + 0.01, vy - 0.015,
            f'$C_L / C_{{D_i}}$ = {ld:.1f}' if ld > 0 else 'Simulation failed',
            fontsize=10, color=DIM, va='top')

    # -- Best reward (top-right) --
    best = data['best_reward'][idx]
    ax.text(0.96, 0.94, f'Best {best:.1f}',
            fontsize=14, color=ACCENT, va='top', ha='right', fontweight='bold')
    ax.text(0.96, 0.89, f'L/D {best + 5.45:.1f}',
            fontsize=10, color=DIM, va='top', ha='right')

    # -- Reward chart --
    draw_reward_chart(fig, [0.60, 0.07, 0.36, 0.28], data, idx, n)
    return fig


# -- Main --------------------------------------------------------------
def make_animation(results_dir, every_n=2):
    data = load_results_csv(results_dir)
    n = len(data['iteration'])
    if n == 0:
        print("No data.")
        return

    plt.rcParams.update(STYLE)

    # Include every_n-th + all "new best" milestones
    indices = sorted(set(
        list(range(0, n, every_n))
        + ([n - 1] if n > 0 else [])
        + [i for i in range(1, n)
           if data['reward'][i] > data['best_reward'][max(i - 1, 0)]]
    ))

    frames = []
    durations = []
    total = len(indices) * 2
    print(f"Rendering {total} frames for {len(indices)} iterations ...")

    for fi, idx in enumerate(indices):
        ctx = load_ctx(results_dir, idx)
        quote = extract_quote(ctx['rationale'])

        # Phase A
        fig_a = render_phase_a(results_dir, data, idx, n, ctx, quote)
        fig_a.canvas.draw()
        buf = np.asarray(fig_a.canvas.buffer_rgba())[..., :3].copy()
        frames.append(Image.fromarray(buf))
        durations.append(900)
        plt.close(fig_a)

        # Phase B
        fig_b = render_phase_b(results_dir, data, idx, n, ctx, quote)
        fig_b.canvas.draw()
        buf = np.asarray(fig_b.canvas.buffer_rgba())[..., :3].copy()
        frames.append(Image.fromarray(buf))
        durations.append(1100)
        plt.close(fig_b)

        done = (fi + 1) * 2
        if done % 20 == 0 or fi == len(indices) - 1:
            print(f"  {done}/{total} frames")

    # Hold last frame
    for _ in range(6):
        frames.append(frames[-1])
        durations.append(1500)

    out_path = os.path.join(results_dir, 'llm_agent.gif')
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=True,
    )
    print(f"Saved -> {out_path}  ({len(frames)} frames, "
          f"{os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python animate_llm_agent_3d.py <results_dir> [every_n]")
        sys.exit(1)
    d = sys.argv[1]
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    make_animation(d, every_n=every)
