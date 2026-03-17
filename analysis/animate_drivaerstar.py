#!/usr/bin/env python3
"""
animate_drivaerstar.py -- Evolution GIF, lineage tree, and summary for
DrivAerStar vehicle aerodynamic optimisation runs.

Left  : reward (-Cd) progression with island colouring and lineage edges.
Right : Pressure iso view from the current design's save/sol/Pressure_iso.png.

Usage:
    python animate_drivaerstar.py <results_dir> [step]
"""

import os, sys, json, argparse
import csv as csv_mod
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

STYLE = {
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset':  'cm',
    'font.size':         9,
    'axes.labelsize':    11,
    'axes.titlesize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   8.5,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'figure.dpi':        200,
}

C_BEST    = '#1f4e8c'
C_CURRENT = '#d4600e'
C_LINEAGE = '#b0b0b0'


def load_csv(results_dir):
    csv_path = os.path.join(results_dir, 'results.csv')
    data = {'iteration': [], 'design': [], 'reward': [], 'best_reward': [],
            'sample': [], 'drag': [], 'Cd': [], 'lift': [], 'L_D': [], 'island': []}
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            data['iteration'].append(int(row['iteration']))
            data['design'].append(row['design'])
            data['reward'].append(float(row['reward']))
            data['best_reward'].append(float(row['best_reward']))
            sample_val = row.get('sample')
            if sample_val is None or sample_val == '':
                data['sample'].append(-1)
            else:
                data['sample'].append(int(float(sample_val)))
            data['drag'].append(float(row.get('drag', 0)))
            cd_val = row.get('Cd', row.get('CD', 0))
            data['Cd'].append(float(cd_val))
            data['lift'].append(float(row.get('lift', 0)))
            data['L_D'].append(float(row.get('L_D', 0)))
            data['island'].append(int(row['island']) if 'island' in row else 0)
    for k in data:
        if k != 'design':
            data[k] = np.array(data[k])
    return data


def load_lineage(results_dir):
    parent_of = {}
    lp = os.path.join(results_dir, 'lineage.json')
    if os.path.exists(lp):
        with open(lp) as f:
            lineage = json.load(f)
        for e in lineage:
            cid, pid = e.get('id'), e.get('parent_id')
            if isinstance(cid, int):
                parent_of[cid] = pid
    return parent_of


def _design_dir_candidates(design_ref):
    if isinstance(design_ref, str) and design_ref:
        yield design_ref
        if design_ref.startswith('design_'):
            suffix = design_ref.split('design_', 1)[1]
            if suffix.isdigit():
                yield f'design_{int(suffix)}'
    if isinstance(design_ref, (int, np.integer)):
        yield f'design_{int(design_ref)}'


def get_design_image(results_dir, design_ref):
    for design_dir in _design_dir_candidates(design_ref):
        for img in ['Pressure_iso.png', 'Pressure_top.png', 'Pressure_side.png']:
            p = os.path.join(results_dir, design_dir, 'save', 'sol', img)
            if os.path.exists(p):
                return Image.open(p)
        # NeuralFoil shape image fallback
        p = os.path.join(results_dir, design_dir, 'save', 'shape.png')
        if os.path.exists(p):
            return Image.open(p)
    # Last fallback for old numeric-index usage
    for img in ['Pressure_iso.png', 'Pressure_top.png', 'Pressure_side.png']:
        p = os.path.join(results_dir, f'design_{design_ref}', 'save', 'sol', img)
        if os.path.exists(p):
            return Image.open(p)
    p = os.path.join(results_dir, f'design_{design_ref}', 'save', 'shape.png')
    if os.path.exists(p):
        return Image.open(p)
    return None


def _select_metric(data):
    cd = data['Cd']
    ld = data['L_D']
    if np.any(np.abs(cd) > 1e-12):
        return 'Cd', cd, r'$C_d$'
    if np.any(np.abs(ld) > 1e-12):
        return 'L_D', ld, r'$L/D$'
    return None, None, None


def _compute_plot_x(data):
    xvals = data['iteration'].astype(float).copy()
    samples = data.get('sample')
    if samples is None or len(samples) == 0:
        return xvals, 'LLM calls'
    valid = samples >= 0
    if not np.any(valid):
        return xvals, 'LLM calls'
    denom = int(np.max(samples[valid])) + 1
    if denom <= 1:
        return xvals, 'LLM calls'
    xvals[valid] = xvals[valid] + samples[valid] / float(denom)
    return xvals, 'LLM calls'


def _compute_global_crop(results_dir, design_ids, sample=30, pad=20, white_thresh=240):
    """Sample up to `sample` design images and return a fixed (r0,r1,c0,c1) crop
    that covers the union of all non-white content bounding boxes."""
    n = len(design_ids)
    if n == 0:
        return None
    indices = sorted(set(np.linspace(0, n - 1, min(sample, n), dtype=int)))
    r0_g, r1_g, c0_g, c1_g = None, None, None, None
    H, W = None, None
    for idx in indices:
        img = get_design_image(results_dir, design_ids[idx])
        if img is None:
            continue
        arr = np.array(img.convert('RGB'))
        H, W = arr.shape[:2]
        mask = ~((arr[..., 0] > white_thresh) &
                 (arr[..., 1] > white_thresh) &
                 (arr[..., 2] > white_thresh))
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            continue
        r0, r1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        c0, c1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
        r0_g = r0 if r0_g is None else min(r0_g, r0)
        r1_g = r1 if r1_g is None else max(r1_g, r1)
        c0_g = c0 if c0_g is None else min(c0_g, c0)
        c1_g = c1 if c1_g is None else max(c1_g, c1)
    if r0_g is None:
        return None
    r0_g = max(0, r0_g - pad)
    r1_g = min(H, r1_g + pad)
    c0_g = max(0, c0_g - pad)
    c1_g = min(W, c1_g + pad)
    return (r0_g, r1_g, c0_g, c1_g)


def _crop_img(img, crop):
    """Apply (r0,r1,c0,c1) crop to a PIL Image, return numpy array."""
    if img is None or crop is None:
        return None
    arr = np.array(img.convert('RGB'))
    r0, r1, c0, c1 = crop
    return arr[r0:r1, c0:c1]


def _show_img_panel(ax, arr, label, label_color='#333'):
    ax.axis('off')
    if arr is not None:
        ax.imshow(arr, aspect='equal')
    else:
        ax.set_facecolor('#f0f0f0')
        ax.text(0.5, 0.5, 'no image', ha='center', va='center',
                fontsize=9, color='#aaa', transform=ax.transAxes)
    ax.set_title(label, fontsize=8.5, fontweight='medium', pad=4,
                 color=label_color)


def make_animation(results_dir, step=1):
    data = load_csv(results_dir)
    n = len(data['iteration'])
    if n == 0:
        print("No data.")
        return

    parent_of = load_lineage(results_dir)
    # Convert reward to a minimization objective for visualization:
    # objective = -reward (strictly positive for log-scale plotting).
    rewards = np.maximum(-data['reward'].copy(), 1e-9)
    best_reward = np.maximum(-data['best_reward'], 1e-9)
    design_ids = data['design']
    metric_key, metric_vals, metric_label = _select_metric(data)
    xvals, x_label = _compute_plot_x(data)
    samples = data.get('sample')
    islands = data['island']
    n_islands = int(islands.max()) + 1
    island_cmap = plt.cm.Set2(np.linspace(0, 1, max(n_islands, 2)))

    p5, p95 = np.percentile(rewards, 5), np.percentile(rewards, 95)
    y_lo = max(min(np.min(rewards) * 0.8, p5 * 0.8), 1e-9)
    y_hi = max(np.max(rewards) * 1.2, p95 * 1.2)

    # Compute a stable crop bbox once from a sample of images
    print(f"[{os.path.basename(results_dir)}] Computing global image crop ...")
    crop = _compute_global_crop(results_dir, design_ids, sample=40)
    if crop:
        print(f"  Crop: rows {crop[0]}-{crop[1]}, cols {crop[2]}-{crop[3]}")
    x_min = float(min(0.0, np.min(xvals)) - 0.1)
    x_max = float(np.max(xvals) + 0.1)
    if x_max - x_min < 5.0:
        x_max = x_min + 5.0

    plt.rcParams.update(STYLE)
    frames = []
    frame_indices = list(range(0, n, step))
    if frame_indices[-1] != n - 1:
        frame_indices.append(n - 1)

    # Track the index of the current best design so we can show its image
    best_idx = 0

    print(f"[{os.path.basename(results_dir)}] Rendering {len(frame_indices)} frames ...")

    for fi, up_to in enumerate(frame_indices):
        k = up_to + 1

        # Update best design index up to this frame
        best_idx = int(np.argmin(rewards[:k]))

        # Layout: left = reward plot, right = two stacked image panels
        fig = plt.figure(figsize=(14, 6), facecolor='white')
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.03)
        gs_right = gs[1].subgridspec(2, 1, hspace=0.08)

        # ── Reward scatter plot ───────────────────────────────────────────────
        ax = fig.add_subplot(gs[0])

        for child in range(k):
            p = parent_of.get(child)
            if p is not None and isinstance(p, int) and p < k:
                ax.plot([xvals[p], xvals[child]],
                        [np.clip(rewards[p], y_lo, y_hi),
                         np.clip(rewards[child], y_lo, y_hi)],
                        color=C_LINEAGE, linewidth=0.35, alpha=0.35, zorder=1)

        cols = [island_cmap[int(isl)] for isl in islands[:k]]
        ax.scatter(xvals[:k],
                   np.clip(rewards[:k], y_lo, y_hi),
                   c=cols, s=14, alpha=0.7,
                   edgecolors='k', linewidths=0.15, zorder=2)

        # Highlight current iteration
        ax.scatter([xvals[up_to]],
                   [np.clip(rewards[up_to], y_lo, y_hi)],
                   s=100, facecolors='none', edgecolors=C_CURRENT,
                   linewidths=2, zorder=5, label='_nolegend_')

        # Highlight best so far
        ax.scatter([xvals[best_idx]],
                   [np.clip(rewards[best_idx], y_lo, y_hi)],
                   s=120, marker='*', facecolors='gold', edgecolors='#555',
                   linewidths=0.8, zorder=6, label='_nolegend_')

        ax.plot(xvals[:k],
                np.clip(best_reward[:k], y_lo, y_hi),
                color=C_BEST, linewidth=1.4, zorder=3)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(x_label)
        ax.set_yscale('log')
        ax.set_ylabel('Objective  (-reward, lower is better)')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        handles = [plt.Line2D([0], [0], color=C_BEST, lw=1.4, label='Best so far')]
        if n_islands > 1:
            for isl in range(n_islands):
                handles.append(plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=island_cmap[isl], markersize=5,
                               label=f'Island {isl}'))
        handles += [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                       markeredgecolor=C_CURRENT, markeredgewidth=2,
                       markersize=7, label='Current'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                       markeredgecolor='#555', markeredgewidth=0.8,
                       markersize=9, label='Best'),
        ]
        ax.legend(handles=handles, loc='upper left', frameon=True, fancybox=False,
                  edgecolor='#bbb', framealpha=1, borderpad=0.4,
                  ncol=2 if n_islands > 1 else 1, fontsize=8)

        r_now = rewards[up_to]
        r_best = best_reward[up_to]
        isl_str = f'  Island {int(islands[up_to])}' if n_islands > 1 else ''
        sim_txt = f'Sims {k}/{n}'
        if samples is not None and len(samples) == n and samples[up_to] >= 0:
            sim_txt += f' (sample {int(samples[up_to])})'
        metric_txt = ''
        if metric_key is not None:
            metric_txt = rf'    {metric_label}={metric_vals[up_to]:.4f}'
        ax.set_title(
            f'{sim_txt}    LLM call={xvals[up_to]:.2f}{isl_str}    '
            rf'$obj={r_now:.4f}$    Best$={r_best:.4f}$' + metric_txt,
            fontweight='medium', pad=8)

        # ── Top image: best-so-far design ────────────────────────────────────
        ax_best = fig.add_subplot(gs_right[0])
        arr_best = _crop_img(get_design_image(results_dir, design_ids[best_idx]), crop)
        _show_img_panel(
            ax_best, arr_best,
            (
                f'Best  ({design_ids[best_idx]},  {metric_label}={metric_vals[best_idx]:.4f})'
                if metric_key is not None else f'Best  ({design_ids[best_idx]})'
            ),
            label_color=C_BEST)

        # ── Bottom image: current iteration design ────────────────────────────
        ax_curr = fig.add_subplot(gs_right[1])
        arr_curr = _crop_img(get_design_image(results_dir, design_ids[up_to]), crop)
        _show_img_panel(
            ax_curr, arr_curr,
            (
                f'Current  ({design_ids[up_to]},  {metric_label}={metric_vals[up_to]:.4f})'
                if metric_key is not None else f'Current  ({design_ids[up_to]})'
            ),
            label_color=C_CURRENT)

        fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.10)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(Image.fromarray(buf))
        plt.close(fig)

        if (fi + 1) % 20 == 0:
            print(f"  {fi + 1}/{len(frame_indices)} frames")

    for _ in range(8):
        frames.append(frames[-1])

    out = os.path.join(results_dir, 'evolution.gif')
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=150, loop=0, optimize=True)
    print(f"  -> {out}  ({len(frames)} frames, {os.path.getsize(out)/1e6:.1f} MB)")


def make_lineage_tree(results_dir):
    lp = os.path.join(results_dir, 'lineage.json')
    if not os.path.exists(lp):
        print(f"  No lineage.json in {results_dir}, skipping tree.")
        return
    try:
        import networkx as nx
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm_module
        from matplotlib.lines import Line2D
    except ImportError:
        print("  networkx not installed, skipping lineage tree.")
        return

    with open(lp) as f:
        lineage = json.load(f)
    if not lineage:
        return

    G = nx.DiGraph()
    for e in lineage:
        G.add_node(e['id'], reward=e['reward'], island=e.get('island', 0))
    node_ids = set(e['id'] for e in lineage)
    for e in lineage:
        if e['parent_id'] is not None and e['parent_id'] in node_ids:
            G.add_edge(e['parent_id'], e['id'])

    EDGE_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    num_nodes = len(G.nodes())
    fig_h = max(10, 4 + num_nodes * 0.15)
    fig, ax = plt.subplots(figsize=(max(14, num_nodes * 0.3), fig_h))

    roots = [n for n, d in G.in_degree() if d == 0]
    root = roots[0] if roots else list(G.nodes())[0]
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', root=root,
            args='-Grankdir=TB -Goverlap=false -Gsep=1.5')
    except Exception:
        try:
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot', root=root)
        except Exception:
            pos = _hierarchical_layout(G, roots)

    rewards = [G.nodes[n]['reward'] for n in G.nodes()]
    vmin, vmax = min(rewards), max(rewards)
    if vmin == vmax:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm_module.get_cmap('viridis')

    best_node = max(G.nodes(), key=lambda n: G.nodes[n]['reward'])
    best_path_edges = []
    try:
        path = nx.shortest_path(G, root, best_node)
        best_path_edges = list(zip(path[:-1], path[1:]))
    except nx.NetworkXNoPath:
        pass

    regular_edges = [e for e in G.edges() if e not in best_path_edges]
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, arrows=True,
        arrowsize=10, width=1.0, edge_color='gray', alpha=0.5, ax=ax)
    if best_path_edges:
        nx.draw_networkx_edges(G, pos, edgelist=best_path_edges, arrows=True,
            arrowsize=14, width=2.5, edge_color='black', alpha=0.8, ax=ax)

    sf = max(0.4, min(1.0, 15 / (num_nodes ** 0.4)))
    for node in G.nodes():
        attrs = G.nodes[node]
        isl = attrs.get('island', 0)
        r = attrs['reward']
        ec = EDGE_COLORS[isl % len(EDGE_COLORS)]
        if node == best_node:
            nc, ns, shape, lw = 'gold', int(1200 * sf), '*', 2.5
        else:
            nc, ns, shape, lw = mcolors.to_hex(cmap(norm(r))), int(600 * sf), 'o', 2.0
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=ns,
            node_color=nc, edgecolors=ec, linewidths=lw, node_shape=shape, ax=ax)

    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=max(6, int(10 * sf)),
        font_weight='bold', font_color='white', ax=ax)

    sm = cm_module.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.6)
    cb.set_label('Reward', size=14, weight='bold')

    n_isl = max((e.get('island', 0) for e in lineage), default=0) + 1
    handles = []
    for isl in range(n_isl):
        handles.append(Line2D([0], [0], marker='o', color='w',
            markerfacecolor='gray', markeredgecolor=EDGE_COLORS[isl % len(EDGE_COLORS)],
            markersize=10, markeredgewidth=2.5, label=f'Island {isl}'))
    handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
        markersize=16, label='Best'))
    handles.append(Line2D([0], [0], color='black', linewidth=2.5, label='Path to Best'))
    ax.legend(handles=handles, loc='upper right', fontsize=10)
    ax.set_title(f'Design Lineage — {os.path.basename(results_dir)}',
                 fontsize=18, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()
    out = os.path.join(results_dir, 'lineage_tree.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out}")


def _hierarchical_layout(G, roots):
    pos = {}
    if not roots:
        roots = [list(G.nodes())[0]]
    visited = set()
    layers = {}
    queue = [(r, 0) for r in roots]
    for r in roots:
        visited.add(r)
    while queue:
        node, depth = queue.pop(0)
        layers.setdefault(depth, []).append(node)
        for child in G.successors(node):
            if child not in visited:
                visited.add(child)
                queue.append((child, depth + 1))
    for node in G.nodes():
        if node not in visited:
            layers.setdefault(0, []).append(node)
    for depth, nodes in layers.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = ((i - (n - 1) / 2.0) * 2.0, -depth * 2.0)
    return pos


def make_summary(results_dir):
    data = load_csv(results_dir)
    n = len(data['iteration'])
    if n == 0:
        return

    iters, rewards, best = data['iteration'], data['reward'], data['best_reward']
    islands = data['island']
    metric_key, metric_vals, metric_label = _select_metric(data)
    n_isl = int(islands.max()) + 1
    island_cmap = plt.cm.Set2(np.linspace(0, 1, max(n_isl, 2)))

    # Percentile-clipped y range for rewards (excludes wild outliers)
    p2, p98 = np.percentile(rewards, 2), np.percentile(rewards, 98)
    margin = (p98 - p2) * 0.15
    r_lo, r_hi = p2 - margin, p98 + margin

    # Same for optional physical metric (Cd or L/D)
    if metric_key is not None:
        m_p2, m_p98 = np.percentile(metric_vals, 2), np.percentile(metric_vals, 98)
        m_margin = (m_p98 - m_p2) * 0.15
        m_lo, m_hi = m_p2 - m_margin, m_p98 + m_margin

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f'{os.path.basename(results_dir)} — {n} iterations, {n_isl} island(s)   '
        rf'Best reward = {best[-1]:.4f}',
        fontsize=12, fontweight='bold', y=0.995)

    # ── Top-left: reward scatter per island + best-so-far line ──────────────
    ax = axes[0, 0]
    for isl in range(n_isl):
        m = islands == isl
        ax.scatter(iters[m], np.clip(rewards[m], r_lo, r_hi),
                   color=island_cmap[isl], alpha=0.55, s=10, lw=0,
                   label=f'Island {isl}' if n_isl > 1 else 'Reward', zorder=2)
    ax.plot(iters, np.clip(best, r_lo, r_hi), color=C_BEST, lw=1.6,
            label='Best so far', zorder=3)
    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(r_lo, r_hi)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Reward')
    ax.set_title('Reward per Iteration')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9, frameon=True,
              fancybox=False, edgecolor='#aaa')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    # ── Top-right: best-reward convergence curve ─────────────────────────────
    ax = axes[0, 1]
    b_lo = best.min() - (best.max() - best.min()) * 0.08
    b_hi = best.max() + (best.max() - best.min()) * 0.25
    ax.plot(iters, best, color=C_BEST, lw=1.8, zorder=3)
    ax.fill_between(iters, best, b_lo, alpha=0.12, color=C_BEST)
    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(b_lo, b_hi)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Reward')
    ax.set_title('Best Reward Convergence')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    best_iter = int(iters[np.argmax(rewards)])
    ax.text(0.97, 0.07,
            rf'Final best: ${best[-1]:.4f}$'
            '\n' rf'Best iter: ${best_iter}$'
            '\n' rf'$n$ = ${n}$',
            transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#ccc', alpha=0.9))
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    # ── Bottom-left: metric boxplot/hist if available, else reward by island ─
    ax = axes[1, 0]
    if metric_key is not None:
        if n_isl > 1:
            data_box = [np.clip(metric_vals[islands == isl], m_lo, m_hi)
                        for isl in range(n_isl)]
            bp = ax.boxplot(data_box,
                            tick_labels=[f'Island {i}' for i in range(n_isl)],
                            patch_artist=True, widths=0.5,
                            medianprops=dict(color='black', linewidth=1.5),
                            whiskerprops=dict(linewidth=0.8),
                            capprops=dict(linewidth=0.8),
                            flierprops=dict(marker='.', markersize=3,
                                            alpha=0.4, linestyle='none'))
            for patch, color in zip(bp['boxes'], island_cmap):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_title(f'{metric_label} Distribution per Island')
        else:
            clipped_m = metric_vals[(metric_vals >= m_lo) & (metric_vals <= m_hi)]
            ax.hist(clipped_m, bins=40, color='steelblue',
                    edgecolor='white', linewidth=0.4, alpha=0.85)
            ax.set_title(f'{metric_label} Distribution')
        ax.set_ylim(m_lo, m_hi)
        ax.set_ylabel(metric_label)
    else:
        if n_isl > 1:
            data_box = [np.clip(rewards[islands == isl], r_lo, r_hi)
                        for isl in range(n_isl)]
            bp = ax.boxplot(data_box,
                            tick_labels=[f'Island {i}' for i in range(n_isl)],
                            patch_artist=True, widths=0.5,
                            medianprops=dict(color='black', linewidth=1.5),
                            whiskerprops=dict(linewidth=0.8),
                            capprops=dict(linewidth=0.8),
                            flierprops=dict(marker='.', markersize=3,
                                            alpha=0.4, linestyle='none'))
            for patch, color in zip(bp['boxes'], island_cmap):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_title('Reward Distribution per Island')
        else:
            clipped_r = rewards[(rewards >= r_lo) & (rewards <= r_hi)]
            ax.hist(clipped_r, bins=40, color='steelblue',
                    edgecolor='white', linewidth=0.4, alpha=0.85)
            ax.set_title('Reward Distribution')
        ax.set_ylim(r_lo, r_hi)
        ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.25, linewidth=0.5, axis='y')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    # ── Bottom-right: reward histogram (clipped range) ───────────────────────
    ax = axes[1, 1]
    clipped_r = rewards[(rewards >= r_lo) & (rewards <= r_hi)]
    ax.hist(clipped_r, bins=50, color='#4a7fc1',
            edgecolor='white', linewidth=0.4, alpha=0.85)
    ax.axvline(best[-1], color='#c0392b', ls='--', lw=1.6,
               label=rf'Best: ${best[-1]:.4f}$')
    ax.axvline(float(np.mean(rewards)), color=C_CURRENT, ls='--', lw=1.4,
               label=rf'Mean: ${np.mean(rewards):.4f}$')
    ax.set_xlim(r_lo, r_hi)
    ax.set_xlabel('Reward')
    ax.set_ylabel('Count')
    ax.set_title('Reward Distribution')
    ax.legend(fontsize=9, framealpha=0.9, frameon=True,
              fancybox=False, edgecolor='#aaa')
    ax.grid(True, alpha=0.25, linewidth=0.5, axis='y')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.993])
    out = os.path.join(results_dir, 'summary_plot.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir')
    parser.add_argument('step', nargs='?', type=int, default=1)
    args = parser.parse_args()

    make_summary(args.results_dir)
    make_lineage_tree(args.results_dir)
    make_animation(args.results_dir, step=args.step)
