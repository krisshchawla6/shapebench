#!/usr/bin/env python3
"""
animate_blendednet.py -- Evolution GIF, lineage tree, and summary for
BlendedNet BWB surrogate optimisation runs.

Left  : reward (L/D) progression with island colouring and lineage edges.
Right : Cp iso view from the current design's save/sol/Cp_iso.png.

Usage:
    python animate_blendednet.py <results_dir> [step]
    python animate_blendednet.py --compare <dir1> <dir2> [--labels L1 L2]
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
            'Cp_mean': [], 'Cfx_mean': [], 'L_D': [], 'island': []}
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            data['iteration'].append(int(row['iteration']))
            data['design'].append(row['design'])
            data['reward'].append(float(row['reward']))
            data['best_reward'].append(float(row['best_reward']))
            data['Cp_mean'].append(float(row.get('Cp_mean', 0)))
            data['Cfx_mean'].append(float(row.get('Cfx_mean', 0)))
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


def get_design_image(results_dir, design_idx):
    for img in ['Cp_iso.png', 'Cp_top.png']:
        p = os.path.join(results_dir, f'design_{design_idx}', 'save', 'sol', img)
        if os.path.exists(p):
            return Image.open(p)
    return None


# ── individual evolution GIF ────────────────────────────────────────────

def make_animation(results_dir, step=1):
    data = load_csv(results_dir)
    n = len(data['iteration'])
    if n == 0:
        print("No data.")
        return

    parent_of = load_lineage(results_dir)
    rewards = data['reward'].copy()
    islands = data['island']
    n_islands = int(islands.max()) + 1
    island_cmap = plt.cm.Set2(np.linspace(0, 1, max(n_islands, 2)))

    p5, p95 = np.percentile(rewards, 5), np.percentile(rewards, 95)
    margin = max((p95 - p5) * 0.25, 2.0)
    y_lo, y_hi = p5 - margin, p95 + margin

    plt.rcParams.update(STYLE)
    frames = []
    frame_indices = list(range(0, n, step))
    if frame_indices[-1] != n - 1:
        frame_indices.append(n - 1)

    print(f"[{os.path.basename(results_dir)}] Rendering {len(frame_indices)} frames ...")

    for fi, up_to in enumerate(frame_indices):
        k = up_to + 1
        fig = plt.figure(figsize=(13, 5), facecolor='white')
        gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.02)

        ax = fig.add_subplot(gs[0])

        for child in range(k):
            p = parent_of.get(child)
            if p is not None and isinstance(p, int) and p < k:
                ax.plot([p, child],
                        [np.clip(rewards[p], y_lo, y_hi),
                         np.clip(rewards[child], y_lo, y_hi)],
                        color=C_LINEAGE, linewidth=0.35, alpha=0.35, zorder=1)

        cols = [island_cmap[int(isl)] for isl in islands[:k]]
        ax.scatter(data['iteration'][:k],
                   np.clip(rewards[:k], y_lo, y_hi),
                   c=cols, s=16, alpha=0.75,
                   edgecolors='k', linewidths=0.2, zorder=2)

        ax.scatter([data['iteration'][up_to]],
                   [np.clip(rewards[up_to], y_lo, y_hi)],
                   s=90, facecolors='none', edgecolors=C_CURRENT,
                   linewidths=2, zorder=5)

        ax.plot(data['iteration'][:k],
                np.clip(data['best_reward'][:k], y_lo, y_hi),
                color=C_BEST, linewidth=1.4, zorder=3, label='Best reward')

        ax.set_xlim(-0.5, max(n - 0.5, 10))
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Reward  ($L/D$)')
        handles = [plt.Line2D([0], [0], color=C_BEST, lw=1.4, label='Best reward')]
        if n_islands > 1:
            for isl in range(n_islands):
                handles.append(plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=island_cmap[isl], markersize=5,
                               label=f'Island {isl}'))
        ax.legend(handles=handles, loc='upper left', frameon=True, fancybox=False,
                  edgecolor='#666', framealpha=1, borderpad=0.4,
                  ncol=2 if n_islands > 1 else 1)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        r_now = rewards[up_to]
        r_best = data['best_reward'][up_to]
        ld_now = data['L_D'][up_to]
        isl_str = f'  (Island {int(islands[up_to])})' if n_islands > 1 else ''
        ax.set_title(
            f'Iteration {up_to}{isl_str}    '
            rf'$r={r_now:.1f}$    Best $={r_best:.1f}$    $L/D={ld_now:.1f}$',
            fontweight='medium', pad=8)

        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        img = get_design_image(results_dir, up_to)
        if img is not None:
            ax2.imshow(img)
            ax2.set_title(f'$C_p$ iso  (design {up_to})', fontsize=9,
                          fontweight='medium', pad=6)
        else:
            ax2.text(0.5, 0.5, f'design_{up_to}\n(no image)',
                     ha='center', va='center', fontsize=10,
                     transform=ax2.transAxes, color='#888')

        fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.14, wspace=0.04)
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


# ── lineage tree (islands run) ──────────────────────────────────────────

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
    cb.set_label('Reward (L/D)', size=14, weight='bold')

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


# ── summary plot ────────────────────────────────────────────────────────

def make_summary(results_dir):
    data = load_csv(results_dir)
    n = len(data['iteration'])
    if n == 0:
        return

    iters, rewards, best = data['iteration'], data['reward'], data['best_reward']
    islands, ld = data['island'], data['L_D']
    n_isl = int(islands.max()) + 1
    cmap = plt.cm.Set2(np.linspace(0, 1, max(n_isl, 2)))

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{os.path.basename(results_dir)} — {n} iterations, {n_isl} island(s)',
                 fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    for isl in range(n_isl):
        m = islands == isl
        ax.scatter(iters[m], rewards[m], color=cmap[isl], alpha=0.6, s=20,
                   label=f'Island {isl}' if n_isl > 1 else 'Reward')
    ax.plot(iters, best, 'k-', lw=2, alpha=0.7, label='Best so far')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Reward (L/D)')
    ax.set_title('Reward per Iteration')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iters, best, 'r-', lw=2)
    ax.fill_between(iters, best, alpha=0.15, color='red')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Best Reward (L/D)')
    ax.set_title('Best Reward Progression'); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95,
            f'Final Best: {best[-1]:.2f}\nBest Iter: {iters[np.argmax(rewards)]}\nTotal: {n}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    ax = axes[1, 0]
    if n_isl > 1:
        data_box = [ld[islands == isl] for isl in range(n_isl)]
        bp = ax.boxplot(data_box,
                        tick_labels=[f'Island {i}' for i in range(n_isl)],
                        patch_artist=True)
        for patch, color in zip(bp['boxes'], cmap):
            patch.set_facecolor(color)
        ax.set_title('L/D Distribution per Island')
    else:
        ax.hist(ld, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title('L/D Distribution')
    ax.set_ylabel('L/D'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(rewards, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(best[-1], color='red', ls='--', lw=2, label=f'Best: {best[-1]:.2f}')
    ax.axvline(np.mean(rewards), color='orange', ls='--', lw=2,
               label=f'Mean: {np.mean(rewards):.2f}')
    ax.set_xlabel('Reward (L/D)'); ax.set_ylabel('Count')
    ax.set_title('Reward Distribution'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, 'summary_plot.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  -> {out}")


# ── comparison GIF (stacked: plots left, bird's-eye Cp right) ───────────

def _get_top_image(results_dir, design_idx):
    p = os.path.join(results_dir, f'design_{design_idx}', 'save', 'sol', 'Cp_top.png')
    if os.path.exists(p):
        return Image.open(p)
    return None


def _best_so_far_idx(data, up_to):
    """Index of the design with highest reward up to (inclusive)."""
    return int(np.argmax(data['reward'][:up_to + 1]))


def make_comparison(dir1, dir2, label1=None, label2=None, output_dir=None):
    d1, d2 = load_csv(dir1), load_csv(dir2)
    label1 = label1 or os.path.basename(dir1)
    label2 = label2 or os.path.basename(dir2)
    n = max(len(d1['iteration']), len(d2['iteration']))
    if n == 0:
        return

    output_dir = output_dir or os.path.dirname(dir1)

    all_rewards = np.concatenate([d1['reward'], d2['reward']])
    p5, p95 = np.percentile(all_rewards, 5), np.percentile(all_rewards, 95)
    margin = max((p95 - p5) * 0.25, 2.0)
    y_lo, y_hi = p5 - margin, p95 + margin

    plt.rcParams.update(STYLE)
    frames = []
    step = max(1, n // 120)
    frame_indices = list(range(0, n, step))
    if frame_indices[-1] != n - 1:
        frame_indices.append(n - 1)

    print(f"[comparison] Rendering {len(frame_indices)} frames ...")

    for fi, up_to in enumerate(frame_indices):
        # 2 rows x 2 cols: left = reward plots stacked, right = Cp_top stacked
        fig = plt.figure(figsize=(14, 10), facecolor='white')
        gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1],
                              hspace=0.30, wspace=0.08)

        for row, (data, rdir, label) in enumerate(
                [(d1, dir1, label1), (d2, dir2, label2)]):
            k = min(up_to + 1, len(data['iteration']))

            # ── left: reward scatter + best line ──
            ax = fig.add_subplot(gs[row, 0])
            if k == 0:
                ax.set_title(f'{label}  (no data yet)')
            else:
                islands = data['island'][:k]
                n_isl = int(islands.max()) + 1
                ic = plt.cm.Set2(np.linspace(0, 1, max(n_isl, 2)))
                cols = [ic[int(isl)] for isl in islands]

                ax.scatter(data['iteration'][:k],
                           np.clip(data['reward'][:k], y_lo, y_hi),
                           c=cols, s=14, alpha=0.7,
                           edgecolors='k', linewidths=0.15, zorder=2)
                ax.plot(data['iteration'][:k],
                        np.clip(data['best_reward'][:k], y_lo, y_hi),
                        color=C_BEST, lw=1.4, zorder=3)
                ax.scatter([data['iteration'][k - 1]],
                           [np.clip(data['reward'][k - 1], y_lo, y_hi)],
                           s=80, facecolors='none', edgecolors=C_CURRENT,
                           linewidths=2, zorder=5)

                ax.set_xlim(-0.5, n - 0.5)
                ax.set_ylim(y_lo, y_hi)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('L/D')
                for sp in ['top', 'right']:
                    ax.spines[sp].set_visible(False)

                r_best = data['best_reward'][k - 1]
                ax.set_title(f'{label}      Iter {k-1}    Best $L/D={r_best:.1f}$',
                             fontweight='medium', pad=6)

            # ── right: Cp_top bird's-eye view ──
            ax_img = fig.add_subplot(gs[row, 1])
            ax_img.axis('off')
            if k > 0:
                best_idx = _best_so_far_idx(data, k - 1)
                img = _get_top_image(rdir, k - 1)
                best_img = _get_top_image(rdir, best_idx) if best_idx != k - 1 else img
                show_img = best_img if best_img is not None else img
                if show_img is not None:
                    ax_img.imshow(show_img)
                    ax_img.set_title(
                        f'$C_p$ top — best design {best_idx}',
                        fontsize=9, fontweight='medium', pad=4)
                else:
                    ax_img.text(0.5, 0.5, '(no image)',
                                ha='center', va='center', fontsize=10,
                                transform=ax_img.transAxes, color='#888')

        fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.06)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(Image.fromarray(buf))
        plt.close(fig)

    for _ in range(8):
        frames.append(frames[-1])

    out = os.path.join(output_dir, 'comparison_evolution.gif')
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=150, loop=0, optimize=True)
    print(f"  -> {out}  ({len(frames)} frames, {os.path.getsize(out)/1e6:.1f} MB)")


# ── main ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', nargs='?')
    parser.add_argument('step', nargs='?', type=int, default=1)
    parser.add_argument('--compare', nargs=2, metavar=('DIR1', 'DIR2'))
    parser.add_argument('--labels', nargs=2, metavar=('L1', 'L2'))
    args = parser.parse_args()

    if args.compare:
        l1, l2 = (args.labels or [None, None])
        make_comparison(args.compare[0], args.compare[1], l1, l2)
    elif args.results_dir:
        make_summary(args.results_dir)
        make_lineage_tree(args.results_dir)
        make_animation(args.results_dir, step=args.step)
    else:
        parser.print_help()
