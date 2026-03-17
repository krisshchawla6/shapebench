"""Lineage tree visualization for island-based airfoil evolution.
Plots a directed graph where nodes are designs, colored by reward,
shaped/bordered by island, with edges showing parent-child relationships."""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm_module
import networkx as nx
from matplotlib.lines import Line2D


# One color per island (extend if more islands needed)
ISLAND_EDGE_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

def plot_lineage_tree(lineage, output_path, title="Design Lineage Tree"):
    """
    Plot an evolutionary lineage tree from lineage data.

    Args:
        lineage: list of dicts, each with keys:
            id (int), parent_id (int or None), reward (float), island (int)
        output_path: path to save the PNG
        title: plot title
    """
    if not lineage:
        return

    G = nx.DiGraph()

    # Add nodes
    for entry in lineage:
        G.add_node(entry['id'], reward=entry['reward'], island=entry['island'])

    # Add edges (parent -> child)
    node_ids = set(entry['id'] for entry in lineage)
    for entry in lineage:
        if entry['parent_id'] is not None and entry['parent_id'] in node_ids:
            G.add_edge(entry['parent_id'], entry['id'])

    num_nodes = len(G.nodes())
    fig_h = max(10, 4 + num_nodes * 0.15)
    fig, ax = plt.subplots(figsize=(max(14, num_nodes * 0.3), fig_h))

    # Layout
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

    # Reward color mapping
    rewards = [G.nodes[n]['reward'] for n in G.nodes()]
    vmin, vmax = min(rewards), max(rewards)
    if vmin == vmax:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm_module.get_cmap('viridis')

    # Best node
    best_node = max(G.nodes(), key=lambda n: G.nodes[n]['reward'])

    # Path from root to best
    best_path_edges = []
    try:
        path = nx.shortest_path(G, root, best_node)
        best_path_edges = list(zip(path[:-1], path[1:]))
    except nx.NetworkXNoPath:
        pass

    # Draw regular edges
    regular_edges = [e for e in G.edges() if e not in best_path_edges]
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, arrows=True,
        arrowsize=10, width=1.0, edge_color='gray', alpha=0.5, ax=ax)
    # Draw best-path edges
    if best_path_edges:
        nx.draw_networkx_edges(G, pos, edgelist=best_path_edges, arrows=True,
            arrowsize=14, width=2.5, edge_color='black', alpha=0.8, ax=ax)

    # Draw nodes per-island
    size_factor = max(0.4, min(1.0, 15 / (num_nodes ** 0.4)))
    base_size = int(600 * size_factor)
    best_size = int(1200 * size_factor)

    for node in G.nodes():
        attrs = G.nodes[node]
        island = attrs['island']
        reward = attrs['reward']
        edge_color = ISLAND_EDGE_COLORS[island % len(ISLAND_EDGE_COLORS)]

        if node == best_node:
            node_color = 'gold'
            node_size = best_size
            shape = '*'
            lw = 2.5
        else:
            node_color = mcolors.to_hex(cmap(norm(reward)))
            node_size = base_size
            shape = 'o'
            lw = 2.0

        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=node_size,
            node_color=node_color, edgecolors=edge_color, linewidths=lw,
            node_shape=shape, ax=ax)

    # Labels: show iteration number
    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=max(6, int(10 * size_factor)),
        font_weight='bold', font_color='white', ax=ax)

    # Colorbar
    sm = cm_module.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.6)
    cb.set_label('Reward', size=14, weight='bold')

    # Legend for islands
    num_islands = max(entry['island'] for entry in lineage) + 1
    handles = []
    for isl in range(num_islands):
        handles.append(Line2D([0], [0], marker='o', color='w',
            markerfacecolor='gray', markeredgecolor=ISLAND_EDGE_COLORS[isl % len(ISLAND_EDGE_COLORS)],
            markersize=10, markeredgewidth=2.5, label=f'Island {isl}'))
    handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
        markersize=16, label='Best'))
    handles.append(Line2D([0], [0], color='black', linewidth=2.5, label='Path to Best'))
    ax.legend(handles=handles, loc='upper right', fontsize=10)

    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _hierarchical_layout(G, roots):
    """Simple hierarchical layout fallback when graphviz is unavailable."""
    pos = {}
    if not roots:
        roots = [list(G.nodes())[0]]
    # BFS to assign layers
    visited = set()
    layers = {}
    queue = [(r, 0) for r in roots]
    for r in roots:
        visited.add(r)
    while queue:
        node, depth = queue.pop(0)
        if depth not in layers:
            layers[depth] = []
        layers[depth].append(node)
        for child in G.successors(node):
            if child not in visited:
                visited.add(child)
                queue.append((child, depth + 1))
    # Assign any unvisited nodes
    for node in G.nodes():
        if node not in visited:
            depth = 0
            if depth not in layers:
                layers[depth] = []
            layers[depth].append(node)
    # Position
    for depth, nodes in layers.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (n - 1) / 2.0) * 2.0
            y = -depth * 2.0
            pos[node] = (x, y)
    return pos


def save_lineage_json(lineage, output_path):
    """Save lineage data to JSON for later re-plotting."""
    with open(output_path, 'w') as f:
        json.dump(lineage, f, indent=2)


if __name__ == '__main__':
    """Re-plot from a saved lineage JSON file."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lineage_json', help='Path to lineage.json')
    parser.add_argument('--output', default=None, help='Output PNG path')
    args = parser.parse_args()
    with open(args.lineage_json) as f:
        lineage = json.load(f)
    out = args.output or args.lineage_json.replace('.json', '.png')
    plot_lineage_tree(lineage, out)
    print(f"Saved to {out}")
