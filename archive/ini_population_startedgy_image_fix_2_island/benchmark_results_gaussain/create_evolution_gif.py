#!/usr/bin/env python3
"""Create an animated GIF showing the evolutionary progression of CFD designs
with a lineage tree visualization and smooth crossfade transitions."""

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
ISLAND_COLORS = {
    0: (66, 133, 244),   # Blue
    1: (234, 67, 53),    # Red
    2: (52, 168, 83),    # Green
}
ISLAND_NAMES = {0: "Island A", 1: "Island B", 2: "Island C"}

# Layout: wider canvas to fit tree on the right
CANVAS_W, CANVAS_H = 1100, 560
IMG_X, IMG_Y = 24, 72
IMG_W, IMG_H = 440, 293
INFO_X = IMG_X + IMG_W + 14
INFO_W = 130
TREE_X = INFO_X + INFO_W + 8
TREE_Y = 72
TREE_W = CANVAS_W - TREE_X - 12
TREE_H = 380
GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H = 24, 478, CANVAS_W - 48, 68
BORDER = 3

# Crossfade settings: more steps = smoother
FADE_STEPS = 6
FADE_MS = 35
HOLD_MS_NORMAL = 250
HOLD_MS_BEST = 900
HOLD_MS_FIRST = 600
HOLD_MS_LAST = 1800

# --- Font helpers ---

def get_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def get_font_regular(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


# --- Pre-cache images to avoid reloading during crossfade ---

_img_cache = {}

def load_design_image(design_id, size=(IMG_W, IMG_H)):
    key = (design_id, size)
    if key not in _img_cache:
        path = BASE_DIR / f"design_{design_id}" / "save" / "sol" / "1_u.png"
        if path.exists():
            img = Image.open(path).convert("RGB")
        else:
            img = Image.new("RGB", (500, 333), (40, 40, 50))
            d = ImageDraw.Draw(img)
            d.text((250, 166), f"Design #{design_id}\n(no image)", fill=(120, 120, 140),
                   font=get_font(18), anchor="mm")
        _img_cache[key] = img.resize(size, Image.LANCZOS)
    return _img_cache[key]


def load_lineage():
    with open(BASE_DIR / "lineage.json") as f:
        return json.load(f)


def draw_rr(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


# --- Tree layout computation (done once) ---

def compute_tree_layout(lineage):
    """Compute (x, y) positions for each design in a top-down tree layout.
    X = depth from root, Y = spread within each depth level."""
    lookup = {d["id"]: d for d in lineage}
    children = defaultdict(list)
    roots = []
    for d in lineage:
        if d["parent_id"] is None:
            roots.append(d["id"])
        else:
            children[d["parent_id"]].append(d["id"])

    # Sort children by ID for consistent ordering
    for k in children:
        children[k].sort()

    # Compute depth for each node
    depth = {}
    def set_depth(nid, d):
        depth[nid] = d
        for c in children[nid]:
            set_depth(c, d + 1)
    for r in roots:
        set_depth(r, 0)

    max_depth = max(depth.values()) if depth else 0

    # Assign y positions via a recursive walk that preserves subtree ordering
    y_counter = [0]
    pos = {}

    def layout(nid):
        kids = children[nid]
        if not kids:
            pos[nid] = (depth[nid], y_counter[0])
            y_counter[0] += 1
        else:
            child_ys = []
            for c in kids:
                layout(c)
                child_ys.append(pos[c][1])
            mid_y = (child_ys[0] + child_ys[-1]) / 2.0
            pos[nid] = (depth[nid], mid_y)

    for r in sorted(roots):
        layout(r)

    total_leaves = y_counter[0]
    return pos, children, roots, max_depth, total_leaves


def draw_lineage_tree(draw, lineage, current_idx, pos, children_map, roots,
                      max_depth, total_leaves, x, y, w, h):
    """Draw the lineage tree with current design highlighted."""
    lookup = {d["id"]: d for d in lineage}
    current_id = lineage[current_idx]["id"]
    revealed = {d["id"] for d in lineage[:current_idx + 1]}

    # Trace ancestors of current design
    ancestors = set()
    cid = current_id
    while cid is not None:
        ancestors.add(cid)
        cid = lookup[cid]["parent_id"]

    # Background panel
    draw_rr(draw, (x, y, x + w, y + h), radius=8, fill=(20, 20, 30, 220))

    # Label
    font_label = get_font_regular(11)
    draw.text((x + w // 2, y + 6), "Lineage Tree", fill=(160, 160, 180),
              font=font_label, anchor="mt")

    # Mapping from tree coordinates to pixel coordinates
    pad_x, pad_y = 20, 22
    tree_area_w = w - 2 * pad_x
    tree_area_h = h - pad_y - 14

    def to_px(dx, dy):
        if max_depth == 0:
            px = x + pad_x + tree_area_w // 2
        else:
            px = x + pad_x + int(dx / max_depth * tree_area_w)
        if total_leaves <= 1:
            py = y + pad_y + tree_area_h // 2
        else:
            py = y + pad_y + int(dy / (total_leaves - 1) * tree_area_h)
        return px, py

    # Draw edges first (only for revealed nodes)
    for d in lineage[:current_idx + 1]:
        nid = d["id"]
        if nid not in pos:
            continue
        for cid in children_map.get(nid, []):
            if cid not in revealed or cid not in pos:
                continue
            px1, py1 = to_px(*pos[nid])
            px2, py2 = to_px(*pos[cid])
            # Highlight path to current
            if nid in ancestors and cid in ancestors:
                edge_color = (255, 215, 0, 200)
                edge_w = 2
            else:
                edge_color = (60, 60, 80, 140)
                edge_w = 1
            # Bezier-ish: horizontal step
            mid_x = (px1 + px2) // 2
            draw.line([(px1, py1), (mid_x, py1), (mid_x, py2), (px2, py2)],
                      fill=edge_color, width=edge_w)

    # Draw nodes
    node_r = 4
    for d in lineage[:current_idx + 1]:
        nid = d["id"]
        if nid not in pos:
            continue
        px, py = to_px(*pos[nid])
        island = d["island"]
        color = ISLAND_COLORS[island]

        if nid == current_id:
            # Current: large bright node with white ring
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=color,
                         outline=(255, 255, 255), width=2)
        elif nid in ancestors:
            # Ancestor: medium gold-outlined node
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color,
                         outline=(255, 215, 0), width=1)
        else:
            # Regular: small dim node
            dim = tuple(max(c // 2, 30) for c in color)
            draw.ellipse((px - node_r, py - node_r, px + node_r, py + node_r),
                         fill=dim)

    # Label current node
    if current_id in pos:
        px, py = to_px(*pos[current_id])
        font_tiny = get_font(9)
        draw.text((px, py - 11), f"#{current_id}", fill=(255, 255, 255),
                  font=font_tiny, anchor="mb")


def draw_fitness_graph(draw, lineage, current_idx, x, y, w, h):
    """Draw a mini fitness-over-time graph."""
    rewards = [d["reward"] for d in lineage]
    r_min, r_max = min(rewards), max(rewards)
    r_range = r_max - r_min if r_max != r_min else 1.0
    n = len(lineage)

    draw_rr(draw, (x, y, x + w, y + h), radius=8, fill=(20, 20, 30, 200))

    for frac in [0.25, 0.5, 0.75]:
        gy = y + h - int(frac * (h - 16)) - 8
        draw.line([(x + 8, gy), (x + w - 8, gy)], fill=(50, 50, 65), width=1)

    # Best-so-far line
    best_so_far = []
    cur_best = -999
    for d in lineage[:current_idx + 1]:
        cur_best = max(cur_best, d["reward"])
        best_so_far.append(cur_best)

    def idx_to_px(i, val):
        px = x + 10 + int(i / (n - 1) * (w - 20))
        py = y + h - 8 - int((val - r_min) / r_range * (h - 16))
        return px, py

    if len(best_so_far) > 1:
        pts = [idx_to_px(i, v) for i, v in enumerate(best_so_far)]
        draw.line(pts, fill=(255, 215, 0, 160), width=2)

    for i in range(current_idx + 1):
        d = lineage[i]
        px, py = idx_to_px(i, d["reward"])
        color = ISLAND_COLORS[d["island"]]
        r = 2 if i < current_idx else 4
        alpha = 100 if i < current_idx else 255
        draw.ellipse((px - r, py - r, px + r, py + r), fill=(*color, alpha))

    # Current highlight
    d = lineage[current_idx]
    px, py = idx_to_px(current_idx, d["reward"])
    draw.ellipse((px - 6, py - 6, px + 6, py + 6),
                 outline=(255, 255, 255, 180), width=2)

    font_sm = get_font_regular(11)
    draw.text((x + w - 8, y + 3), "Fitness over time", fill=(140, 140, 160),
              font=font_sm, anchor="rt")


def create_frame(lineage, design_idx, tree_layout):
    """Create a single animation frame."""
    pos, children_map, roots, max_depth, total_leaves = tree_layout
    d = lineage[design_idx]
    design_id = d["id"]
    reward = d["reward"]
    island = d["island"]
    parent_id = d["parent_id"]

    best_reward = max(e["reward"] for e in lineage[:design_idx + 1])
    is_new_best = reward == best_reward and (
        design_idx == 0 or reward > max(e["reward"] for e in lineage[:design_idx])
    )

    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (15, 15, 25, 255))
    draw = ImageDraw.Draw(canvas)

    font_title = get_font(20)
    font_info = get_font(15)
    font_sm = get_font_regular(12)
    font_badge = get_font(11)

    # --- Title bar ---
    draw_rr(draw, (0, 0, CANVAS_W, 56), radius=0, fill=(25, 25, 40))
    draw.text((18, 7), "Evolutionary Design Optimization",
              fill=(255, 255, 255), font=font_title)
    draw.text((CANVAS_W - 18, 7), f"Design {design_idx + 1} / {len(lineage)}",
              fill=(160, 160, 180), font=font_sm, anchor="rt")

    # Progress bar
    bar_y, bar_h = 38, 7
    bar_x, bar_w = 18, CANVAS_W - 36
    draw_rr(draw, (bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=3,
            fill=(50, 50, 70))
    fill_w = int((design_idx + 1) / len(lineage) * bar_w)
    if fill_w > 0:
        draw_rr(draw, (bar_x, bar_y, bar_x + fill_w, bar_y + bar_h),
                radius=3, fill=ISLAND_COLORS[island])

    # --- Design image ---
    border_color = ISLAND_COLORS[island]
    if is_new_best:
        for off in range(5, 0, -1):
            a = int(35 * (6 - off) / 5)
            draw_rr(draw,
                    (IMG_X - off, IMG_Y - off, IMG_X + IMG_W + off, IMG_Y + IMG_H + off),
                    radius=7 + off, fill=None, outline=(255, 215, 0, a), width=2)

    draw_rr(draw,
            (IMG_X - BORDER, IMG_Y - BORDER, IMG_X + IMG_W + BORDER, IMG_Y + IMG_H + BORDER),
            radius=6, fill=(*border_color, 255))
    design_img = load_design_image(design_id)
    canvas.paste(design_img, (IMG_X, IMG_Y))

    # --- Info panel (narrow column between image and tree) ---
    px = INFO_X
    pw = INFO_W

    # Island badge
    draw_rr(draw, (px, IMG_Y, px + pw, IMG_Y + 24), radius=5,
            fill=(*border_color, 220))
    draw.text((px + pw // 2, IMG_Y + 12), ISLAND_NAMES[island],
              fill=(255, 255, 255), font=font_badge, anchor="mm")

    # Design ID
    draw.text((px, IMG_Y + 34), f"#{design_id}",
              fill=(255, 255, 255), font=get_font(24))

    # Fitness
    draw.text((px, IMG_Y + 68), "Fitness", fill=(130, 130, 150), font=font_sm)
    rc = (100, 255, 100) if reward > 0 else (255, 100, 100)
    draw.text((px, IMG_Y + 83), f"{reward:.2f}", fill=rc, font=font_info)

    # Parent
    draw.text((px, IMG_Y + 110), "Parent", fill=(130, 130, 150), font=font_sm)
    ps = f"#{parent_id}" if parent_id is not None else "Initial"
    draw.text((px, IMG_Y + 125), ps, fill=(200, 200, 220), font=font_info)

    # Best so far
    draw.text((px, IMG_Y + 154), "Best", fill=(130, 130, 150), font=font_sm)
    draw.text((px, IMG_Y + 169), f"{best_reward:.2f}", fill=(255, 215, 0), font=font_info)

    # New best badge
    if is_new_best:
        ny = IMG_Y + 200
        draw_rr(draw, (px, ny, px + pw, ny + 22), radius=5,
                fill=(255, 215, 0, 230))
        draw.text((px + pw // 2, ny + 11), "NEW BEST!", fill=(20, 20, 30),
                  font=font_badge, anchor="mm")

    # Island legend
    ly = IMG_Y + IMG_H - 60
    draw.text((px, ly), "Islands", fill=(130, 130, 150), font=font_sm)
    for isl in range(3):
        iy = ly + 18 + isl * 16
        c = ISLAND_COLORS[isl]
        draw.ellipse((px, iy, px + 10, iy + 10), fill=c)
        draw.text((px + 14, iy - 1), ISLAND_NAMES[isl],
                  fill=(180, 180, 200), font=get_font_regular(10))

    # --- Lineage tree ---
    draw_lineage_tree(draw, lineage, design_idx, pos, children_map, roots,
                      max_depth, total_leaves, TREE_X, TREE_Y, TREE_W, TREE_H)

    # --- Fitness graph ---
    draw_fitness_graph(draw, lineage, design_idx, GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H)

    return canvas.convert("RGB")


def main():
    print("Loading lineage data...")
    lineage = load_lineage()
    print(f"Found {len(lineage)} designs")

    print("Computing tree layout...")
    tree_layout = compute_tree_layout(lineage)

    print("Pre-loading all design images...")
    for d in lineage:
        load_design_image(d["id"])
    print(f"  Cached {len(_img_cache)} images")

    # --- Render key frames ---
    print("Rendering key frames...")
    key_frames = []
    for i in range(len(lineage)):
        print(f"  Frame {i}/{len(lineage)-1} (#{lineage[i]['id']}, "
              f"reward={lineage[i]['reward']:.3f})")
        key_frames.append(create_frame(lineage, i, tree_layout))

    # --- Build final frame list with smooth crossfades ---
    print("Building smooth animation with crossfades...")
    frames = []
    durations = []

    for i in range(len(key_frames)):
        best_so_far = max(d["reward"] for d in lineage[:i + 1])
        is_new_best = lineage[i]["reward"] == best_so_far and (
            i == 0 or lineage[i]["reward"] > max(d["reward"] for d in lineage[:i])
        )

        # Crossfade from previous
        if i > 0:
            prev = key_frames[i - 1]
            curr = key_frames[i]
            for step in range(1, FADE_STEPS + 1):
                alpha = step / (FADE_STEPS + 1)
                blended = Image.blend(prev, curr, alpha)
                frames.append(blended)
                durations.append(FADE_MS)

        # Hold on key frame
        frames.append(key_frames[i])
        if is_new_best:
            durations.append(HOLD_MS_BEST)
        elif i == 0:
            durations.append(HOLD_MS_FIRST)
        elif i == len(key_frames) - 1:
            durations.append(HOLD_MS_LAST)
        else:
            durations.append(HOLD_MS_NORMAL)

    # --- Save main GIF ---
    out = BASE_DIR / "evolution_animation.gif"
    print(f"\nSaving {out} ({len(frames)} frames)...")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=False)
    print(f"  Done! {out}")

    # --- Best-lineage-only GIF ---
    print("\nCreating best-lineage animation...")
    lookup = {d["id"]: d for d in lineage}
    best_id = max(lineage, key=lambda d: d["reward"])["id"]
    chain = []
    cid = best_id
    while cid is not None:
        chain.append(cid)
        cid = lookup[cid]["parent_id"]
    chain.reverse()
    id_to_idx = {d["id"]: idx for idx, d in enumerate(lineage)}

    chain_kf = []
    for did in chain:
        idx = id_to_idx[did]
        print(f"  Chain: #{did} (reward={lookup[did]['reward']:.3f})")
        chain_kf.append(create_frame(lineage, idx, tree_layout))

    cf, cd = [], []
    for ci in range(len(chain_kf)):
        if ci > 0:
            for step in range(1, 8):
                alpha = step / 8
                cf.append(Image.blend(chain_kf[ci - 1], chain_kf[ci], alpha))
                cd.append(50)
        cf.append(chain_kf[ci])
        cd.append(2000 if ci == len(chain_kf) - 1 else (800 if ci == 0 else 500))

    out2 = BASE_DIR / "evolution_best_lineage.gif"
    print(f"Saving {out2} ({len(cf)} frames)...")
    cf[0].save(out2, save_all=True, append_images=cf[1:],
               duration=cd, loop=0, optimize=False)
    print(f"  Done! {out2}")
    print(f"  Best: #{best_id} ({lookup[best_id]['reward']:.3f})")
    print(f"  Chain: {' -> '.join(f'#{d}' for d in chain)}")


if __name__ == "__main__":
    main()
