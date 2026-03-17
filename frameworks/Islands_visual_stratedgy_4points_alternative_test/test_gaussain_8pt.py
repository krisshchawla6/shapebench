"""
Test script: Generate 5 new designs using gaussain action with example_parent as parent.
Builds context exactly like run_benchmark_action.py's generate_design().
"""
import os
import sys
import json
import numpy as np

# Setup paths exactly like run_benchmark_action.py
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'modified_env'))
sys.path.insert(0, os.path.join(base_dir, 'modified_env', 'LLM_Actions'))

from LLM_agent import run_llm_action

# === Config ===
EXAMPLE_PARENT_DIR = os.path.join(base_dir, 'example_parent')
OUTPUT_DIR = os.path.join(base_dir, 'test_gaussain_8pt_output')
N_DESIGNS = 5
ACTION = 'gaussain'

# === Build parent action CSV (Design 1 from context.txt) ===
parent_action = [1.0, 0.004328, -0.93674, 0.833076, 0.303893, -0.690358,
                 0.879421, 0.670615, -0.250714, 1.0, 0.937862, 0.125332,
                 0.932845, 0.870981, 1.0, -1.0, 0.935066, 0.175294,
                 -0.853371, 0.498291, -0.355829, -0.96408, 0.011296, -0.87506]
parent_reward = 1.0360
parent_rank = 0

# Save parent action CSV
os.makedirs(OUTPUT_DIR, exist_ok=True)
parent_csv = os.path.join(OUTPUT_DIR, 'parent_action.csv')
np.savetxt(parent_csv, [parent_action], delimiter=',', fmt='%.6f')

# === Build parent results exactly like post_process_results() ===
save_dir = os.path.join(EXAMPLE_PARENT_DIR, 'save')

# drag/lift from last line of drag_lift
drag, lift = 0.0, 0.0
dl_path = os.path.join(save_dir, 'drag_lift')
if os.path.exists(dl_path):
    with open(dl_path, 'r') as f:
        lines = f.readlines()
        if lines:
            last = lines[-1].split()
            if len(last) >= 3:
                drag = float(last[1])
                lift = float(last[2])

# sol images
sol_images = [
    os.path.join(save_dir, 'sol', '1_p.png'),
    os.path.join(save_dir, 'sol', '1_u.png'),
    os.path.join(save_dir, 'sol', '1_v.png')
]

# shape image
shape_image = None
png_dir = os.path.join(save_dir, 'png')
if os.path.exists(png_dir):
    shape_pngs = sorted([f for f in os.listdir(png_dir)
                         if f.startswith('shape_') and f.endswith('.png')])
    if shape_pngs:
        shape_image = os.path.join(png_dir, shape_pngs[-1])

# feedback (analysis text) - use existing if available
feedback = ""
analysis_path = os.path.join(EXAMPLE_PARENT_DIR, 'context', 'llm_analysis.txt')
if os.path.exists(analysis_path):
    with open(analysis_path, 'r') as f:
        feedback = f.read().strip()

# === Build context exactly like generate_design() in run_benchmark_action.py ===
parent_images = []
if shape_image and os.path.exists(shape_image):
    parent_images.append(shape_image)
for sol_img in sol_images:
    if sol_img and os.path.exists(sol_img):
        parent_images.append(sol_img)

llm_context = [{
    'vector': parent_action,
    'reward': parent_reward,
    'ranking': parent_rank,
    'drag': drag,
    'lift': lift,
    'feedback': feedback,
    'images': parent_images
}]
# 0 inspirations - only parent

print("=" * 60)
print(f"Parent action vector: {len(parent_action)} values (8 control points)")
print(f"Parent reward: {parent_reward}")
print(f"Parent drag: {drag:.4f}, lift: {lift:.4f}")
print(f"Parent images: {len(parent_images)} ({[os.path.basename(p) for p in parent_images]})")
print(f"Feedback: {feedback[:100]}..." if len(feedback) > 100 else f"Feedback: {feedback}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Action: {ACTION}")
print(f"Generating {N_DESIGNS} designs...")
print("=" * 60)

# === Generate N_DESIGNS new designs ===
for i in range(N_DESIGNS):
    print(f"\n--- Design {i+1}/{N_DESIGNS} ---")
    design_name = f"design_{i}"
    debug_dir = os.path.join(OUTPUT_DIR, design_name, 'context')

    csv_path = run_llm_action(
        ACTION,
        llm_context,
        OUTPUT_DIR,
        base_csv=parent_csv,
        name=design_name,
        skip_vis=True,
        debug_dir=debug_dir,
        random_strategy=True
    )

    if csv_path and os.path.exists(csv_path):
        action = np.loadtxt(csv_path, delimiter=',')
        if action.ndim == 2:
            action = action[0]
        print(f"  Generated: {csv_path}")
        print(f"  Action size: {len(action)} ({len(action)//3} control points)")
        print(f"  Values: [{', '.join(f'{v:.4f}' for v in action[:6])}...{', '.join(f'{v:.4f}' for v in action[-3:])}]")
    else:
        print(f"  FAILED: No CSV generated")

print("\n" + "=" * 60)
print(f"Done. Results in: {OUTPUT_DIR}")
print("=" * 60)
