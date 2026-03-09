#!/usr/bin/env python3
"""
Test all 4 LLM actions with real context from a benchmark run.
Generates designs and produces PNG visualizations.

Usage:
    python test_all_actions_with_context.py --context-dir <path_to_design_context> --n-designs 10 --output-dir ./test_outputs
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLM_agent import run_llm_action
from shapes_utils import Shape


def load_context_from_dir(context_dir: str):
    """Load context from a benchmark output directory.
    
    Expected structure:
        context_dir/
            design_X_0.csv (or any *_0.csv action file)
            save/
                reward_penalization
                png/shape_0.png (optional)
    """
    context = []
    
    # Find action CSV files (flat format, not shape CSVs)
    csv_files = []
    for f in os.listdir(context_dir):
        if f.endswith('.csv') and 'shape' not in f.lower():
            csv_path = os.path.join(context_dir, f)
            # Check if it's a flat action CSV (single line, 12 values)
            try:
                with open(csv_path, 'r') as fp:
                    first_line = fp.readline().strip()
                    vals = first_line.split(',')
                    if len(vals) == 12:  # 4 control points x 3 params
                        csv_files.append(csv_path)
            except:
                pass
    
    if not csv_files:
        print(f"No action CSVs found in {context_dir}")
    
    for csv_path in csv_files:
        try:
            action_vector = np.loadtxt(csv_path, delimiter=',')
            if action_vector.ndim > 1:
                action_vector = action_vector.flatten()
            
            # Read reward from save/reward_penalization
            reward = 0.0
            reward_file = os.path.join(context_dir, 'save', 'reward_penalization')
            if os.path.exists(reward_file):
                with open(reward_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Last line, column 2 is the reward
                        reward = float(lines[-1].split()[1])
            
            # Check for shape image
            images = []
            shape_png = os.path.join(context_dir, 'save', 'png', 'shape_0.png')
            if os.path.exists(shape_png):
                images.append(shape_png)
            
            context.append({
                'vector': action_vector.tolist(),
                'reward': reward,
                'ranking': len(context),
                'images': images
            })
            print(f"Loaded: {os.path.basename(csv_path)}, reward={reward:.4f}")
            
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
    
    if context:
        print(f"Loaded {len(context)} designs from {context_dir}")
        return context
    
    # Fallback: use baseline
    print("Using baseline context")
    baseline_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  'baseline_action.csv')
    
    if os.path.exists(baseline_path):
        action_vector = np.loadtxt(baseline_path, delimiter=',')
        if action_vector.ndim > 1:
            action_vector = action_vector.flatten()
        
        return [{
            'vector': action_vector.tolist(),
            'reward': 0.0264,
            'ranking': 0,
            'images': []
        }]
    else:
        # Absolute fallback
        return [{
            'vector': [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            'reward': 0.0264,
            'ranking': 0,
            'images': []
        }]


def action_to_shape(action_csv: str) -> Shape:
    """Convert action CSV to Shape object for visualization."""
    shape = Shape()
    
    # Load baseline
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    baseline_path = os.path.join(base_dir, 'reset', '4', 'shape_0.csv')
    
    if not os.path.exists(baseline_path):
        # Try alternative path
        baseline_path = os.path.join(base_dir, 'baseline_action.csv')
        if os.path.exists(baseline_path):
            # This is already an action, use circular baseline
            shape.n_control_pts = 4
            shape.n_sampling_pts = 10
    else:
        shape.read_csv(baseline_path)
    
    # Load action and convert to deformation
    action = np.loadtxt(action_csv, delimiter=',')
    if action.ndim > 1:
        action = action.flatten()
    
    n_pts = len(action) // 3
    MAX_DEFORMATION = 3.0
    
    deformation = np.zeros((n_pts, 3))
    action_reshaped = action.reshape((n_pts, 3))
    dangle = 360.0 / float(n_pts)
    
    for i in range(n_pts):
        radius_param = action_reshaped[i, 0]
        angle_param = action_reshaped[i, 1]
        edgy_param = action_reshaped[i, 2]
        
        # Environment conversion
        import math
        radius = max(abs(radius_param), 0.2) * MAX_DEFORMATION
        angle = dangle * float(i) + angle_param * dangle / 2.0
        x = radius * math.cos(math.radians(angle))
        y = radius * math.sin(math.radians(angle))
        edgy = 0.5 + 0.5 * abs(edgy_param)
        
        deformation[i, 0] = x
        deformation[i, 1] = y
        deformation[i, 2] = edgy
    
    # Apply deformation
    all_pts_list = list(range(n_pts))
    shape.modify_shape_from_field(deformation, replace=True, pts_list=all_pts_list)
    shape.generate(centering=False)
    
    return shape


def generate_shape_png(csv_path: str, output_png: str):
    """Generate shape visualization PNG from action CSV."""
    try:
        shape = action_to_shape(csv_path)
        
        # Generate image
        plt.figure(figsize=(8, 6))
        plt.xlim([-15, 30])
        plt.ylim([-15, 15])
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Plot shape
        if hasattr(shape, 'curve_pts') and shape.curve_pts is not None:
            plt.plot(shape.curve_pts[:, 0], shape.curve_pts[:, 1], 'b-', linewidth=2)
            plt.fill(shape.curve_pts[:, 0], shape.curve_pts[:, 1], 'lightblue', alpha=0.5)
        
        # Plot control points
        if hasattr(shape, 'control_pts') and shape.control_pts is not None:
            plt.plot(shape.control_pts[:, 0], shape.control_pts[:, 1], 'ro', markersize=8)
        
        plt.savefig(output_png, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Warning: Failed to generate PNG for {csv_path}: {e}")
        return False


def test_action(action_name: str, context: list, output_dir: str, 
                n_designs: int = 10, base_csv: str = None):
    """Test a single action type with multiple generations."""
    print(f"\n{'='*60}")
    print(f"Testing action: {action_name}")
    print(f"{'='*60}")
    
    action_output_dir = os.path.join(output_dir, action_name)
    os.makedirs(action_output_dir, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for i in range(n_designs):
        print(f"\n[{i+1}/{n_designs}] Generating design_{i}...")
        
        try:
            csv_path = run_llm_action(
                action=action_name,
                context=context,
                output_dir=action_output_dir,
                base_csv=base_csv,
                name=f'design_{i}',
                temperature=1.0,
                skip_vis=True
            )
            
            if csv_path and os.path.exists(csv_path):
                print(f"  ✓ CSV generated: {os.path.basename(csv_path)}")
                
                # Generate PNG visualization
                png_path = csv_path.replace('.csv', '_geometry.png')
                if generate_shape_png(csv_path, png_path):
                    print(f"  ✓ PNG generated: {os.path.basename(png_path)}")
                else:
                    print(f"  ✗ PNG generation failed")
                
                successful += 1
            else:
                print(f"  ✗ Generation failed")
                failed += 1
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
    
    print(f"\n{action_name} Summary: {successful} successful, {failed} failed")
    return successful, failed


def main():
    parser = argparse.ArgumentParser(description='Test all LLM actions with context')
    parser.add_argument('--context-dir', type=str, 
                        default='/scratch/LLM_Evolve/AirFoil_becnhmark/benchmark_results_generate_direct/design_3/context',
                        help='Path to context directory from a benchmark run')
    parser.add_argument('--n-designs', type=int, default=10,
                        help='Number of designs to generate per action')
    parser.add_argument('--output-dir', type=str, 
                        default='/scratch/LLM_Evolve/AirFoil_becnhmark/modified_env/LLM_Actions/test_all_actions_output',
                        help='Output directory for test results')
    parser.add_argument('--actions', nargs='+', 
                        default=['generate', 'generate_direct', 'modify', 'modify_direct'],
                        help='Actions to test')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LLM Actions Test Suite")
    print("="*60)
    print(f"Context source: {args.context_dir}")
    print(f"Designs per action: {args.n_designs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Actions to test: {', '.join(args.actions)}")
    
    # Load context
    context = load_context_from_dir(args.context_dir)
    print(f"\nLoaded context with {len(context)} designs")
    for i, ctx in enumerate(context):
        print(f"  Design {i}: Reward={ctx['reward']:.6f}, Rank=#{ctx['ranking']}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get base CSV for modify actions
    base_csv = None
    if context and 'vector' in context[0]:
        base_csv_path = os.path.join(args.output_dir, 'base_action.csv')
        np.savetxt(base_csv_path, [context[0]['vector']], delimiter=',', fmt='%.6f')
        base_csv = base_csv_path
        print(f"\nSaved base CSV for modify actions: {base_csv}")
    
    # Test each action
    results = {}
    for action in args.actions:
        successful, failed = test_action(
            action_name=action,
            context=context,
            output_dir=args.output_dir,
            n_designs=args.n_designs,
            base_csv=base_csv if 'modify' in action else None
        )
        results[action] = {'successful': successful, 'failed': failed}
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for action, res in results.items():
        total = res['successful'] + res['failed']
        success_rate = res['successful'] / total * 100 if total > 0 else 0
        print(f"{action:20s}: {res['successful']}/{total} successful ({success_rate:.1f}%)")
    
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("\nTo view results:")
    print(f"  ls -la {args.output_dir}/*/")
    

if __name__ == '__main__':
    main()
