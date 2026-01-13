#!/usr/bin/env python3
"""
Parallel batch runner for executing Flow360 simulations across multiple design folders.
Launches multiple simulations in parallel for faster processing.
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


def find_design_folders(designs_dir):
    """Find all design folders containing STL files."""
    design_folders = []
    for item in designs_dir.iterdir():
        if item.is_dir() and item.name.startswith('design_'):
            # Check if folder has any STL files
            stl_files = list(item.glob('*.stl'))
            if stl_files:
                design_folders.append(item)
    return sorted(design_folders)


def find_stl_files(design_folder):
    """Find all STL files in a design folder."""
    return sorted(design_folder.glob('*.stl'))


def run_simulation_worker(task_info):
    """Worker function to run a single simulation (called in separate process)."""
    stl_path, params_path, runner_script, project_root, task_id = task_info
    
    # Use relative paths from project root
    stl_rel = stl_path.relative_to(project_root)
    params_rel = params_path.relative_to(project_root)
    runner_rel = runner_script.relative_to(project_root)
    
    cmd = ["python", str(runner_rel), str(stl_rel), str(params_rel)]
    
    start_time = datetime.now()
    print(f"[Task {task_id}] 🚀 Starting: {stl_path.name}")
    print(f"[Task {task_id}]    Command: {' '.join(cmd)}")
    
    try:
        # Run without capturing output - let it stream
        result = subprocess.run(cmd, check=True, cwd=project_root)
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[Task {task_id}] Completed: {stl_path.name} ({elapsed:.1f}s)\n")
        return {
            'success': True,
            'stl': stl_path.name,
            'design': stl_path.parent.name,
            'task_id': task_id,
            'elapsed': elapsed
        }
    except subprocess.CalledProcessError as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[Task {task_id}] Failed: {stl_path.name} (exit code {e.returncode}, {elapsed:.1f}s)\n")
        return {
            'success': False,
            'stl': stl_path.name,
            'design': stl_path.parent.name,
            'task_id': task_id,
            'elapsed': elapsed,
            'error': e.returncode
        }
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[Task {task_id}] Exception: {stl_path.name} - {e}\n")
        return {
            'success': False,
            'stl': stl_path.name,
            'design': stl_path.parent.name,
            'task_id': task_id,
            'elapsed': elapsed,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Parallel batch run Flow360 simulations for multiple designs')
    parser.add_argument('designs_dir', help='Path to designs directory containing design folders')
    parser.add_argument('params_json', help='Path to parameters JSON file to use for all simulations')
    parser.add_argument('--max-workers', type=int, default=3, 
                       help='Maximum number of parallel simulations (default: 3)')
    args = parser.parse_args()
    
    # Get project root and resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Resolve designs directory
    designs_dir = Path(args.designs_dir)
    if not designs_dir.is_absolute():
        designs_dir = project_root / designs_dir
    
    if not designs_dir.exists():
        print(f"Designs directory not found: {designs_dir}")
        return 1
    
    # Resolve params file
    params_path = Path(args.params_json)
    if not params_path.is_absolute():
        params_path = project_root / params_path
    
    if not params_path.exists():
        print(f"Parameters file not found: {params_path}")
        return 1
    
    # Locate runner script
    runner_script = project_root / "simulation_agents" / "Aerospace" / "External_flow" / "run_sweep.py"
    if not runner_script.exists():
        print(f"Runner script not found: {runner_script}")
        return 1
    
    try:
        print(f" Designs Directory: {designs_dir.name}")
        print(f" Parameters: {params_path.name}")
        print(f" Runner: {runner_script.name}")
        print(f" Max Parallel Workers: {args.max_workers}")
        print()
        
        # Find design folders
        design_folders = find_design_folders(designs_dir)
        if not design_folders:
            print("  No design folders found with STL files")
            return 1
        
        print(f" Found {len(design_folders)} design folder(s)")
        print()
        
        # Build list of all tasks (STL files to process)
        tasks = []
        task_id = 1
        for design_folder in design_folders:
            stl_files = find_stl_files(design_folder)
            for stl_file in stl_files:
                tasks.append((stl_file, params_path, runner_script, project_root, task_id))
                task_id += 1
        
        if not tasks:
            print("  No STL files found to process")
            return 1
        
        print(f" Total simulations to run: {len(tasks)}")
        print(f" Launching parallel execution...\n")
        print("="*60)
        
        # Run simulations in parallel
        results = []
        start_time = datetime.now()
        
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(run_simulation_worker, task): task for task in tasks}
            
            # Process results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"  Task exception: {e}")
        
        total_elapsed = (datetime.now() - start_time).total_seconds()
        
        # Summary
        print("="*60)
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f" Parallel Batch Results:")
        print(f"    Successful: {len(successful)}/{len(results)}")
        print(f"    Failed: {len(failed)}/{len(results)}")
        print(f"    Total Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        
        if successful:
            avg_time = sum(r['elapsed'] for r in successful) / len(successful)
            print(f"  Avg Time per Simulation: {avg_time:.1f}s")
        
        if failed:
            print(f"\n   Failed simulations:")
            for r in failed:
                print(f"      {r['design']}/{r['stl']}")
        
        print("="*60)
        
        return 0 if len(failed) == 0 else 1
    
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user - shutting down workers...")
        return 1
    except Exception as e:
        print(f" Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())





