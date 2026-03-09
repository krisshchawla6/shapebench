#!/usr/bin/env python3
"""
Benchmark script for running CFD simulations on action CSVs via sbatch.
Submits jobs, waits for completion, and reports timing statistics.
"""

import os
import sys
import glob
import shutil
import subprocess
import time
import re

def setup_case_directory(base_dir, case_name, action_csv, output_root):
    """Setup isolated directory for each case with all required files."""
    case_dir = os.path.join(output_root, case_name)
    
    # Clean and create directory
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(case_dir)
    
    # Files to copy
    env_files = [
        'run_case.py', 'parametered_env.py', 'environment.py',
        'shapes_utils.py', 'meshes_utils.py', 'fenics_solver.py',
    ]
    
    # Directories to symlink
    env_dirs = ['reset', 'LLM_Actions']
    
    for f in env_files:
        src = os.path.join(base_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, case_dir)
    
    for d in env_dirs:
        src = os.path.join(base_dir, d)
        dst = os.path.join(case_dir, d)
        if os.path.exists(src):
            if os.path.exists(dst) or os.path.islink(dst):
                os.unlink(dst)
            os.symlink(src, dst)
    
    # Copy action CSV
    csv_name = os.path.basename(action_csv)
    shutil.copy2(action_csv, os.path.join(case_dir, csv_name))
    
    return case_dir, csv_name


def create_slurm_script(case_dir, case_name, csv_name):
    """Create sbatch script with timing."""
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={case_name}
#SBATCH --output={case_name}_%j.out
#SBATCH --error={case_name}_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

echo "Starting job for {case_name}"
date

# Activate conda environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate fenics_env

# Run with timing
/usr/bin/time -v python run_case.py {csv_name}

echo "Job finished"
date
"""
    slurm_path = os.path.join(case_dir, 'run.slurm')
    with open(slurm_path, 'w') as f:
        f.write(slurm_content)
    return slurm_path


def submit_job(case_dir):
    """Submit sbatch job and return job ID."""
    result = subprocess.run(
        ['/opt/slurm/bin/sbatch', 'run.slurm'],
        cwd=case_dir,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        # Parse job ID from "Submitted batch job 123"
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if match:
            return int(match.group(1))
    print(f"Submit failed: {result.stderr}")
    return None


def wait_for_jobs(job_ids, poll_interval=5):
    """Wait for all jobs to complete."""
    print(f"Waiting for {len(job_ids)} jobs to complete...")
    while True:
        # Check queue
        result = subprocess.run(
            ['/opt/slurm/bin/squeue', '-j', ','.join(map(str, job_ids)), '-h'],
            capture_output=True, text=True
        )
        # If output is empty, all jobs finished
        if not result.stdout.strip():
            break
        running = len(result.stdout.strip().split('\n'))
        print(f"  {running} jobs still running...", end='\r')
        time.sleep(poll_interval)
    print("\nAll jobs completed.")


def parse_timing(case_dir, case_name):
    """Parse elapsed time from .err file."""
    # Find the .err file
    err_files = glob.glob(os.path.join(case_dir, f"{case_name}_*.err"))
    if not err_files:
        return None, None, "No .err file found"
    
    err_file = err_files[0]
    with open(err_file, 'r') as f:
        content = f.read()
    
    # Parse elapsed time
    # Format: "Elapsed (wall clock) time (h:mm:ss or m:ss): 2:31.13"
    match = re.search(r'Elapsed \(wall clock\) time.*?: (\d+):(\d+\.?\d*)', content)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        total_seconds = minutes * 60 + seconds
        return total_seconds, f"{minutes}:{seconds:05.2f}", None
    
    # Check for exit status
    exit_match = re.search(r'Exit status: (\d+)', content)
    exit_code = int(exit_match.group(1)) if exit_match else -1
    
    if exit_code != 0:
        return None, None, f"Job failed with exit code {exit_code}"
    
    return None, None, "Could not parse timing"


def run_benchmark(actions_dir, output_root):
    """Main benchmark function."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Find action CSVs (exclude *_shape.csv)
    csv_files = glob.glob(os.path.join(actions_dir, '*.csv'))
    action_csvs = []
    for f in csv_files:
        if '_shape.csv' in f:
            continue
        # Verify it's an action CSV (12 values)
        try:
            with open(f, 'r') as fp:
                vals = fp.readline().strip().split(',')
                if len(vals) == 12:
                    action_csvs.append(f)
        except:
            pass
    
    if not action_csvs:
        print("No valid action CSVs found!")
        return
    
    print(f"Found {len(action_csvs)} action CSVs to benchmark")
    print(f"Output directory: {output_root}")
    
    # Setup and submit jobs
    jobs = []  # List of (job_id, case_name, case_dir)
    
    for csv_path in action_csvs:
        case_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"Setting up: {case_name}")
        
        case_dir, csv_name = setup_case_directory(base_dir, case_name, csv_path, output_root)
        slurm_path = create_slurm_script(case_dir, case_name, csv_name)
        job_id = submit_job(case_dir)
        
        if job_id:
            print(f"  Submitted job {job_id}")
            jobs.append((job_id, case_name, case_dir))
        else:
            print(f"  Failed to submit")
    
    if not jobs:
        print("No jobs submitted!")
        return
    
    # Wait for completion
    job_ids = [j[0] for j in jobs]
    wait_for_jobs(job_ids)
    
    # Parse results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    times = []
    for job_id, case_name, case_dir in jobs:
        elapsed_sec, elapsed_str, error = parse_timing(case_dir, case_name)
        if elapsed_sec is not None:
            times.append(elapsed_sec)
            print(f"{case_name}: {elapsed_str} ({elapsed_sec:.2f}s)")
        else:
            print(f"{case_name}: FAILED - {error}")
    
    print("-"*60)
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"Successful runs: {len(times)}/{len(jobs)}")
        print(f"Average time:    {avg_time:.2f}s ({avg_time/60:.2f} min)")
        print(f"Min time:        {min_time:.2f}s")
        print(f"Max time:        {max_time:.2f}s")
    else:
        print("No successful runs to average.")
    print("="*60)
    
    return times


if __name__ == "__main__":
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    actions_dir = os.path.join(script_dir, 'geometry_actions/generate_bounds')
    output_root = os.path.join(script_dir, 'benchmark_output')
    
    # Allow override via command line
    if len(sys.argv) > 1:
        actions_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_root = sys.argv[2]
    
    run_benchmark(actions_dir, output_root)
