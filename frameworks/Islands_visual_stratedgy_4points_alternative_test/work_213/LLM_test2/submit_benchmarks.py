import os
import glob
import shutil
import subprocess

def setup_and_submit():
    # Paths
    base_dir = os.path.abspath('AirFoil_becnhmark/modified_env')
    actions_dir = os.path.join(base_dir, 'LLM_test2/geometry_actions/generate_bounds')
    output_root = os.path.join(base_dir, 'hpc_output')
    
    # Python executable for Fenics
    python_exec = '/home/ubuntu/miniconda3/envs/fenics_env/bin/python'
    
    # Environment files to copy
    env_files = [
        'run_case.py',
        'parametered_env.py',
        'environment.py',
        'shapes_utils.py',
        'meshes_utils.py',
        'fenics_solver.py',
    ]
    
    # Environment dirs to symlink
    env_dirs = [
        'reset',
        'LLM_Actions'
    ]
    
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    # Find CSVs
    csv_files = glob.glob(os.path.join(actions_dir, '*.csv'))
    # Filter out shape CSVs, keep only action CSVs
    # Action CSVs: single line with 12 comma-separated values (4 points x 3 values)
    # Shape CSVs: multi-line with header "4 10" or similar
    action_csvs = []
    for f in csv_files:
        if '_shape.csv' in f:
            continue
        # Check if it's a valid action CSV (single line, 12 values)
        try:
            with open(f, 'r') as fp:
                first_line = fp.readline().strip()
                # Action CSVs have 12 comma-separated floats
                values = first_line.split(',')
                if len(values) == 12:
                    action_csvs.append(f)
        except:
            pass
    
    print(f"Found {len(action_csvs)} action CSVs.")
    
    for csv_path in action_csvs:
        csv_name = os.path.basename(csv_path)
        case_name = os.path.splitext(csv_name)[0]
        case_dir = os.path.join(output_root, case_name)
        
        print(f"Setting up case: {case_name}")
        
        # 1. Create Directory
        if os.path.exists(case_dir):
            shutil.rmtree(case_dir)
        os.makedirs(case_dir)
        
        # 2. Copy/Link Files
        for f in env_files:
            src = os.path.join(base_dir, f)
            dst = os.path.join(case_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"Warning: {src} not found")
                
        for d in env_dirs:
            src = os.path.join(base_dir, d)
            dst = os.path.join(case_dir, d)
            if os.path.exists(src):
                if os.path.exists(dst) or os.path.islink(dst):
                    os.unlink(dst)
                os.symlink(src, dst)
            else:
                print(f"Warning: {src} directory not found")
        
        # 3. Copy Action CSV
        shutil.copy2(csv_path, os.path.join(case_dir, csv_name))
        
        # 4. Create Slurm Script
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name={case_name}
#SBATCH --output={case_name}_%j.out
#SBATCH --error={case_name}_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

echo "Starting job for {case_name}"
date

# Activate conda environment properly
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate fenics_env

# Use time to benchmark
/usr/bin/time -v python run_case.py {csv_name}

echo "Job finished"
date
"""
        slurm_path = os.path.join(case_dir, 'run.slurm')
        with open(slurm_path, 'w') as f:
            f.write(slurm_content)
            
        # 5. Submit
        try:
            # We must be inside the dir to submit or use --chdir if sbatch supports it (newer versions)
            # Safe way: cd then submit
            print(f"Submitting job for {case_name}...")
            # subprocess.run(['sbatch', 'run.slurm'], cwd=case_dir, check=True)
            # Using absolute path for safety
            proc = subprocess.run(['/opt/slurm/bin/sbatch', 'run.slurm'], cwd=case_dir, capture_output=True, text=True)
            if proc.returncode == 0:
                print(f"  Submitted: {proc.stdout.strip()}")
            else:
                print(f"  Submission failed: {proc.stderr}")
        except Exception as e:
            print(f"  Error submitting: {e}")

if __name__ == "__main__":
    setup_and_submit()
