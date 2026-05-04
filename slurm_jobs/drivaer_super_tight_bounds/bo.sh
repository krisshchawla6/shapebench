#!/bin/bash
#SBATCH --job-name=drvr_stb_bo
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/scratch/ShapeEvolve/environments/DrivAer_Star/results/logs_super_tight_bounds/slurm_bo_%j.log

# BO_torch — 10 seeds, vtk_E, super-tight bounds ablation.
# All 17 high-saturation parameters tightened by 50% vs tight_bounds.
# 128-vCPU node: 16 concurrent × 8 threads = 128
MAX_JOBS=16
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
source /scratch/ShapeEvolve/slurm_jobs/drivaer_super_tight_bounds/_common.sh

for seed in $(seq 0 9); do
  launch "run_BO_torch_cd_only_super_tight_bounds_vtk_E_seed${seed}_n1000" \
    --framework BO_torch \
    --reward cd_only \
    --n_calls 1000 \
    --n_initial 30 \
    --random_state "$seed" \
    --gradient_infeasible
done

wait
echo "BO super_tight_bounds node done."
