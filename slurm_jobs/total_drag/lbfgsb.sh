#!/bin/bash
#SBATCH --job-name=bn_lbfgsb_totaldrag
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/scratch/ShapeEvolve/environments/BlendedNet/results/logs_total_drag_runs/slurm_lbfgsb_%j.log

# L-BFGS-B — 20 runs (10 seeds × 2 rewards), single-process per run
# 128-vCPU node: 16 concurrent × 8 threads = 128
MAX_JOBS=16
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
source /scratch/ShapeEvolve/slurm_jobs/total_drag/_common.sh

for REWARD in shapebench_5_total_drag shapebench_5_max_LD_total_drag; do
  for seed in $(seq 0 9); do
    launch "run_lbfgsb_${REWARD}_seed${seed}_nr10" \
      --framework lbfgsb \
      --reward "$REWARD" \
      --n_restarts 10 \
      --maxiter 200 \
      --eps 0.0001 \
      --random_state "$seed" \
      --gradient_infeasible
  done
done

wait
echo "L-BFGS-B node done."
