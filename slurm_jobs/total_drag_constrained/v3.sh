#!/bin/bash
#SBATCH --job-name=bn_v3_tdcon
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/scratch/ShapeEvolve/environments/BlendedNet/results/logs_total_drag_constrained_runs/slurm_v3_%j.log

# ShapeEvolve (v3) — 20 runs (10 attempts × 2 constrained total-drag rewards)
# AR ≥ 2.5 constraint blocks surrogate-exploitation corner.
# LLM API calls to gemini-2.5-flash; 4 concurrent × 32 threads = 128
MAX_JOBS=4
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
source /scratch/ShapeEvolve/slurm_jobs/total_drag_constrained/_common.sh

for REWARD in shapebench_5_total_drag_constrained; do
  for attempt in $(seq 1 10); do
    launch "run_v3_flash2_5_${REWARD}_attempt_${attempt}_n2000" \
      --framework v3_dynamic_optimizer \
      --reward "$REWARD" \
      --iterations 200 \
      --batch_size 10 \
      --sampler_model gemini-2.5-flash \
      --inspirations 10
  done
done

wait
echo "v3 constrained node done."
