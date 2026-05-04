# Sourced by each method script — defines shared helpers.
REPO=/scratch/ShapeEvolve
VENV=/home/jack/venv_torch210/bin/activate
RESULTS=$REPO/environments/BlendedNet/results
LOGDIR=$RESULTS/logs_total_drag_runs

_wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 5
  done
}

_run_one() {
  local name="$1"; shift
  local outdir="$RESULTS/$name"
  local log="$LOGDIR/${name}.log"
  if [ -d "$outdir" ] && [ -f "$outdir/results.csv" ]; then
    echo "[SKIP]  $name"
    return 0
  fi
  mkdir -p "$outdir"
  echo "[START] $name"
  source "$VENV"
  python "$REPO/run_benchmark.py" "$@" \
    --environment BlendedNet \
    --output-dir "$outdir" \
    > "$log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && echo "[DONE]  $name" || echo "[FAIL]  $name  (exit $rc)"
}

launch() { _wait_for_slot; _run_one "$@" & }
