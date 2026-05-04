# Sourced by each method script — defines shared helpers for DrivAer super_tight_bounds runs.
REPO=/scratch/ShapeEvolve
VENV=/home/jack/venv_torch210/bin/activate
RESULTS=$REPO/environments/DrivAer_Star/results
LOGDIR=$RESULTS/logs_super_tight_bounds

# All 17 high-saturation parameters tightened by 50% of their tight_bounds range.
# car_size kept at [0.9,1.1] (already tightened, not at bound in either design).
# car_width, car_green_house_angle unchanged (interior in both best designs).
BOUNDS_OVERRIDE='{"car_size":[0.9,1.1],"car_len":[-0.05,0.05],"ramp_angle":[-4.0,4.0],"front_bumper_length":[-0.05,0.05],"wind_screen_x":[-0.025,0.025],"wind_screen_z":[-0.025,0.025],"side_mirrors_x":[-0.025,0.025],"side_mirrors_z":[-0.025,0.025],"rear_window_x":[-0.025,0.025],"rear_window_z":[-0.025,0.025],"trunklid_angle":[-4.0,4.0],"trunklid_x":[-0.025,0.025],"trunklid_z":[-0.025,0.025],"diffusor_angle":[-2.0,2.0],"car_front_hood_angle":[-4.0,4.0],"car_air_intake_angle":[-4.0,4.0],"tires_diameter":[-0.0065,0.0065],"tires_width":[-0.0075,0.0075]}'

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
    --environment DrivAer_Star \
    --base_vtk "$REPO/environments/DrivAer_Star/data/vtk_E/00000.vtk" \
    --bounds_override "$BOUNDS_OVERRIDE" \
    --rho 1.25 --u 40.0 --area_ref 2.37 \
    --output-dir "$outdir" \
    > "$log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && echo "[DONE]  $name" || echo "[FAIL]  $name  (exit $rc)"
}

launch() { _wait_for_slot; _run_one "$@" & }
