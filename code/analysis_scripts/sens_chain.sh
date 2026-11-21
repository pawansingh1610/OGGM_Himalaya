#!/bin/bash
# Sensitivity runs on the 681-glacier test subset (author request 2026-11-12). Uses the COPY of the model code
# the sensitivity copy of the pipeline (switches: OGGM_V3_MELT_SCALE, OGGM_V3_LAPSE_DELTA, OGGM_V3_CALIB_ONLY_PERIOD).
# All runs: sigma_T = 4, no area-consistent target, ERA5-Land with corrected precipitation, 12 workers. New run names only.
set -u
OGGM=${HMA_OGGM_ROOT:?set HMA_OGGM_ROOT}
CODE=${HMA_SENS_CODE_DIR:?set HMA_SENS_CODE_DIR (code/pipeline/provenance + the other pipeline files)}
PY=${HMA_PYTHON:-python}
ST=${HMA_STATUS_DIR:-.}/SENS_STATUS.md
say() { echo "$(date '+%F %T') $*" | tee -a $OGGM/sens_chain.log >> $ST; }
[ -f $ST ] || echo "# sensitivity runs status" > $ST
export OGGM_V3_ONLY_IDS=$(cat $OGGM/subset681_ids.txt) OGGM_V3_WORKERS=${OGGM_V3_WORKERS:-6}
cd $CODE
run() {   # name, env assignments...
  name=$1; shift
  if [ -d $OGGM/compiled_v3_$name/per_forcing ]; then say "skip $name (already compiled)"; return 0; fi
  say "start $name ($*)"
  env "$@" OGGM_V3_RUN=$name $PY 01_MAIN_PIPELINE_run_this.py >> $OGGM/run_$name.log 2>&1 || { say "FAILED: $name model"; return 1; }
  env OGGM_V3_RUN=$name $PY 02_COMPILE_RESULTS_run_after.py >> $OGGM/compile_$name.txt 2>&1 || { say "FAILED: $name compile"; return 1; }
  say "done $name"
}
run run_s01_base        OGGM_V3_MELT_SCALE=1.0
run run_s02_hindcast    OGGM_V3_CALIB_ONLY_PERIOD=2000-01-01_2010-01-01
run run_s03_melt_p20    OGGM_V3_MELT_SCALE=1.2
run run_s04_melt_m20    OGGM_V3_MELT_SCALE=0.8
run run_s05_lapse_p001  OGGM_V3_LAPSE_DELTA=0.001
run run_s06_lapse_m001  OGGM_V3_LAPSE_DELTA=-0.001
say "ALL SENSITIVITY RUNS FINISHED"
