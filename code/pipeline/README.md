# OGGM HMA glacier calibration pipeline (v3)

## Quick start
```bash
conda activate oggm
python precheck.py                                        # must pass, incl. the orography check
OGGM_V3_TEST=200 python main_calibration_parallel_v2.py   # test subset first
python compile_v2.py
python 06_ANALYSIS_v3.py --hma-shp /path/to/HMA_15_subregions.shp
#   -> read <output_root>/../compiled_v3/analysis/REPORT.txt
python main_calibration_parallel_v2.py                    # full run (or: sbatch submit_calibration_v2.sh)
python compile_v2.py && python 06_ANALYSIS_v3.py --hma-shp ...
python monitor.py                                         # live progress
```
Before the full run, delete the test output (`calibration_output_v3/`) or change `OGGM_V3_RUN`; otherwise the
checkpoints skip the test glaciers. The settings of the published runs are in `../manifests/` and the top-level README.

## Do not rename (imported by other scripts)
- `climate_loader_unified.py`: climate loading, grid-cell elevation, gap-safe monthly axis
- `calibration_v3.py`: calibration, volume-area geometry, consensus volume
- `main_calibration_parallel_v2.py`: imported by `monitor.py`

## File guide (pairs are identical copies)
| Descriptive name | Actual file | Purpose |
|---|---|---|
| 01_MAIN_PIPELINE_run_this.py | main_calibration_parallel_v2.py | Calibration + geometry |
| 02_COMPILE_RESULTS_run_after.py | compile_v2.py | Compile to parquet |
| 03_PRECHECK_run_first.py | precheck.py | System, data and orography check |
| — | monitor.py | Live progress monitor |
| 05_SLURM_submit.sh | submit_calibration_v2.sh | SLURM job |
| 06_ANALYSIS_v3.py | — | Analysis tables on the full data |
| 07_GET_OROGRAPHY.py | — | Download/build the grid-cell elevation files |
| 08_DEBUG_one_glacier.py | — | Run one glacier in the foreground |

## Grid-cell orography (one file per forcing)
Each file must be on the same grid as that forcing's climate files (`07_GET_OROGRAPHY.py` builds them). Add the paths to
`OROG_PATHS` in `climate_loader_unified.py` if your file names differ.

| Forcing | Field | Converted |
|---|---|---|
| ERA5 | invariant `geopotential` (z) | ÷ 9.80665 |
| ERA5-Land | ERA5-Land invariant `geopotential` | ÷ 9.80665 |
| CRU TS4 | CRU CL2.0 elevation `elv` (km) | × 1000 |
| CHELSA-W5E5 | `orog` (m) | — |
| HARv2 | static `hgt` (m) | — |

`precheck.py` fails if any file is missing and prints the glacier-vs-grid elevation offset.

## Method summary
- Climate `ref_hgt` = elevation of the climate grid cell.
- Calibration per glacier to the Hugonnet et al. (2021) geodetic mass balance, 2000–2019 (CHELSA-W5E5: 2000–2009, as
  the product ends in 2016). For each precipitation factor the temperature bias is found by a root search; the solution
  minimises (dT/σ_T)² + (ln(pf/1.5)/ln 2)² with σ_T = 4 °C. Bounds: dT ±10 °C, pf 0.3–6. `calib_status` and
  at-bound flags are stored per glacier. Glaciers without a Hugonnet value fail and are logged.
- Area-consistent target (`OGGM_V3_AREA_CONSISTENT=1`): target × A_RGI / mean(A over the calibration period),
  factors from `OGGM_V3_AREA_FACTORS`.
- Reference volume = Farinotti et al. (2019) consensus volume at the end of 1999; ΔV = B·A/0.9 (ice); volume can reach
  zero after 2000; glaciers that vanish going backwards are flagged.
- Melt factor fixed per RGI region (4.5 / 5.5 / 5.0); lapse rate −0.0055 / −0.0055 / −0.0060 K m⁻¹.
- Start years: ERA5 / ERA5-Land / CRU TS4 1950, CHELSA-W5E5 1979, HARv2 1980. Only complete calendar years are used.
- Flowlines: OGGM elevation bands (centerlines as fallback).
- OGGM 1.6 computes annual mass balance over **calendar years**; `hydro_month_nh` does not change this, so field
  glaciological years (Oct–Sep) are offset by about three months.
