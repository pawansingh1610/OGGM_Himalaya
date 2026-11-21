# OGGM HMA Glacier Calibration Pipeline — v3
Updated: 2026-11-04 (from v2, 2026-07-09)

## Quick start
```bash
conda activate oggm
pip install pyarrow                       # parquet output (new)
python precheck.py                        # must pass, incl. orography check
OGGM_V3_TEST=200 python main_calibration_parallel_v2.py   # test subset FIRST
python compile_v2.py
python 06_ANALYSIS_v3.py --hma-shp /path/to/HMA_15_subregions.shp
#   -> read <output_root>/../compiled_v3/analysis/REPORT.txt
python main_calibration_parallel_v2.py    # full run (or: sbatch submit_calibration_v2.sh)
python compile_v2.py && python 06_ANALYSIS_v3.py --hma-shp ...
python monitor.py                         # live progress (unchanged)
```
Before the full run, delete the test output (`calibration_output_v3/`) or change `RUN_SUFFIX`. Otherwise the checkpoints will skip the test glaciers.

## Do NOT rename (imported by other scripts)
- `climate_loader_unified.py` (v1.2: grid elevation + gap-safe monthly axis)
- `calibration_v3.py` (**new**: calibration, geometry, consensus volume)
- `main_calibration_parallel_v2.py` (imported by `monitor.py`)
- `calibration_functions.py`, `improved_ela_computation.py`, `glacier_summary_statistics.py`, `compile_yearly_data.py` (v2; the v3 pipeline no longer imports them)

## File guide (pairs are identical copies)
| Descriptive name | Actual file | Purpose |
|---|---|---|
| 01_MAIN_PIPELINE_run_this.py | main_calibration_parallel_v2.py | Calibration + geometry (v3) |
| 02_COMPILE_RESULTS_run_after.py | compile_v2.py | Compile to parquet (v3) |
| 03_PRECHECK_run_first.py | precheck.py | System + data + orography check (v3) |
| 04_MONITOR_progress.py | monitor.py | Live monitor (unchanged) |
| 05_SLURM_submit.sh | submit_calibration_v2.sh | SLURM job (v3) |
| 06_ANALYSIS_v3.py | — | All paper numbers, full data (**new**) |

## New data you must add: grid-cell orography (one file per forcing)
Each file must be on the same grid as that forcing's climate files. Add the paths to `OROG_PATHS` in `climate_loader_unified.py` if your filenames differ.

| Forcing | Field | Converted |
|---|---|---|
| ERA5 | invariant `geopotential` (z) | ÷ 9.80665 |
| ERA5-Land | ERA5-Land invariant `geopotential` | ÷ 9.80665 |
| CRU TS4 | CRU CL2.0 elevation `elv` (km) | × 1000 |
| CHELSA-W5E5 | `orog` (m) | — |
| HARv2 | static `hgt` (m) | — |

`precheck.py` fails if any file is missing. It also prints the glacier-vs-grid elevation offset (and the equivalent °C) that v2 ignored.

## What changed from v2, and why the results will change
| | v2 | v3 |
|---|---|---|
| Climate `ref_hgt` | glacier (Zmin+Zmax)/2, so ΔT absorbed a 1–2 km height error (CRU ΔT −7.8 °C; 19 % of CHELSA glaciers at bounds) | grid-cell elevation |
| Calibration | DE, 5 seeds, forcing-specific bounds (CRU ΔT −14…3); 2 parameters fitted to 1 number gives an arbitrary point on the solution curve | exact solve + nearest-to-prior; one set of bounds ΔT ±6, pf 0.5–6; `calib_status` flag; ~120 MB evaluations/glacier |
| Calibration years | 2000–2020 inclusive (21 yr); CHELSA fitted to the 2000–2020 mean using 2000–2016 | 2000–2019 (OGGM/Hugonnet); target must be fully covered (CHELSA uses 2000–2010) |
| Missing Hugonnet | silently −0.5 ± 0.3 | glacier fails, logged |
| V₂₀₀₀ | OGGM inversion run with an uncalibrated MB; consensus only as fallback | consensus volume (Σ thickness·dx²) |
| ΔV | km³ w.e. added to km³ ice | ÷ 0.9 (ice density) |
| Extinction | impossible (1 % volume floor) | V → 0 after 2000; backward "growth from nothing" flagged |
| Flowlines | centerlines; 27 glaciers failed permanently | elevation bands (OGGM standard, exact area), centerlines as fallback |
| Climate series | `dropna()` could drop one month and shift all earlier months | gap-free axis; gaps ≤ 3 months filled with monthly climatology (counted), longer gaps fail |
| ERA5-Land partial 2026 | last year included | only complete calendar years |
| Analysis | notebook on `nrows` slices (~1,400 glaciers) | `06_ANALYSIS_v3.py`, full data |

**For the methods section:** OGGM 1.6 computes annual MB over **calendar years**. `hydro_month_nh=10` has no effect on `get_specific_mb`, so field glaciological years (Oct–Sep) are offset by about 3 months.

## Settings (main_calibration_parallel_v2.py / calibration_v3.py)
- Forcings: ERA5, ERA5-Land, CRU_TS4, CHELSA, HARv2; regions RGI60-13/14/15
- Start years: ERA5 / ERA5-Land / CRU 1950, CHELSA 1979, HARv2 1980
- Melt factor fixed per region (4.5 / 5.5 / 5.0); lapse rate −0.0055 / −0.0055 / −0.0060 K m⁻¹
- Prior: `PRIOR_PF0 = 1.5`, `PRIOR_SIGMA_T = 1.5 °C`, `σ_lnP = ln 2`. Test the sensitivity on the test subset with PF0 = 1.0 and 2.5 and report it.
- `FLOWLINE_TYPE = 'elevation_band'`

## Reliability
Do not claim a forcing ranking until `06_ANALYSIS_v3.py` sections A (calibration health), D (validation on the common subset) and E (split-sample test) have been run on v3 output. The v2 ranking was produced with the wrong reference height.
