"""
================================================================================
MASTER BATCH CALIBRATION — MULTI-FORCING  v3
(file names kept: 01_MAIN_PIPELINE_run_this.py == main_calibration_parallel_v2.py)
================================================================================
Same structure as v2 (5 forcings × 3 RGI regions, per-glacier workers,
checkpointing, live progress), with these corrections:

  [F1] ref_hgt = elevation of the CLIMATE GRID CELL (climate_loader_unified
       .grid_elevation),
       not glacier mid-elevation. Fails if the orography file is missing.
  [F2] V_2000 from the Farinotti (2019) consensus thickness directly.
       (v2 ran an OGGM inversion with an UNCALIBRATED MB model, dT=0 pf=2.5,
       and took an unweighted mean of flowline points; consensus was only
       a fallback.)
  [F3] Calibration: exact, deterministic, regularised (calibration_v3).
       One bounds definition (calibration_v3.BOUNDS), written to metadata.
       calib_status / at-bound flags stored for every glacier.
  [F4] Calibration years follow OGGM/Hugonnet convention: period
       2000-01-01_2020-01-01 -> calendar years 2000..2019 (v2 used 21 years).
       The target period must be FULLY covered by the forcing:
       CHELSA (ends 2016) -> 2000-2010 target, not a truncated 2000-2020.
  [F5] All three Hugonnet periods stored (for the split-sample test).
       Missing Hugonnet -> glacier skipped & logged (v2 silently used -0.5).
  [F6] Geometry v3: ice/water density, no 1 % floor, real extinction,
       backward vanishing flagged, consistent end-of-year timing, dm_gt.
  [F7] Hugonnet table loaded once per worker process (v2: once per glacier).
  [F7b] Flowlines: OGGM elevation-band flowlines by default (OGGM standard
       for regional MB; exact RGI area; ~0.05 s/glacier). Centerlines are
       tried as fallback (or set FLOWLINE_TYPE='centerlines'). Recovers the
       v2 "centerline too short" / "altitude range too small" failures.
  [F8] TEST MODE: set env OGGM_V3_TEST=200 to run all field glaciers + 200
       random glaciers per region. Do this first and inspect before the
       full run.

Output:  config.json 'output_root' (default <cache>/../calibration_output_v3)
         /<FORCING>/<REGION>_run_03/<RGIId>/
         NEW folder on purpose: the v2 checkpoints would otherwise make v3
         skip every glacier that v2 already processed.
================================================================================
"""

import sys
import os
import gc
import json
import time
import shutil
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')

for _proj_candidate in [
    Path(sys.executable).parent.parent / "share/proj",
    Path("/usr/share/proj"),
]:
    if _proj_candidate.exists():
        os.environ.setdefault("PROJ_DATA", str(_proj_candidate))
        os.environ.setdefault("PROJ_LIB",  str(_proj_candidate))
        break


def _find_script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


SCRIPT_DIR = _find_script_dir()
sys.path.insert(0, str(SCRIPT_DIR))

from climate_loader_unified import detect_cache_root, load_climate, \
    write_oggm_climate_file, grid_elevation, find_orography_file, \
    FORCING_NAMES                                    # noqa: E402

CACHE_ROOT = detect_cache_root()


def _output_root() -> Path:
    """config.json 'output_root' if it is a v3 folder, else default."""
    cf = SCRIPT_DIR / 'config.json'
    if cf.exists():
        try:
            p = json.loads(cf.read_text()).get('output_root', '')
            if p and 'v2' not in Path(p).name and Path(p).parent.exists():
                return Path(p)
        except Exception:
            pass
    return CACHE_ROOT.parent / "calibration_output_v3"


OUTPUT_ROOT = _output_root()

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

ALL_FORCINGS = ['ERA5', 'ERA5-Land', 'CRU_TS4', 'CHELSA', 'HARv2']
ALL_REGIONS  = ['RGI60-13', 'RGI60-14', 'RGI60-15']


def _env_subset(name, allowed):
    """Run switch (environment only, no physics change): comma-separated subset,
    e.g. OGGM_V3_ONLY_FORCING=ERA5 OGGM_V3_ONLY_REGION=RGI60-13. Unset = all."""
    raw = os.environ.get(name, '').strip()
    if not raw:
        return list(allowed)
    sel = [s.strip() for s in raw.split(',') if s.strip()]
    bad = [s for s in sel if s not in allowed]
    if bad:
        raise SystemExit(f"{name}={raw!r}: unknown {bad}; allowed {allowed}")
    return [a for a in allowed if a in sel]


FORCINGS_TO_RUN = _env_subset('OGGM_V3_ONLY_FORCING', ALL_FORCINGS)
REGIONS_TO_RUN  = _env_subset('OGGM_V3_ONLY_REGION', ALL_REGIONS)
RUN_SUFFIX      = os.environ.get('OGGM_V3_RUN', 'run_03')   # separate folder per sensitivity run
# Testing on chosen glaciers: comma list of RGI ids (empty = all glaciers of the region)
ONLY_IDS = [s.strip() for s in os.environ.get('OGGM_V3_ONLY_IDS', '').split(',') if s.strip()]

# Area-consistent calibration target (run_06): Hugonnet per-glacier rates are mass change / FIXED RGI area,
# but the model integrates dM = B * A(t) with the shrinking V-A area. With OGGM_V3_AREA_CONSISTENT=1 the
# target (and its sigma) is multiplied by factor = A_RGI / mean(A_start over the calibration period) of a
# reference run, read from OGGM_V3_AREA_FACTORS (csv: rgi_id, forcing, calib_period, factor, capped).
AREA_CONSISTENT   = os.environ.get('OGGM_V3_AREA_CONSISTENT', '0') == '1'
AREA_FACTORS_FILE = os.environ.get('OGGM_V3_AREA_FACTORS', '')
if AREA_CONSISTENT and not (AREA_FACTORS_FILE and Path(AREA_FACTORS_FILE).is_file()):
    raise SystemExit(f"OGGM_V3_AREA_CONSISTENT=1 needs OGGM_V3_AREA_FACTORS=<csv>; got {AREA_FACTORS_FILE!r}")

REFERENCE_YEAR = 2000            # RGI epoch: V_ref = state at end of 1999
GEODETIC_START = 2000            # (kept for 04_MONITOR_progress.py)
GEODETIC_END   = 2020
SKIP_EXISTING  = True            # checkpoint-based resume
FLOWLINE_TYPE  = 'elevation_band'   # or 'centerlines' (other used as fallback)

# Hugonnet periods in order of preference; a period is used only if the
# forcing covers ALL of its (calendar) years.
HUGONNET_PERIODS = [
    '2000-01-01_2020-01-01',
    '2000-01-01_2010-01-01',
    '2010-01-01_2020-01-01',
]

HIST_START_PER_FORCING = {
    'ERA5': 1950, 'ERA5-Land': 1950, 'CRU_TS4': 1950,
    'CHELSA': 1979, 'HARv2': 1980,
}
HIST_START_DEFAULT = 1950

CPU_UTILIZATION = 0.9
N_WORKERS = max(1, int((os.cpu_count() or 4) * CPU_UTILIZATION))
if os.environ.get("OGGM_V3_WORKERS"):        # e.g. OGGM_V3_WORKERS=6 on a laptop
    N_WORKERS = max(1, int(os.environ["OGGM_V3_WORKERS"]))
elif os.environ.get("SLURM_NTASKS"):
    try:
        N_WORKERS = max(1, int(os.environ["SLURM_NTASKS"]))
    except ValueError:
        pass

TEST_N = int(os.environ.get('OGGM_V3_TEST', '0'))   # 0 = full run
FAIL_FAST_N = 30     # abort if the first 30 glaciers of a forcing/region all fail
TEST_SEED = 42

FARINOTTI_UNCERTAINTY = {'RGI60-13': 0.28, 'RGI60-14': 0.34, 'RGI60-15': 0.30}

REGIONAL_SETTINGS = {
    'RGI60-13': {'lapse_rate': -0.0055, 'region_name': 'Central Asia',    'melt_f': 4.5},
    'RGI60-14': {'lapse_rate': -0.0055, 'region_name': 'South Asia West', 'melt_f': 5.5},
    'RGI60-15': {'lapse_rate': -0.0060, 'region_name': 'South Asia East', 'melt_f': 5.0},
}

# --- SENSITIVITY COPY ONLY (OGGM_Complete_sens, paper sensitivity runs; the production code in OGGM_Complete is unchanged) ---
_MELT_SCALE = float(os.environ.get('OGGM_V3_MELT_SCALE', '1.0'))        # multiplies the regional melt factors
_LAPSE_DELTA = float(os.environ.get('OGGM_V3_LAPSE_DELTA', '0.0'))      # added to the regional lapse rates (K m-1)
for _k in REGIONAL_SETTINGS:
    REGIONAL_SETTINGS[_k]['melt_f'] *= _MELT_SCALE
    REGIONAL_SETTINGS[_k]['lapse_rate'] += _LAPSE_DELTA
_CAL_ONLY = os.environ.get('OGGM_V3_CALIB_ONLY_PERIOD', '').strip()     # e.g. 2000-01-01_2010-01-01: calibrate on this period only

_RGI_CANDIDATES = {
    'RGI60-13': ["RGI/rgi60/13_rgi60_CentralAsia/13_rgi60_CentralAsia.shp",
                 "RGI/13_rgi60_CentralAsia/13_rgi60_CentralAsia.shp"],
    'RGI60-14': ["RGI/rgi60/14_rgi60_SouthAsiaWest/14_rgi60_SouthAsiaWest.shp",
                 "RGI/14_rgi60_SouthAsiaWest/14_rgi60_SouthAsiaWest.shp"],
    'RGI60-15': ["RGI/rgi60/15_rgi60_SouthAsiaEast/15_rgi60_SouthAsiaEast.shp",
                 "RGI/15_rgi60_SouthAsiaEast/15_rgi60_SouthAsiaEast.shp"],
}


def _find_rgi(region_code: str) -> Path:
    for rel in _RGI_CANDIDATES[region_code]:
        p = CACHE_ROOT / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"RGI shapefile for {region_code} not found")


_TERM_TYPES  = {0: 'land', 1: 'marine', 2: 'lake', 3: 'dry_calving',
                4: 'regenerated', 5: 'shelf'}
_SURGE_TYPES = {0: 'no_evidence', 1: 'possible', 2: 'probable', 3: 'observed'}


def classify_glacier(rgi_row) -> dict:
    """Terminus / surge / size flags (unchanged from v2)."""
    term_type  = int(rgi_row.get('TermType', 0))
    # RGI6 shapefiles call it 'Surging'; other exports 'SurgeType'
    surge_type = rgi_row.get('SurgeType', rgi_row.get('Surging', 0))
    surge_type = int(surge_type) if pd.notna(surge_type) else 0
    if surge_type == 9:          # RGI 'not assigned'
        surge_type = 0
    area_km2   = float(rgi_row.get('Area', 1.0))
    is_lake  = term_type == 2
    is_surge = surge_type >= 2
    is_small = area_km2 < 0.5
    if is_surge or is_lake:
        conf, reliable = 'low', False
        reason = ','.join(filter(None, [f'surge_{surge_type}' if is_surge else '',
                                        'lake_terminating' if is_lake else '']))
    elif is_small or surge_type == 1:
        conf, reliable, reason = 'medium', True, 'small_or_possible_surge'
    else:
        conf, reliable, reason = 'high', True, ''
    return {
        'terminus_type': _TERM_TYPES.get(term_type, 'land'),
        'surge_type': _SURGE_TYPES.get(surge_type, 'no_evidence'),
        'is_lake_terminating': is_lake,
        'is_marine_terminating': term_type == 1,
        'is_surge_type': is_surge,
        'is_small_glacier': is_small,
        'geom_confidence': conf,
        'geom_reliable': reliable,
        'geom_flag_reason': reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CODE / PRIOR GUARD (added for run_05 after the run_04_full sigma_T mix-up)
# run_04_full: calibration_v3.py was edited (default sigma_T 1.5 -> 4) while the
# run was going; 'spawn' workers re-import the module for every new pool, so the
# first pool (ERA5 x RGI60-13) used 1.5 and all later pools 4. Now:
#   - main() resolves the prior ONCE and exports it as OGGM_V3_PRIOR_* env vars,
#     which every spawned worker inherits (file defaults can no longer leak in);
#   - SHA-256 of the code that workers import is stored at start-up and checked
#     before every new pool and in every worker; any change stops the run.
# ─────────────────────────────────────────────────────────────────────────────
import hashlib

_CODE_DIR = Path(__file__).resolve().parent
GUARDED_FILES = [_CODE_DIR / 'calibration_v3.py',
                 _CODE_DIR / 'climate_loader_unified.py',
                 Path(__file__).resolve()]
if AREA_CONSISTENT:                       # the factor table must not change during the run either
    GUARDED_FILES.append(Path(AREA_FACTORS_FILE).resolve())


class CodeChangedError(RuntimeError):
    pass


def _code_hashes() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in GUARDED_FILES}


def _lock_prior_and_code() -> dict:
    """Call once in main(), before any pool exists."""
    import calibration_v3 as c
    prior = {'OGGM_V3_PRIOR_T0': float(c.PRIOR_T0),
             'OGGM_V3_PRIOR_SIGMA_T': float(c.PRIOR_SIGMA_T),
             'OGGM_V3_PRIOR_PF0': float(c.PRIOR_PF0),
             'OGGM_V3_PRIOR_PF_FACTOR': float(np.exp(c.PRIOR_SIGMA_LNP))}
    for k, v in prior.items():
        os.environ[k] = repr(v)
    os.environ['OGGM_V3_CODE_SHA256'] = json.dumps(_code_hashes(), sort_keys=True)
    return prior


def _check_code_unchanged(where: str) -> None:
    locked = json.loads(os.environ['OGGM_V3_CODE_SHA256'])
    now = _code_hashes()
    changed = [k for k in locked if locked[k] != now.get(k)]
    if changed:
        raise CodeChangedError(f"{where}: {changed} changed since start-up. "
                               f"Refusing to continue (results would mix code versions).")


def _worker_init() -> None:
    """ProcessPoolExecutor initializer: runs once in every spawned worker."""
    _check_code_unchanged('worker start-up')
    import calibration_v3 as c
    exp = float(os.environ['OGGM_V3_PRIOR_SIGMA_T'])
    if abs(float(c.PRIOR_SIGMA_T) - exp) > 1e-12:
        raise CodeChangedError(f"worker PRIOR_SIGMA_T={c.PRIOR_SIGMA_T} != locked {exp}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_dir(forcing, region_code) -> Path:
    d = OUTPUT_ROOT / forcing / f"{region_code}_{RUN_SUFFIX}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _checkpoint_path(forcing, region_code) -> Path:
    return _run_dir(forcing, region_code) / f"checkpoint_{region_code}.json"


def _load_completed(forcing, region_code) -> set:
    cp = _checkpoint_path(forcing, region_code)
    if cp.exists():
        try:
            return set(json.loads(cp.read_text()).get('completed', []))
        except Exception:
            pass
    return set()


def _save_checkpoint(rgi_ids: list, forcing, region_code) -> None:
    """Called from the MAIN process only -> no lock needed."""
    cp = _checkpoint_path(forcing, region_code)
    data = {'completed': []}
    if cp.exists():
        try:
            data = json.loads(cp.read_text())
        except Exception:
            pass
    s = set(data.get('completed', []))
    s.update(rgi_ids)
    data = {'completed': sorted(s), 'total': len(s),
            'last_updated': datetime.now().isoformat()}
    tmp = cp.with_suffix('.tmp')
    tmp.write_text(json.dumps(data))
    tmp.replace(cp)


# ─────────────────────────────────────────────────────────────────────────────
# WORKER-LEVEL CACHES
# ─────────────────────────────────────────────────────────────────────────────

_GEO_DF = None
_AREA_F = None


def _area_factor(rgi_id, forcing):
    """(factor, capped, calib_period) from OGGM_V3_AREA_FACTORS, or None if the glacier is not in it."""
    global _AREA_F
    if _AREA_F is None:
        d = pd.read_csv(AREA_FACTORS_FILE, usecols=['rgi_id', 'forcing', 'calib_period', 'factor', 'capped'])
        _AREA_F = {(r.rgi_id, r.forcing): (float(r.factor), bool(r.capped), str(r.calib_period))
                   for r in d.itertuples(index=False)}
    return _AREA_F.get((rgi_id, forcing))


def _hugonnet_rows(rgi_id):
    """All Hugonnet periods for one glacier -> {period: (dmdtda, err)}."""
    global _GEO_DF
    from oggm import utils
    if _GEO_DF is None:
        _GEO_DF = utils.get_geodetic_mb_dataframe()
    if rgi_id not in _GEO_DF.index:
        return {}
    rows = _GEO_DF.loc[[rgi_id]]
    out = {}
    for _, r in rows.iterrows():
        out[str(r['period'])] = (float(r['dmdtda']), float(r['err_dmdtda']))
    return out


def _period_years(period: str):
    """'2000-01-01_2020-01-01' -> years 2000..2019 (OGGM: np.arange(y0, y1))."""
    a, b = period.split('_')
    return list(range(int(a[:4]), int(b[:4])))


# ─────────────────────────────────────────────────────────────────────────────
# WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _worker(task: dict) -> dict:
    import xarray as xr                                  # noqa: F401
    import warnings
    warnings.filterwarnings('ignore')

    from oggm import cfg, workflow, tasks, GlacierDirectory
    from oggm.shop import bedtopo as _bedtopo
    from shapely import wkt as _wkt

    from calibration_v3 import (calibrate_regularized, set_mb_params,
                                make_mb_model,
                                period_mean_mb, consensus_volume_km3,
                                compute_geometry_timeseries_v3,
                                C_VA_GLOBAL, GAMMA_VA, BOUNDS)

    rgi_id      = task['rgi_id']
    region_code = task['region_code']
    forcing     = task['forcing']
    rgi_entity  = task['rgi_entity']

    result = {'rgi_id': rgi_id, 'region': region_code, 'forcing': forcing,
              'success': False, 'error': None, 'error_short': ''}

    oggm_work = None
    try:
        out_root  = _run_dir(forcing, region_code)
        gl_out    = out_root / rgi_id
        diag_dir  = gl_out / 'diagnostics'
        oggm_work = out_root / f'oggm_workspace_worker_{os.getpid()}'
        for d in [gl_out, diag_dir, oggm_work]:
            d.mkdir(parents=True, exist_ok=True)

        rs = REGIONAL_SETTINGS[region_code]
        cfg.initialize(logging_level='CRITICAL')
        cfg.PARAMS['use_tar_shapefiles']    = False
        cfg.PARAMS['use_intersects']        = False
        cfg.PARAMS['hydro_month_nh']        = 10
        cfg.PARAMS['temp_default_gradient'] = rs['lapse_rate']
        cfg.PATHS['working_dir']            = str(oggm_work)
        cfg.PATHS['dl_cache_dir']           = str(CACHE_ROOT)


        rgi_dict = dict(rgi_entity)
        if isinstance(rgi_dict.get('geometry'), str):
            rgi_dict['geometry'] = _wkt.loads(rgi_dict['geometry'])
        per_gl = oggm_work / 'per_glacier'
        per_gl.mkdir(exist_ok=True)
        gdir = GlacierDirectory(pd.Series(rgi_dict), base_dir=str(per_gl))

        # ── [1] DEM + flowlines ───────────────────────────────────────────────
        dem_used = None
        for dem_src in ['SRTM', 'COPDEM30', 'COPDEM90', 'ASTER']:
            try:
                tasks.define_glacier_region(gdir, source=dem_src)
                dem_used = dem_src
                break
            except Exception:
                continue
        if dem_used is None:
            raise RuntimeError("No DEM source worked")
        tasks.glacier_masks(gdir)

        def _centerlines():
            for fn in [tasks.compute_centerlines, tasks.initialize_flowlines,
                       tasks.catchment_area, tasks.catchment_intersections,
                       tasks.catchment_width_geom,
                       tasks.catchment_width_correction]:
                fn(gdir)

        def _elev_bands():
            tasks.elevation_band_flowline(gdir)
            tasks.fixed_dx_elevation_band_flowline(gdir)

        order = ([('elevation_band', _elev_bands), ('centerlines', _centerlines)]
                 if FLOWLINE_TYPE == 'elevation_band' else
                 [('centerlines', _centerlines), ('elevation_band', _elev_bands)])
        flowline_type, fl_err = None, ''
        for name, fn in order:
            try:
                fn()
                flowline_type = name
                break
            except Exception as e:
                fl_err += f'{name}: {str(e)[:80]}; '
        if flowline_type is None:
            raise RuntimeError(f"Flowlines failed: {fl_err}")

        # ── [2] Reference volume from consensus thickness  [F2] ───────────────
        A_ref = float(rgi_entity['Area'])
        V_ref, mean_thick, v_source = None, np.nan, 'Bahr_fallback'
        try:
            _bedtopo.add_consensus_thickness(gdir)
            V_ref, mean_thick = consensus_volume_km3(gdir, A_ref)
            if V_ref is not None:
                v_source = 'Farinotti2019_consensus'
        except Exception:
            pass
        V_bahr = C_VA_GLOBAL * A_ref ** GAMMA_VA
        if V_ref is None:
            V_ref = V_bahr
            mean_thick = V_ref / A_ref * 1e3
        bahr_bias = (V_bahr - V_ref) / V_ref * 100.0

        # ── [3] Climate + grid-cell elevation  [F1] ───────────────────────────
        lon = float(rgi_entity['CenLon'])
        lat = float(rgi_entity['CenLat'])
        climate_df, climate_meta = load_climate(forcing, lon, lat,
                                                cache_root=CACHE_ROOT)
        if len(climate_df) < 24:
            raise RuntimeError(f"Climate too short: {len(climate_df)} months")

        orog = grid_elevation(forcing, lon, lat, CACHE_ROOT)
        ref_hgt = orog['grid_elev_m']
        z_med = float(rgi_entity.get('Zmed', np.nan))
        if not np.isfinite(z_med) or z_med <= 0:
            z_med = (float(rgi_entity['Zmin']) + float(rgi_entity['Zmax'])) / 2

        write_oggm_climate_file(climate_df, gdir, climate_meta, ref_hgt)

        forcing_start = HIST_START_PER_FORCING.get(forcing, HIST_START_DEFAULT)
        t = climate_df['time']
        data_start = int(t[t.dt.month == 1].dt.year.min())   # complete years
        end_year   = int(t[t.dt.month == 12].dt.year.max())  # only (2026 partial!)
        hist_start = max(data_start, forcing_start)
        # NOTE: OGGM 1.6 MonthlyTIModel computes annual MB over CALENDAR
        # years (Jan-Dec; climate files must be full calendar years and
        # _get_2d_annual_climate selects months by calendar year).
        # cfg.PARAMS['hydro_month_nh'] does NOT change this. Field
        # glaciological years (Oct-Sep, labelled by end year) are therefore
        # offset by 3 months — state this in the methods.
        first_mb_year = hist_start

        # ── [4] Geodetic target  [F4][F5] ─────────────────────────────────────
        hug = _hugonnet_rows(rgi_id)
        if not hug:
            raise RuntimeError("No Hugonnet geodetic MB for this glacier")

        # MB model (calendar-year annual MB, see note above)
        mb_model = make_mb_model(gdir, 0.0, 1.0, rs['melt_f'], forcing)

        # Verify fast setter == rebuilt model (OGGM-version guard)
        test_year = max(hist_start + 1, 2001)
        set_mb_params(mb_model, 1.0, 2.0)
        fast = mb_model.get_specific_mb(year=test_year)
        ref_model = make_mb_model(gdir, 1.0, 2.0, rs['melt_f'], forcing)
        slow = ref_model.get_specific_mb(year=test_year)
        use_fast = abs(fast - slow) < 1e-6 * max(1.0, abs(slow))

        def mb_fun_factory(years):
            def mb_fun(tb, pf):
                if use_fast:
                    set_mb_params(mb_model, tb, pf)
                    m = mb_model
                else:
                    m = make_mb_model(gdir, tb, pf, rs['melt_f'], forcing)
                return period_mean_mb(m, years)
            return mb_fun

        cal_period = None
        for p in HUGONNET_PERIODS:
            if p not in hug:
                continue
            if _CAL_ONLY and p != _CAL_ONLY:
                continue
            yrs = _period_years(p)
            if yrs[0] >= first_mb_year and yrs[-1] <= end_year:
                cal_period = p
                break
        if cal_period is None:
            raise RuntimeError("No Hugonnet period fully covered by forcing")

        target, sigma = hug[cal_period]
        target_hug, sigma_hug = target, sigma
        area_factor, area_flag = 1.0, 'off'
        if AREA_CONSISTENT:
            fr = _area_factor(rgi_id, forcing)
            if fr is None:
                area_flag = 'missing'                       # not in the reference run -> factor 1
            else:
                if fr[2] != cal_period:
                    raise RuntimeError(f"area factor period {fr[2]} != calibration period {cal_period}")
                area_factor, area_flag = fr[0], ('capped' if fr[1] else 'ok')
            target, sigma = target * area_factor, sigma * area_factor
        if not (np.isfinite(target) and abs(target) < 5):
            raise RuntimeError(f"Implausible geodetic target {target}")
        cal_years = _period_years(cal_period)

        # ── [5] Calibration  [F3] ─────────────────────────────────────────────
        calib = calibrate_regularized(mb_fun_factory(cal_years), target, sigma)

        # Final model with calibrated params (always rebuilt -> writes json)
        mb_model = make_mb_model(gdir, calib['temp_bias'], calib['prcp_fac'],
                                 rs['melt_f'], forcing)

        # Modelled means for every Hugonnet period (split-sample test)
        hug_model = {}
        for p in HUGONNET_PERIODS:
            yrs = _period_years(p)
            if yrs[0] >= first_mb_year and yrs[-1] <= end_year:
                hug_model[p] = period_mean_mb(mb_model, yrs)

        # ── [6] MB time series ────────────────────────────────────────────────
        mb_rows = []
        for yr in range(first_mb_year, end_year + 1):
            try:
                mb_rows.append({'year': yr,
                                'mb': mb_model.get_specific_mb(year=yr) / 1000.0})
            except Exception:
                mb_rows.append({'year': yr, 'mb': np.nan})
        pd.DataFrame(mb_rows).to_csv(gl_out / 'mb_timeseries.csv', index=False)

        # ── [7] Geometry  [F6] ────────────────────────────────────────────────
        flags = classify_glacier(rgi_entity)
        geom = compute_geometry_timeseries_v3(
            mb_rows, V_ref, A_ref, FARINOTTI_UNCERTAINTY[region_code],
            ref_year=REFERENCE_YEAR)
        geom['forcing'] = forcing
        geom['rgi_id'] = rgi_id
        geom.to_csv(gl_out / 'glacier_geometry.csv', index=False)

        # ── [8] Metadata ──────────────────────────────────────────────────────
        meta = {
            'rgi_id': rgi_id, 'region': region_code, 'forcing': forcing,
            'pipeline_version': 'v3',
            'climate_period': f"{hist_start}-{end_year}",
            'first_mb_year': first_mb_year, 'end_year': end_year,
            'dem_source': dem_used,
            'flowline_type': flowline_type,
            # climate reference height
            'ref_hgt_m': ref_hgt, 'glacier_zmed_m': z_med,
            'elev_offset_m': z_med - ref_hgt,
            'orog_file': orog['orog_file'],
            'grid_lat': orog['orog_lat'], 'grid_lon': orog['orog_lon'],
            # calibration
            'calib_period': cal_period,
            'geodetic_target': target, 'geodetic_sigma': sigma,
            'geodetic_target_hugonnet': target_hug, 'geodetic_sigma_hugonnet': sigma_hug,
            'area_consistent': AREA_CONSISTENT, 'area_factor': area_factor,
            'area_factor_flag': area_flag,
            'geodetic_modeled': calib['modeled_mb'],
            'geodetic_residual': calib['residual'],
            'temp_bias': calib['temp_bias'], 'prcp_fac': calib['prcp_fac'],
            'melt_f': rs['melt_f'], 'lapse_rate': rs['lapse_rate'],
            'calib_status': calib['calib_status'],
            'at_tb_bound': calib['at_tb_bound'],
            'at_pf_bound': calib['at_pf_bound'],
            'n_mb_evals': calib['n_mb_evals'],
            'bounds_temp_bias': calib['bounds_temp_bias'],
            'bounds_prcp_fac': calib['bounds_prcp_fac'],
            'prior': calib['prior'],
            'fast_param_setter': use_fast,
            # Hugonnet obs + model per period
            **{f'hug_obs_{p[:4]}_{p[11:15]}': hug.get(p, (np.nan, np.nan))[0]
               for p in HUGONNET_PERIODS},
            **{f'hug_err_{p[:4]}_{p[11:15]}': hug.get(p, (np.nan, np.nan))[1]
               for p in HUGONNET_PERIODS},
            **{f'hug_mod_{p[:4]}_{p[11:15]}': hug_model.get(p, np.nan)
               for p in HUGONNET_PERIODS},
            # geometry
            'A_ref_km2': A_ref, 'V_ref_km3': V_ref,
            'mean_thickness_m': mean_thick, 'V_bahr_km3': V_bahr,
            'bahr_bias_pct': bahr_bias, 'volume_source': v_source,
            'extinct_year': (None if pd.isna(geom['extinct_year'].iloc[0])
                             else int(geom['extinct_year'].iloc[0])),
            'backward_vanished': bool(geom['backward_vanished'].iloc[0]),
            **flags,
            **{k: str(v) for k, v in climate_meta.items()},
        }
        with open(diag_dir / 'calibration_metadata.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        result['success'] = True

    except Exception as e:
        result['error'] = traceback.format_exc()
        result['error_short'] = str(e)[:160]
    finally:
        try:
            if oggm_work is not None and oggm_work.exists():
                shutil.rmtree(oggm_work, ignore_errors=True)
        except Exception:
            pass
        gc.collect()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TASKS + PROGRESS
# ─────────────────────────────────────────────────────────────────────────────

def _rgi_row_to_dict(row):
    d = dict(row)
    if 'geometry' in d and hasattr(d['geometry'], 'wkt'):
        d['geometry'] = d['geometry'].wkt
    return d


class ProgressLogger:
    def __init__(self, forcing, region_code, total):
        self.forcing, self.region_code, self.total = forcing, region_code, total
        self.n_ok = self.n_fail = 0
        self.t0 = time.time()
        self.errors = []
        d = _run_dir(forcing, region_code)
        self.json_path, self.log_path = d / 'progress.json', d / 'progress.log'
        self.fail_path = d / 'failures.csv'
        self._write()
        self._log(f"Started: {forcing} / {region_code} — {total} glaciers")

    def update(self, rgi_id, ok, err=''):
        if ok:
            self.n_ok += 1
        else:
            self.n_fail += 1
            self.errors.append({'rgi_id': rgi_id, 'error': err[:160]})
            with open(self.fail_path, 'a') as f:
                f.write(f'{rgi_id},"{err[:160].replace(chr(34), chr(39))}"\n')
        self._write()
        self._log(f"{'OK' if ok else 'FAIL'}  {rgi_id}  "
                  f"[{self.n_ok + self.n_fail}/{self.total}]"
                  + (f"  {err[:80]}" if not ok else ''))

    def eta(self):
        done = self.n_ok + self.n_fail
        if done == 0:
            return 'unknown'
        rate = done / (time.time() - self.t0)
        s = int((self.total - done) / rate) if rate > 0 else 0
        return f"{s // 3600}h {(s % 3600) // 60}m"

    def rate(self):
        el = time.time() - self.t0
        return (self.n_ok + self.n_fail) / (el / 3600) if el > 1 else 0.0

    def _write(self):
        done = self.n_ok + self.n_fail
        data = {'forcing': self.forcing, 'region': self.region_code,
                'total': self.total, 'completed': done, 'success': self.n_ok,
                'failed': self.n_fail,
                'pct': round(done / self.total * 100, 1) if self.total else 0,
                'elapsed_min': round((time.time() - self.t0) / 60, 1),
                'eta': self.eta(), 'rate_per_hr': round(self.rate(), 1),
                'workers': N_WORKERS,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'recent_errors': self.errors[-5:]}
        tmp = self.json_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.json_path)

    def _log(self, msg):
        with open(self.log_path, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")


def _field_ids(cache_root: Path) -> set:
    for p in [cache_root / "Field_MB/Mass_Balance.csv",
              cache_root / "field_data/Mass_Balance.csv",
              SCRIPT_DIR / "Mass_Balance.csv"]:
        if p.exists():
            df = pd.read_csv(p)
            col = next((c for c in df.columns
                        if c.upper().replace('_', '') == 'RGIID'), None)
            if col:
                return {str(x).replace('RGI50-', 'RGI60-') for x in df[col].unique()}
    return set()


def run_forcing_region(forcing, region_code, field_ids):
    print(f"\n{'=' * 70}\n  FORCING: {forcing}   REGION: {region_code}\n{'=' * 70}",
          flush=True)
    rgi = gpd.read_file(_find_rgi(region_code))

    if TEST_N > 0:
        fld = rgi[rgi['RGIId'].isin(field_ids)]
        rest = rgi[~rgi['RGIId'].isin(field_ids)]
        rnd = rest.sample(min(TEST_N, len(rest)), random_state=TEST_SEED)
        rgi = pd.concat([fld, rnd])
        print(f"  TEST MODE: {len(fld)} field + {len(rnd)} random glaciers")

    if ONLY_IDS:
        rgi = rgi[rgi['RGIId'].isin(ONLY_IDS)]
        print(f"  ONLY_IDS: {len(rgi)} of {len(ONLY_IDS)} requested ids are in {region_code}")

    completed = _load_completed(forcing, region_code)
    todo = [r for _, r in rgi.iterrows() if r['RGIId'] not in completed]
    tasks_ = [{'rgi_id': r['RGIId'], 'rgi_entity': _rgi_row_to_dict(r),
               'region_code': region_code, 'forcing': forcing} for r in todo]

    print(f"  Total: {len(rgi):,}  done: {len(completed):,}  "
          f"to process: {len(tasks_):,}  workers: {N_WORKERS}", flush=True)
    if not tasks_:
        return

    prog = ProgressLogger(forcing, region_code, len(tasks_))
    pending_ckpt = []
    first_error_shown = False
    early = []                      # outcomes of the first FAIL_FAST_N glaciers
    _check_code_unchanged(f"before pool {forcing}/{region_code}")
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as pool:
        futures = {pool.submit(_worker, t): t['rgi_id'] for t in tasks_}
        for fut in as_completed(futures):
            rgi_id = futures[fut]
            try:
                res = fut.result()
                ok, err = res['success'], res.get('error_short', '')
                full = res.get('error') or ''
            except Exception as e:
                ok, err, full = False, str(e)[:160], traceback.format_exc()
            if not ok:
                with open(_run_dir(forcing, region_code) / 'errors_full.log',
                          'a') as fh:
                    fh.write(f"\n===== {rgi_id} =====\n{full}\n")
                if not first_error_shown:
                    print(f"\n  FIRST FAILURE ({rgi_id}) — full traceback:\n"
                          f"{full}", flush=True)
                    first_error_shown = True
            if len(early) < FAIL_FAST_N:
                early.append((ok, err))
                if len(early) == FAIL_FAST_N and not any(o for o, _ in early):
                    from collections import Counter
                    top = Counter(e for _, e in early).most_common(3)
                    print(f"\n  FAIL-FAST: first {FAIL_FAST_N} glaciers of "
                          f"{forcing}/{region_code} all failed. Most common "
                          f"errors:", flush=True)
                    for e, n in top:
                        print(f"    {n:3d}x  {e}", flush=True)
                    print("  Stopping the whole run. Fix the error, then rerun "
                          "(checkpoints resume). Full tracebacks: "
                          f"{_run_dir(forcing, region_code) / 'errors_full.log'}",
                          flush=True)
                    for f_ in futures:
                        f_.cancel()
                    pool.shutdown(wait=False, cancel_futures=True)
                    os._exit(2)
            if ok:
                pending_ckpt.append(rgi_id)
            prog.update(rgi_id, ok, err)
            done = prog.n_ok + prog.n_fail
            if len(pending_ckpt) >= 50 or done == len(tasks_):
                _save_checkpoint(pending_ckpt, forcing, region_code)
                pending_ckpt = []
            if done % 100 == 0 or done == len(tasks_):
                print(f"  [{datetime.now():%H:%M:%S}] {done}/{len(tasks_)} "
                      f"({prog.n_ok} ok, {prog.n_fail} fail) ETA {prog.eta()} "
                      f"{prog.rate():.0f}/hr", flush=True)
    if pending_ckpt:
        _save_checkpoint(pending_ckpt, forcing, region_code)
    prog._log(f"DONE: {prog.n_ok} ok, {prog.n_fail} failed")


def main():
    print("=" * 70)
    print("  MULTI-FORCING GLACIER CALIBRATION  v3")
    print(f"  Cache root : {CACHE_ROOT}\n  Output root: {OUTPUT_ROOT}")
    print(f"  Forcings   : {FORCINGS_TO_RUN}\n  Regions    : {REGIONS_TO_RUN}")
    print(f"  Workers    : {N_WORKERS}   Test mode: {TEST_N or 'off'}   "
          f"Run: {RUN_SUFFIX}")
    prior = _lock_prior_and_code()
    import calibration_v3 as _c
    print(f"  Prior lock : {prior}")
    print(f"  Code SHA256: {os.environ['OGGM_V3_CODE_SHA256']}")
    print(f"  Prior      : dT0={_c.PRIOR_T0} sigma_T={_c.PRIOR_SIGMA_T} "
          f"pf0={_c.PRIOR_PF0} pf_factor={float(np.exp(_c.PRIOR_SIGMA_LNP)):.2f}"
          f"   bounds {_c.BOUNDS}")
    print(f"  Area-consistent target: {AREA_CONSISTENT}"
          + (f"  factors {AREA_FACTORS_FILE}" if AREA_CONSISTENT else ""))
    if ONLY_IDS:
        print(f"  ONLY_IDS   : {ONLY_IDS}")
    print("=" * 70)

    # Fail fast if any orography file is missing  [F1]
    for f in FORCINGS_TO_RUN:
        print(f"  orography {f:10s}: {find_orography_file(f, CACHE_ROOT)}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT_ROOT / f"run_manifest_{RUN_SUFFIX}.json"
    manifest.write_text(json.dumps({
        'run': RUN_SUFFIX, 'started': datetime.now().isoformat(timespec='seconds'),
        'forcings': FORCINGS_TO_RUN, 'regions': REGIONS_TO_RUN, 'prior': prior,
        'bounds': {k: list(v) for k, v in _c.BOUNDS.items()},
        'code_sha256': json.loads(os.environ['OGGM_V3_CODE_SHA256']),
        'test_n': TEST_N, 'workers': N_WORKERS, 'only_ids': ONLY_IDS,
        'area_consistent': AREA_CONSISTENT,
        'area_factors_file': AREA_FACTORS_FILE if AREA_CONSISTENT else None}, indent=2))
    print(f"  Manifest   : {manifest}")
    field_ids = _field_ids(CACHE_ROOT)
    print(f"  Field-MB glaciers: {len(field_ids)}")

    for forcing in FORCINGS_TO_RUN:
        for region_code in REGIONS_TO_RUN:
            try:
                run_forcing_region(forcing, region_code, field_ids)
            except CodeChangedError as e:
                print(f"\nABORT: {e}", flush=True)
                sys.exit(3)
            except Exception as e:
                print(f"\nERROR in {forcing}/{region_code}: {e}")
                traceback.print_exc()
    print("\n" + "=" * 70 + "\n  ALL DONE\n" + "=" * 70)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
