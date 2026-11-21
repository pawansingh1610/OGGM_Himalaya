"""
================================================================================
LIVE PROGRESS MONITOR
================================================================================
Run in a separate terminal while the calibration job is running:

    python monitor.py              # refresh every 30 seconds
    python monitor.py --interval 10    # refresh every 10 seconds
    python monitor.py --once           # print once and exit

Works for both interactive runs and SLURM jobs.
Reads progress.json files written by the pipeline — no SSH into the job needed.
================================================================================
"""

import json
import time
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# ── same auto-detection as the main pipeline ─────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / 'config.json'


def load_config() -> dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    # fallback — walk up looking for calibration_output_v2
    for parent in SCRIPT_DIR.parents:
        p = parent / 'calibration_output_v2'
        if p.exists():
            return {'output_root': str(p)}
    raise FileNotFoundError(
        "config.json not found and cannot auto-detect output root.\n"
        "Run precheck.py first.")


def _bar(pct: float, width: int = 30) -> str:
    """Simple ASCII progress bar."""
    filled = int(width * pct / 100)
    return '[' + '█' * filled + '░' * (width - filled) + f'] {pct:.1f}%'


def _colour(text, code):
    """ANSI colour if terminal supports it."""
    if sys.stdout.isatty():
        return f'\033[{code}m{text}\033[0m'
    return text


def find_all_progress_files(output_root: Path) -> list:
    """
    Walk output_root and collect all progress.json files.
    Structure: output_root/FORCING/REGION_run_XX/progress.json
    Worker temp directories are deleted mid-run so we must
    handle missing directories gracefully.
    """
    found = []
    if not output_root.exists():
        return found
    # Walk manually instead of rglob to handle disappearing directories
    try:
        for forcing_dir in sorted(output_root.iterdir()):
            if not forcing_dir.is_dir():
                continue
            try:
                for region_dir in sorted(forcing_dir.iterdir()):
                    if not region_dir.is_dir():
                        continue
                    progress = region_dir / 'progress.json'
                    if progress.exists():
                        found.append(progress)
            except (PermissionError, FileNotFoundError):
                continue
    except (PermissionError, FileNotFoundError):
        pass
    return found


def read_progress(path: Path):
    """Read one progress.json safely."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')


def print_dashboard(output_root: Path, show_errors: bool = False,
                    current_run_only: bool = True,
                    staleness_hours: float = 2.0):
    """
    Print dashboard for running/completed forcing×region combos.

    current_run_only : if True, only show progress files updated in
                       the last staleness_hours — filters out old runs
    staleness_hours  : progress files older than this are considered
                       from a previous run and hidden
    """
    progress_files = find_all_progress_files(output_root)

    now     = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'═'*72}")
    print(f"  GLACIER PIPELINE — LIVE MONITOR          {now_str}")
    print(f"  Output root: {output_root}")
    print(f"{'═'*72}")

    # ── Dynamic summary from pipeline config ─────────────────────────────────
    try:
        import sys, importlib, json as _json
        import xarray as xr
        import pandas as _pd

        script_dir = Path(__file__).resolve().parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))

        pipeline = importlib.import_module('main_calibration_parallel_v2')

        forcings  = pipeline.FORCINGS_TO_RUN
        regions   = pipeline.REGIONS_TO_RUN
        workers   = pipeline.N_WORKERS
        geo_s     = pipeline.GEODETIC_START
        geo_e     = pipeline.GEODETIC_END
        hist_map  = getattr(pipeline, 'HIST_START_PER_FORCING', {})
        skip      = pipeline.SKIP_EXISTING
        reg_set   = getattr(pipeline, 'REGIONAL_SETTINGS', {})
        out_root  = pipeline.OUTPUT_ROOT

        # Load cache root
        try:
            cache = Path(_json.loads(
                (script_dir / 'config.json').read_text())['data_root'])
        except Exception:
            cache = None

        print(f"\n  {'─'*68}")
        print(f"  PIPELINE CONFIGURATION")
        print(f"  {'─'*68}")
        print(f"  Script        : main_calibration_parallel_v2.py")
        print(f"  Forcings      : {forcings}")
        print(f"  Regions       : {regions}")
        print(f"  Workers       : {workers}  |  SKIP_EXISTING: {skip}")
        print(f"  Calib target  : Hugonnet {geo_s}–{geo_e} (geodetic MB)")
        print(f"  Volume anchor : Farinotti 2019 (consensus thickness)")
        print(f"  Geometry      : SRTM Feb-2000 + RGI v6.0")

        # Year range — from actual climate files
        REL_TEMP = {
            'ERA5':      'ERA5/era5_t2m.nc',
            'ERA5-Land': 'era5_land_data/raw/era5_land_t2m_1950_2026.nc',
            'CRU_TS4':   'cru_ts4/cru_ts4.04.1901.2019.tmp.dat.nc',
            'CHELSA':    'chelsa_w5e5/chelsa-w5e5_obsclim_tas_monthlymean_1979-2016_HMA.nc',
            'HARv2':     'HRAv2/HARv2_d10km_m_2d_t2_1980_2025.nc',
        }
        print(f"\n  Year range per forcing (configured — what current run will produce):")
        total_combinations = 0
        for f in forcings:
            h_start = hist_map.get(f, getattr(pipeline,'HIST_START_DEFAULT',1980))
            end_yr  = '?'
            if cache and f in REL_TEMP:
                fp = cache / REL_TEMP[f]
                if fp.exists():
                    try:
                        with xr.open_dataset(fp) as ds:
                            tdim = next((d for d in ['time','valid_time']
                                         if d in ds.dims), None)
                            if tdim:
                                end_yr = int(_pd.to_datetime(
                                    ds[tdim].values).year.max())
                    except Exception:
                        pass
            dur = (end_yr - h_start + 1) if isinstance(end_yr,int) else '?'
            total_combinations += len(regions)
            print(f"    {f:12s}: {h_start} – {end_yr}  ({dur} yrs)  "
                  f"[start from HIST_START_PER_FORCING]")

        # RGI counts from shapefiles
        RGI_SHP = {
            'RGI60-13': 'RGI/rgi60/13_rgi60_CentralAsia/13_rgi60_CentralAsia.shp',
            'RGI60-14': 'RGI/rgi60/14_rgi60_SouthAsiaWest/14_rgi60_SouthAsiaWest.shp',
            'RGI60-15': 'RGI/rgi60/15_rgi60_SouthAsiaEast/15_rgi60_SouthAsiaEast.shp',
        }
        print(f"\n  Glacier counts (from RGI shapefiles):")
        total_gl = 0
        try:
            import geopandas as gpd
            for region in regions:
                if cache and region in RGI_SHP:
                    rp = cache / RGI_SHP[region]
                    if rp.exists():
                        rgi  = gpd.read_file(rp)
                        n    = len(rgi)
                        area = rgi['Area'].sum()
                        total_gl += n
                        print(f"    {region}: {n:,} glaciers  "
                              f"area={area:.0f} km²  "
                              f"mean={area/n:.2f} km²/gl")
            print(f"    {'TOTAL':8s}: {total_gl:,} glaciers  "
                  f"× {len(forcings)} forcings = "
                  f"{total_gl*len(forcings):,} combinations")
        except Exception as e:
            print(f"    (RGI unavailable: {e})")

        # Regional melt factors
        if reg_set:
            print(f"\n  Regional melt factors:")
            for region, vals in reg_set.items():
                print(f"    {region}: melt_f={vals.get('melt_f','?')}  "
                      f"lapse={vals.get('lapse_rate','?')}  "
                      f"({vals.get('region_name','')})")

        # Compiled dataset if available
        compiled_root = out_root.parent / 'compiled_v2'
        summary_f     = compiled_root / 'MERGED' / 'DATASET_SUMMARY.csv'
        rel_f         = compiled_root / 'MERGED' / 'FORCING_RELIABILITY.csv'
        if summary_f.exists():
            df  = _pd.read_csv(summary_f)
            rel = _pd.read_csv(rel_f) if rel_f.exists() else _pd.DataFrame()
            print(f"\n  Last compiled dataset ({compiled_root.name}/)  ⚠ from previous run — may differ from current:")
            for _, row in df.iterrows():
                r = rel[rel['forcing']==row['forcing']]['reliability'].values
                reliability = r[0] if len(r)>0 else '?'
                col = ('92' if reliability=='high'
                       else '93' if reliability=='moderate' else '91')
                v2000 = float(row.get('V_2000_total_km3', 0))
                print(f"    {row['forcing']:12s}: "
                      f"{int(row['n_glaciers']):,} gl  "
                      f"{int(row['year_min'])}–{int(row['year_max'])}  "
                      f"MB={float(row['mb_mean_mwe_yr']):+.3f} m/yr  "
                      f"V₂₀₀₀={v2000:.0f} km³  "
                      f"{_colour(reliability, col)}")
        else:
            print(f"\n  Compiled dataset: not yet available "
                  f"(run compile_v2.py after pipeline completes)")

        print(f"  {'─'*68}")

    # ── Per-forcing processed/remaining status ───────────────────────────
        print(f"\n  {'─'*68}")
        print(f"  FORCING STATUS  (✓=processed  ▶=running  ⏳=pending)")
        print(f"  {'─'*68}")
        print(f"  {'Forcing':<12}  {'Configured':^12}  {'Glaciers':>9}  "
              f"{'OK':>7}  {'Fail':>5}  {'MB mean':>9}  {'V2000':>8}  "
              f"Status  [prev compiled]")
        print(f"  {'─'*68}")

        from datetime import datetime as _dt2
        import pandas as _pd3

        # Aggregate progress per forcing across all regions
        all_pf2    = find_all_progress_files(out_root)
        fagg = {}
        for pf2 in all_pf2:
            d2 = read_progress(pf2)
            if d2 is None:
                continue
            f2  = d2.get('forcing','?')
            upd = d2.get('last_update','')
            stale2 = False
            try:
                age2 = (_dt2.now() - _dt2.strptime(
                    upd,'%Y-%m-%d %H:%M:%S')).total_seconds()/3600
                stale2 = age2 > 2.0
            except Exception:
                pass
            if f2 not in fagg:
                fagg[f2] = {'total':0,'done':0,'ok':0,'fail':0,
                            'stale':True,'last_eta':'?','active_region':''}
            fagg[f2]['total'] += d2.get('total',0)
            fagg[f2]['done']  += d2.get('completed',0)
            fagg[f2]['ok']    += d2.get('success',0)
            fagg[f2]['fail']  += d2.get('failed',0)
            if not stale2:
                fagg[f2]['stale']         = False
                fagg[f2]['last_eta']      = d2.get('eta','?')
                fagg[f2]['active_region'] = d2.get('region','?')

        # Load compiled stats once
        compiled_stats = {}
        if summary_f.exists():
            try:
                df_s = _pd3.read_csv(summary_f)
                for _, row in df_s.iterrows():
                    compiled_stats[row['forcing']] = {
                        'mb':   float(row.get('mb_mean_mwe_yr',0)),
                        'v':    float(row.get('V_2000_total_km3',0)),
                        'yr1':  int(row.get('year_min',0)),
                        'yr2':  int(row.get('year_max',0)),
                        'n_gl': int(row.get('n_glaciers',0)),
                    }
            except Exception:
                pass

        for f3 in forcings:
            # Year range from HIST_START_PER_FORCING + actual file end
            h_start = hist_map.get(f3,
                      getattr(pipeline,'HIST_START_DEFAULT',1980))
            end_yr3 = '?'
            if cache and f3 in REL_TEMP:
                fp3 = cache / REL_TEMP[f3]
                if fp3.exists():
                    try:
                        import xarray as _xr2
                        with _xr2.open_dataset(fp3) as ds3:
                            tdim3 = next((d for d in ['time','valid_time']
                                          if d in ds3.dims), None)
                            if tdim3:
                                end_yr3 = int(_pd3.to_datetime(
                                    ds3[tdim3].values).year.max())
                    except Exception:
                        pass

            # Year range: always use config (what current run produces)
            # compiled stats are shown separately as previous run reference
            cs     = compiled_stats.get(f3, {})
            yr_str = f"{h_start}–{end_yr3}"   # from config + actual file

            # Compiled stats (previous run) — shown as reference only
            if cs.get('yr1') and cs.get('yr2'):
                mb_str  = f"{cs['mb']:+.3f}" if cs.get('mb') else "  –  "
                v_str   = f"{cs['v']:.0f}"   if cs.get('v')  else "  –"
                old_yrs = f"[prev: {cs['yr1']}–{cs['yr2']}]"
            else:
                mb_str  = "  –  "
                v_str   = "  –"
                old_yrs = "[not compiled]"

            agg3 = fagg.get(f3)
            if agg3 is None:
                print(f"  {f3:<12}  {yr_str:^12}  {'–':>9}  "
                      f"{'–':>7}  {'–':>5}  {mb_str:>9}  {v_str:>8}  "
                      f"{_colour('⏳ NOT STARTED', '33')}")
                continue

            tot3   = agg3['total']
            done3  = agg3['done']
            ok3    = agg3['ok']
            fail3  = agg3['fail']
            pct3   = done3/tot3*100 if tot3 > 0 else 0
            stale3 = agg3['stale']
            eta3   = agg3['last_eta']
            areg   = agg3['active_region']

            if pct3 >= 100 and ok3 > 0:
                status3 = _colour('✓ PROCESSED', '1;32')
            elif not stale3 and pct3 < 100:
                status3 = _colour(
                    f'▶ RUNNING {pct3:.0f}% | {areg} ETA {eta3}', '1;34')
            elif stale3 and pct3 < 100:
                status3 = _colour(
                    f'⏳ INCOMPLETE {pct3:.0f}%', '33')
            else:
                status3 = _colour('✓ DONE', '1;32')

            print(f"  {f3:<12}  {yr_str:^12}  {done3:>9,}  "
                  f"{ok3:>7,}  {fail3:>5,}  {mb_str:>9}  {v_str:>8}  "
                  f"{status3}  {old_yrs}")

        print(f"  {'─'*68}")

    except Exception as e:
        import traceback
        print(f"  (pipeline summary error: {e})")
        print(f"  {'─'*68}")

    if not progress_files:
        print(f"\n  No progress files found yet.")
        print(f"  Waiting for pipeline to start writing output...")
        print(f"  (files appear after first glacier completes)")
        return

    # ── Filter to current run only ────────────────────────────────────────────
    all_data = []
    stale_count = 0
    for pf in progress_files:
        data = read_progress(pf)
        if data is None:
            continue

        # Check staleness using last_update field
        stale = False
        if current_run_only:
            last_upd_str = data.get('last_update', '')
            if last_upd_str:
                try:
                    last_upd_dt = datetime.strptime(
                        last_upd_str, '%Y-%m-%d %H:%M:%S')
                    age_hours = (now - last_upd_dt).total_seconds() / 3600
                    if age_hours > staleness_hours:
                        stale = True
                        stale_count += 1
                except Exception:
                    pass

        data['_stale'] = stale
        data['_path']  = str(pf)
        all_data.append(data)

    if stale_count > 0:
        print(f"  (hiding {stale_count} stale entries from previous runs "
              f"— use --all to show)")

    # ── Group by forcing ──────────────────────────────────────────────────────
    by_forcing = {}
    for data in all_data:
        forcing = data.get('forcing', 'unknown')
        if forcing not in by_forcing:
            by_forcing[forcing] = []
        by_forcing[forcing].append(data)

    total_glaciers = total_done = total_ok = total_fail = 0

    for forcing, regions in sorted(by_forcing.items()):
        # Check if any region in this forcing is active (non-stale)
        active = [d for d in regions if not d['_stale']]
        stale  = [d for d in regions if d['_stale']]

        if not active and current_run_only:
            continue   # skip entirely if all stale

        print(f"\n  {'─'*68}")
        print(f"  FORCING: {_colour(forcing, '1;36')}")
        print(f"  {'─'*68}")

        for data in sorted(active + stale,
                           key=lambda d: d.get('region', '')):
            if data['_stale'] and current_run_only:
                continue

            region   = data.get('region', '?')
            tot      = data.get('total', 0)
            done     = data.get('completed', 0)
            ok_n     = data.get('success', 0)
            fail_n   = data.get('failed', 0)
            pct      = data.get('pct', 0.0)
            eta      = data.get('eta', '?')
            rate     = data.get('rate_per_hr', 0)
            workers  = data.get('workers', '?')
            elapsed  = data.get('elapsed_min', 0)
            last_upd = data.get('last_update', '?')
            errors   = data.get('recent_errors', [])

            # Age-aware status
            if data['_stale']:
                status = _colour('STALE (prev run)', '90')
            elif done == tot and tot > 0 and ok_n > 0:
                status = _colour('COMPLETE', '1;32')
            elif fail_n > 0 and fail_n == done:
                status = _colour('ALL FAILING', '1;31')
            elif fail_n > tot * 0.1:
                status = _colour('HIGH FAIL RATE', '1;33')
            else:
                status = _colour('RUNNING', '1;34')

            print(f"\n    {region}  [{status}]")
            print(f"    {_bar(pct)}")
            print(f"    Glaciers : {done:,} / {tot:,}  "
                  f"({_colour(str(ok_n)+' ok', '32')}, "
                  f"{_colour(str(fail_n)+' failed', '31' if fail_n else '90')})")
            print(f"    Workers  : {workers}   "
                  f"Rate: {rate:.0f} glaciers/hr   "
                  f"Elapsed: {elapsed:.0f} min")
            print(f"    ETA      : {eta}   Last update: {last_upd}")

            if show_errors and errors:
                print(f"    Recent failures:")
                for e in errors[-3:]:
                    print(f"      {e['rgi_id']}: {e['error'][:65]}")

            if not data['_stale']:
                total_glaciers += tot
                total_done     += done
                total_ok       += ok_n
                total_fail     += fail_n

    # Overall summary (current run only)
    overall_pct = (total_done / total_glaciers * 100
                   if total_glaciers else 0)
    print(f"\n{'═'*72}")
    print(f"  CURRENT RUN: {_colour(str(total_done), '1')}/{total_glaciers} glaciers  "
          f"{_colour(str(total_ok)+' ok', '32')}  "
          f"{_colour(str(total_fail)+' failed', '31' if total_fail else '90')}  "
          f"({overall_pct:.1f}%)")
    print(f"  {_bar(overall_pct, width=50)}")
    print(f"{'═'*72}")

    try:
        import subprocess
        result = subprocess.run(
            ['squeue', '--me', '--format=%i %j %T %M %l'],
            capture_output=True, text=True, timeout=3)
        if result.returncode == 0 and result.stdout.strip():
            print(f"\n  SLURM jobs:")
            for line in result.stdout.strip().split('\n'):
                print(f"    {line}")
    except Exception:
        pass
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Live monitor for glacier calibration pipeline')
    parser.add_argument('--interval', type=int, default=30,
                        help='Refresh interval in seconds (default: 30)')
    parser.add_argument('--once', action='store_true',
                        help='Print once and exit')
    parser.add_argument('--errors', action='store_true',
                        help='Show recent error messages')
    parser.add_argument('--all', action='store_true',
                        help='Show all runs including previous (default: current run only)')
    parser.add_argument('--stale-hours', type=float, default=2.0,
                        help='Hours before a progress file is considered stale (default: 2)')
    args = parser.parse_args()

    cfg         = load_config()
    output_root = Path(cfg['output_root'])

    if args.once:
        print_dashboard(output_root,
                        show_errors=args.errors,
                        current_run_only=not args.all,
                        staleness_hours=args.stale_hours)
        return

    print(f"Monitoring {output_root}")
    print(f"Refreshing every {args.interval}s  (Ctrl+C to stop)\n")

    try:
        while True:
            clear_screen()
            print_dashboard(output_root,
                            show_errors=args.errors,
                            current_run_only=not args.all,
                            staleness_hours=args.stale_hours)
            print(f"  Next refresh in {args.interval}s  (Ctrl+C to stop)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == '__main__':
    main()
