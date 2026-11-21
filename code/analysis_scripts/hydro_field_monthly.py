"""Monthly mass balance for the field glaciers, all five forcings, with the calibrated parameters of the main run.

Run name: run_h01_field_monthly (output in $HMA_OGGM_ROOT, nothing in the main run is touched).
No recalibration: each glacier is rebuilt exactly as in the pipeline (DEM, elevation-band flowlines, climate file at the
grid-cell reference height) by importing the model code read-only, and the run_07 temp_bias / prcp_fac / melt_f /
lapse_rate of that glacier and forcing are applied. Glacier-wide monthly specific MB is computed on the same fixed
inversion flowlines and with the same width weights as OGGM's get_specific_mb, so the Jan-Dec sum must reproduce the
run_07 mb_annual_mwe (checked per glacier-year, column cal_check_diff).

Output: <OUT>/monthly_<forcing>.csv (rgi_id, year, month, mb_mwe) and <OUT>/check_<forcing>.csv.
Usage: python hydro_field_monthly.py [n_workers]
"""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import OGGM_ROOT, CODE_DIR  # noqa: E402
import os, sys, shutil, tempfile, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd

CODE = CODE_DIR
RUN07 = OGGM_ROOT / "compiled_v3_run_07_tpfix" / "per_forcing"
OUT = OGGM_ROOT / "run_h01_field_monthly"
PAIRS = Path(__file__).resolve().parents[1] / "analysis" / "D_validation_pairs.csv"
FORC = ["ERA5", "ERA5-Land", "CRU_TS4", "CHELSA", "HARv2"]


def _init():
    sys.path.insert(0, str(CODE))
    os.chdir(CODE)


def work(task):
    import warnings; warnings.filterwarnings("ignore")
    import main_calibration_parallel_v2 as mc
    from calibration_v3 import make_mb_model
    from climate_loader_unified import load_climate, grid_elevation, write_oggm_climate_file
    from oggm import cfg, tasks, GlacierDirectory, utils
    from oggm.cfg import SEC_IN_MONTH
    from shapely import wkt
    rid, forcing, ent, par = task["rgi_id"], task["forcing"], task["entity"], task["par"]
    reg = rid[:8]
    rs = mc.REGIONAL_SETTINGS[reg]
    tmp = Path(tempfile.mkdtemp(prefix="h01_", dir=OUT / "work"))
    try:
        cfg.initialize(logging_level="CRITICAL")
        cfg.PARAMS["use_tar_shapefiles"] = False
        cfg.PARAMS["use_intersects"] = False
        cfg.PARAMS["hydro_month_nh"] = 10
        cfg.PARAMS["temp_default_gradient"] = float(par["lapse_rate"])
        cfg.PATHS["working_dir"] = str(tmp)
        cfg.PATHS["dl_cache_dir"] = str(mc.CACHE_ROOT)
        d = dict(ent); d["geometry"] = wkt.loads(d["geometry"])
        (tmp / "pg").mkdir()
        gdir = GlacierDirectory(pd.Series(d), base_dir=str(tmp / "pg"))
        tasks.define_glacier_region(gdir, source=par["dem_source"])
        tasks.glacier_masks(gdir)
        tasks.elevation_band_flowline(gdir)
        tasks.fixed_dx_elevation_band_flowline(gdir)
        lon, lat = float(ent["CenLon"]), float(ent["CenLat"])
        clim, meta = load_climate(forcing, lon, lat, cache_root=mc.CACHE_ROOT)
        ref_hgt = grid_elevation(forcing, lon, lat, mc.CACHE_ROOT)["grid_elev_m"]
        write_oggm_climate_file(clim, gdir, meta, ref_hgt)
        m = make_mb_model(gdir, float(par["temp_bias"]), float(par["prcp_fac"]), float(par["melt_f"]), forcing)
        rho = cfg.PARAMS["ice_density"]
        rows = []
        y0, y1 = int(par["first_mb_year"]), int(par["end_year"])
        for y in range(y0, y1 + 1):
            for mo in range(1, 13):
                fy = utils.date_to_floatyear(y, mo)
                vals, wts = [], []
                for i, fl in enumerate(m.fls):
                    w = getattr(fl, "widths_m", fl.widths)
                    mb = m.get_monthly_mb(fl.surface_h, year=fy, fl_id=i) * SEC_IN_MONTH * rho
                    vals.append(mb); wts.append(w)
                rows.append((rid, y, mo, float(np.average(np.concatenate(vals), weights=np.concatenate(wts))) / 1000.0))
        mon = pd.DataFrame(rows, columns=["rgi_id", "year", "month", "mb_mwe"])
        ann = {y: m.get_specific_mb(year=y) / 1000.0 for y in range(y0, y1 + 1)}
        return dict(ok=True, rgi_id=rid, forcing=forcing, monthly=mon, annual=ann, ref_hgt=ref_hgt)
    except Exception:
        return dict(ok=False, rgi_id=rid, forcing=forcing, err=traceback.format_exc()[-600:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    (OUT / "work").mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(CODE)); os.chdir(CODE)
    import geopandas as gpd
    import main_calibration_parallel_v2 as mc
    ids = sorted(pd.read_csv(PAIRS).rgi_id.dropna().unique())
    ents = {}
    for reg in sorted({i[:8] for i in ids}):
        g = gpd.read_file(mc._find_rgi(reg))
        for _, r in g[g.RGIId.isin(ids)].iterrows():
            ents[r.RGIId] = mc._rgi_row_to_dict(r)
    print(f"{len(ids)} field glaciers, {len(ents)} found in RGI", flush=True)
    for f in FORC:
        if (OUT / f"monthly_{f}.csv").exists():
            print(f"skip {f} (done)", flush=True); continue
        md = pd.read_csv(RUN07 / f / "metadata.csv").set_index("rgi_id")
        y = pd.read_parquet(RUN07 / f / "yearly.parquet", columns=["rgi_id", "year", "mb_annual_mwe"])
        y = y[y.rgi_id.isin(ids)]
        tl = [dict(rgi_id=i, forcing=f, entity=ents[i], par=md.loc[i].to_dict()) for i in ids if i in md.index and i in ents]
        mons, chk, fails = [], [], []
        with ProcessPoolExecutor(max_workers=nw, initializer=_init) as pool:
            for fut in as_completed([pool.submit(work, t) for t in tl]):
                r = fut.result()
                if not r["ok"]:
                    fails.append(r); print(f"FAIL {f} {r['rgi_id']}: {r['err'][-200:]}", flush=True); continue
                mons.append(r["monthly"])
                o = md.loc[r["rgi_id"]]
                for yy, a in r["annual"].items():
                    chk.append((r["rgi_id"], yy, a, float(o["ref_hgt_m"]) - r["ref_hgt"]))
        mon = pd.concat(mons, ignore_index=True)
        cal = mon.groupby(["rgi_id", "year"]).mb_mwe.sum().rename("cal_from_monthly")
        c = pd.DataFrame(chk, columns=["rgi_id", "year", "oggm_annual", "ref_hgt_diff"]).set_index(["rgi_id", "year"])
        c = c.join(cal).join(y.set_index(["rgi_id", "year"]).mb_annual_mwe.rename("run07"))
        c["cal_check_diff"] = c.cal_from_monthly - c.run07
        mon.to_csv(OUT / f"monthly_{f}.csv", index=False)
        c.reset_index().to_csv(OUT / f"check_{f}.csv", index=False)
        print(f"{f}: {mon.rgi_id.nunique()} glaciers, {len(fails)} failed; max |monthly sum - run07| = "
              f"{c.cal_check_diff.abs().max():.2e}, max |ref_hgt diff| = {c.ref_hgt_diff.abs().max():.2f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
