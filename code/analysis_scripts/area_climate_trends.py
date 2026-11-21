#!/usr/bin/env python
"""1975-1999 temperature and precipitation of each climate product over the six areas of the KH-9 comparison (same samples as
scripts/kh9_compare.py): area-weighted annual T_zmed (lapse-corrected to the glacier median elevation, as in climate_series.py)
and P (product value), mean and linear trend (per decade). Env oggm. Usage: python scripts/area_climate_trends.py
Output: tables/area_climate_trends.csv"""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import CODE_DIR  # noqa: E402
import sys
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(CODE_DIR))
from climate_loader_unified import _convert_precip, _resolve, detect_cache_root
from climate_series import KEYS, META, dims_of
LK = pd.read_csv(ROOT / "analysis/glacier_lookup.csv").set_index("rgi_id")
Y0, Y1 = 1975, 1999
FIRST = {"ERA5": 1950, "ERA5-Land": 1950, "CRU_TS4": 1950, "CHELSA": 1979, "HARv2": 1980}
him = LK.subregion.isin(["West Himalaya", "Central Himalaya", "East Himalaya"]) & LK.CenLon.between(75, 93)
top = him & (LK.area_rgi_km2.rank(ascending=False, method="first") <= 0)   # placeholder, replaced below
hid = LK[him].sort_values("area_rgi_km2", ascending=False).head(650).index
box = lambda a: LK.index[(LK.CenLon.between(a[0], a[1])) & (LK.CenLat.between(a[2], a[3]))]
AREAS = {"Himalaya 650 largest (Maurer)": set(hid), "Karakoram (Zhou)": set(box((74.33, 78.25, 33.75, 37.5))),
         "Purogangri": set(box((88.5, 89.7, 33.5, 34.4))), "Poiqu": set(box((85.3, 86.2, 27.8, 28.5))),
         "W Nyainqentanglha": set(box((90.0, 91.2, 30.0, 30.8))), "Ak-Shirak": set(box((77.8, 78.8, 41.5, 42.2)))}
rows = []
for F in ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4", "CHELSA"]:
    m = pd.read_csv(META.format(F), usecols=["rgi_id", "grid_lat", "grid_lon", "ref_hgt_m", "glacier_zmed_m", "lapse_rate", "A_ref_km2"]).dropna()
    root = detect_cache_root(); tp, pp = (_resolve(k, root) for k in KEYS[F])
    dt, dp = xr.open_dataset(tp), xr.open_dataset(pp)
    tv = next(v for v in dt.data_vars if dt[v].ndim >= 3 and "bnds" not in v); pv = next(v for v in dp.data_vars if dp[v].ndim >= 3 and "bnds" not in v)
    t_dim, la, lo = dims_of(dt[tv]); p_dim = dims_of(dp[pv])[0]; punit = dp[pv].attrs.get("units", "mm")
    times = pd.to_datetime(dt[t_dim].values); ptimes = pd.to_datetime(dp[p_dim].values)
    sub = {a: m[m.rgi_id.isin(ids)].copy() for a, ids in AREAS.items()}
    allm = pd.concat(sub.values()).drop_duplicates(["grid_lat", "grid_lon"])
    if la == "south_north":
        lat2, lon2 = dt["lat"].values.astype(float), dt["lon"].values.astype(float)
        idx = {(a_, b_): np.unravel_index(np.argmin((lat2 - a_) ** 2 + (lon2 - b_) ** 2), lat2.shape) for a_, b_ in zip(allm.grid_lat, allm.grid_lon)}
    else:
        lav, lov = dt[la].values, dt[lo].values
        lonq = lambda x: x + 360 if (lov.max() > 180 and x < 0) else x
        idx = {(a_, b_): (int(np.abs(lav - a_).argmin()), int(np.abs(lov - lonq(b_)).argmin())) for a_, b_ in zip(allm.grid_lat, allm.grid_lon)}
    iy = np.array([v[0] for v in idx.values()]); ix = np.array([v[1] for v in idx.values()]); keys = list(idx.keys())
    pos = {k: i for i, k in enumerate(keys)}
    ser = {a: {"T": [], "P": []} for a in AREAS}; yrs = [y for y in range(Y0, Y1 + 1)]
    ok_years = [y for y in yrs if y >= FIRST[F]]
    for y in ok_years:
        sel = np.where(times.year == y)[0]; psel = np.where(ptimes.year == y)[0]
        if len(sel) != 12 or len(psel) != 12: ok_years = [z for z in ok_years if z != y]; continue
        def read(ds, v, tdim, s):
            blk = ds[v].isel({tdim: s, la: slice(int(iy.min()), int(iy.max()) + 1), lo: slice(int(ix.min()), int(ix.max()) + 1)}).values
            return blk[:, iy - iy.min(), ix - ix.min()]
        T = read(dt, tv, t_dim, sel).astype(float); P = read(dp, pv, p_dim, psel).astype(float)
        if np.nanmean(T) > 100: T = T - 273.15
        Pa = np.stack([_convert_precip(P[i], punit) for i in range(12)]).sum(0); Ta = T.mean(0)
        for a, g in sub.items():
            if not len(g): continue
            ci = np.array([pos[(r.grid_lat, r.grid_lon)] for r in g.itertuples()]); w = g.A_ref_km2.values
            off = g.lapse_rate.values * (g.glacier_zmed_m.values - g.ref_hgt_m.values)
            ser[a]["T"].append(float(np.average(Ta[ci] + off, weights=w))); ser[a]["P"].append(float(np.average(Pa[ci], weights=w)))
    for a in AREAS:
        n = len(ok_years)
        if not n or not len(sub[a]): continue
        x = np.array(ok_years); tT = np.polyfit(x, ser[a]["T"], 1)[0] * 10; tP = np.polyfit(x, ser[a]["P"], 1)[0] * 10
        rows.append(dict(area=a, forcing=F, years=f"{ok_years[0]}-{ok_years[-1]}", n_years=n, T_mean=np.mean(ser[a]["T"]), T_trend_per_decade=tT, P_mean_mm=np.mean(ser[a]["P"]), P_trend_mm_per_decade=tP, P_trend_pct_per_decade=100 * tP / np.mean(ser[a]["P"]), n_glaciers=len(sub[a])))
    print("done", F, flush=True)
T = pd.DataFrame(rows); T.to_csv(ROOT / "tables/area_climate_trends.csv", index=False)
pd.set_option("display.width", 220)
print(T.pivot_table(index="area", columns="forcing", values="T_trend_per_decade").round(2).to_string())
print(T.pivot_table(index="area", columns="forcing", values="P_trend_pct_per_decade").round(1).to_string())
print(T.pivot_table(index="area", columns="forcing", values="P_mean_mm").round(0).to_string())
