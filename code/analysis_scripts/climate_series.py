#!/usr/bin/env python
"""Glacier-area-weighted annual climate of the five forcings, from the files the model actually used.

Run with the conda env `oggm` (xarray, netCDF4). Reads $HMA_OGGM_ROOT (download_cache + run_06 metadata), writes
tables/climate_series_<forcing>.csv, one row per calendar year:
  T_grid   annual mean 2 m temperature at the climate grid cell (deg C), area-weighted over all glaciers
  T_zmed   the same, lapse-corrected to each glacier's median elevation with the model's regional lapse rate
           (T_cell + lapse * (glacier_zmed - ref_hgt); ref_hgt = grid-cell elevation, as in the model)
  P        annual total precipitation at the grid cell (mm yr-1), area-weighted; NOT multiplied by the calibrated pf
Weights: RGI 6.0 glacier area (A_ref_km2 in per_forcing/<F>/metadata.csv). Cells: the nearest grid cell of each
glacier (metadata grid_lat/grid_lon), identical to the cell used in the calibration. Unit conversions are the
model's own (climate_loader_unified._convert_precip). Only complete calendar years; years with any missing
cell-month are dropped and counted.
Usage: python climate_series.py ERA5 | ERA5-Land | HARv2 | CRU_TS4 | CHELSA
"""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import OGGM_ROOT, CODE_DIR  # noqa: E402
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(CODE_DIR))
from climate_loader_unified import _convert_precip, _resolve, detect_cache_root  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
META = str(OGGM_ROOT / "compiled_v3_run_06_areacons" / "per_forcing") + "/{}/metadata.csv"
KEYS = {"ERA5": ("era5_temp", "era5_precip"), "ERA5-Land": ("era5l_temp", "era5l_precip"),
        "HARv2": ("harv2_temp", "harv2_precip"), "CRU_TS4": ("cru_temp", "cru_precip"),
        "CHELSA": ("chelsa_temp", "chelsa_precip")}
LAST_FULL = {"ERA5": 2025, "ERA5-Land": 2025, "HARv2": 2025, "CRU_TS4": 2019, "CHELSA": 2016}


def dims_of(da):
    t = next(d for d in da.dims if d in ("time", "valid_time"))
    if "south_north" in da.dims:
        return t, "south_north", "west_east"
    la = next(d for d in da.dims if d in ("latitude", "lat"))
    lo = next(d for d in da.dims if d in ("longitude", "lon"))
    return t, la, lo


def main(F):
    m = pd.read_csv(META.format(F), usecols=["grid_lat", "grid_lon", "ref_hgt_m", "glacier_zmed_m", "lapse_rate",
                                           "A_ref_km2"]).dropna()
    A = m.A_ref_km2.values
    off = float((A * m.lapse_rate * (m.glacier_zmed_m - m.ref_hgt_m)).sum() / A.sum())
    cells = m.groupby(["grid_lat", "grid_lon"], as_index=False).A_ref_km2.sum()
    w = cells.A_ref_km2.values / cells.A_ref_km2.sum()
    print(F, "glaciers", len(m), "cells", len(cells), "area-weighted lapse offset (K)", round(off, 3), flush=True)
    root = detect_cache_root()
    tp, pp = (_resolve(k, root) for k in KEYS[F])
    dt, dp = xr.open_dataset(tp), xr.open_dataset(pp)
    tv = next(v for v in dt.data_vars if dt[v].ndim >= 3 and "bnds" not in v)
    pv = next(v for v in dp.data_vars if dp[v].ndim >= 3 and "bnds" not in v)
    t_dim, la, lo = dims_of(dt[tv])
    p_dim = dims_of(dp[pv])[0]
    punit = dp[pv].attrs.get("units", "mm")
    # cell indices
    if la == "south_north":
        lat2, lon2 = dt["lat"].values.astype(float), dt["lon"].values.astype(float)
        idx = [np.unravel_index(np.argmin((lat2 - a) ** 2 + (lon2 - b) ** 2), lat2.shape)
               for a, b in zip(cells.grid_lat, cells.grid_lon)]
        iy, ix = (np.array(z) for z in zip(*idx))
    elif "points" in dt[tv].dims:
        raise SystemExit("flat 'points' grid not handled")
    else:
        lav, lov = dt[la].values, dt[lo].values
        lon_q = np.where((lov.max() > 180) & (cells.grid_lon.values < 0), cells.grid_lon.values + 360, cells.grid_lon.values)
        iy = np.abs(lav[None, :] - cells.grid_lat.values[:, None]).argmin(1)
        ix = np.abs(lov[None, :] - lon_q[:, None]).argmin(1)
    ydim, xdim = (la, lo)
    times = pd.to_datetime(dt[t_dim].values)
    years = sorted(set(times.year[(times.year >= 1950) & (times.year <= LAST_FULL[F])]))
    rows, dropped = [], []
    for y in years:
        sel = np.where(times.year == y)[0]
        if len(sel) != 12:
            dropped.append((y, "months", len(sel)))
            continue
        # a time slice read with the bounding box of the cells, then picked pointwise
        def read(ds, v, tdim, sel_t):
            blk = ds[v].isel({tdim: sel_t, ydim: slice(int(iy.min()), int(iy.max()) + 1),
                              xdim: slice(int(ix.min()), int(ix.max()) + 1)}).values
            return blk[:, iy - iy.min(), ix - ix.min()]
        T = read(dt, tv, t_dim, sel).astype(float)
        ptimes = pd.to_datetime(dp[p_dim].values)
        psel = np.where(ptimes.year == y)[0]
        Pm = read(dp, pv, p_dim, psel).astype(float)
        if np.isnan(T).any() or np.isnan(Pm).any() or len(psel) != 12:
            dropped.append((y, "nan", int(np.isnan(T).sum() + np.isnan(Pm).sum())))
            continue
        if np.nanmean(T) > 100:
            T = T - 273.15
        Pc = np.stack([_convert_precip(Pm[i], punit) for i in range(12)])
        t_ann = T.mean(0) @ w
        rows.append(dict(forcing=F, year=y, T_grid=t_ann, T_zmed=t_ann + off, P=float(Pc.sum(0) @ w)))
        print(y, round(t_ann, 3), round(float(Pc.sum(0) @ w), 1), flush=True)
    out = pd.DataFrame(rows)
    out["n_cells"] = len(cells)
    out["lapse_offset_K"] = off
    out.to_csv(ROOT / "tables" / f"climate_series_{F}.csv", index=False)
    print("dropped years:", dropped)


if __name__ == "__main__":
    main(sys.argv[1])
