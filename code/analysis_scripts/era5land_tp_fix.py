#!/usr/bin/env python
"""Replace ERA5-Land monthly total precipitation for Sep 2022 - Feb 2024 (ECMWF known issue: incorrect accumulated
variables in 'monthly averaged reanalysis', announced 2 Apr 2024; workaround: 'monthly averaged reanalysis by hour of
day' at 00:00).  https://forum.ecmwf.int/t/issue-affecting-era5-land-monthly-averaged-reanalysis-for-the-period-september-2022-to-february-2024/2370

Steps (env `oggm` + `pip install cdsapi`; needs ~/.cdsapirc and the ERA5-Land licence accepted on the CDS website):
  1. download  months 2022-09 .. 2024-02 (affected) and CONTROL months (2022-08, 2024-03, 2021-07, 2019-01) from the
               'by hour of day' product, time 00:00, total_precipitation, global, into <cache>/era5_land_data/raw/fix/
  2. verify    (a) control months: new == old file (max abs diff) - if not, the two products differ in convention and
                   NOTHING is written;
               (b) affected months: HMA-box mean ratio new / ERA5 must be 0.95-1.06 in every month (the unaffected
                   months are 1.00-1.01).
  3. patch     write era5_land_tp_1950_2026_patched.nc = copy of the original with only the 18 affected time steps
               replaced. The original file is never modified. The climate loader
               (climate_loader_unified._REL_PATHS) finds the patched file first.
Usage: python scripts/era5land_tp_fix.py download|verify|patch
"""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import DOWNLOAD_CACHE  # noqa: E402
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = DOWNLOAD_CACHE
RAW = CACHE / "era5_land_data/raw"
FIXDIR = RAW / "fix"
ORIG = RAW / "era5_land_tp_1950_2026.nc"
PATCHED = RAW / "era5_land_tp_1950_2026_patched.nc"
ERA5 = CACHE / "ERA5/era5_tp.nc"
AFFECTED = [p for p in pd.period_range("2022-09", "2024-02", freq="M")]
CONTROL = [pd.Period(x, "M") for x in ("2022-08", "2024-03", "2021-07", "2019-01")]
HMA = dict(lat=(26.7, 46.0), lon=(67.0, 105.0))


def download():
    import cdsapi
    FIXDIR.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()
    for p in AFFECTED + CONTROL:
        out = FIXDIR / f"era5land_tp_hod00_{p}.nc"
        if out.exists():
            continue
        c.retrieve("reanalysis-era5-land-monthly-means", {
            "product_type": "monthly_averaged_reanalysis_by_hour_of_day",
            "variable": "total_precipitation", "year": str(p.year), "month": f"{p.month:02d}",
            "time": "00:00", "data_format": "netcdf", "download_format": "unarchived"}, str(out))
        print("downloaded", out.name, flush=True)


def _field(path):
    import xarray as xr
    ds = xr.open_dataset(path)
    v = next(k for k in ds.data_vars if ds[k].ndim >= 2 and "tp" in k.lower() or k == "tp")
    da = ds[v]
    for d in list(da.dims):
        if d not in ("latitude", "longitude", "lat", "lon"):
            da = da.isel({d: 0})
    return da.rename({"lat": "latitude", "lon": "longitude"}) if "lat" in da.dims else da


def _hma_mean(da):
    la = da.latitude.values
    s = slice(HMA["lat"][1], HMA["lat"][0]) if la[0] > la[-1] else slice(*HMA["lat"])
    return float(da.sel(latitude=s, longitude=slice(*HMA["lon"])).mean())


def verify():
    import xarray as xr
    old, e5 = xr.open_dataset(ORIG), xr.open_dataset(ERA5)
    to = list(pd.to_datetime(old.valid_time.values).to_period("M"))
    te = list(pd.to_datetime(e5.valid_time.values).to_period("M"))
    ok = True
    for p in CONTROL:
        new = _field(FIXDIR / f"era5land_tp_hod00_{p}.nc")
        o = old.tp.isel(valid_time=to.index(p))
        same_mask = bool((np.isnan(new.values) == np.isnan(o.values)).all())      # land-only product: ocean = NaN in both
        diff = float(np.nanmax(np.abs(new.values - o.values)))
        rel = diff / float(np.nanmax(np.abs(o.values)))
        print(f"control {p}: same land mask {same_mask}; max|new-old| = {diff:.3e} m ({rel:.2e} of max)")
        ok &= same_mask and rel < 1e-3
    for p in AFFECTED:
        new = _field(FIXDIR / f"era5land_tp_hod00_{p}.nc")
        e = e5.tp.isel(valid_time=te.index(p))
        r = _hma_mean(new) / _hma_mean(e)
        print(f"affected {p}: HMA-box mean new/ERA5 = {r:.3f}")
        ok &= 0.95 <= r <= 1.06
    print("VERIFY", "PASSED" if ok else "FAILED - do not patch")
    return ok


def patch():
    import shutil
    import xarray as xr
    if not verify():
        raise SystemExit("verification failed; patched file not written")
    if PATCHED.exists():
        raise SystemExit(f"{PATCHED} exists; refusing to overwrite")
    shutil.copy2(ORIG, PATCHED)
    ds = xr.open_dataset(ORIG)
    to = list(pd.to_datetime(ds.valid_time.values).to_period("M"))
    import netCDF4
    with netCDF4.Dataset(PATCHED, "r+") as nc:
        v = nc.variables["tp"]
        for p in AFFECTED:
            new = _field(FIXDIR / f"era5land_tp_hod00_{p}.nc")
            ref = ds.tp.isel(valid_time=to.index(p))
            new = new.reindex(latitude=ref.latitude, longitude=ref.longitude, method="nearest", tolerance=1e-3)
            v[to.index(p), :, :] = new.values.astype("float32")
        nc.setncattr("history_fix", "tp for 2022-09..2024-02 replaced by monthly_averaged_reanalysis_by_hour_of_day 00:00 "
                                    "(ECMWF known issue, 2024-04-02); made by scripts/era5land_tp_fix.py")
    print("wrote", PATCHED)


if __name__ == "__main__":
    {"download": download, "verify": verify, "patch": patch}[sys.argv[1]]()
