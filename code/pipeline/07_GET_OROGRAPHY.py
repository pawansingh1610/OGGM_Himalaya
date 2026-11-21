"""
================================================================================
07_GET_OROGRAPHY.py — get / build the grid-cell elevation file per forcing
================================================================================
Writes files to exactly the paths climate_loader_unified.OROG_PATHS expects:

  ERA5      cluster.klima.uni-bremen.de/~oggm/climate/era5/monthly/v1.1/
            era5_invariant.nc           (OGGM server, ERA5 0.25° geopotential)
  CRU_TS4   cru_ts4/cru_elev.nc         (OGGM's CRU CL2.0 10' elevation,
                                         block-averaged onto the CRU TS 0.5°
                                         grid of YOUR tmp file)
  CHELSA    chelsa_w5e5/chelsa-w5e5_orog_HMA.nc
                                        (ISIMIP CHELSA-W5E5 v1.0 orog,
                                         cropped to HMA)
  HARv2     HRAv2/HARv2_d10km_static_hgt.nc   (TU Berlin HAR v2 static hgt)
  ERA5-Land cluster.klima.uni-bremen.de/~oggm/climate/era5-land/monthly/vhma/
            era5_land_invariant_flat_HMA.nc   (OGGM, flat land-points format)

Usage
  python 07_GET_OROGRAPHY.py inspect          # show what you have + diagnose
  python 07_GET_OROGRAPHY.py era5 cru         # download/build those
  python 07_GET_OROGRAPHY.py all
  python 07_GET_OROGRAPHY.py chelsa --file /path/to/downloaded_orog.nc
  python 07_GET_OROGRAPHY.py harv2  --file /path/to/HARv2_d10km_static_hgt.nc
     (--file: use a file you downloaded by hand in the browser instead)

After this: python precheck.py   (section 4 must PASS for all 5 forcings,
                                   section 5 shows plausible offsets)
================================================================================
"""
import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from climate_loader_unified import (detect_cache_root, OROG_PATHS,   # noqa
                                    find_orography_file, grid_elevation,
                                    _resolve)

CACHE = detect_cache_root()
HMA = dict(lat=(25.0, 47.0), lon=(60.0, 106.0))     # crop box

URLS = {
    'ERA5': 'https://cluster.klima.uni-bremen.de/~oggm/climate/era5/monthly/'
            'v1.1/era5_invariant.nc',
    'ERA5-Land': 'https://cluster.klima.uni-bremen.de/~oggm/climate/era5-land/'
                 'monthly/vhma/era5_land_invariant_flat_HMA.nc',
    'ERA5-Land-global': 'https://cluster.klima.uni-bremen.de/~oggm/climate/'
                        'era5-land/monthly/v1.0/era5_land_invariant_flat.nc',
    'CRU_CL2': 'https://cluster.klima.uni-bremen.de/~oggm/climate/cru/'
               'cru_cl2.nc.zip',
    'CHELSA': 'https://files.isimip.org/ISIMIP3a/InputData/climate/atmosphere/'
              'obsclim/global/daily/historical/CHELSA-W5E5v1.0/'
              'chelsa-w5e5v1.0_obsclim_orog_30arcsec_global.nc',
    'HARv2': 'https://data.klima.tu-berlin.de/HAR/v2/d10km/static/'
             'HARv2_d10km_static_hgt.nc',
}
MANUAL = {
    'CHELSA': "Browse https://data.isimip.org → search 'CHELSA-W5E5 orog' "
              "(ISIMIP3a, obsclim, 30arcsec). Download the orog .nc, then run\n"
              "   python 07_GET_OROGRAPHY.py chelsa --file <downloaded.nc>",
    'HARv2': "Browse https://data.klima.tu-berlin.de/HAR/v2/d10km/ → open the "
             "'static' folder → download the file with 'hgt' in its name, then\n"
             "   python 07_GET_OROGRAPHY.py harv2 --file <downloaded.nc>",
}


def _download(url, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    print(f"  downloading {url}\n        -> {dest}")
    try:
        if shutil.which('wget'):
            r = subprocess.run(['wget', '-c', '-O', str(tmp), url])
            ok = r.returncode == 0
        else:
            urllib.request.urlretrieve(url, tmp)
            ok = True
    except Exception as e:
        print(f"  download failed: {e}")
        ok = False
    if ok and tmp.exists() and tmp.stat().st_size > 1000:
        tmp.replace(dest)
        return True
    tmp.unlink(missing_ok=True)
    return False


def _crop(ds):
    lat = next(c for c in ['lat', 'latitude'] if c in ds.coords)
    lon = next(c for c in ['lon', 'longitude'] if c in ds.coords)
    la = ds[lat].values
    sl = slice(*HMA['lat']) if la[0] < la[-1] else slice(*HMA['lat'][::-1])
    return ds.sel({lat: sl, lon: slice(*HMA['lon'])})


def _check(forcing):
    """Grid elevation at 3 test points + compare with climate-grid spacing."""
    pts = [(77.5, 32.2, 'Chhota Shigri'), (76.5, 35.7, 'Baltoro'),
           (78.0, 42.1, 'Tien Shan')]
    for lon, lat, name in pts:
        try:
            r = grid_elevation(forcing, lon, lat, CACHE)
            print(f"    {forcing:9s} {name:13s} grid cell ({r['orog_lat']:.3f},"
                  f"{r['orog_lon']:.3f}) = {r['grid_elev_m']:6.0f} m "
                  f"{r['orog_note']}")
        except Exception as e:
            print(f"    {forcing:9s} {name:13s} ERROR: {e}")


# ─────────────────────────────────────────────────────────────────────────────

def do_inspect():
    print("=" * 74 + "\n  OROGRAPHY INSPECTION\n" + "=" * 74)
    for f in OROG_PATHS:
        try:
            p = find_orography_file(f, CACHE)
        except FileNotFoundError:
            print(f"\n  {f}: MISSING")
            continue
        print(f"\n  {f}: {p}")
        with xr.open_dataset(p) as ds:
            print(f"    dims : {dict(ds.sizes)}")
            for v in ds.data_vars:
                da = ds[v]
                vals = da.values.astype(float) if da.size < 5e7 else \
                    da.isel({d: slice(None, None, 10) for d in da.dims}).values
                if np.isnan(vals).all():
                    print(f"    *** {v} is 100 % NaN — this file cannot be used "
                          f"for {f} ***")
                print(f"    var  : {v} {da.dims} units='{da.attrs.get('units', '')}'"
                      f"  min={np.nanmin(vals):.1f} max={np.nanmax(vals):.1f} "
                      f"NaN={np.isnan(vals).mean() * 100:.1f}%")
            for c in ['latitude', 'lat', 'longitude', 'lon']:
                if c in ds.coords and ds[c].ndim == 1:
                    x = ds[c].values
                    print(f"    {c:9s}: {x.min():.3f}..{x.max():.3f} "
                          f"step {np.diff(x).mean() if len(x) > 1 else 0:.4f}")
        _check(f)


def do_era5(file=None):
    dest = CACHE / OROG_PATHS['ERA5'][3]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if file:
        shutil.copy(file, dest)
    elif not dest.exists() and not _download(URLS['ERA5'], dest):
        print("  ERA5: download failed. Alternative: Copernicus CDS → 'ERA5 "
              "hourly single levels' → variable 'Geopotential', any 1 hour, "
              "save as ERA5/era5_invariant.nc")
        return
    # grid consistency with the climate file
    with xr.open_dataset(_resolve('era5_temp', CACHE)) as c, \
            xr.open_dataset(dest) as o:
        dc = float(abs(np.diff(c['latitude'].values[:2])[0]))
        do = float(abs(np.diff(o['latitude'].values[:2])[0]))
        print(f"  ERA5 climate grid step {dc}°, orography grid step {do}°"
              + ("  OK" if abs(dc - do) < 1e-6 else "  *** MISMATCH ***"))
    _check('ERA5')


def do_era5land(file=None):
    """OGGM's ERA5-Land invariant (flat 'points' format, land points only)."""
    dest = CACHE / OROG_PATHS['ERA5-Land'][0]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if file:
        shutil.copy(file, dest)
    elif not dest.exists() and not _download(URLS['ERA5-Land'], dest):
        dest = CACHE / OROG_PATHS['ERA5-Land'][1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() and not _download(URLS['ERA5-Land-global'], dest):
            print("  ERA5-Land: both OGGM downloads failed")
            return
    # same 0.1° cell as the climate file?
    pts = [(77.5, 32.2), (76.5, 35.7), (78.0, 42.1)]
    with xr.open_dataset(_resolve('era5l_temp', CACHE)) as c:
        la = next(x for x in ['latitude', 'lat'] if x in c.coords)
        lo = next(x for x in ['longitude', 'lon'] if x in c.coords)
        for lon, lat in pts:
            cl = float(c[la].values[np.argmin(abs(c[la].values - lat))])
            lons = c[lo].values
            q = lon + 360 if (lons.max() > 180 and lon < 0) else lon
            cn = float(lons[np.argmin(abs(lons - q))])
            r = grid_elevation('ERA5-Land', lon, lat, CACHE)
            same = abs(cl - r['orog_lat']) < 0.051 and \
                abs((cn - r['orog_lon'] + 180) % 360 - 180) < 0.051
            print(f"    climate cell ({cl:.2f},{cn:.2f})  orog cell "
                  f"({r['orog_lat']:.2f},{r['orog_lon']:.2f}) "
                  f"{r['grid_elev_m']:6.0f} m  " + ("OK" if same else
                                                   "*** DIFFERENT CELL ***"))


def do_cru(file=None):
    """CRU CL2.0 10' elevation -> mean over each 0.5° CRU TS cell."""
    src = Path(file) if file else None
    if src is None:
        z = CACHE / 'cluster.klima.uni-bremen.de/~oggm/climate/cru/cru_cl2.nc.zip'
        nc = z.with_suffix('')
        if not nc.exists():
            if not z.exists() and not _download(URLS['CRU_CL2'], z):
                print("  CRU: download failed. Get CRU CL2.0 elevation "
                      "(grid_10min_elv) from https://crudata.uea.ac.uk/cru/data/"
                      "hrg/tmc/ and rerun with --file")
                return
            import zipfile
            with zipfile.ZipFile(z) as zf:
                zf.extractall(z.parent)
        src = nc
    with xr.open_dataset(src) as s:
        v = next(x for x in ['elev', 'elv', 'z', 'orog'] if x in s.data_vars)
        e = s[v].load()
        if 'time' in e.dims:
            e = e.isel(time=0)
        if float(np.nanmax(e.values)) < 10:       # km -> m
            e = e * 1000.0
        lat_s = next(c for c in ['lat', 'latitude'] if c in e.dims)
        lon_s = next(c for c in ['lon', 'longitude'] if c in e.dims)
    with xr.open_dataset(_resolve('cru_temp', CACHE)) as c:
        tlat, tlon = c['lat'].values, c['lon'].values
    m = (tlat >= HMA['lat'][0]) & (tlat <= HMA['lat'][1])
    n = (tlon >= HMA['lon'][0]) & (tlon <= HMA['lon'][1])
    tlat, tlon = tlat[m], tlon[n]
    out = np.full((len(tlat), len(tlon)), np.nan)
    slat, slon = e[lat_s].values, e[lon_s].values
    ev = e.transpose(lat_s, lon_s).values
    for i, la in enumerate(tlat):
        mi = np.abs(slat - la) <= 0.25 + 1e-6
        for j, lo in enumerate(tlon):
            mj = np.abs(slon - lo) <= 0.25 + 1e-6
            blk = ev[np.ix_(mi, mj)]
            if np.isfinite(blk).any():
                out[i, j] = np.nanmean(blk)
    dest = CACHE / 'cru_ts4/cru_elev.nc'
    dest.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({'elev': (('lat', 'lon'), out.astype('f4'),
                         {'units': 'm', 'long_name':
                          'CRU CL2.0 elevation averaged to CRU TS 0.5 deg cells'})},
               coords={'lat': tlat, 'lon': tlon}).to_netcdf(dest)
    print(f"  CRU: wrote {dest}  ({len(tlat)}x{len(tlon)} cells, "
          f"{np.isfinite(out).mean() * 100:.0f}% land)")
    _check('CRU_TS4')


def _crop_save(src, dest, var_names, label):
    with xr.open_dataset(src) as ds:
        v = next(x for x in var_names if x in ds.data_vars)
        if 'lat' in ds.coords and ds['lat'].ndim == 1 or 'latitude' in ds.coords:
            sub = _crop(ds[[v]]).load()
        else:
            sub = ds.load()          # 2-D curvilinear (HARv2) — keep as is
    dest.parent.mkdir(parents=True, exist_ok=True)
    sub.to_netcdf(dest)
    print(f"  {label}: wrote {dest}  dims {dict(sub.sizes)}")


def do_chelsa(file=None):
    dest = CACHE / OROG_PATHS['CHELSA'][0]
    src = Path(file) if file else CACHE / 'chelsa_w5e5' / Path(URLS['CHELSA']).name
    if not file and not src.exists() and not _download(URLS['CHELSA'], src):
        print("  CHELSA: automatic download failed.\n  " + MANUAL['CHELSA'])
        return
    _crop_save(src, dest, ['orog', 'elev', 'z', 'hgt'], 'CHELSA')
    with xr.open_dataset(_resolve('chelsa_temp', CACHE)) as c, \
            xr.open_dataset(dest) as o:
        la_c = next(x for x in ['lat', 'latitude'] if x in c.coords)
        la_o = next(x for x in ['lat', 'latitude'] if x in o.coords)
        dc = abs(float(np.diff(c[la_c].values[:2])[0]))
        do = abs(float(np.diff(o[la_o].values[:2])[0]))
        print(f"  CHELSA climate step {dc:.5f}°, orog step {do:.5f}°"
              + ("  OK" if abs(dc - do) < 1e-6 else "  *** MISMATCH ***"))
    _check('CHELSA')


def do_harv2(file=None):
    dest = CACHE / OROG_PATHS['HARv2'][0]
    if file:
        _crop_save(Path(file), dest, ['hgt', 'HGT', 'z'], 'HARv2')
    elif not dest.exists() and not _download(URLS['HARv2'], dest):
        print("  HARv2: automatic download failed.\n  " + MANUAL['HARv2'])
        return
    with xr.open_dataset(_resolve('harv2_temp', CACHE)) as c, \
            xr.open_dataset(dest) as o:
        sc = c['lat'].shape
        v = next(x for x in ['hgt', 'HGT', 'z'] if x in o.data_vars)
        so = o[v].shape[-2:]
        print(f"  HARv2 climate grid {sc}, static grid {so}"
              + ("  OK" if tuple(sc) == tuple(so) else "  *** MISMATCH ***"))
    _check('HARv2')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', nargs='+',
                    choices=['inspect', 'era5', 'era5land', 'cru', 'chelsa', 'harv2',
                             'all'])
    ap.add_argument('--file', default=None,
                    help='use this local file instead of downloading')
    a = ap.parse_args()
    todo = (['era5', 'era5land', 'cru', 'chelsa', 'harv2'] if 'all' in a.what
            else a.what)
    print(f"  cache: {CACHE}")
    for w in todo:
        print(f"\n── {w.upper()} " + "─" * 60)
        {'inspect': do_inspect, 'era5': do_era5, 'era5land': do_era5land,
         'cru': do_cru,
         'chelsa': do_chelsa, 'harv2': do_harv2}[w](
            **({} if w == 'inspect' else {'file': a.file}))
    print("\nNext: python precheck.py")


if __name__ == '__main__':
    main()
