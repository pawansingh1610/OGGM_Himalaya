"""
08_DEBUG_one_glacier.py — run ONE glacier in the foreground, full traceback

    python 08_DEBUG_one_glacier.py                       # first glacier, ERA5, RGI60-14
    python 08_DEBUG_one_glacier.py ERA5 RGI60-14.15990   # a given glacier
    python 08_DEBUG_one_glacier.py CHELSA RGI60-13.00001

Writes into the normal output folder, prints every step's result.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pipe', HERE / 'main_calibration_parallel_v2.py')
pipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipe)

forcing = sys.argv[1] if len(sys.argv) > 1 else 'ERA5'
rgi_id = sys.argv[2] if len(sys.argv) > 2 else None
region = rgi_id[:8] if rgi_id else 'RGI60-14'

print(f"forcing={forcing} region={region} cache={pipe.CACHE_ROOT}\n"
      f"output={pipe.OUTPUT_ROOT}")
rgi = gpd.read_file(pipe._find_rgi(region))
row = rgi[rgi.RGIId == rgi_id].iloc[0] if rgi_id else rgi.iloc[0]
print(f"glacier {row.RGIId}  area {row.Area} km2  CenLon {row.CenLon} "
      f"CenLat {row.CenLat}  Zmed {row.Zmed}")

# step-by-step checks outside OGGM first
from climate_loader_unified import load_climate, grid_elevation
t = time.time()
df, meta = load_climate(forcing, float(row.CenLon), float(row.CenLat), pipe.CACHE_ROOT)
print(f"[ok] climate {len(df)} months {df.time.min():%Y-%m}..{df.time.max():%Y-%m} "
      f"({time.time() - t:.1f}s)")
print(f"[ok] grid elevation {grid_elevation(forcing, float(row.CenLon), float(row.CenLat), pipe.CACHE_ROOT)}")
t = time.time()
hug = pipe._hugonnet_rows(row.RGIId)
print(f"[{'ok' if hug else 'FAIL'}] Hugonnet periods: {hug} ({time.time() - t:.1f}s)")

t = time.time()
res = pipe._worker({'rgi_id': row.RGIId, 'rgi_entity': pipe._rgi_row_to_dict(row),
                    'region_code': region, 'forcing': forcing})
print(f"\nworker success={res['success']} ({time.time() - t:.1f}s)")
if not res['success']:
    print(res['error'])
else:
    d = pipe._run_dir(forcing, region) / row.RGIId / 'diagnostics' / 'calibration_metadata.json'
    m = json.loads(d.read_text())
    for k in ['flowline_type', 'dem_source', 'ref_hgt_m', 'glacier_zmed_m',
              'calib_period', 'geodetic_target', 'temp_bias', 'prcp_fac',
              'calib_status', 'geodetic_residual', 'V_ref_km3', 'volume_source',
              'extinct_year']:
        print(f"  {k:18s} {m.get(k)}")
