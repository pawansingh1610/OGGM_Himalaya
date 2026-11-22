#!/usr/bin/env python
"""Comparison with the HMA values of Malles and Marzeion (2021, The Cryosphere 15, 3135-3157, Table 3: ensemble-mean regional
mass change rate, mm SLE yr-1, RGI regions 13, 14, 15).
Marzeion et al. (2015) give HMA values only as plots (their Fig. 2), so they are not used numerically.
Our side: regional series analysis/B_regional_series.csv (level hma, dm_gt), mean over the same calendar years, per forcing
covering every year. Conversion: 362 Gt per mm SLE (ocean area 3.62e8 km2; mass loss = positive SLE).
Geodetic reference for the overlapping years: Hugonnet et al. (2021) per-glacier rates x RGI area, summed over HMA
(hug_obs_* x A_ref_km2 in the calibration metadata of the paths.py run), for 2000-2009, 2010-2019 and 2000-2019.
Output: tables/literature_compare.csv."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MALLES = {  # period: (R13, R14, R15) mm SLE yr-1, Malles & Marzeion (2021) Table 3; uncertainties summed linearly below
    "1941-1980": ((0.05, 0.01), (0.04, 0.01), (0.02, 0.01)),
    "1971-2018": ((0.049, 0.006), (0.034, 0.005), (0.015, 0.005)),
    "1993-2018": ((0.052, 0.005), (0.034, 0.006), (0.017, 0.003)),
    "2006-2018": ((0.057, 0.007), (0.04, 0.01), (0.018, 0.005)),
}
GT_PER_MM = 362.0
B = pd.read_csv(ROOT / "analysis/B_regional_series.csv")
B = B[B.level == "hma"]
rows = []
for per, v in MALLES.items():
    a, b = map(int, per.split("-"))
    m_sle, m_unc = sum(x[0] for x in v), sum(x[1] for x in v)
    r = dict(period=per, malles_mmSLE=m_sle, malles_unc=m_unc, malles_Gt_per_yr=-m_sle * GT_PER_MM)
    y0 = max(a, 1950)
    for f, g in B.groupby("forcing"):
        s = g.set_index("year").dm_gt.reindex(range(y0, b + 1))
        r[f"{f}_Gt_per_yr"] = s.mean() if s.notna().all() else float("nan")
    r["our_years"] = f"{y0}-{b}"
    rows.append(r)
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from paths import PER_FORCING  # noqa: E402
m = pd.read_csv(PER_FORCING / "ERA5" / "metadata.csv", usecols=["rgi_id", "A_ref_km2", "hug_obs_2000_2010", "hug_obs_2010_2020", "hug_obs_2000_2020"])
hug = {k: (m[f"hug_obs_{k}"] * m.A_ref_km2).sum() / 1000 for k in ["2000_2010", "2010_2020", "2000_2020"]}
print("Hugonnet per-glacier sums (Gt/yr): 2000-2009 %.1f, 2010-2019 %.1f, 2000-2019 %.1f" % tuple(hug.values()))
for r in rows:
    r.update(hugonnet_Gt_2000_2009=hug["2000_2010"], hugonnet_Gt_2010_2019=hug["2010_2020"], hugonnet_Gt_2000_2019=hug["2000_2020"])
T = pd.DataFrame(rows)
T.to_csv(ROOT / "tables/literature_compare.csv", index=False)
print(T.round(2).to_string(index=False))
