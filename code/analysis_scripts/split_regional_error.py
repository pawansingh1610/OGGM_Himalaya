#!/usr/bin/env python
"""Split-test captured fractions with the published REGIONAL uncertainty of Hugonnet et al. (2021), Extended Data Table 1
(5-year mean elevation change rates with 95 % intervals; values read from data_external/Hugonnet2021_ExtendedDataTable1.jpg):
region 13: 0.09/0.07/0.08/0.10, region 14: 0.17/0.11/0.10/0.14, region 15: 0.16/0.13/0.14/0.17 m yr-1 (2000-04/05-09/10-14/15-19).
Decadal mean error = combination of the two 5-year errors; the acceleration error combines both decades; two limits for the temporal
correlation of the 5-year errors (independent / fully correlated); regions independent; 1 sigma = 95 % interval / 1.96; dh -> mass with 0.85 (density
850 kg m-3, as Hugonnet et al.). The captured fraction is dMod / (dObs + eps), eps ~ N(0, sigma_acc) (Monte Carlo, 100000 draws); dMod, dObs from
analysis/E_split_sample.csv (HMA). Output tables/split_regional_error.csv. Env oggm."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
E5 = {"RGI60-13": [0.09, 0.07, 0.08, 0.10], "RGI60-14": [0.17, 0.11, 0.10, 0.14], "RGI60-15": [0.16, 0.13, 0.14, 0.17]}
AREA = {"RGI60-13": 49303, "RGI60-14": 33568, "RGI60-15": 14734}
def sigma_acc(corr):
    out = {}
    for r, e in E5.items():
        e = np.array(e) / 1.96 * 0.85                      # 1 sigma, m w.e. yr-1
        if corr == "independent":
            d1 = np.hypot(e[0], e[1]) / 2; d2 = np.hypot(e[2], e[3]) / 2
        else:
            d1 = (e[0] + e[1]) / 2; d2 = (e[2] + e[3]) / 2
        out[r] = np.hypot(d1, d2)                          # decades independent of each other in the difference
    A = np.array([AREA[r] for r in out]); s = np.array(list(out.values()))
    return out, float(np.sqrt(((A * s) ** 2).sum()) / A.sum())
E = pd.read_csv(ROOT / "analysis/E_split_sample.csv"); E = E[E.subregion == "ALL_HMA"].set_index("forcing")
rng = np.random.default_rng(1); rows = []
for corr in ("independent", "correlated"):
    sr, sh = sigma_acc(corr)
    print(corr, "regional sigma_acc", {k: round(v, 3) for k, v in sr.items()}, "HMA", round(sh, 3))
    eps = rng.standard_normal(100000) * sh
    for f in ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4"]:
        dO, dM = E.loc[f, "dObs_areaw"], E.loc[f, "dMod_areaw"]
        c = dM / (dO + eps); lo, hi = np.percentile(c, [2.5, 97.5])
        rows.append(dict(corr=corr, forcing=f, sigma_acc_HMA=sh, dObs=dO, dMod=dM, captured=dM / dO, lo=lo, hi=hi))
T = pd.DataFrame(rows); T.to_csv(ROOT / "tables/split_regional_error.csv", index=False); print(T.round(3).to_string(index=False))
