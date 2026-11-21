#!/usr/bin/env python
"""Bootstrap confidence intervals (resampling glaciers, 1000 draws, fixed seed) for the split-sample test.

Captured fraction of the observed acceleration, HMA, area-weighted as in 06_ANALYSIS_v3.split_sample:
   c_f = sum_g w_g d_mod_{f,g} / sum_g w_g d_obs_{f,g},   d = (2010-2019) - (2000-2009) specific balance, w = A_ref_km2,
over the in-sample glaciers of forcing f with a 2000-2019 calibration. The SAME resampled glaciers (drawn from the union of
glaciers) are used for all forcings, so that differences between forcings are paired. Also: split skill s_f = max(0, 1 - |1 - c_f|)
(c_f with the sign of the observed change; equals 1 - |dMod - dObs|/|dObs|) and the pairwise differences.
Only the sampling variability of the glacier population is resampled; the measurement errors of the decadal Hugonnet maps are not
(they are not available per glacier in the repository), so these intervals are a lower bound of the uncertainty.
Env `oggm`. Output: tables/split_bootstrap.csv, tables/split_bootstrap_pairs.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paths import PER_FORCING  # noqa: E402

FORC = ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4"]        # CHELSA ends in 2016, no split test
NB, SEED = 1000, 20261112
need = ["rgi_id", "calib_period", "calib_status", "backward_vanished", "A_ref_km2", "hug_obs_2000_2010", "hug_obs_2010_2020",
        "hug_mod_2000_2010", "hug_mod_2010_2020", "hug_err_2000_2010", "hug_err_2010_2020"]
D = {}
for f in FORC:
    m = pd.read_csv(PER_FORCING / f / "metadata.csv", usecols=need, low_memory=False)
    m = m[(m.calib_period == "2000-01-01_2020-01-01") & (m.calib_status == "ok") & ~m.backward_vanished.astype(bool)].dropna(subset=need[5:])
    m["d_obs"] = m.hug_obs_2010_2020 - m.hug_obs_2000_2010
    m["d_mod"] = m.hug_mod_2010_2020 - m.hug_mod_2000_2010
    m["e_ind"] = np.hypot(m.hug_err_2000_2010, m.hug_err_2010_2020)     # per-glacier 1 sigma of d_obs, independent errors
    m["e_cor"] = m.hug_err_2000_2010 + m.hug_err_2010_2020              # same, fully correlated (upper bound)
    D[f] = m.set_index("rgi_id")
ids = sorted(set().union(*[set(d.index) for d in D.values()]))
LKP = pd.read_csv(ROOT / "analysis/glacier_lookup.csv").set_index("rgi_id")
GSUB = LKP.subregion.reindex(ids).fillna("none").values; GREG = LKP.rgi_region.reindex(ids).fillna("none").values
US, ISUB = np.unique(GSUB, return_inverse=True); UR, IREG = np.unique(GREG, return_inverse=True)
ix = {g: i for i, g in enumerate(ids)}
N = len(ids)
W, DO, DM, EI, EC = {}, {}, {}, {}, {}
for f, d in D.items():
    r = np.array([ix[g] for g in d.index])
    for dct, col in ((W, "A_ref_km2"), (DO, "d_obs"), (DM, "d_mod"), (EI, "e_ind"), (EC, "e_cor")):
        a = np.zeros(N); a[r] = d[col].values; dct[f] = a
rng = np.random.default_rng(SEED)
cnt = rng.multinomial(N, np.full(N, 1.0 / N), size=NB).astype(float)       # NB x N resampling counts
def captured(mode):
    """NB bootstrap draws of c_f. mode: 'sample' (glacier resampling only), 'ind' / 'cor' (plus Hugonnet decadal errors,
    1 sigma per glacier, independent between glaciers / fully correlated)."""
    r2 = np.random.default_rng(SEED + 1)
    out = {f: np.empty(NB) for f in FORC}
    z = r2.standard_normal(NB)
    zs = r2.standard_normal((NB, len(US))); zr = r2.standard_normal((NB, len(UR)))
    for k in range(NB):
        n = cnt[k]
        eps = r2.standard_normal(N)
        for f in FORC:
            w = W[f] * n
            do = DO[f]
            if mode == "ind":
                do = do + eps * EI[f]
            elif mode == "cor":
                do = do + z[k] * EC[f]
            elif mode == "sub":        # errors independent between the two decades, fully correlated among glaciers of a sub-region
                do = do + zs[k][ISUB] * EI[f]
            elif mode == "reg":        # same, correlated within each of the three RGI regions
                do = do + zr[k][IREG] * EI[f]
            out[f][k] = (w * DM[f]).sum() / (w * do).sum()
    return out


point = {f: (W[f] * DM[f]).sum() / (W[f] * DO[f]).sum() for f in FORC}
MODES = {m_: captured(m_) for m_ in ("sample", "ind", "sub", "reg", "cor")}
c = MODES["sample"]
rows = []
for f in FORC:
    s_pt = max(0.0, 1 - abs(1 - point[f])); s = np.maximum(0, 1 - np.abs(1 - c[f]))
    r = dict(forcing=f, n_glaciers=int((W[f] > 0).sum()), captured=point[f], skill=s_pt)
    for m_, cc in MODES.items():
        r[f"ci_lo_{m_}"], r[f"ci_hi_{m_}"] = np.percentile(cc[f], [2.5, 97.5])
        ss = np.maximum(0, 1 - np.abs(1 - cc[f])); r[f"skill_lo_{m_}"], r[f"skill_hi_{m_}"] = np.percentile(ss, [2.5, 97.5])
    rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(ROOT / "tables" / "split_bootstrap.csv", index=False)
print(T.round(3).to_string(index=False))
pr = []
for i, a in enumerate(FORC):
    for b in FORC[i + 1:]:
        r = dict(a=a, b=b, diff=point[a] - point[b])
        for m_, cc in MODES.items():
            d = cc[a] - cc[b]; lo, hi = np.percentile(d, [2.5, 97.5])
            r[f"ci_lo_{m_}"], r[f"ci_hi_{m_}"], r[f"excl0_{m_}"] = lo, hi, bool(lo > 0 or hi < 0)
        pr.append(r)
P = pd.DataFrame(pr)
P.to_csv(ROOT / "tables" / "split_bootstrap_pairs.csv", index=False)
print(P.round(3).to_string(index=False))
