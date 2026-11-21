#!/usr/bin/env python
"""Reconciliation of the 2000-2019 HMA volume change: per member and for the ensemble (glacier count, V1999 used, dV, % of V1999).
Ensemble variants: (A) annual weights over the members present in each year (CHELSA 2000-2016 only; what gridded_best.py did first),
(B) members that cover every year of the window (4 members; the rule of Table 7). Env oggm. Output tables/vol_reconcile_2000_2019.csv"""
import sys, io, contextlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
with contextlib.redirect_stdout(io.StringIO()):
    from gridded_best import stack, best, load, LK, FORC
A, ids, yrs = stack(2000, 2019, "dm_gt", FORC)
dV = A / 0.9
v99 = load("ERA5", 1999, 1999).set_index("rgi_id").volume_km3.reindex(ids).values
rows = []
def add(name, X):
    ok = ~np.isnan(X) & ~np.isnan(v99)
    rows.append(dict(member=name, n_glaciers=int(ok.sum()), V1999_km3=v99[ok].sum(), dV_km3=X[ok].sum(), pct=100 * X[ok].sum() / v99[ok].sum()))
for i, f in enumerate(FORC):
    add(f, np.where(np.isnan(dV[i]).any(-1), np.nan, dV[i].sum(-1)))
Ba = best(dV, FORC); add("ensemble A: members present each year (5 -> 4)", np.where(np.isnan(Ba).any(-1), np.nan, Ba.sum(-1)))
four = [f for f in FORC if f != "CHELSA"]; sel = [FORC.index(f) for f in four]
Bb = best(dV[sel], four); add("ensemble B: 4 members covering all years", np.where(np.isnan(Bb).any(-1), np.nan, Bb.sum(-1)))
# the 2000-2016 sub-window, five members (CHELSA present) vs four
k = (yrs <= 2016)
for nm, D, fl in (("2000-2016 five members", dV[:, :, k], FORC), ("2000-2016 four members", dV[sel][:, :, k], four)):
    Bx = best(D, fl); add("ensemble " + nm, np.where(np.isnan(Bx).any(-1), np.nan, Bx.sum(-1)))
for i, f in enumerate(FORC):
    X = dV[i][:, k]; add(f + " 2000-2016", np.where(np.isnan(X).any(-1), np.nan, X.sum(-1)))
T = pd.DataFrame(rows); T.to_csv(ROOT / "tables/vol_reconcile_2000_2019.csv", index=False)
print(T.round(2).to_string(index=False))
