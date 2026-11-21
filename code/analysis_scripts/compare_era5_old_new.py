#!/usr/bin/env python
"""Old (run_04_full, ERA5 x RGI-13 at sigma_T = 1.5) vs new (run_04_full_fix, sigma_T = 4 everywhere):
every ERA5 number quoted in FULL_RUN_RESULTS, plus the cross-forcing quantities that depend on ERA5
(spreads, rankings, consensus extinction). Writes tables/era5_old_vs_new.csv and .txt.

Usage: compare_era5_old_new.py OLD_ANALYSIS_DIR NEW_ANALYSIS_DIR
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD, NEW = Path(sys.argv[1]), Path(sys.argv[2])
F = "ERA5"
rows = []


def add(sec, item, o, n, fmt="{:.2f}"):
    f = lambda v: "" if pd.isna(v) else (fmt.format(v) if isinstance(v, (int, float)) else str(v))
    d = (n - o) if all(isinstance(v, (int, float)) and not pd.isna(v) for v in (o, n)) else None
    rows.append({"section": sec, "item": item, "old": f(o), "new": f(n), "change": "" if d is None else f(d)})


def both(name, **kw):
    return pd.read_csv(OLD / name, **kw), pd.read_csv(NEW / name, **kw)


# A. calibration (HMA)
ao, an = both("A_calibration_diagnostics.csv")
ao, an = [d[(d.forcing == F) & (d.subregion == "ALL_HMA")].iloc[0] for d in (ao, an)]
for c, lab in [("pct_ok", "calib ok (% glaciers)"), ("pct_area_unreachable", "unreachable (% area)"),
               ("pct_tb_bound", "dT at bound (%)"), ("pct_pf_bound", "pf at bound (%)"),
               ("pct_backward_vanished", "backward_vanished (% glaciers)"), ("median_dT", "median dT (degC)"),
               ("iqr_dT", "IQR dT (degC)"), ("median_pf", "median pf"),
               ("median_elev_offset_m", "median Zmed - grid (m)")]:
    add("A calibration", lab, float(ao[c]), float(an[c]))
# per-region medians (from metadata would need the compiled data; sub-regions in RGI-13 instead)
so, sn = both("A_calibration_diagnostics.csv")
for s in ["Pamir (Safed Khirs / West Tarim)", "West Kun Lun", "Karakoram", "Inner Tibet", "West Tien Shan"]:
    for c in ["median_dT", "median_pf"]:
        o = so[(so.forcing == F) & (so.subregion == s)][c]
        n = sn[(sn.forcing == F) & (sn.subregion == s)][c]
        if len(o) and len(n):
            add("A calibration (sub-region)", f"{s}: {c}", float(o.iloc[0]), float(n.iloc[0]))

# B. sample: from regional series n (HMA) - number of glaciers used
bo, bn = both("B_regional_series.csv")
for d, tag in [(bo, "o"), (bn, "n")]:
    pass
no = bo[(bo.forcing == F) & (bo.level == "hma")].n.max()
nn = bn[(bn.forcing == F) & (bn.level == "hma")].n.max()
add("B sample", "glaciers in regional series", float(no), float(nn), "{:,.0f}")

# C. volume windows (HMA)
co, cn = both("C_volume_change_windows.csv")
key = ["forcing", "level", "unit", "window"]
co = co[co.level == "hma"].set_index(key)
cn = cn[cn.level == "hma"].set_index(key)
for w in sorted(set(co.xs(F).index.get_level_values("window"))):
    o, n = co.loc[(F, "hma", "ALL_HMA", w)], cn.loc[(F, "hma", "ALL_HMA", w)]
    add("C volume", f"{w}: dV % of window start", float(o.dV_pct), float(n.dV_pct))
    add("C volume", f"{w}: dV km3", float(o.dV_km3), float(n.dV_km3), "{:.1f}")
    add("C volume", f"{w}: dM Gt", float(o.dM_Gt), float(n.dM_Gt), "{:.1f}")
o = co.loc[(F, "hma", "ALL_HMA", "fwd_2000-2019")].V_start_km3
n = cn.loc[(F, "hma", "ALL_HMA", "fwd_2000-2019")].V_start_km3
add("C volume", "V end-1999 in sample (km3)", float(o), float(n), "{:,.1f}")
# spreads across forcings
for w in sorted(set(co.index.get_level_values("window"))):
    po = co.xs(w, level="window").dV_pct
    pn = cn.xs(w, level="window").dV_pct
    add("C spread", f"{w}: min..max across forcings (%)", f"{po.min():.2f}..{po.max():.2f} (n={len(po)})",
        f"{pn.min():.2f}..{pn.max():.2f} (n={len(pn)})")
    add("C spread", f"{w}: spread (pp)", float(po.max() - po.min()), float(pn.max() - pn.min()))

# D. field validation
do, dn = both("D_validation_overall.csv")
for sub in ["all_pairs", "common_all5", "common_ERA5_ERA5L_HARv2"]:
    o = do[(do.subset == sub) & (do.forcing == F)].iloc[0]
    n = dn[(dn.subset == sub) & (dn.forcing == F)].iloc[0]
    for c in ["N", "bias", "RMSE", "r", "r_anomaly", "std_mod"]:
        add(f"D field ({sub})", c, float(o[c]), float(n[c]), "{:.0f}" if c == "N" else "{:.3f}")
for sub in ["common_all5"]:
    for c in ["bias", "RMSE", "r_anomaly"]:
        asc = c != "r_anomaly"
        ro = do[do.subset == sub].sort_values(c, ascending=asc, key=lambda s: s.abs() if c == "bias" else s).forcing
        rn = dn[dn.subset == sub].sort_values(c, ascending=asc, key=lambda s: s.abs() if c == "bias" else s).forcing
        add("D ranking (best first)", f"{sub} {c}", " > ".join(ro), " > ".join(rn))
ddo, ddn = both("D_validation_decadal.csv")
for dec in [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]:
    o = ddo[(ddo.forcing == F) & (ddo.decade == dec)]
    n = ddn[(ddn.forcing == F) & (ddn.decade == dec)]
    if len(o):
        add("D field decadal (all pairs)", f"{dec}s bias", float(o.bias.iloc[0]), float(n.bias.iloc[0]), "{:.3f}")

# E. split test
eo, en = both("E_split_sample.csv")
for d in (eo, en):
    d["captured_pct"] = d.dMod_areaw / d.dObs_areaw * 100
o = eo[(eo.forcing == F) & (eo.subregion == "ALL_HMA")].iloc[0]
n = en[(en.forcing == F) & (en.subregion == "ALL_HMA")].iloc[0]
add("E split test", "observed dB (m w.e./yr)", o.dObs_areaw, n.dObs_areaw, "{:.3f}")
add("E split test", "modelled dB (m w.e./yr)", o.dMod_areaw, n.dMod_areaw, "{:.3f}")
add("E split test", "captured (%)", o.captured_pct, n.captured_pct, "{:.0f}")
add("E split test", "glacier r (>=1 km2)", o.r_glacier_ge1km2, n.r_glacier_ge1km2, "{:.3f}")
rk = lambda d: " > ".join(d[d.subregion == "ALL_HMA"].sort_values("dMod_minus_dObs", key=abs).forcing)
add("E ranking (best first)", "|dMod - dObs| HMA", rk(eo), rk(en))
for s in ["West Himalaya", "Hindu Kush", "East Himalaya", "Karakoram", "West Tien Shan", "Inner Tibet"]:
    o = eo[(eo.forcing == F) & (eo.subregion == s)].dMod_minus_dObs
    n = en[(en.forcing == F) & (en.subregion == s)].dMod_minus_dObs
    add("E split test (sub-region)", f"{s}: dMod - dObs", float(o.iloc[0]), float(n.iloc[0]), "{:+.3f}")

# F. extinction
fo, fn = both("F_extinction_by_year.csv")
for d in (fo, fn):
    d.set_index("year", inplace=True)
for y in [2019, 2024, 2025]:
    add("F extinction (cumulative)", f"ERA5 by {y}", float(fo.loc[:y, F].sum()), float(fn.loc[:y, F].sum()), "{:.0f}")
    add("F extinction (cumulative)", f"consensus by {y}", float(fo.loc[:y, "consensus"].sum()),
        float(fn.loc[:y, "consensus"].sum()), "{:.0f}")

# G. positive-MB fraction (ERA5, area) mean over decades
go, gn = both("G_positive_mb_fraction.csv")
for a, b in [(1950, 1979), (1980, 1999), (2000, 2019)]:
    o = go[(go.forcing == F) & go.year.between(a, b)].frac_area.mean() * 100
    n = gn[(gn.forcing == F) & gn.year.between(a, b)].frac_area.mean() * 100
    add("G positive-MB area fraction", f"{a}-{b} mean (%)", o, n, "{:.1f}")

t = pd.DataFrame(rows)
(ROOT / "tables").mkdir(exist_ok=True)
t.to_csv(ROOT / "tables/era5_old_vs_new.csv", index=False)
txt = t.to_string(index=False)
(ROOT / "tables/era5_old_vs_new.txt").write_text(
    f"# generated by scripts/{Path(sys.argv[0]).name}\n# old: {OLD}\n# new: {NEW}\n" + txt + "\n")
print(txt)
