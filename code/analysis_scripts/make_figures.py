#!/usr/bin/env python
r"""LaTeX tables of the paper (sigma_T = 4 C everywhere).

Inputs (read-only):
  analysis/*.csv                                   analysis folder of the run in paths.py
  $HMA_OGGM_ROOT/compiled_v3_<run>/per_forcing/<F>/metadata.csv   per-glacier calibration
  $HMA_OGGM_ROOT/download_cache/HMA/HMA_region.shp  sub-region polygons

Outputs:
  (figures: see scripts/figures/)
  tables/T*.tex, tables/S*.tex  (Copernicus \tophline/\middlehline/\bottomhline)
  tables/*.csv     the exact numbers each table prints

Run:  conda activate oggm && python scripts/make_figures.py
Sign convention: negative = loss. Units: m w.e. yr-1, km3 ice, Gt.
"""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import OGGM_ROOT  # noqa: E402
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "analysis"
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
TAB = ROOT / "tables"; TAB.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PER_FORCING, RUN, OGGM_ROOT as EXT  # noqa: E402
META = str(PER_FORCING / "{}" / "metadata.csv")
SHP = EXT / "download_cache/HMA/HMA_region.shp"

# ---------------------------------------------------------------- style
# Fixed entity -> colour mapping (validated palette, adjacent pairs pass CVD/normal-vision checks).
# Secondary encoding: line style by forcing family (solid = ERA5 family, dashed = CRU family).
FORC = ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4", "CHELSA"]
LABEL = {"ERA5": "ERA5", "ERA5-Land": "ERA5-Land", "HARv2": "HARv2",
         "CRU_TS4": "CRU TS4", "CHELSA": "CHELSA-W5E5"}
COL = {"ERA5": "#2a78d6", "ERA5-Land": "#eb6834", "HARv2": "#1baf7a",
       "CRU_TS4": "#eda100", "CHELSA": "#e87ba4"}
LS = {"ERA5": "-", "ERA5-Land": "-", "HARv2": "-", "CRU_TS4": "--", "CHELSA": "--"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NEUTRAL = "#f0efec"
# diverging: red = loss / negative, grey midpoint, blue = gain / positive
DIV = LinearSegmentedColormap.from_list(
    "div", ["#8a1f1f", "#e34948", "#f2a3a2", NEUTRAL, "#86b6ef", "#2a78d6", "#0d366b"])

SINGLE = 8.3 / 2.54   # Copernicus one column (in)
DOUBLE = 17.0 / 2.54  # full width (in)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "legend.frameon": False,
    "savefig.dpi": 600, "figure.dpi": 100, "axes.titlelocation": "left",
})

SUBREG_ORDER = [  # shapefile order (o2region codes 13-01 ... 15-03)
    "Hissar Alay", "Pamir (Safed Khirs / West Tarim)", "West Tien Shan",
    "East Tien Shan (Dzhungaria)", "West Kun Lun", "East Kun Lun (Altyn Tagh)",
    "Qilian Shan", "Inner Tibet", "Southeast Tibet", "Hindu Kush", "Karakoram",
    "West Himalaya", "Central Himalaya", "East Himalaya", "Hengduan Shan"]
SHORT = {"Pamir (Safed Khirs / West Tarim)": "Pamir",
         "East Tien Shan (Dzhungaria)": "East Tien Shan",
         "East Kun Lun (Altyn Tagh)": "East Kun Lun"}
CHELSA_FLAG = {"West Kun Lun", "East Kun Lun (Altyn Tagh)", "East Tien Shan (Dzhungaria)"}


def short(s):
    return SHORT.get(s, s)


def legend_handles(forcings=FORC):
    return [Line2D([], [], color=COL[f], ls=LS[f], lw=1.4, label=LABEL[f]) for f in forcings]


def save(fig, name):
    p = FIG / f"{name}.png"
    fig.savefig(p, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {p.relative_to(ROOT)}")


def shade_periods(ax, y0=1950, calib=True, extrap=True):
    """grey = before 1980 (not constrained); blue = 2000-2019 geodetic calibration;
    hatched = 2020-2025 unconstrained extrapolation (pending GlaMBIE check)."""
    ax.axvspan(y0 - 0.5, 1979.5, color="#f0efec", lw=0, zorder=0)
    if calib:
        ax.axvspan(1999.5, 2019.5, color="#cde2fb", alpha=0.45, lw=0, zorder=0)
    if extrap:
        ax.axvspan(2019.5, 2025.5, facecolor="none", edgecolor="#c3c2b7", hatch="////", lw=0, zorder=0)


# ---------------------------------------------------------------- LaTeX helpers
def fmt(x, nd=2, sign=True, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    s = s.replace("-", "$-$")
    return s + ("\\,\\%" if pct else "")


def write_table(name, caption, label, colspec, header_rows, body_rows, note=None,
                star=False, size="\\small"):
    env = "table*" if star else "table"
    L = [f"% generated by scripts/{Path(sys.argv[0]).name} -- do not edit by hand",
         f"\\begin{{{env}}}[t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         size, f"\\begin{{tabular}}{{{colspec}}}", "\\tophline"]
    L += [r + " \\\\" for r in header_rows]
    L.append("\\middlehline")
    for r in body_rows:
        L.append("\\middlehline" if r == "MID" else r + " \\\\")
    L.append("\\bottomhline")
    L.append("\\end{tabular}")
    if note:
        L.append(f"\\belowtable{{{note}}}")
    L.append(f"\\end{{{env}}}")
    p = TAB / f"{name}.tex"
    p.write_text("\n".join(L) + "\n")
    print(f"  -> {p.relative_to(ROOT)}")


def tex_esc(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


# ---------------------------------------------------------------- load
A = pd.read_csv(AN / "A_calibration_diagnostics.csv")
B = pd.read_csv(AN / "B_regional_series.csv")
BE = pd.read_csv(AN / "B_ensemble_series.csv")
from volume_ref import windows_with_v1999  # noqa: E402  (% of in-sample V_1999)
C = windows_with_v1999(pd.read_csv(AN / "C_volume_change_windows.csv"), pd.read_csv(AN / "B_regional_series.csv"))
DO = pd.read_csv(AN / "D_validation_overall.csv")
DD = pd.read_csv(AN / "D_validation_decadal.csv")
DP = pd.read_csv(AN / "D_validation_pairs.csv")
DG = pd.read_csv(AN / "D_geodetic_vs_glaciological.csv")
E = pd.read_csv(AN / "E_split_sample.csv")
FX = pd.read_csv(AN / "F_extinct_glaciers.csv")
FY = pd.read_csv(AN / "F_extinction_by_year.csv")
G = pd.read_csv(AN / "G_positive_mb_fraction.csv")
LK = pd.read_csv(AN / "glacier_lookup.csv")


def load_meta():
    cols = ["rgi_id", "temp_bias", "prcp_fac", "elev_offset_m", "calib_status",
            "resolution", "temp_var", "prcp_var", "first_mb_year", "end_year",
            "calib_period", "area_factor_flag", "V_ref_km3"]
    out = {}
    for f in FORC:
        p = Path(str(META).format(f))
        out[f] = pd.read_csv(p, usecols=cols) if p.exists() else None
    return out


MD, HAVE_META = {}, False   # filled in __main__ (keeps imports from other scripts cheap)


# ================================================================ FIGURES
# Figures: scripts/figures/figNN_*.py (one script per figure).
WLAB = {"recon_1980-1999": "1980–1999", "full_1980-2016": "1980–2016", "fwd_2000-2016": "2000–2016",
        "fwd_2000-2019": "2000–2019", "fwd_2000-2025": "2000–2025 (extrap.)", "recon_1950-1979": "1950–1979",
        "recon_1950-1999": "1950–1999", "full_1950-2019": "1950–2019", "full_1950-2025": "1950–2025"}


# ================================================================ TABLES
def t1_forcings():
    print("T1 forcings")
    if not HAVE_META:
        print("   skipped (metadata not available)"); return
    desc = {"ERA5": "global reanalysis", "ERA5-Land": "land reanalysis (ERA5-driven)",
            "HARv2": "regional reanalysis (WRF)", "CRU_TS4": "station-interpolated",
            "CHELSA": "downscaled W5E5 (ERA5 bias-adjusted to CRU)"}
    VERSION = {"CRU_TS4": " (v4.04)"}   # product version; ends in 2019
    rows, rec = [], []
    for f in FORC:
        m = MD[f].iloc[0]
        cp = m.calib_period.split("_")
        cal = f"{cp[0][:4]}--{int(cp[1][:4]) - 1}"
        res = m.resolution.replace("deg", "$^\\circ$").replace("~", "$\\sim$")
        rows.append(f"{LABEL[f]}{VERSION.get(f, '')} & {desc[f]} & {res} & {int(m.first_mb_year)}--{int(m.end_year)} & {cal} & "
                    f"\\texttt{{{tex_esc(m.temp_var)}}}, \\texttt{{{tex_esc(m.prcp_var)}}}")
        rec.append(dict(forcing=f, resolution=m.resolution, years=f"{m.first_mb_year}-{m.end_year}", calib=cal,
                        vars=f"{m.temp_var},{m.prcp_var}"))
    pd.DataFrame(rec).to_csv(TAB / "T1_forcings.csv", index=False); print(pd.DataFrame(rec).to_string(index=False))
    write_table("T1_forcings", "Climate forcings. Years are complete calendar years with modelled mass balance; "
                "the calibration period is the Hugonnet et al.\\ (2021) period fully covered by the forcing. "
                "The reference height is the elevation of the nearest grid cell in each product's orography.",
                "tab:forcings", "llllll",
                ["Forcing & Type & Grid & Model years & Calibration & Variables"],
                rows, star=True, size="\\scriptsize")


def t2_calibration():
    print("T2 calibration diagnostics (HMA)")
    a = A[A.subregion == "ALL_HMA"].set_index("forcing").reindex(FORC)
    cols = ["n", "pct_ok", "pct_area_unreachable", "pct_tb_bound", "pct_pf_bound", "pct_backward_vanished",
            "median_dT", "iqr_dT", "median_pf", "median_elev_offset_m"]
    a[cols].to_csv(TAB / "T2_calibration.csv"); print(a[cols].round(2).to_string())
    rows = [f"{LABEL[f]} & {r.pct_ok:.2f} & {r.pct_area_unreachable:.2f} & {r.pct_tb_bound:.2f} & {r.pct_pf_bound:.2f} & "
            f"{r.pct_backward_vanished:.2f} & {fmt(r.median_dT)} & {r.iqr_dT:.2f} & {r.median_pf:.2f} & {r.median_elev_offset_m:.0f}"
            for f, r in a.iterrows()]
    write_table("T2_calibration", "Calibration diagnostics for all " + f"{int(a.n.iloc[0]):,}".replace(",", "\\,") + " glaciers per forcing "
                "($\\sigma_T = 4\\,^\\circ$C). Bounds: $\\Delta T \\in [-10, 10]\\,^\\circ$C, $p_f \\in [0.3, 6]$. "
                "Elevation offset = glacier median elevation minus climate grid-cell elevation.",
                "tab:calibration", "lrrrrrrrrr",
                ["Forcing & ok & Unreach. & $\\Delta T$ at & $p_f$ at & Backward & Median & IQR & Median & Elev.",
                 " & (\\%) & area (\\%) & bound (\\%) & bound (\\%) & vanished (\\%) & $\\Delta T$ ($^\\circ$C) & $\\Delta T$ & $p_f$ & offset (m)"],
                rows, star=True, size="\\scriptsize")


def t3_volume():
    print("T3 HMA volume change windows")
    h = C[C.level == "hma"]
    wins = ["recon_1980-1999", "full_1980-2016", "fwd_2000-2016", "fwd_2000-2019", "fwd_2000-2025",
            "MID", "recon_1950-1979", "recon_1950-1999", "full_1950-2019", "full_1950-2025"]
    rows, rec = [], []
    for w in wins:
        if w == "MID": rows.append("MID"); continue
        d = h[h.window == w].set_index("forcing").reindex(FORC)
        for k, (col, unit, nd) in enumerate([("dV_km3", "km$^3$ ice", 0), ("dM_Gt", "Gt", 0),
                                             ("dV_pct_V1999", "\\% of $V_{1999}$", 2)]):
            v = d[col].dropna()
            spread = (f"{v.max() - v.min():.{max(nd, 1) if col == 'dV_pct_V1999' else 0}f}" if len(v) > 1 else "--")
            cells = ["--" if pd.isna(x) else fmt(x, nd) for x in d[col]]
            rows.append(f"{WLAB[w] if k == 0 else ''} & {unit} & " + " & ".join(cells) + f" & {spread}")
        for f, r in d.iterrows():
            rec.append(dict(window=w, forcing=f, dV_km3=r.dV_km3, dM_Gt=r.dM_Gt, V_1999_km3=r.V_1999_km3,
                            dV_pct_V1999=r.dV_pct_V1999, dV_pct_window_start=r.dV_pct))
    df = pd.DataFrame(rec); df.to_csv(TAB / "T3_volume_windows.csv", index=False)
    print(df.pivot(index="window", columns="forcing", values="dV_km3").round(1).to_string())
    print(df.pivot(index="window", columns="forcing", values="dV_pct_V1999").round(2).to_string())
    v99 = h.groupby("forcing").V_1999_km3.first().reindex(FORC)
    vtxt = ", ".join(f"{LABEL[f]} {v:,.0f}" for f, v in v99.items() if pd.notna(v))
    import tables_tex; tables_tex.t3()   # compact layout


def t4_validation():
    print("T4 field validation")
    rows, keep = [], []
    for sub, lab in [("common_all5", "Common subset"), ("all_pairs", "All available pairs")]:
        d = DO[DO.subset == sub].set_index("forcing").reindex(FORC)
        if rows: rows.append("MID")
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{lab}}}}}")
        for f, r in d.iterrows():
            rows.append(f"{LABEL[f]} & {int(r.N)} & {int(r.n_glaciers)} & {fmt(r.bias, 3)} & {r.RMSE:.3f} & {r.r:.3f} & "
                        f"{r.r_anomaly:.3f} & {r.std_mod:.3f} ({r.std_obs:.3f})")
        keep.append(d.assign(subset=sub))
    df = pd.concat(keep); df.to_csv(TAB / "T4_validation.csv"); print(df.round(3).to_string())
    write_table("T4_validation", "Annual mass balance (m w.e.\\ yr$^{-1}$) compared with glaciological field data "
                "(bias = model $-$ observation). Anomaly $r$ uses each glacier's mean removed.",
                "tab:validation", "lrrrrrrr",
                ["Forcing & $N$ & Glaciers & Bias & RMSE & $r$ & Anomaly $r$ & SD mod (obs)"], rows, size="\\scriptsize")


def t5_split():
    print("T5 split-sample test")
    F4 = [f for f in FORC if f in set(E.forcing)]
    h = E[E.subregion == "ALL_HMA"].set_index("forcing").reindex(F4)
    rows = [f"Observed $\\Delta B$ (m w.e.\\ yr$^{{-1}}$) & " + " & ".join(fmt(x, 3) for x in h.dObs_areaw),
            f"Modelled $\\Delta B$ & " + " & ".join(fmt(x, 3) for x in h.dMod_areaw),
            "Captured (\\%) & " + " & ".join(f"{100 * m / o:.0f}" for m, o in zip(h.dMod_areaw, h.dObs_areaw)),
            "Glacier-level $r$ ($\\geq$1 km$^2$) & " + " & ".join(f"{x:.2f}" for x in h.r_glacier_ge1km2),
            "MID",
            "\\multicolumn{5}{l}{\\textit{Modelled $-$ observed $\\Delta B$ by sub-region}}"]
    M = E[E.subregion != "ALL_HMA"].pivot(index="subregion", columns="forcing", values="dMod_minus_dObs").reindex(SUBREG_ORDER)[F4]
    O = E[E.subregion != "ALL_HMA"].groupby("subregion").dObs_areaw.mean()
    for sr, r in M.iterrows():
        rows.append(f"{tex_esc(short(sr))} (obs {fmt(O[sr], 2)}) & " + " & ".join(fmt(x, 2) for x in r))
    out = M.copy(); out["dObs"] = O; out.to_csv(TAB / "T5_split_sample.csv")
    print(h[["dObs_areaw", "dMod_areaw", "r_glacier_ge1km2"]].assign(captured=100 * h.dMod_areaw / h.dObs_areaw).round(3).to_string())
    print(M.round(3).to_string())
    write_table("T5_split_sample", "Split-sample geodetic test: change in area-weighted mass balance from 2000--2009 to "
                "2010--2019 (Hugonnet et al., 2021). Models were calibrated to the 2000--2019 mean only. "
                "Positive misfit = modelled acceleration too weak. CHELSA-W5E5 ends in 2016 and is not tested.",
                "tab:split", "l" + "r" * len(F4), ["& " + " & ".join(LABEL[f] for f in F4)], rows, size="\\scriptsize")


def s1_calibration_subregion():
    print("S1 calibration by sub-region")
    a = A[A.subregion != "ALL_HMA"]
    for key, name, cap, nd in [
        (["median_dT", "median_pf"], "S1a_calibration_subregion_params",
         "Median calibrated temperature bias $\\Delta T$ ($^\\circ$C) and precipitation factor $p_f$ by sub-region.", 2),
        (["pct_area_unreachable", "pct_backward_vanished"], "S1b_calibration_subregion_flags",
         "Glacier area with unreachable calibration (\\%) and glaciers flagged backward-vanished (\\%) by sub-region.", 2)]:
        rows, rec = [], []
        P = {k: a.pivot(index="subregion", columns="forcing", values=k).reindex(SUBREG_ORDER)[FORC] for k in key}
        for sr in SUBREG_ORDER:
            cells = [fmt(P[k].loc[sr, f], nd, sign=(k == "median_dT")) for k in key for f in FORC]
            rows.append(f"{tex_esc(short(sr))} & " + " & ".join(cells))
        pd.concat(P, axis=1).to_csv(TAB / f"{name}.csv"); print(pd.concat(P, axis=1).round(2).to_string())
        ab = ["ERA5", "E5L", "HAR", "CRU", "CHE"]
        HEAD = {"median_dT": "Median $\\Delta T$ ($^\\circ$C)", "median_pf": "Median $p_f$",
                "pct_area_unreachable": "Unreachable area (\\%)", "pct_backward_vanished": "Backward-vanished glaciers (\\%)"}
        hdr = [" & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{{HEAD[k]}}}" for k in key),
               "Sub-region & " + " & ".join(ab * len(key))]
        write_table(name, cap + " E5L = ERA5-Land, HAR = HARv2, CHE = CHELSA-W5E5.", f"tab:{name}",
                    "l" + "r" * (5 * len(key)), hdr, rows, star=True, size="\\scriptsize")


def s2_volume_subregion():
    print("S2 volume change by sub-region")
    s = C[C.level == "subregion"]
    wins = ["recon_1980-1999", "fwd_2000-2016"]
    K = {w: s[s.window == w].pivot(index="unit", columns="forcing", values="dV_km3").reindex(SUBREG_ORDER)[FORC] for w in wins}
    P = {w: s[s.window == w].pivot(index="unit", columns="forcing", values="dV_pct_V1999").reindex(SUBREG_ORDER)[FORC] for w in wins}
    v99 = s[(s.window == "fwd_2000-2016") & (s.forcing == "HARv2")].set_index("unit").V_1999_km3.reindex(SUBREG_ORDER)
    rows = []
    for sr in SUBREG_ORDER:
        for k, (D, nd, lab) in enumerate([(K, 1, "km$^3$"), (P, 1, "\\%")]):
            cells = []
            for w in wins:
                for f in FORC:
                    x = fmt(D[w].loc[sr, f], nd)
                    cells.append(x + ("$^*$" if (f == "CHELSA" and sr in CHELSA_FLAG and x != "--") else ""))
            head = f"{tex_esc(short(sr))} & {v99[sr]:.0f}" if k == 0 else " & "
            rows.append(f"{head} & {lab} & " + " & ".join(cells))
    out = pd.concat({**{("km3",) + (w,): K[w] for w in wins}, **{("pct_V1999",) + (w,): P[w] for w in wins}}, axis=1)
    out[("V1999_HARv2_km3", "", "")] = v99
    out.to_csv(TAB / "S2_volume_subregion.csv")
    print(out.round(2).to_string())
    ab = ["ERA5", "E5L", "HAR", "CRU", "CHE"]
    write_table("S2_volume_subregion", "Sub-region ice-volume change in km$^3$ ice (first line; Gt = 0.9 $\\times$ km$^3$) and in per cent of the "
                "sub-region's consensus volume at the end of 1999 in the same forcing's sample (second line); negative = loss. "
                "The $V_{1999}$ column is from the HARv2 sample (largest retained sample). $^*$ CHELSA-W5E5 flagged (unreachable area $\\geq$ 9\\,\\%).",
                "tab:S2", "lrl" + "r" * 10,
                [" & & & \\multicolumn{5}{c}{1980--1999} & \\multicolumn{5}{c}{2000--2016}",
                 "Sub-region & $V_{1999}$ (km$^3$) & & " + " & ".join(ab * 2)], rows, star=True, size="\\scriptsize")


def s3_decadal():
    print("S3 decadal validation")
    d = DD.copy(); d["fo"] = d.forcing.map({f: i for i, f in enumerate(FORC)})
    d = d.sort_values(["decade", "fo"])
    rows = [f"{int(r.decade)}s & {LABEL[r.forcing]} & {int(r.N)} & {int(r.n_glaciers)} & {fmt(r.bias, 3)} & {r.RMSE:.3f} & "
            f"{fmt(r.r, 2, sign=False)} & {fmt(r.r_anomaly, 2, sign=False)}" for _, r in d.iterrows()]
    d.drop(columns="fo").to_csv(TAB / "S3_validation_decadal.csv", index=False)
    print(d.pivot(index="decade", columns="forcing", values="bias").round(3).to_string())
    write_table("S3_validation_decadal", "Field validation by decade (all available pairs). Decades with few glacier-years ($N$) are uncertain; CHELSA-W5E5 starts in 1979.",
                "tab:S3", "llrrrrrr", ["Decade & Forcing & $N$ & Glaciers & Bias & RMSE & $r$ & Anomaly $r$"], rows,
                size="\\scriptsize")


def s4_prior():
    """Prior sensitivity from the analysis folders of the two test runs (681 glaciers) in $HMA_OGGM_ROOT:
    compiled_v3 (sigma_T = 1.5) and compiled_v3_run_03_sT4 (sigma_T = 4). Volume change in % of V_1999."""
    print("S4 prior sensitivity (test runs compiled_v3 / compiled_v3_run_03_sT4)")
    runs = {"s15": EXT / "compiled_v3" / "analysis", "s4": EXT / "compiled_v3_run_03_sT4" / "analysis"}
    if not all((d / "C_volume_change_windows.csv").exists() for d in runs.values()):
        print("   skipped (test-run analysis folders not available)"); return
    cols = {}
    for tag, an in runs.items():
        Bt = pd.read_csv(an / "B_regional_series.csv"); Ct = pd.read_csv(an / "C_volume_change_windows.csv")
        v99 = Bt[(Bt.year == 1999) & (Bt.level == "hma")].set_index("forcing").V
        c = Ct[(Ct.level == "hma") & (Ct.window == "recon_1980-1999")].set_index("forcing")
        d = pd.read_csv(an / "D_validation_overall.csv"); d = d[d.subset == "common_all5"].set_index("forcing")
        a = pd.read_csv(an / "A_calibration_diagnostics.csv"); a = a[a.subregion == "ALL_HMA"].set_index("forcing")
        cols[f"rmse_{tag}"] = d.RMSE; cols[f"dv8099_{tag}"] = 100 * c.dV_km3 / v99; cols[f"pf_{tag}"] = a.median_pf
    df = pd.DataFrame(cols).reindex(FORC).rename_axis("forcing").reset_index()
    df.to_csv(TAB / "S4_prior_sensitivity.csv", index=False); print(df.round(3).to_string(index=False))
    rows = [f"{LABEL[r.forcing]} & {r.rmse_s15:.3f} & {r.rmse_s4:.3f} & {fmt(r.dv8099_s15)} & {fmt(r.dv8099_s4)} & {r.pf_s15:.2f} & {r.pf_s4:.2f}"
            for r in df.itertuples()]
    write_table("S4_prior_sensitivity", "Prior sensitivity on the 681-glacier test subset for $\\sigma_T = 1.5$ and $4\\,^\\circ$C: "
                "field-validation RMSE (m w.e.\\ yr$^{-1}$, common subset), 1980--1999 volume change (\\% of $V_{1999}$) and median precipitation factor. "
                "The $\\sigma_T = 4\\,^\\circ$C run is an earlier run of the subset than the baseline of Table~\\ref{tab:sens}.",
                "tab:S4", "lrrrrrr", [" & \\multicolumn{2}{c}{RMSE} & \\multicolumn{2}{c}{$\\Delta V$ 1980--1999 (\\%)} & \\multicolumn{2}{c}{Median $p_f$}",
                                      "Forcing & $\\sigma_T=1.5$ & $\\sigma_T=4$ & $\\sigma_T=1.5$ & $\\sigma_T=4$ & $\\sigma_T=1.5$ & $\\sigma_T=4$"], rows,
                size="\\scriptsize")


def s5_extinction():
    print("S5 extinction to 2024 / to 2025")
    rec = []
    for col in FORC + ["consensus"]:
        if col not in FY: continue
        d = FY[["year", col, "n_forcings_available"]].dropna(subset=[col])
        rec.append(dict(set=col, to_2019=int(d.loc[d.year <= 2019, col].sum()),
                        to_2024=int(d.loc[d.year <= 2024, col].sum()) if d.year.max() >= 2024 else np.nan,
                        to_2025=int(d.loc[d.year <= 2025, col].sum()) if d.year.max() >= 2025 else np.nan,
                        last_year=int(d.year.max())))
    df = pd.DataFrame(rec); df.to_csv(TAB / "S5_extinction.csv", index=False); print(df.to_string(index=False))
    a24 = FX[FX.consensus_year <= 2024]
    print(f"   consensus to 2024: n={len(a24)} median A={a24.A_ref_km2.median():.3f} km2; to 2025: n={len(FX)} median A={FX.A_ref_km2.median():.3f} km2")
    # Supplement layout: first run (specific-rate targets, analysis_superseded_run_04_full_fix) next to the current run.
    OLD = ROOT / "analysis_superseded_run_04_full_fix"
    if not HAVE_META or not (OLD / "F_extinction_by_year.csv").exists():
        print("   S5 .tex skipped (needs the per-glacier metadata and the first-run analysis)"); return
    fy0, fx0 = pd.read_csv(OLD / "F_extinction_by_year.csv"), pd.read_csv(OLD / "F_extinct_glaciers.csv")
    m = MD["ERA5"].set_index("rgi_id")          # V_ref and RGI area do not depend on the forcing; cap flag of ERA5
    a_hma, v_hma = LK.area_rgi_km2.sum(), m.V_ref_km3.sum()
    cum = lambda fy, c, y: f"{int(fy.loc[fy.year <= y, c].sum()):,}".replace(",", "\\,")
    rows = [f"to {y} & " + " & ".join(cum(fy0, c, y) for c in ("ERA5", "ERA5-Land", "consensus")) + " & & "
            + " & ".join(cum(FY, c, y) for c in ("ERA5", "ERA5-Land", "consensus")) for y in (2019, 2024, 2025)]
    x0, x1 = fx0[fx0.consensus_year <= 2025], FX[FX.consensus_year <= 2025]
    v0, v1 = m.V_ref_km3.reindex(x0.rgi_id).sum(), m.V_ref_km3.reindex(x1.rgi_id).sum()
    ncap = int(m.area_factor_flag.reindex(x1.rgi_id).astype(str).str.contains("cap").sum())
    two = lambda l, a, b: f"{l} & & & {a} & & & & {b}"
    rows += ["MID", two("Median RGI area (km$^2$)", f"{x0.A_ref_km2.median():.3f}", f"{x1.A_ref_km2.median():.3f}"),
             two("RGI area of 2025 set (km$^2$; \\% of HMA)", f"{x0.A_ref_km2.sum():.1f}; {100 * x0.A_ref_km2.sum() / a_hma:.2f}",
                 f"{x1.A_ref_km2.sum():.1f}; {100 * x1.A_ref_km2.sum() / a_hma:.2f}"),
             two("Volume of 2025 set (km$^3$; \\% of $V_{1999}$)", f"{v0:.2f}; {100 * v0 / v_hma:.3f}", f"{v1:.2f}; {100 * v1 / v_hma:.3f}"),
             two("Capped factor (ERA5)", "--", f"{ncap:,}".replace(",", "\\,"))]
    write_table("S5_extinction", "Glaciers whose modelled volume reaches zero after 2000 (consensus: $\\geq$50\\,\\% of the forcings available, "
                "minimum two, at least three available), for the calibration to the specific geodetic rate and for the area-consistent "
                "targets (used in the paper). Cumulative counts and, for the 2025 set, the share of the RGI area and of $V_{1999}$. "
                "Values after 2019 rely on unconstrained extrapolation and on volume--area scaling; the counts are not robust to the "
                "calibration target and only the area and volume shares should be used.",
                "tab:S5", "lrrrrrrr",
                [" & \\multicolumn{3}{c}{Specific-rate targets} & & \\multicolumn{3}{c}{Area-consistent targets}",
                 " & ERA5 & ERA5-L & Consensus & & ERA5 & ERA5-L & Consensus"], rows, size="\\scriptsize")


def s6_backward_vanished():
    print("S6 backward-vanished glaciers (count vs area)")
    if not HAVE_META:
        print("   skipped"); return
    rec = []
    for f in FORC:
        m = pd.read_csv(Path(str(META).format(f)), usecols=["backward_vanished", "A_ref_km2", "geodetic_target"])
        bv = m[m.backward_vanished == True]
        rec.append(dict(forcing=f, n=len(bv), pct_glaciers=100 * len(bv) / len(m),
                        area_km2=bv.A_ref_km2.sum(), pct_area=100 * bv.A_ref_km2.sum() / m.A_ref_km2.sum(),
                        median_area_km2=bv.A_ref_km2.median(), pct_lt_0p5km2=100 * (bv.A_ref_km2 < 0.5).mean(),
                        median_geod_target=bv.geodetic_target.median()))
    df = pd.DataFrame(rec); df.to_csv(TAB / "S6_backward_vanished.csv", index=False); print(df.round(3).to_string(index=False))
    rows = [f"{LABEL[r.forcing]} & " + f"{r.n:,}".replace(",", "\\,") + f" & {r.pct_glaciers:.2f} & {r.area_km2:.1f} & {r.pct_area:.2f} & {r.median_area_km2:.3f} & "
            f"{r.pct_lt_0p5km2:.0f} & {fmt(r.median_geod_target)}" for r in df.itertuples()]
    write_table("S6_backward_vanished", "Glaciers whose backward integration reaches zero volume before the start of the "
                "forcing (flag \\texttt{backward\\_vanished}); excluded from regional series. Last column: median Hugonnet et al.\\ (2021) "
                "2000--2019 target (m w.e.\\ yr$^{-1}$) of the flagged glaciers.",
                "tab:S6", "lrrrrrrr",
                ["Forcing & $n$ & \\% glaciers & Area (km$^2$) & \\% area & Median area (km$^2$) & \\% $<$0.5 km$^2$ & Median target"],
                rows, size="\\scriptsize")


if __name__ == "__main__":
    MD = load_meta()
    HAVE_META = all(v is not None for v in MD.values())
    print(f"per-glacier metadata: {'yes' if HAVE_META else 'NO'}")
    for fn in [t1_forcings, t2_calibration, t3_volume, t4_validation, t5_split,
               s1_calibration_subregion, s2_volume_subregion, s3_decadal, s4_prior,
               s5_extinction, s6_backward_vanished]:
        fn()
