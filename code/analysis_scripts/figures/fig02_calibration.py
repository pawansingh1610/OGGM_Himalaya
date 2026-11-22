#!/usr/bin/env python
"""Figure 2 - calibrated parameters per forcing: median temperature bias, precipitation factor and
elevation offset (glacier median elevation minus climate grid-cell elevation), for HMA (top row, with the
inter-quartile range of dT) and for each of the 15 sub-regions.

Input: analysis/A_calibration_diagnostics.csv
  columns forcing, subregion, n, median_dT, iqr_dT, median_pf, median_elev_offset_m
Uses the per-sub-region medians of the analysis (no per-glacier metadata needed).
"""
import numpy as np
import pandas as pd

from style import (AN, CRU_FAMILY, COL, DOUBLE, FORC, INK, INK2, LABEL, LW_THIN, SUBREG_ORDER, Line2D,
                   figure, header, save, short, zero_line)

header(2, "calibration parameters")
A = pd.read_csv(AN / "A_calibration_diagnostics.csv")
rows = ["ALL_HMA"] + SUBREG_ORDER
P = {k: A.pivot(index="subregion", columns="forcing", values=k).reindex(rows)[FORC]
     for k in ["median_dT", "median_pf", "median_elev_offset_m", "iqr_dT"]}
h = A[A.subregion == "ALL_HMA"].set_index("forcing").reindex(FORC)
print(h[["n", "median_dT", "iqr_dT", "median_pf", "median_elev_offset_m"]].round(2).to_string())
for k in ["median_dT", "median_pf", "median_elev_offset_m"]:
    print(f"   sub-region {k}: min/max per forcing")
    print(P[k].iloc[1:].agg(["min", "max"]).round(2).to_string())

fig = figure(DOUBLE, 9.0)
gs = fig.add_gridspec(1, 3)
items = [("median_dT", "(a) Temperature bias ΔT (°C)", 0), ("median_pf", "(b) Precipitation factor $p_f$", 1.5),
         ("median_elev_offset_m", "(c) Elevation offset (m)", 0)]
off = dict(zip(FORC, np.linspace(-0.28, 0.28, len(FORC))))
y = np.arange(len(rows))
axs = []
for k, (col, title, ref) in enumerate(items):
    ax = fig.add_subplot(gs[0, k], sharey=axs[0] if axs else None)
    axs.append(ax)
    ax.axhspan(-0.5, 0.5, color="#f3f2ee", lw=0, zorder=0)
    zero_line(ax, "x", ref)
    for f in FORC:
        v = P[col][f].values
        if col == "median_dT":   # IQR of dT, HMA row only
            q = P["iqr_dT"].loc["ALL_HMA", f]
            ax.plot([v[0] - q / 2, v[0] + q / 2], [off[f]] * 2, color=COL[f], lw=1.2, zorder=2,
                    solid_capstyle="butt")
        cru = f in CRU_FAMILY
        ax.scatter(v, y + off[f], s=11, marker="D" if cru else "o", facecolor=COL[f], edgecolor="white",
                   lw=0.4, zorder=3)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    if k:
        ax.tick_params(labelleft=False)
axs[0].set_yticks(y, ["HMA (all glaciers)"] + [short(s) for s in SUBREG_ORDER])
axs[0].get_yticklabels()[0].set_fontweight("bold")
axs[0].set_ylim(len(rows) - 0.5, -0.6)
axs[1].set_xlim(0.9, None)
fig.legend(handles=[Line2D([], [], marker="D" if f in CRU_FAMILY else "o", ls="none", mfc=COL[f], mec="white",
                           mew=0.4, ms=4.5, label=LABEL[f]) for f in FORC],
           loc="outside lower center", ncol=5)
save(fig, "fig02_calibration")
print("   HMA row: medians over all glaciers (= Table 2); bar in (a) = inter-quartile range of dT. "
      "Other rows: sub-region medians. Circles = ERA5 family, diamonds = CRU family.")
