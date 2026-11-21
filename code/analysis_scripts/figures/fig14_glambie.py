#!/usr/bin/env python
"""Figure 9 in Draft_2 (file 14) - annual mass balance 2000-2025 of the forcings and the best estimate
against GlaMBIE (calendar years, 95 % interval) for RGI 13, 14, 15 and HMA (a-d); field glaciers with data in both
2010-2019 and 2020-2024 (e).

Inputs (written by scripts/glambie_check.py): tables/glambie_annual.csv, tables/glambie_field_2020s.csv
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from style import (COL, CRU_FAMILY, DOUBLE, FORC, HATCH_EXT, INK, INK2, LABEL, LS, MUTED, TAB, Line2D, figure,
                   header, legend_handles, save, zero_line)

header(14, "GlaMBIE check")
ann = pd.read_csv(TAB / "glambie_annual.csv")
F = pd.read_csv(TAB / "glambie_field_2020s.csv", index_col=0)
NAME = {"RGI60-13": "RGI 13 Central Asia", "RGI60-14": "RGI 14 South Asia West", "RGI60-15": "RGI 15 South Asia East",
        "ALL_HMA": "HMA"}
UNITS = list(NAME)
for u in UNITS:
    a = ann[ann.unit == u]
    for p in [(2000, 2019), (2020, 2023)]:
        d = a[a.year.between(*p)]
        print(f"   {NAME[u]:24s} {p[0]}-{p[1]}: best {d.B_best.mean():+.3f}  GlaMBIE cal {d.glambie_cal_mwe.mean():+.3f}")
print("   field glaciers (mean of glacier means):\n" + F.round(3).to_string())

fig = figure(DOUBLE, 11.5)
gs = fig.add_gridspec(2, 3)
axs = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(5)]
ylims = []
for k, u in enumerate(UNITS):
    ax = axs[k]
    a = ann[ann.unit == u].sort_values("year")
    ax.axvspan(2019.5, 2025.5, facecolor="none", edgecolor=HATCH_EXT, hatch="////", lw=0, zorder=0)
    g = a.dropna(subset=["glambie_cal_mwe"])
    ax.fill_between(g.year, g.glambie_cal_mwe - g.glambie_cal_mwe_err, g.glambie_cal_mwe + g.glambie_cal_mwe_err,
                    color="#d9d8d2", lw=0, zorder=1)
    ax.plot(g.year, g.glambie_cal_mwe, color=INK2, lw=1.2, marker="s", ms=2.5, zorder=3)
    for f in FORC:
        d = a.dropna(subset=[f"B_{f}"])
        ax.plot(d.year, d[f"B_{f}"], color=COL[f], ls=LS[f], lw=0.8, zorder=2)
    ax.plot(a.year, a.B_best, color=INK, lw=1.5, zorder=4)
    zero_line(ax)
    ax.set_title(f"({'abcd'[k]}) {NAME[u]}")
    ax.set_xlim(1999.5, 2025.5)
    if k in (0, 3):
        ax.set_ylabel("Mass balance (m w.e. yr$^{-1}$)")
ax = axs[4]
x = np.array([0, 1])
ax.bar(x - 0.27, [F.loc["obs", "2010-2019"], F.loc["obs", "2020-2024"]], color=INK2, width=0.22, lw=0, label="observed")
for i, f in enumerate(["ERA5", "ERA5-Land", "HARv2"]):
    ax.bar(x - 0.08 + 0.16 * i, [F.loc[f, "2010-2019"], F.loc[f, "2020-2024"]], width=0.15, color=COL[f], lw=0)
ax.set_xticks(x, ["2010–2019", "2020–2024"])
zero_line(ax)
ax.grid(axis="x", visible=False)
ax.set_ylabel("Mass balance (m w.e. yr$^{-1}$)")
ax.set_title(f"(e) Field glaciers ({int(F.n_glaciers.iloc[0])}, same in both)")
lg = fig.add_subplot(gs[1, 2]); lg.axis("off")
handles = [Line2D([], [], color=INK2, lw=1.2, marker="s", ms=2.5, label="GlaMBIE (calendar years)"),
           Patch(color="#d9d8d2", label="GlaMBIE 95 % interval"),
           Line2D([], [], color=INK, lw=1.5, label="best estimate"),
           Patch(color=INK2, label="observed (e)")] + legend_handles(lw=0.9)
lg.legend(handles=handles, loc="upper left", ncol=1, fontsize=7, borderaxespad=0)
lg.text(0, 0.0, "Hatched: 2020–2025, beyond the\ngeodetic calibration period.\nGlaMBIE ends in 2023.",
        va="bottom", color=INK2, transform=lg.transAxes)
save(fig, "fig14_glambie")
