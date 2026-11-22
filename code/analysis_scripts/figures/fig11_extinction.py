#!/usr/bin/env python
"""Supplement figure (file 11) - glaciers reaching zero volume after 2000 (indicative),
run_06_areacons (area-consistent targets) and run_04_full_fix (specific-rate targets) side by side: (a) per year, consensus
(bars) and run_06 per forcing (lines); (b) RGI area of the consensus sets.

Inputs: analysis/F_extinction_by_year.csv (year, <forcing>, consensus), analysis/F_extinct_glaciers.csv (A_ref_km2, consensus_year)
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from style import (AN, ROOT, COL, CRU_FAMILY, DOUBLE, FORC, GAIN, HATCH_EXT, INK2, LABEL, LS, figure, header,
                   legend_handles, save)

header(11, "extinction (indicative)")
FY = pd.read_csv(AN / "F_extinction_by_year.csv")
FX = pd.read_csv(AN / "F_extinct_glaciers.csv")
OLD = ROOT / "analysis_superseded_run_04_full_fix"
FYo = pd.read_csv(OLD / "F_extinction_by_year.csv"); FXo = pd.read_csv(OLD / "F_extinct_glaciers.csv")
fig = figure(DOUBLE, 6.5)   # panels side by side
axs = fig.subplots(1, 2, width_ratios=[1.4, 1])
ax = axs[0]
ax.axvspan(2019.5, 2025.5, facecolor="none", edgecolor=HATCH_EXT, hatch="////", lw=0, zorder=0)
d = FY[FY.year >= 2005]
do = FYo[FYo.year >= 2005]
ax.bar(d.year - 0.2, d.consensus, color=INK2, width=0.4, zorder=2)
ax.bar(do.year + 0.2, do.consensus, color="#b0b0b0", width=0.4, zorder=2)
for f in FORC:
    if f in d:
        ax.plot(d.year, d[f], color=COL[f], ls=LS[f], marker="D" if f in CRU_FAMILY else "o", ms=2.2, zorder=3)
ax.set_ylabel("Glaciers reaching V = 0 per year")
ax.set_title("(a) Modelled disappearance per year")
ax.text(2019.3, 0.6, "2020–2025:\nextrapolation", transform=ax.get_xaxis_transform(), ha="right", va="center",
        color=INK2)
ax.set_xlim(2004.5, 2025.8); ax.set_xticks(range(2005, 2026, 5))
ax.legend(handles=[Patch(color=INK2, label="area-consistent target"), Patch(color="#b0b0b0", label="specific-rate target (first run)")] + legend_handles(), loc="upper left", fontsize=7, ncol=2, handlelength=1.8)
ax = axs[1]
bins = np.logspace(np.log10(max(FX.A_ref_km2.min(), 1e-3)), np.log10(FX.A_ref_km2.max()), 20)
ax.hist(FX.A_ref_km2, bins=bins, color=GAIN, edgecolor="white", lw=0.75, label="area-consistent")
ax.hist(FXo.A_ref_km2, bins=bins, color="#b0b0b0", edgecolor="white", lw=0.75, label="specific-rate")
ax.legend(fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("RGI area (km$^2$)")
ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5], ["0.02", "0.05", "0.1", "0.2", "0.5"]); ax.minorticks_off()
ax.set_ylabel("Glaciers")
n24 = int((FX.consensus_year <= 2024).sum())
ax.set_title("(b) RGI area of the consensus sets")
print(f"   consensus set: {n24} to 2024, {len(FX)} to 2025")
save(fig, "fig11_extinction")
cum = {c: {y: int(FY.loc[FY.year <= y, c].sum()) for y in (2019, 2024, 2025)} for c in ["consensus"] + [f for f in FORC if f in FY]}
print("   cumulative (to 2019 / 2024 / 2025):", cum)
print(f"   consensus n={len(FX)} median area={FX.A_ref_km2.median():.3f} km2  <0.05 km2: {100 * (FX.A_ref_km2 < 0.05).mean():.0f} %"
      f"  years before 2005 with consensus>0: {int((FY[FY.year < 2005].consensus > 0).sum())}")
