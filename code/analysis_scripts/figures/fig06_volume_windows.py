#!/usr/bin/env python
"""Figure file 06 - volume change per window: (a) HMA, km3 ice, five forcings;
(b, c) sub-regions, % of the sub-region's V_1999, 1980-1999 and 2000-2016 (common colour scale).

Inputs: analysis/C_volume_change_windows.csv (level, unit, forcing, window, dV_km3)
        analysis/B_regional_series.csv (V at 1999 -> V_1999, via scripts/volume_ref.py)
"""
import numpy as np
import pandas as pd

from style import (AN, CHELSA_FLAG, COL, CRU_FAMILY, DOUBLE, FORC, INK2, LABEL, SUBREG_ORDER, figure, header,
                   heat, save, zero_line)
from volume_ref import windows_with_v1999

header(6, "volume-change windows")
C = windows_with_v1999(pd.read_csv(AN / "C_volume_change_windows.csv"), pd.read_csv(AN / "B_regional_series.csv"))
WIN = ["recon_1980-1999", "full_1980-2016", "fwd_2000-2016", "fwd_2000-2019", "fwd_2000-2025"]
WLAB = {"recon_1980-1999": "1980–1999", "full_1980-2016": "1980–2016", "fwd_2000-2016": "2000–2016",
        "fwd_2000-2019": "2000–2019", "fwd_2000-2025": "2000–2025\n(extrapolated)"}
h = C[C.level == "hma"]
P = h.pivot(index="window", columns="forcing", values="dV_km3").reindex(WIN)[FORC]
print("   HMA dV (km3 ice):\n" + P.round(0).to_string())
print("   HMA dV (% of V_1999):\n" + h.pivot(index="window", columns="forcing", values="dV_pct_V1999").reindex(WIN)[FORC].round(2).to_string())

fig = figure(DOUBLE, 16.5)
gs = fig.add_gridspec(2, 3, height_ratios=[0.75, 1.6], width_ratios=[1, 1, 0.05])
ax = fig.add_subplot(gs[0, :])
w = 0.16
for k, f in enumerate(FORC):
    x = np.arange(len(WIN)) + (k - 2) * w
    ax.bar(x, P[f], width=w * 0.86, color=COL[f], label=LABEL[f], lw=0,
           hatch="////" if f in CRU_FAMILY else None, edgecolor="white")
ax.set_xticks(range(len(WIN)), [WLAB[x] for x in WIN])
zero_line(ax)
ax.grid(axis="x", visible=False)
ax.set_ylabel("ΔV (km$^3$ ice)")
ax.legend(ncol=5, loc="lower left")
v1 = h.V_1999_km3.median() / 100
ax.text(1.0, 1.01, f"1 % of $V_{{1999}}$ ≈ {v1:.0f} km$^3$ ice ≈ {0.9 * v1:.0f} Gt", transform=ax.transAxes,
        ha="right", va="bottom", color=INK2)  # above the axes, clear of the bars
ax.set_title("(a) HMA volume change per window")
print(f"   1 % of V_1999 (median over forcings) = {v1:.1f} km3 = {0.9 * v1:.1f} Gt")

s = C[C.level == "subregion"]
flags = {(sr, "CHELSA") for sr in CHELSA_FLAG}
Ms = {win: s[s.window == win].pivot(index="unit", columns="forcing", values="dV_pct_V1999").reindex(SUBREG_ORDER)[FORC]
      for win in ["recon_1980-1999", "fwd_2000-2016"]}
vl = max(np.nanmax(np.abs(M.values)) for M in Ms.values())
for col, (win, t) in enumerate([("recon_1980-1999", "(b) 1980–1999, % of sub-region $V_{1999}$"),
                                ("fwd_2000-2016", "(c) 2000–2016, % of sub-region $V_{1999}$")]):
    a = fig.add_subplot(gs[1, col])
    mesh = heat(a, Ms[win], t, vl, flags=flags)
    if col:
        a.tick_params(labelleft=False)
    print(f"   sub-regions {WLAB[win]} (% of V_1999):\n" + Ms[win].round(1).to_string())
cb = fig.colorbar(mesh, cax=fig.add_subplot(gs[1, 2]))
cb.set_label("ΔV (% of $V_{1999}$; negative = loss)"); cb.outline.set_visible(False)
save(fig, "fig06_volume_windows")
print(f"   colour scale: ±{vl:.1f} %; * = CHELSA-W5E5 flagged (West/East Kun Lun, East Tien Shan; explained in the caption)")
