#!/usr/bin/env python
"""Figure 5 in Draft_2 (file 07) - split-sample geodetic test: (a) HMA change in area-weighted mass balance
from 2000-2009 to 2010-2019, modelled vs observed; (b) modelled minus observed change by sub-region.

Input: analysis/E_split_sample.csv (forcing, subregion, dObs_areaw, dMod_areaw, dMod_minus_dObs)
"""
import numpy as np
import pandas as pd

from style import (AN, COL, CRU_FAMILY, DOUBLE, FORC, INK, LABEL, SUBREG_ORDER, figure, header, heat, save,
                   zero_line)

header(7, "split-sample test")
E = pd.read_csv(AN / "E_split_sample.csv")
F4 = [f for f in FORC if f in set(E.forcing)]
h = E[E.subregion == "ALL_HMA"].set_index("forcing").reindex(F4)
obs = h.dObs_areaw.mean()
print(h[["dObs_areaw", "dMod_areaw"]].assign(captured_pct=100 * h.dMod_areaw / h.dObs_areaw).round(3).to_string())
M = E[E.subregion != "ALL_HMA"].pivot(index="subregion", columns="forcing", values="dMod_minus_dObs").reindex(SUBREG_ORDER)[F4]
print("   modelled - observed by sub-region (m w.e. yr-1):\n" + M.round(2).to_string())

fig = figure(DOUBLE, 9.5)
gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.25, 0.05])
ax = fig.add_subplot(gs[0])
y = np.arange(len(F4))
for i, f in enumerate(F4):
    v = h.dMod_areaw.iloc[i]
    ax.barh(i, v, color=COL[f], height=0.6, hatch="////" if f in CRU_FAMILY else None, edgecolor="white", lw=0)
    ax.text(v - 0.003, i, f"{100 * v / h.dObs_areaw.iloc[i]:.0f} %", ha="right", va="center", color=INK)
ax.axvline(obs, color=INK, lw=1.0, ls=(0, (1.5, 1.5)))
ax.text(obs + 0.003, len(F4) - 0.45, f"observed\n{obs:+.3f}".replace("-", "−"), ha="left", va="bottom")
zero_line(ax, "x")
ax.set_xlim(-0.165, 0.0)
ax.set_ylim(len(F4) - 0.5 + 0.9, -0.6)
ax.set_yticks(y, [LABEL[f] for f in F4])
ax.grid(axis="y", visible=False)
ax.set_xlabel("ΔB, 2010–2019 minus 2000–2009\n(m w.e. yr$^{-1}$)")
ax.set_title("(a) HMA: share of acceleration captured")
a = fig.add_subplot(gs[1])
mesh = heat(a, M, "(b) Modelled − observed ΔB (m w.e. yr$^{-1}$)", np.nanmax(np.abs(M.values)), fmtstr="{:+.2f}")
cb = fig.colorbar(mesh, cax=fig.add_subplot(gs[2]))
cb.set_label("+ = too little acceleration"); cb.outline.set_visible(False)
save(fig, "fig07_split_sample")
