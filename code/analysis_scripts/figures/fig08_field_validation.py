#!/usr/bin/env python
"""Figure 6 in Draft_2 (file 08) - modelled vs glaciological annual mass balance on the common five-forcing
subset (a-e) and bias by decade on all pairs (f).

Inputs: analysis/D_validation_pairs.csv (rgi_id, year, obs, <forcing>), analysis/D_validation_overall.csv
        (subset == common_all5; bias, RMSE, r, r_anomaly), analysis/D_validation_decadal.csv (decade, forcing, bias, N)
"""
import numpy as np
import pandas as pd

from style import (AN, COL, CRU_FAMILY, DOUBLE, FORC, INK, INK2, LABEL, LS, LW_THIN, ZERO, figure, header,
                   legend_handles, save, zero_line)

header(8, "field validation")
DP = pd.read_csv(AN / "D_validation_pairs.csv")
DO = pd.read_csv(AN / "D_validation_overall.csv")
DD = pd.read_csv(AN / "D_validation_decadal.csv")
common = DP.dropna(subset=FORC)
print(f"   common subset: {len(common)} glacier-years, {common.rgi_id.nunique()} glaciers")
fig = figure(DOUBLE, 12.5)
axs = fig.subplots(2, 3)
lim = (-2.6, 1.7)
for k, (ax, f) in enumerate(zip(axs.flat, FORC)):
    ax.plot(lim, lim, color=ZERO, lw=LW_THIN, zorder=1)
    ax.scatter(common.obs, common[f], s=5, marker="D" if f in CRU_FAMILY else "o", facecolor=COL[f],
               edgecolor="none", alpha=0.7, zorder=2)
    ref = DO[(DO.subset == "common_all5") & (DO.forcing == f)].iloc[0]
    d = common[f] - common.obs
    assert abs(d.mean() - ref.bias) < 1e-3 and abs(np.sqrt((d ** 2).mean()) - ref.RMSE) < 1e-3, f
    ax.text(0.04, 0.96, f"bias {ref.bias:+.2f}\nRMSE {ref.RMSE:.2f}\n$r$ {ref.r:.2f}\nanomaly $r$ {ref.r_anomaly:.2f}".replace("-", "−"),
            transform=ax.transAxes, va="top")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_title(f"({'abcde'[k]}) {LABEL[f]}")
    ax.set_xlabel("Observed (m w.e. yr$^{-1}$)")
    if k % 3 == 0:
        ax.set_ylabel("Modelled (m w.e. yr$^{-1}$)")
    print(f"   {LABEL[f]:12s} bias {ref.bias:+.3f} RMSE {ref.RMSE:.3f} r {ref.r:.3f} anomaly r {ref.r_anomaly:.3f} (cross-checked)")
ax = axs[1, 2]
for f in FORC:
    d = DD[DD.forcing == f].sort_values("decade")
    ax.plot(d.decade + 5, d.bias, color=COL[f], ls=LS[f], marker="D" if f in CRU_FAMILY else "o", ms=2.8)
n = DD.groupby("decade").N.max()
for dec, nn in n.items():
    ax.text(dec + 5, -0.33, f"{int(nn)}", ha="center", color=INK2)
zero_line(ax)
ax.set_ylim(-0.4, 0.75)
ax.set_title("(f) Bias by decade, all pairs")
ax.set_xlabel("Decade (numbers: max. N)"); ax.set_ylabel("Bias (m w.e. yr$^{-1}$)")
ax.legend(handles=legend_handles(), loc="upper right", fontsize=7, handlelength=1.8)
print("   decadal bias:\n" + DD.pivot(index="decade", columns="forcing", values="bias")[FORC].round(2).to_string())
print("   max N per decade:", n.astype(int).to_dict())
save(fig, "fig08_field_validation")
