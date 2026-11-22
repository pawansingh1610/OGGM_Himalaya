#!/usr/bin/env python
"""Figure file 09 - glaciological minus geodetic (Hugonnet et al. 2021) mean mass balance
2000-2019, glaciers with >= 3 glaciological years in that period.

Input: analysis/D_geodetic_vs_glaciological.csv (name, n_2000_2019, glac_minus_geod)
"""
import numpy as np
import pandas as pd

from style import AN, GAIN, INK, LOSS, SINGLE, figure, header, save, zero_line

header(9, "glaciological minus geodetic")
d = pd.read_csv(AN / "D_geodetic_vs_glaciological.csv").sort_values("glac_minus_geod")
fig = figure(SINGLE, 14.5)
ax = fig.subplots()
y = np.arange(len(d))
ax.barh(y, d.glac_minus_geod, color=np.where(d.glac_minus_geod < 0, LOSS, GAIN), height=0.7, lw=0)
ax.set_yticks(y, [n.title() for n in d.name], fontsize=7)
m = d.glac_minus_geod.mean()
ax.axvline(m, color=INK, ls=(0, (1.5, 1.5)), lw=1.0)
ax.text(m - 0.02, -1.2, f"mean {m:+.2f}".replace("-", "−"), ha="right", va="center")
zero_line(ax, "x")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Glaciological $-$ geodetic MB,\n2000–2019 (m w.e. yr$^{-1}$)", fontsize=6.5)
fig.subplots_adjust(left=0.5, right=0.98, top=0.98, bottom=0.12)
ax.set_ylim(len(d) - 0.4, -1.8)
save(fig, "fig09_glac_vs_geod")
print(f"   n={len(d)} mean={m:+.3f} median={d.glac_minus_geod.median():+.3f} n_negative={(d.glac_minus_geod < 0).sum()}")
print("   four most negative:", ", ".join(f"{r.name.title()} {r.glac_minus_geod:+.2f}" for r in d.head(4).itertuples()))
