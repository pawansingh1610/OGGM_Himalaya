#!/usr/bin/env python
"""Figures 16/17 - 1 deg x 1 deg gridded maps of the best-estimate ensemble (weights of scripts/best_mb_ensemble.py).

fig16_gridded_mb      (a) mean mass balance 2000-2019 (calibration period), (b) 1980-1999 (reconstruction), hatched where the
                      five forcings disagree on the sign of the cell mean. Colour: m w.e. yr-1; square size: glacier area.
figS_gridded_mb_forcings   the five forcings, 1980-1999 (Supplement)
fig17_gridded_vol     (a) ice-volume change 2000-2019 in km3, (b) in % of the cell's V_1999
figS_gridded_vol_1950 volume change 1950-1979, ERA5 / ERA5-Land / CRU_TS4 separately (Supplement; not constrained)

Inputs: tables/gridded_{mb,vol}_*.csv (scripts/gridded_best.py). The 15 sub-region outlines are drawn only when
$HMA_OGGM_ROOT/download_cache/HMA/HMA_region.shp is present.
"""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[2]))
from config import DOWNLOAD_CACHE  # noqa: E402
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from style import (DIV, DOUBLE, FORC, INK2, LABEL, MUTED, TAB, figure, header, save)

VMIN = 1.0      # km3: cells with less ice than this at the end of 1999 are not drawn in the per-cent maps (a few small glaciers make the % meaningless)
SHP = DOWNLOAD_CACHE / "HMA" / "HMA_region.shp"
EXT = (66.5, 105.5, 26.0, 46.5)


def outline():
    try:
        import geopandas as gpd
        return gpd.read_file(SHP).to_crs(4326)
    except Exception as e:                          # pragma: no cover
        print("   (no sub-region outline:", e, ")")
        return None


G = outline()


def base(ax):
    if G is not None:
        G.boundary.plot(ax=ax, color="#b5b3aa", lw=0.5, zorder=1)
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(36)))
    ax.grid(False)
    ax.set_xticks(range(70, 106, 10), [f"{x}°E" for x in range(70, 106, 10)])
    ax.set_yticks(range(30, 46, 5), [f"{y}°N" for y in range(30, 46, 5)])
    ax.tick_params(length=2)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.75)


def squares(ax, d, col, norm, cmap, hatch=None, size_by="area_km2", amin=1.0):
    """one square per 1 deg cell; side scales with sqrt(glacier area) (full cell at >= 800 km2)."""
    d = d[d[col].notna() & (d.area_km2 >= amin)]
    for _, r in d.iterrows():
        s = float(np.clip(np.sqrt(r[size_by] / 800.0), 0.25, 1.0))
        x, y = r.lon + 0.5, r.lat + 0.5
        ax.add_patch(plt.Rectangle((x - s / 2, y - s / 2), s, s, fc=cmap(norm(r[col])), ec="none", zorder=2))
        if hatch is not None and r.get(hatch, False):
            ax.add_patch(plt.Rectangle((x - s / 2, y - s / 2), s, s, fc="none", ec="#222222", hatch="/////", lw=0, zorder=3))


def cbar(fig, ax, norm, cmap, label, **kw):
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.8, pad=0.04, aspect=30, **kw)
    cb.set_label(label)
    cb.outline.set_linewidth(0.75)
    return cb


def size_legend(ax):
    for i, a in enumerate((50, 200, 800)):
        s = np.sqrt(a / 800.0)
        ax.add_patch(plt.Rectangle((67.3 + 3.4 * i, 26.4), s, s, fc="#8c8a84", ec="none", zorder=4))
        ax.text(67.3 + 3.4 * i + s / 2, 26.4 + s + 0.15, f"{a}", ha="center", va="bottom", color=INK2, fontsize=6)
    ax.text(67.3, 28.9, "glacier area\n(km² per cell)", color=INK2, fontsize=6, va="bottom")


# ------------------------------------------------------------------------------------------------ MB, best estimate
header(16, "gridded mass balance, best estimate")
mbA = pd.read_csv(TAB / "gridded_mb_2000_2019.csv")
mbB = pd.read_csv(TAB / "gridded_mb_1980_1999.csv")
for t, d in (("2000-2019", mbA), ("1980-1999", mbB)):
    print(f"   {t}: {len(d)} cells, {int(d.n_glaciers.sum())} glaciers; B_best cell range {d.B_best.min():+.2f} .. {d.B_best.max():+.2f}; "
          f"sign disagreement in {int(d.get('sign_disagree', pd.Series(False)).sum())} cells")
norm = TwoSlopeNorm(vcenter=0, vmin=-1.0, vmax=0.5)
fig = figure(DOUBLE, 7.6)
axs = fig.subplots(1, 2)
for ax, d, ttl, hat in ((axs[0], mbA, "(a) 2000–2019 (geodetic calibration)", None),
                        (axs[1], mbB, "(b) 1980–1999 (reconstruction)", "sign_disagree")):
    base(ax)
    squares(ax, d, "B_best", norm, DIV, hatch=hat)
    ax.set_title(ttl)
size_legend(axs[0])
cbar(fig, axs, norm, DIV, "Best-estimate mass balance (m w.e. yr$^{-1}$); hatched in (b): the five forcings disagree on the sign")
save(fig, "fig16_gridded_mb")

# ------------------------------------------------------------------------------------------------ MB, per forcing
fig = figure(DOUBLE, 12.0)
axs = fig.subplots(2, 3).ravel()
for ax, f in zip(axs, FORC):
    base(ax); squares(ax, mbB, "B_" + f, norm, DIV); ax.set_title(LABEL[f])
    ax.set_xticks([]); ax.set_yticks([])
axs[5].axis("off")
cbar(fig, axs[5], norm, DIV, "Mass balance 1980–1999 (m w.e. yr$^{-1}$)")
save(fig, "figS_gridded_mb_forcings")

# ------------------------------------------------------------------------------------------------ volume change
header(17, "gridded volume change, best estimate")
v = pd.read_csv(TAB / "gridded_vol_2000_2019.csv")
v["pct"] = 100 * v.dV_best / v.V1999_best
print(f"   2000-2019: {len(v)} cells, HMA dV {v.dV_best.sum():+.1f} km3 = {100 * v.dV_best.sum() / v.V1999_best.sum():+.2f} % of V1999; "
      f"cell dV {v.dV_best.min():+.1f} .. {v.dV_best.max():+.1f} km3; cell pct {v.pct.min():+.1f} .. {v.pct.max():+.1f}")
fig = figure(DOUBLE, 7.6)
axs = fig.subplots(1, 2)
lim = np.nanpercentile(np.abs(v.dV_best), 98)
n1 = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
base(axs[0]); squares(axs[0], v, "dV_best", n1, DIV); axs[0].set_title("(a) Volume change 2000–2019")
size_legend(axs[0])
cbar(fig, axs[0], n1, DIV, "km$^3$ ice per cell")
vm = v[v.V1999_best >= VMIN]
print(f"   per-cent panel: cells with V1999 >= {VMIN} km3: {len(vm)} of {len(v)}; 5 most extreme: " +
      "; ".join(f"({r.lon:.0f}E,{r.lat:.0f}N) {r.pct:+.1f} % n={int(r.n_best)} V1999={r.V1999_best:.2f}" for _, r in vm.reindex(vm.pct.abs().sort_values(ascending=False).index).head(5).iterrows()))
lp = np.nanpercentile(np.abs(vm.pct), 98)
n2 = TwoSlopeNorm(vcenter=0, vmin=-lp, vmax=lp)
base(axs[1]); squares(axs[1], vm, "pct", n2, DIV); axs[1].set_title("(b) Relative to the cell's $V_{1999}$")
cbar(fig, axs[1], n2, DIV, "% of $V_{1999}$")
save(fig, "fig17_gridded_vol")

p = TAB / "gridded_vol_1950_1979.csv"
if p.exists():
    w = pd.read_csv(p)
    fig = figure(DOUBLE, 6.6)
    axs = fig.subplots(1, 3)
    for ax, f in zip(axs, ["ERA5", "ERA5-Land", "CRU_TS4"]):
        w["pct_" + f] = 100 * w["dV_" + f] / w["V1999_" + f]
        base(ax); squares(ax, w[w["V1999_" + f] >= VMIN], "pct_" + f, n2, DIV); ax.set_title(LABEL[f]); ax.set_xticks([]); ax.set_yticks([])
        print(f"   1950-1979 {f}: HMA {w['dV_' + f].sum():+.1f} km3 = {100 * w['dV_' + f].sum() / w['V1999_' + f].sum():+.2f} % of V1999")
    cbar(fig, axs, n2, DIV, "Volume change 1950–1979, % of $V_{1999}$ (not constrained by observations)")
    save(fig, "figS_gridded_vol_1950")

# ------------------------------------------------------------------------------------------------ 2000-2025 (Supplement)
pa, pb = TAB / "gridded_vol_2000_2025.csv", TAB / "gridded_vol_2020_2025.csv"
if pa.exists() and pb.exists():
    a, b = pd.read_csv(pa), pd.read_csv(pb)
    a["pct"] = 100 * a.dV_best / a.V1999_best
    b["pct"] = 100 * b.dV_best / b.V1999_best
    print(f"   2000-2025 (ERA5 family): HMA {a.dV_best.sum():+.1f} km3 = {100 * a.dV_best.sum() / a.V1999_best.sum():+.2f} %; "
          f"2020-2025: {b.dV_best.sum():+.1f} km3 = {100 * b.dV_best.sum() / b.V1999_best.sum():+.2f} %")
    fig = figure(DOUBLE, 7.6)
    axs = fig.subplots(1, 2)
    for ax, d, t in ((axs[0], a, "(a) 2000–2025"), (axs[1], b, "(b) 2020–2025 only (unconstrained extrapolation)")):
        base(ax); squares(ax, d[d.V1999_best >= VMIN], "pct", n2, DIV); ax.set_title(t)
    size_legend(axs[0])
    cbar(fig, axs, n2, DIV, "Volume change, % of $V_{1999}$ (ERA5, ERA5-Land, HARv2; 2020–2025 not constrained)")
    save(fig, "figS_gridded_vol_2025")
