#!/usr/bin/env python
"""Glacier volume change and population exposure in glacier-fed river basins (HMA).

Basins: HydroBASINS v1c level 8 (Lehner & Grill, 2013), Asia + Europe tiles, grouped into
major river basins by MAIN_BAS (identified by a reference point on each river, not by
hand-picked ids). Ganges and Brahmaputra share one outlet in HydroBASINS; they are split by
tracing upstream (NEXT_DOWN) from Farakka (Ganges) and Bahadurabad (Brahmaputra); the
remaining delta + Meghna part is reported separately. Endorheic basins that are not named
are grouped by the location of their sink.

"Glacierised upstream part" = HydroBASINS units (level 6, 7, 8; Pfafstetter prefix of the
level-8 code) containing at least one glacier centroid. Level 7 is used in T6/F12.

Glacier volume: per forcing, glaciers in the regional sample (calib_status == ok and not
backward_vanished), same rule as 06_ANALYSIS_v3.py; window change = V(y1) - V(y0 - 1), in % of V(y0 - 1).
Population: GHS-POP R2023A, 30 arc-second, WGS84 (Schiavina et al., 2023), epochs 1975,
2000, 2020 (main) and 2025 (projection, supplementary). Pixel-centre zonal sums.

Wording for the paper: "population in glacier-fed basins" / "exposure". This is ambient
residential population, NOT glacier-melt water users; no runoff is modelled.

Outputs: tables/T6_basin_glacier_population.{tex,csv}, tables/T6_basin_volume_by_forcing.csv,
tables/S7_subregion_population.{tex,csv}, figures/F12_basins_volume_population.png
Run: conda activate oggm && python scripts/population_basins.py
"""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import DOWNLOAD_CACHE  # noqa: E402
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.windows import from_bounds
from shapely.geometry import Point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import (FORC, LABEL, COL, LS, INK, INK2, MUTED, AXIS, DOUBLE,  # noqa: E402
                          write_table, fmt, tex_esc, save, TAB, legend_handles)

RAW = ROOT / "data_external/raw"
from paths import PER_FORCING as EXT  # noqa: E402
import os
SHP_SUB = Path(os.environ.get("HMA_SUBREGION_SHP", DOWNLOAD_CACHE / "HMA" / "HMA_region.shp"))
EPOCHS = [1975, 2000, 2020, 2025]          # 2025 = projection in R2023A
BBOX = (55.0, 5.0, 125.0, 52.0)            # lon_min, lat_min, lon_max, lat_max (all basins inside)
WINDOWS = {"1980-1999": (1980, 1999), "2000-2019": (2000, 2019), "2000-2025": (2000, 2025)}
RHO = 0.9

# Reference points (lon, lat) on each river; the basin = MAIN_BAS of the level-8 unit there.
REF = {
    "Indus": (72.24, 33.89),             # Attock
    "Amu Darya": (67.28, 37.22),         # Termez
    "Syr Darya": (69.60, 40.28),         # Khujand
    "Tarim": (80.26, 41.17),             # Aksu
    "Ili-Balkhash": (77.0, 43.9),        # Ili river (Kapchagay)
    "Yangtze": (104.6, 28.8),            # Yibin
    "Yellow": (103.8, 36.1),             # Lanzhou
    "Mekong": (100.1, 21.5),             # upper Mekong (Lancang)
    "Salween": (98.9, 25.9),             # Nu river
    "Irrawaddy": (96.08, 21.97),         # Mandalay
    "GBM": (90.4, 23.7),                 # Dhaka (shared Ganges-Brahmaputra-Meghna outlet)
}
SPLIT = {"Ganges": (87.92, 24.81),        # Farakka
         "Brahmaputra": (89.70, 25.10)}  # Bahadurabad
ORDER = ["Indus", "Ganges", "Brahmaputra", "GBM delta & Meghna", "Amu Darya", "Syr Darya", "Tarim",
         "Ili-Balkhash", "Yangtze", "Yellow", "Mekong", "Salween", "Irrawaddy",
         "Tibetan Plateau & Qaidam endorheic", "Other Central Asian endorheic", "Other"]


def load_basins():
    parts = []
    for reg in ["as", "eu"]:
        b = gpd.read_file(RAW / f"hybas_{reg}_lev08_v1c/hybas_{reg}_lev08_v1c.shp")
        parts.append(b)
    b = pd.concat(parts, ignore_index=True)
    b = gpd.GeoDataFrame(b, geometry="geometry", crs=4326)
    b = b.cx[BBOX[0]:BBOX[2], BBOX[1]:BBOX[3]].copy()
    b["PFAF"] = b.PFAF_ID.astype(int).astype(str)
    return b


def unit_at(b, lon, lat):
    hit = b[b.contains(Point(lon, lat))]
    if len(hit) != 1:
        raise RuntimeError(f"reference point {lon},{lat} hits {len(hit)} units")
    return hit.iloc[0]


def upstream(b, hid):
    up = b.groupby("NEXT_DOWN").HYBAS_ID.apply(list).to_dict()
    out, stack = set(), [hid]
    while stack:
        h = stack.pop()
        if h in out: continue
        out.add(h); stack.extend(up.get(h, []))
    return out


def assign_basins(b, glaciers_mainbas):
    name = pd.Series("Other", index=b.index)
    ids = {}
    for k, (lon, lat) in REF.items():
        ids[k] = unit_at(b, lon, lat).MAIN_BAS
    # Exorheic rivers: only units that really drain to the outlet (NEXT_SINK == outlet).
    # HydroBASINS attaches closed inland sinks (ENDO = 2, e.g. Paiku Co in the GBM basin)
    # to the main basin virtually; those are re-grouped below by their own sink.
    exorheic = {"Indus", "Yangtze", "Yellow", "Mekong", "Salween", "Irrawaddy", "GBM"}
    for k, mb in ids.items():
        sel = b.MAIN_BAS == mb
        if k in exorheic:
            sel &= b.NEXT_SINK == mb
        name[sel] = "GBM delta & Meghna" if k == "GBM" else k
    for k, (lon, lat) in SPLIT.items():
        u = unit_at(b, lon, lat)
        assert u.MAIN_BAS == ids["GBM"], k
        name[b.HYBAS_ID.isin(upstream(b, u.HYBAS_ID))] = k
    # remaining glacier-bearing endorheic basins, grouped by sink location
    sink = b.set_index("HYBAS_ID").geometry.representative_point()
    endo_sink = set(b.loc[b.ENDO == 2, "HYBAS_ID"])
    moved = 0
    for sk in glaciers_mainbas:          # NEXT_SINK ids of glacier-bearing units
        sel = (b.NEXT_SINK == sk) & (name == "Other")
        if not sel.any() or sk not in endo_sink: continue
        p = sink[sk]
        name[sel] = ("Tibetan Plateau & Qaidam endorheic" if 78 <= p.x <= 100 and 27 <= p.y < 39
                     else "Other Central Asian endorheic")
        moved += int(sel.sum())
    print(f"   units in closed inland sinks grouped as endorheic: {moved}")
    b["basin"] = name
    print("   basin outlet MAIN_BAS ids:", ids)
    return b


def ghs_path(ep):
    d = RAW / f"GHS_POP_E{ep}_GLOBE_R2023A_4326_30ss_V1_0"
    tif = list(d.glob("*.tif"))
    if not tif:
        raise FileNotFoundError(f"GHS-POP {ep} not found in {d}")
    return tif[0]


def zonal_population(b, sub):
    """Population per level-8 unit and per HMA sub-region polygon, all epochs."""
    with rasterio.open(ghs_path(EPOCHS[0])) as src:
        win = from_bounds(*BBOX, transform=src.transform).round_offsets().round_lengths()
        tr = src.window_transform(win); shape = (int(win.height), int(win.width))
        meta = dict(crs=src.crs, res=src.res, nodata=src.nodata, dtype=src.dtypes[0])
    print(f"   GHS-POP grid: res={meta['res']} crs={meta['crs']} nodata={meta['nodata']} dtype={meta['dtype']} window={shape}")
    code = np.arange(1, len(b) + 1, dtype=np.int32)
    lab = features.rasterize(zip(b.geometry, code), out_shape=shape, transform=tr, fill=0, dtype="int32")
    slab = (features.rasterize(zip(sub.geometry, np.arange(1, len(sub) + 1)), out_shape=shape,
                               transform=tr, fill=0, dtype="int32") if sub is not None else None)
    pu, ps, tot = {}, {}, {}
    for ep in EPOCHS:
        with rasterio.open(ghs_path(ep)) as src:
            # identical grids across epochs (resolution differs only at ~1e-16 float precision)
            assert np.allclose(src.res, meta["res"], rtol=0, atol=1e-12)
            assert np.allclose(tuple(src.window_transform(win))[:6], tuple(tr)[:6], rtol=0, atol=1e-9)
            a = src.read(1, window=win).astype(np.float64)
            if src.nodata is not None:
                a[a == src.nodata] = 0
            a[~np.isfinite(a) | (a < 0)] = 0
        pu[ep] = np.bincount(lab.ravel(), weights=a.ravel(), minlength=len(b) + 1)[1:]
        if slab is not None:
            ps[ep] = np.bincount(slab.ravel(), weights=a.ravel(), minlength=len(sub) + 1)[1:]
        tot[ep] = a.sum()
        print(f"   {ep}: window total {a.sum():,.0f}  in level-8 units {pu[ep].sum():,.0f}")
    ps_df = pd.DataFrame(ps, index=sub.full_name.values) if sub is not None else None
    return pd.DataFrame(pu, index=b.HYBAS_ID.values), ps_df, tot


def glacier_volumes(lk, g2u):
    """Per forcing: basin-summed volume at the years needed, regional-sample glaciers only."""
    years = sorted({y0 - 1 for y0, _ in WINDOWS.values()} | {y1 for _, y1 in WINDOWS.values()})
    out = []
    for f in FORC:
        m = pd.read_csv(EXT / f / "metadata.csv", usecols=["rgi_id", "calib_status", "backward_vanished"])
        keep = set(m.loc[(m.calib_status == "ok") & (m.backward_vanished != True), "rgi_id"])
        y = pd.read_parquet(EXT / f / "yearly.parquet", columns=["rgi_id", "year", "volume_km3"],
                            filters=[("year", "in", years)])
        y = y[y.rgi_id.isin(keep)].merge(g2u[["rgi_id", "basin"]], on="rgi_id", how="left")
        v = y.groupby(["basin", "year"]).volume_km3.sum().unstack()
        v["forcing"] = f
        out.append(v.reset_index())
        hma = y.groupby("year").volume_km3.sum()
        print(f"   {f}: glaciers kept {len(keep):,}; HMA V(1999)={hma.get(1999, np.nan):.1f} km3")
    return pd.concat(out, ignore_index=True)


def main():
    print("load HydroBASINS level 8")
    b = load_basins()
    lk = pd.read_csv(ROOT / "analysis/glacier_lookup.csv")
    g = gpd.GeoDataFrame(lk, geometry=gpd.points_from_xy(lk.CenLon, lk.CenLat), crs=4326)
    j = gpd.sjoin(g, b[["HYBAS_ID", "MAIN_BAS", "NEXT_SINK", "PFAF", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    assert j.HYBAS_ID.notna().all(), "glacier centroid outside HydroBASINS"
    b = assign_basins(b, j.NEXT_SINK.unique())
    left = j.merge(b[["HYBAS_ID", "basin"]], on="HYBAS_ID")
    assert (left.basin != "Other").all(), left.loc[left.basin == "Other", ["rgi_id", "CenLon", "CenLat"]].head()
    j = j.merge(b[["HYBAS_ID", "basin"]], on="HYBAS_ID", how="left")

    # consensus volume (end 1999) per glacier from metadata (same for every forcing)
    vref = pd.read_csv(EXT / "ERA5" / "metadata.csv", usecols=["rgi_id", "V_ref_km3"])
    j = j.merge(vref, on="rgi_id", how="left")
    print(f"   glaciers {len(j):,}  area {j.area_rgi_km2.sum():,.1f} km2  V_ref {j.V_ref_km3.sum():,.1f} km3")

    # glacierised units at levels 6, 7, 8 (Pfafstetter prefixes)
    for lev in (6, 7, 8):
        gl = set(j.PFAF.str[:lev])
        b[f"glac_l{lev}"] = b.PFAF.str[:lev].isin(gl)
        ua = b[b[f"glac_l{lev}"]].groupby(b.PFAF.str[:lev]).SUB_AREA.sum()
        UNIT_STATS[lev] = (len(ua), float(ua.median()), float(ua.sum()))
        print(f"   level {lev}: {len(ua):,} glacierised units, median {ua.median():,.0f} km2, total {ua.sum():,.0f} km2")

    print("population zonal sums (GHS-POP R2023A 30 arcsec)")
    have_sub = SHP_SUB.exists()
    sub = gpd.read_file(SHP_SUB) if have_sub else None
    pu, ps, tot = zonal_population(b, sub)
    if not have_sub:
        print(f"   {SHP_SUB} not found: skipping Table S7 (sub-region population); basin tables are unaffected")
    pu = pu.reindex(b.HYBAS_ID.values)

    rows = []
    for name in ORDER:
        sel = (b.basin == name).values
        if not sel.any(): continue
        jj = j[j.basin == name]
        r = dict(basin=name, basin_area_km2=b.loc[sel, "SUB_AREA"].sum(), n_glaciers=len(jj),
                 glacier_area_km2=jj.area_rgi_km2.sum(), V1999_consensus_km3=jj.V_ref_km3.sum())
        for ep in EPOCHS:
            r[f"pop{ep}"] = pu.loc[sel, ep].sum()
            for lev in (6, 7, 8):
                s2 = sel & b[f"glac_l{lev}"].values
                r[f"pop{ep}_glac_l{lev}"] = pu.loc[s2, ep].sum()
        r["glac_l7_area_km2"] = b.loc[sel & b.glac_l7.values, "SUB_AREA"].sum()
        rows.append(r)
    T = pd.DataFrame(rows).set_index("basin")

    print("glacier volume per basin and forcing (per_forcing/yearly.parquet)")
    V = glacier_volumes(lk, j)
    rec = []
    for _, r in V.iterrows():
        for w, (y0, y1) in WINDOWS.items():
            v0, v1 = r.get(y0 - 1), r.get(y1)
            if pd.isna(v0) or pd.isna(v1): continue
            v99 = r.get(1999)   # decision B: one reference, in-sample consensus V at end of 1999
            rec.append(dict(basin=r.basin, forcing=r.forcing, window=w, V_start=v0, V_1999=v99, dV_km3=v1 - v0,
                            dM_Gt=RHO * (v1 - v0),
                            dV_pct_V1999=100 * (v1 - v0) / v99 if v99 > 0 else np.nan,
                            dV_pct_window_start=100 * (v1 - v0) / v0 if v0 > 0 else np.nan))
    W = pd.DataFrame(rec)
    # sanity check: basin sums must reproduce the HMA windows in C_volume_change_windows.csv
    C = pd.read_csv(ROOT / "analysis/C_volume_change_windows.csv")
    cmap = {"1980-1999": "recon_1980-1999", "2000-2019": "fwd_2000-2019", "2000-2025": "fwd_2000-2025"}
    for (f, w), d in W.groupby(["forcing", "window"]):
        ref = C[(C.level == "hma") & (C.forcing == f) & (C.window == cmap[w])]
        if ref.empty: continue
        diff = d.dV_km3.sum() - ref.dV_km3.iloc[0]
        assert abs(diff) < 0.5, (f, w, diff)
    print("   check: basin sums reproduce HMA windows of C_volume_change_windows.csv (|diff| < 0.5 km3)")
    W.to_csv(TAB / "T6_basin_volume_by_forcing.csv", index=False)
    E = W.groupby(["basin", "window"]).agg(n=("dV_pct_V1999", "count"), pct_mean=("dV_pct_V1999", "mean"),
                                           pct_min=("dV_pct_V1999", "min"), pct_max=("dV_pct_V1999", "max"),
                                           km3_mean=("dV_km3", "mean"), km3_min=("dV_km3", "min"),
                                           km3_max=("dV_km3", "max")).reset_index()
    for w in WINDOWS:
        e = E[E.window == w].set_index("basin")
        for c in ["n", "pct_mean", "pct_min", "pct_max", "km3_mean", "km3_min", "km3_max"]:
            T[f"dV{w}_{c}"] = e[c]
    for a_, b_ in [(1975, 2000), (2000, 2020)]:
        T[f"dpop{a_}_{b_}"] = T[f"pop{b_}"] - T[f"pop{a_}"]
        T[f"dpop{a_}_{b_}_pct"] = 100 * T[f"dpop{a_}_{b_}"] / T[f"pop{a_}"]
        T[f"dpop{a_}_{b_}_glac_l7_pct"] = 100 * (T[f"pop{b_}_glac_l7"] - T[f"pop{a_}_glac_l7"]) / T[f"pop{a_}_glac_l7"]
    T.to_csv(TAB / "T6_basin_glacier_population.csv")
    pd.set_option("display.width", 250)
    print(T[["n_glaciers", "glacier_area_km2", "V1999_consensus_km3"]].round(1).to_string())
    print(T[[c for c in T.columns if c.startswith("dV")]].round(2).to_string())
    print((T[[f"pop{e}" for e in EPOCHS] + [f"pop{e}_glac_l7" for e in EPOCHS]] / 1e6).round(2).to_string())
    print(T[["dpop1975_2000_pct", "dpop2000_2020_pct", "dpop1975_2000_glac_l7_pct", "dpop2000_2020_glac_l7_pct"]].round(1).to_string())
    TT = T.drop(index="Other", errors="ignore")
    print("   totals, glacier-fed basins incl. GBM delta:", {e: f"{TT[f'pop{e}'].sum()/1e6:.1f} M" for e in EPOCHS},
          "| glacierised units 2020 (l6/l7/l8):", [f"{TT[f'pop2020_glac_l{l}'].sum()/1e6:.1f} M" for l in (6, 7, 8)])

    # ---------- T6 (main text): glaciers + volume change (km3, % of V_1999) + population
    # glacier-bearing basins + the shared Ganges-Brahmaputra delta (downstream of both, no glaciers)
    main = T[(T.glacier_area_km2 > 0) | (T.index == "GBM delta & Meghna")]
    rows = []
    for name, r in main.iterrows():
        def rng(w, kind):
            lo, hi, nd = (f"dV{w}_km3_min", f"dV{w}_km3_max", 1) if kind == "km3" else (f"dV{w}_pct_min", f"dV{w}_pct_max", 1)
            if pd.isna(r[lo]): return "--"
            return f"{fmt(r[lo], nd)} to {fmt(r[hi], nd)}" if r[f"dV{w}_n"] > 1 else fmt(r[lo], nd)
        pops = (f"{r.pop1975/1e6:.1f} & {r.pop2000/1e6:.1f} & {r.pop2020/1e6:.1f} & "
                f"{glac(r)} & {r.pop2025/1e6:.1f}")
        if r.glacier_area_km2 == 0:
            rows.append(f"{tex_esc(name)}$^a$ & 0 & 0 & & -- & -- & -- & " + pops)
            continue
        rows.append(f"{tex_esc(name)} & {r.glacier_area_km2:,.0f} & {r.V1999_consensus_km3:,.0f} & km$^3$ & "
                    + " & ".join(rng(w, "km3") for w in WINDOWS) + " & " + pops)
        rows.append(" & & & \\% & " + " & ".join(rng(w, "pct") for w in WINDOWS) + " & & & & & ")
    s = main.sum(numeric_only=True)
    rows.append("MID")
    rows.append(f"All glacier-fed basins & {s.glacier_area_km2:,.0f} & {s.V1999_consensus_km3:,.0f} & & & & & "
                f"{s.pop1975/1e6:,.1f} & {s.pop2000/1e6:,.1f} & {s.pop2020/1e6:,.1f} & {glac(s)} & {s.pop2025/1e6:,.1f}")
    write_table("S9_basin_detail",   # full table -> Supplement S9; main-text Table 7 from scripts/tables_tex.py
                "Glaciers, modelled ice-volume change and population in glacier-fed river basins (Sect.~\\ref{sec:population}). "
                "Volume change: range across forcings, in km$^3$ ice (first line; Gt = 0.9\\,$\\times$\\,km$^3$) and in per cent of the "
                "basin's consensus volume at the end of 1999 in each forcing's sample (second line); negative = loss. 1980--1999: five "
                "forcings, 2000--2019: four, 2000--2025: three (2020--2025 extrapolated). Population (GHS-POP R2023A, millions; 2025 "
                "projected): whole basin, then the 2020 population of the glacierised HydroBASINS units at level~7 (main value; levels "
                "8 and 6 in brackets, as a sensitivity range). GBM = Ganges--Brahmaputra--Meghna. "
                "$^a$ Downstream of both the Ganges (below Farakka) and the Brahmaputra (below Bahadurabad); no glaciers.",
                "tab:S9", "lrrlrrrrrrrr",
                [" & Glacier & $V_{1999}$ & & \\multicolumn{3}{c}{$\\Delta V$} & \\multicolumn{5}{c}{Population (M)}",
                 "Basin & area (km$^2$) & (km$^3$) & & 1980--1999 & 2000--2019 & 2000--2025 & 1975 & 2000 & 2020 & " + GLAC_HEAD + " & 2025 proj."],
                rows, star=True, size="\\tiny\\setlength{\\tabcolsep}{2pt}")

    import tables_tex; tables_tex.t6()   # simplified main-text Table 7 (reads the CSV written above)

    # ---------- S7: sub-region population (GHS-POP within HMA sub-region polygons)
    if ps is not None:
        ps.to_csv(TAB / "S7_subregion_population.csv")
        rws = [f"{tex_esc(n)} & {r[1975]/1e6:.2f} & {r[2000]/1e6:.2f} & {r[2020]/1e6:.2f} & {r[2025]/1e6:.2f}" for n, r in ps.iterrows()]
        rws += ["MID", f"Total & {ps[1975].sum()/1e6:.2f} & {ps[2000].sum()/1e6:.2f} & {ps[2020].sum()/1e6:.2f} & {ps[2025].sum()/1e6:.2f}"]
        print((ps / 1e6).round(2).to_string())
        write_table("S7_subregion_population", "Population (millions) inside the 15 HMA sub-region polygons, GHS-POP R2023A "
                    "(2025 = projection).", "tab:S7", "lrrrr", ["Sub-region & 1975 & 2000 & 2020 & 2025 proj."], rws, size="\\scriptsize")
    else:
        print("   S7_subregion_population skipped (no sub-region shapefile); kept as previously generated")

    # ---------- F12
    # figure: scripts/figures/fig12_basins.py (reads the tables written here)
    print("done")


GLAC_HEAD = "2020 glac. L7 (L8--L6)"
UNIT_STATS = {}   # level -> (n units, median unit area km2, total area km2), filled in main()


def glac(r):
    return f"{r.pop2020_glac_l7/1e6:.1f} ({r.pop2020_glac_l8/1e6:.1f}--{r.pop2020_glac_l6/1e6:.1f})"


if __name__ == "__main__":
    main()
