#!/usr/bin/env python
"""Consistency check of the modelled mass balance with published geodetic estimates from declassified imagery (1960s-2000).
Model: calendar years y0 .. y1-1 for a published period y0-y1 (DEM dates y0 and y1), per forcing and for the best estimate
(equal family weights, members present in each year; scripts/gridded_best.py). A forcing that does not cover all years of a window
(HARv2 from 1980, CHELSA from its start year) is shown as 'partial' (mean over the years available) and is not used in the summary.
Samples: Maurer et al. 2019 = 650 largest RGI glaciers of the West/Central/East Himalaya sub-regions between 75 and 93 E (their glacier list is not
used); Zhou et al. 2017 = all glaciers within the study-area bounds they print (33.75-37.5 N, 74.33-78.25 E); Bhattacharya et al. 2021 sites: the paper
gives no coordinates in the text, the boxes are ours (approximate). Output: tables/kh9_comparison.csv, tables/kh9_summary.csv, tables/kh9_sites.csv. Env oggm."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from gridded_best import stack, best, LK, FORC
Y0, Y1 = 1968, 2003
A, ids, yrs = stack(Y0, Y1, "mb_annual_mwe", FORC)           # forcing x glacier x year
area = LK.loc[ids, "area_rgi_km2"].values; lon = LK.loc[ids, "CenLon"].values; lat = LK.loc[ids, "CenLat"].values; sub = LK.loc[ids, "subregion"].values
B = best(A, FORC)
him = np.isin(sub, ["West Himalaya", "Central Himalaya", "East Himalaya"]) & (lon >= 75) & (lon <= 93)
top = np.zeros(len(ids), bool); top[np.argsort(-np.where(him, area, -1))[:650]] = True
print(f"Himalaya 75-93E: {him.sum()} glaciers, {area[him].sum():.0f} km2; 650 largest = {area[top].sum():.0f} km2 ({100 * area[top].sum() / area[him].sum():.0f} % of area), smallest of them {area[top].min():.2f} km2")
box = lambda a: (lon >= a[0]) & (lon <= a[1]) & (lat >= a[2]) & (lat <= a[3])
SITES = [  # name, published, sigma, y0, y1 (DEM dates), mask, bib key, extent note
 ("Himalaya, 650 largest glaciers (Maurer 2019)", -0.22, 0.13, 1975, 2000, top, "Maurer2019", "RGI West, Central and East Himalaya, 75--93$^\\circ$E (paper: ``transect from Spiti Lahaul to Bhutan, 75--93$^\\circ$E'', 650 largest glaciers); latitude bounds not given", True),
 ("Himalaya, all glaciers (same extent)", -0.22, 0.13, 1975, 2000, him, "Maurer2019", "as above, all RGI glaciers", False),
 ("Karakoram (Zhou 2017)", -0.09, 0.03, 1973, 2000, box((74.33, 78.25, 33.75, 37.5)), "Zhou2017", "paper: 33$^\\circ$45'--37$^\\circ$30'\\,N, 74$^\\circ$20'--78$^\\circ$15'\\,E; the 12\\,366\\,km$^2$ of the estimate are the area covered by the KH-9 DEMs, ours is the whole box", True),
 ("Purogangri (Bhattacharya 2021)", -0.22, 0.07, 1975, 2000, box((88.5, 89.7, 33.5, 34.4)), "Bhattacharya2021", "no coordinates in the paper's text; box ours (approximate)", True),
 ("Poiqu (Bhattacharya 2021)", -0.30, 0.10, 1974, 2004, box((85.3, 86.2, 27.8, 28.5)), "Bhattacharya2021", "no coordinates in the paper's text; box ours (approximate)", True),
 ("W Nyainqentanglha (Bhattacharya 2021)", -0.26, 0.13, 1968, 2001, box((90.0, 91.2, 30.0, 30.8)), "Bhattacharya2021", "no coordinates in the paper's text; box ours (approximate)", True),
 ("Ak-Shirak (Bhattacharya 2021)", -0.54, 0.10, 1980, 2002, box((77.8, 78.8, 41.5, 42.2)), "Bhattacharya2021", "period printed as 1980--2002 (Results, Comparison) and 1980--2000 (Non-climatic factors); box ours (approximate)", True),
]
rows, sites = [], []
for name, pub, sig, y0, y1, m, src, ext, counted in SITES:
    k = (yrs >= y0) & (yrs <= y1 - 1)
    r = dict(area=name, published=pub, sigma=sig, period=f"{y0}-{y1}", n_glaciers=int(m.sum()), glacier_area_km2=float(area[m].sum()), counted=counted)
    for i, f in enumerate(FORC):
        X = A[i][:, k]
        full = (~np.isnan(X[m])).all(1)
        h = X[m].astype("float64"); w = area[m]
        ok = ~np.isnan(h).all(1)
        mean_g = np.where(np.isnan(h).all(1), np.nan, np.nanmean(np.where(np.isnan(h), np.nan, h), axis=1)) if ok.any() else np.array([])
        r[f] = np.average(mean_g[ok], weights=w[ok]) if ok.any() else np.nan
        r[f + "_years"] = int((~np.isnan(X[m])).any(0).sum()); r[f + "_partial"] = bool(r[f + "_years"] < k.sum())
    bb = B[:, k][m]; ok = ~np.isnan(bb).any(1)
    r["best"] = np.average(bb[ok].mean(1), weights=area[m][ok]) if ok.any() else np.nan
    r["window_years"] = int(k.sum())
    rows.append(r)
    sites.append(dict(area=name, source=src, extent=ext, n_glaciers=int(m.sum()), glacier_area_km2=round(float(area[m].sum()), 1)))
T = pd.DataFrame(rows); T.to_csv(ROOT / "tables/kh9_comparison.csv", index=False)
pd.DataFrame(sites).to_csv(ROOT / "tables/kh9_sites.csv", index=False)
# Ak-Shirak alternative window 1980-2000 (years 1980-1999)
k = (yrs >= 1980) & (yrs <= 1999); m = SITES[6][5]
alt = {f: np.average(np.nanmean(A[i][m][:, k], axis=1), weights=area[m]) for i, f in enumerate(FORC)}
print("Ak-Shirak 1980-2000 alternative:", {a: round(b, 3) for a, b in alt.items()}, "best", round(np.average(B[:, k][m].mean(1), weights=area[m]), 3))
cols = FORC + ["best"]
print(T[["area", "published", "sigma", "period", "n_glaciers", "glacier_area_km2"] + cols].round(3).to_string(index=False))
print(T[["area"] + [f + "_partial" for f in FORC]].to_string(index=False))
S = []
for c in cols:
    d = []; inside = []
    for _, r in T[T.counted].iterrows():
        if c != "best" and r[c + "_partial"]: continue
        d.append(r[c] - r.published); inside.append(abs(r[c] - r.published) <= r.sigma)
    S.append(dict(member=c, n=len(d), n_within=int(sum(inside)), mean_signed_diff=float(np.mean(d)) if d else np.nan, mean_abs_diff=float(np.mean(np.abs(d))) if d else np.nan))
S = pd.DataFrame(S); S.to_csv(ROOT / "tables/kh9_summary.csv", index=False); print(S.round(3).to_string(index=False))

# ---- LaTeX tables (main: T9_kh9.tex; Supplement: S11_kh9_sites.tex)
# Column order of the printed table (header below). Must match the header.
DISP = ["ERA5", "ERA5-Land", "CRU_TS4", "HARv2", "CHELSA"]
assert sorted(DISP) == sorted(FORC)
f2 = lambda v: ("$+$" if v >= 0.005 else "$-$") + f"{abs(v):.2f}" if abs(v) >= 0.005 else "0.00"
L = ["% generated by scripts/kh9_compare.py -- do not edit by hand", "\\begin{table*}[t]",
     "\\caption{Modelled mass balance (m w.e. yr$^{-1}$) for the periods and areas of published geodetic estimates from declassified imagery; $\\pm$ as printed in the source. Model years are the calendar years from the first to the year before the last year of the published period. Best: equal family weights, members present in each year. $^p$: the forcing does not cover the whole period; the mean is over the years available and is not used in the summary. Summary: comparisons within the printed uncertainty and mean signed difference (model minus published) over the six rows without $^p$. Samples: Himalaya, the 650 largest glaciers of the West, Central and East Himalaya between 75 and 93$^\\circ$\\,E; Karakoram, glaciers within the bounds of Zhou et al.; other sites, our boxes (Table~\\ref{S-tab:kh9_sites}).}",
     "\\label{tab:kh9}", "\\scriptsize", "\\begin{tabular}{lcrrrrrrr}", "\\tophline",
     "Area (source) & Period & Published & " + " & ".join({"CRU_TS4": "CRU~TS4"}.get(f, f) for f in DISP) + " & Best \\\\", "\\middlehline"]
lab = {"Himalaya, 650 largest glaciers (Maurer 2019)": "Himalaya, 650 largest (Maurer et al., 2019)", "Himalaya, all glaciers (same extent)": "Himalaya, all glaciers (same extent)",
       "Karakoram (Zhou 2017)": "Karakoram (Zhou et al., 2017)", "Purogangri (Bhattacharya 2021)": "Purogangri (Bhattacharya et al., 2021)",
       "Poiqu (Bhattacharya 2021)": "Poiqu (Bhattacharya et al., 2021)", "W Nyainqentanglha (Bhattacharya 2021)": "W Nyainqentanglha (Bhattacharya et al., 2021)",
       "Ak-Shirak (Bhattacharya 2021)": "Ak-Shirak (Bhattacharya et al., 2021)"}
for _, r in T.iterrows():
    cells = [f2(r[f]) + ("$^p$" if r[f + "_partial"] else "") for f in DISP] + [f2(r["best"])]
    L.append(f"{lab[r.area]} & {r.period.replace('-', '--')} & {f2(r.published)}\\,$\\pm$\\,{r.sigma:.2f} & " + " & ".join(cells) + " \\\\")
L.append("\\middlehline")
inner = {r["member"]: r for _, r in S.iterrows()}
mk = lambda c: "--" if inner[c]["n"] < 3 else f"{int(inner[c].n_within)}/{int(inner[c].n)}"
L.append("Within printed uncertainty & & & " + " & ".join(mk(c) for c in DISP + ["best"]) + " \\\\")
L.append("Mean signed difference & & & " + " & ".join("--" if inner[c]["n"] < 3 else f2(inner[c].mean_signed_diff) for c in DISP + ["best"]) + " \\\\")
L += ["\\bottomhline", "\\end{tabular}", "\\end{table*}"]
(ROOT / "tables/T9_kh9.tex").write_text("\n".join(L) + "\n")
Q = pd.read_csv(ROOT / "tables/kh9_sites.csv")
esc = lambda s: s.replace("%", "\\%").replace("&", "\\&")
thin = lambda n: f"{n:,}".replace(",", "\\,")
SL = ["% generated by scripts/kh9_compare.py -- do not edit by hand", "\\begin{table}[t]",
      "\\caption{Areas used in the comparison with published pre-2000 geodetic estimates (Table~\\ref{tab:kh9}): source and the extent used, with the number of RGI glaciers and their area.}", "\\label{tab:kh9_sites}", "\\scriptsize",
      "\\begin{tabular}{p{3.2cm}p{2.4cm}p{6.5cm}rr}", "\\tophline", "Area & Source & Extent & Glaciers & Area (km$^2$) \\\\", "\\middlehline"]
for _, q in Q.iterrows():
    SL.append(f"{esc(q.area)} & \\citet{{{q.source}}} & {q.extent} & {thin(q.n_glaciers)} & {thin(round(q.glacier_area_km2))} \\\\")
SL += ["\\bottomhline", "\\end{tabular}", "\\end{table}"]
(ROOT / "tables/S11_kh9_sites.tex").write_text("\n".join(SL) + "\n")
print("wrote tables/T9_kh9.tex and S11_kh9_sites.tex")
