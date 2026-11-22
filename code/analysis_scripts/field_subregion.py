#!/usr/bin/env python
"""Field (glaciological) against modelled annual mass balance by sub-region and period, all available glacier-years per forcing
(analysis/D_validation_pairs.csv, which carries the sub-region).
Output: tables/field_bias_subregion.csv (one row per sub-region x period x forcing: n glacier-years, glaciers, obs mean, model mean)."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = pd.read_csv(ROOT / "analysis/D_validation_pairs.csv")


P["period"] = pd.cut(P.year, [1949, 1979, 1999, 2019, 2030], labels=["1950-1979", "1980-1999", "2000-2019", "2020-2024"])
rows = []
for f in ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4", "CHELSA"]:
    d = P.dropna(subset=[f, "obs"])
    for (s, p), g in d.groupby(["subregion", "period"], observed=True):
        rows.append(dict(subregion=s, period=p, forcing=f, n_years=len(g), n_glaciers=g.rgi_id.nunique(),
                         obs=g.obs.mean(), model=g[f].mean(), bias=(g[f] - g.obs).mean()))
    for (n, p), g in d[d.name.str.contains("ABRAMOV", case=False, na=False)].groupby(["name", "period"], observed=True):
        rows.append(dict(subregion="glacier: " + n, period=p, forcing=f, n_years=len(g), n_glaciers=1, obs=g.obs.mean(), model=g[f].mean(), bias=(g[f] - g.obs).mean()))
T = pd.DataFrame(rows); T.to_csv(ROOT / "tables/field_bias_subregion.csv", index=False)
q = T[T.subregion.str.contains("Hissar|Karakoram|ABRAMOV", case=False) & T.period.isin(["1950-1979", "1980-1999", "2000-2019"])]
print(q.round(2).to_string(index=False))
