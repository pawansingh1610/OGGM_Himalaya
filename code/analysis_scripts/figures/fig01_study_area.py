#!/usr/bin/env python
"""Figure 1 - study area. The author's own map (Draft_1), not generated here.

Source (original, never modified): <your figure folder>/Study Area.png (hand-drawn base map, not included)
Copied once (cp, 2026-11-09) to figures/fig01_study_area.png. This script only checks the copy
and reports its pixel size and the effective resolution at the printed width (17 cm, full width).
LaTeX: \\includegraphics[width=17cm]{../figures/fig01_study_area.png} (raster: width = printed width).
"""
from PIL import Image

from style import FIG, header

header(1, "study area (author's map)")
p = FIG / "fig01_study_area.png"
assert p.exists(), f"{p} missing - copy 'Study Area.png' to figures/ (not included in this archive)"
im = Image.open(p)
w, h = im.size
for cm in (17.0, 8.3):
    dpi = w / (cm / 2.54)
    print(f"   {w} x {h} px, mode {im.mode}; printed width {cm} cm -> height {cm * h / w:.1f} cm, "
          f"effective {dpi:.0f} dpi {'(below 300 dpi)' if dpi < 300 else '(>= 300 dpi)'}")
