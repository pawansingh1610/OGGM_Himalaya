"""Shared figure style for Draft_2 (Copernicus / The Cryosphere).

Every figure is drawn at its final printed size and included in LaTeX without scaling:
  SINGLE = 8.3 cm (one column), DOUBLE = 17.0 cm (full width).
Fonts: one sans font (Arial, fallback DejaVu Sans), 7-8 pt. Lines >= 0.75 pt.
Vector output: PDF with TrueType (Type 42) fonts embedded, plus a 600 dpi PNG preview.
Raster figures (maps / images): 600 dpi PNG only.
No bbox_inches="tight": the saved page size is exactly the drawn size.

Inputs: only files inside my-paper/ (analysis/, tables/, data_external/).
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "analysis"
TAB = ROOT / "tables"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))   # volume_ref.py (reads analysis/ only)

CM = 1 / 2.54
SINGLE = 8.3 * CM
DOUBLE = 17.0 * CM

# ------------------------------------------------------------------ entities and colours
# Fixed forcing -> colour mapping (validated palette, unchanged from Draft_2 v1 figures).
# Secondary encoding: line style by family (solid = ERA5 family, dashed = CRU family).
FORC = ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4", "CHELSA"]
LABEL = {"ERA5": "ERA5", "ERA5-Land": "ERA5-Land", "HARv2": "HARv2",
         "CRU_TS4": "CRU TS4", "CHELSA": "CHELSA-W5E5"}
COL = {"ERA5": "#2a78d6", "ERA5-Land": "#eb6834", "HARv2": "#1baf7a",
       "CRU_TS4": "#eda100", "CHELSA": "#e87ba4"}
LS = {"ERA5": "-", "ERA5-Land": "-", "HARv2": "-", "CRU_TS4": (0, (4, 1.6)), "CHELSA": (0, (4, 1.6))}
CRU_FAMILY = {"CRU_TS4", "CHELSA"}

INK = "#0b0b0b"      # text, best estimate
INK2 = "#3a3936"     # secondary text, ticks (darker than v1 to avoid a washed-out look)
MUTED = "#6f6d68"    # notes
AXIS = "#3a3936"     # axis spines
ZERO = "#8c8a84"     # zero lines
GRID = "#e4e3dd"
NEUTRAL = "#f0efec"
SHADE_PRE = "#ecebe6"   # before 1980: not constrained
SHADE_CAL = "#d6e6fa"   # 2000-2019 geodetic calibration
HATCH_EXT = "#b5b3aa"   # 2020-2025 extrapolation hatch
LOSS, GAIN = "#c83a39", "#2a78d6"
DIV = LinearSegmentedColormap.from_list(
    "div", ["#8a1f1f", "#e34948", "#f2a3a2", NEUTRAL, "#86b6ef", "#2a78d6", "#0d366b"])

LW = 1.0          # default data line
LW_THIN = 0.75    # minimum line width anywhere

SUBREG_ORDER = [  # o2region order 13-01 ... 15-03
    "Hissar Alay", "Pamir (Safed Khirs / West Tarim)", "West Tien Shan",
    "East Tien Shan (Dzhungaria)", "West Kun Lun", "East Kun Lun (Altyn Tagh)",
    "Qilian Shan", "Inner Tibet", "Southeast Tibet", "Hindu Kush", "Karakoram",
    "West Himalaya", "Central Himalaya", "East Himalaya", "Hengduan Shan"]
SHORT = {"Pamir (Safed Khirs / West Tarim)": "Pamir",
         "East Tien Shan (Dzhungaria)": "East Tien Shan",
         "East Kun Lun (Altyn Tagh)": "East Kun Lun"}
CHELSA_FLAG = {"West Kun Lun", "East Kun Lun (Altyn Tagh)", "East Tien Shan (Dzhungaria)"}


def short(s):
    return SHORT.get(s, s)


plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.default": "regular",
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "figure.titlesize": 8,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 4,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.linewidth": LW_THIN, "xtick.major.width": LW_THIN, "ytick.major.width": LW_THIN,
    "xtick.minor.width": LW_THIN, "ytick.minor.width": LW_THIN,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": LW_THIN,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "lines.linewidth": LW, "patch.linewidth": LW_THIN, "hatch.linewidth": LW_THIN,
    "legend.frameon": False, "legend.handlelength": 2.2, "legend.borderaxespad": 0.3,
    "legend.columnspacing": 1.0, "legend.handletextpad": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 600, "figure.dpi": 150, "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
    "image.interpolation": "none",
})


def figure(width, height_cm, **kw):
    """Figure of exact final size (width in inches = SINGLE or DOUBLE, height in cm)."""
    assert width in (SINGLE, DOUBLE), "width must be SINGLE (8.3 cm) or DOUBLE (17 cm)"
    return plt.figure(figsize=(width, height_cm * CM), **kw)


def subplots(width, height_cm, *a, **kw):
    assert width in (SINGLE, DOUBLE), "width must be SINGLE (8.3 cm) or DOUBLE (17 cm)"
    return plt.subplots(*a, figsize=(width, height_cm * CM), **kw)


def legend_handles(forcings=FORC, lw=1.2):
    return [Line2D([], [], color=COL[f], ls=LS[f], lw=lw, label=LABEL[f]) for f in forcings]


def shade_periods(ax, y0=1950, calib=True, extrap=True):
    """grey = before 1980 (not constrained); blue = 2000-2019 geodetic calibration;
    hatched = 2020-2025 unconstrained extrapolation."""
    ax.axvspan(y0 - 0.5, 1979.5, color=SHADE_PRE, lw=0, zorder=0)
    if calib:
        ax.axvspan(1999.5, 2019.5, color=SHADE_CAL, lw=0, zorder=0)
    if extrap:
        ax.axvspan(2019.5, 2025.5, facecolor="none", edgecolor=HATCH_EXT, hatch="////", lw=0, zorder=0)


def zero_line(ax, axis="y", v=0):
    (ax.axhline if axis == "y" else ax.axvline)(v, color=ZERO, lw=LW_THIN, zorder=1)


def save(fig, name, vector=True):
    """vector=True: figures/<name>.pdf (Type 42 fonts) + 600 dpi PNG preview; else 600 dpi PNG only."""
    w, h = fig.get_size_inches()
    out = []
    if vector:
        p = FIG / f"{name}.pdf"
        fig.savefig(p, metadata={"Creator": f"scripts/figures/{name}.py", "CreationDate": None})
        out.append(p)
    p = FIG / f"{name}.png"
    fig.savefig(p, dpi=600)
    out.append(p)
    plt.close(fig)
    for p in out:
        print(f"  -> {p.relative_to(ROOT)}  ({w / CM:.1f} x {h / CM:.1f} cm)")


def header(num, title):
    print(f"\n=== Figure file {num:02d}: {title}")


def heat(ax, M, title, vlim, fmtstr="{:+.1f}", flags=None, cmap=DIV):
    """Annotated diverging heat map (vector: pcolormesh with white cell gaps). Returns the mesh."""
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim)
    V = np.ma.masked_invalid(M.values.astype(float))
    mesh = ax.pcolormesh(np.arange(M.shape[1] + 1) - 0.5, np.arange(M.shape[0] + 1) - 0.5, V, cmap=cmap,
                         norm=norm, edgecolors="white", linewidth=LW_THIN)
    ax.set_xlim(-0.5, M.shape[1] - 0.5); ax.set_ylim(M.shape[0] - 0.5, -0.5)
    ax.set_xticks(range(M.shape[1]), [LABEL.get(c, c) for c in M.columns], rotation=30, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(range(M.shape[0]), [short(i) for i in M.index])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if v != v:   # NaN
                ax.text(j, i, "–", ha="center", va="center", color=MUTED)
                continue
            dark = abs(norm(v) - 0.5) > 0.30
            star = "*" if flags and (M.index[i], M.columns[j]) in flags else ""
            ax.text(j, i, fmtstr.format(v).replace("-", "−") + star, ha="center", va="center",
                    color="white" if dark else INK)
    ax.set_title(title)
    return mesh
