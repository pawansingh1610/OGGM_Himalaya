"""
|F|Drang Drung Glacier — Multi-SSP Simulation & Visualization Pipeline
====================================================================
Glacier  : Drang Drung (RGI60-14.18948) · Karakoram · India
Period   : 2000 – 2100
Scenarios: SSP1-2.6 · SSP2-4.5 · SSP3-7.0 · SSP5-8.5
GCM      : MRI-ESM2-0 r1i1p1f1  (ISIMIP3b)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW THE SCRIPT WORKS  (read this before changing anything)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 1 — OGGM SIMULATION
  • Downloads OGGM Level-3 pre-processed glacier directory ONCE.
    Level 3 is required because it contains gridded_data.nc (the 2-D
    topography / mask file).  Level 5 strips this file out.
  • Runs a historical baseline spin-up (shared by all SSPs).
  • Downloads ISIMIP3b climate data for ssp370 ONCE (~500 MB).
    All other SSPs are derived by applying a temperature bias offset
    (FAST mode, USE_BIAS_OFFSETS=True) — no extra downloads needed.
  • For each SSP: runs the 2000-2100 projection, distributes the
    1-D flowline output to a 2-D grid, saves a NetCDF file.

  ⚡ FAST MODE (default, USE_BIAS_OFFSETS=True)
    Only ssp370 climate is downloaded.  ssp126/ssp245/ssp585 are
    derived by shifting the ssp370 temperature field by a constant
    offset based on IPCC AR6 Table SPM.1 warming differences.
    Total download: ~500 MB.  Total time: ~10 min.

  🔬 FULL MODE (USE_BIAS_OFFSETS=False)
    Each SSP downloads its own ISIMIP3b climate data independently.
    Total download: ~2 GB.  Scientifically more rigorous.

PHASE 2 — PER-SSP 3-D ANIMATIONS
  For every SSP scenario the script renders:
    • Static snapshots at years 2000 / 2050 / 2100
    • Full 2000-2100 animation  (terrain colormap)
    • Same animation with globe inset in the corner
    • Satellite texture version: snapshots + animation
      (bedrock draped with real satellite imagery tiles)

PHASE 3 — 4-PANEL COMPARISON VIDEOS
  All 4 SSPs side-by-side in one figure:
    • comparison_4panel.mp4              (fixed camera, terrain)
    • comparison_4panel_with_globe.mp4   (+ globe inset)
    • comparison_4panel_rotating.mp4     (rotating camera, terrain)
    • comparison_4panel_sat_rotating.mp4 (rotating, satellite texture)
    • comparison_4panel_sat_with_globe.mp4 (sat + globe)
    • comparison_4panel_2000/2050/2100.png (static maps)
    • comparison_timeseries.png          (area + thickness over time)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON THINGS YOU MIGHT WANT TO CHANGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  To change the glacier            → edit RGI_ID
  To change the GCM model          → edit ISIMIP_MEMBER
  To change scenarios simulated    → edit SSP_SCENARIOS list
  To change scenario colours       → edit SSP_COLORS dict
  To change the camera angle       → edit camera_args in COMMON_VIZ
  To change the rotating arc       → edit start_angle/end_angle in ROTATE_CFG
  To change animation speed        → edit framerate= in export_animation calls
  To change video resolution       → edit window_size in PLOTTER_ARGS
  To change Doda Peak label        → edit doda_peak PointAnnotation below
  To add another point annotation  → copy doda_peak block, change coords
  To disable satellite texture     → set RENDER_SATELLITE = False
  To re-run simulations            → delete the relevant .nc file and re-run
  To re-download everything        → set RESET_GDIR = True and delete ~/oggm_run

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  conda activate glacier3dviz
  python durung_drong_pipeline.py

  First run:   set RESET_GDIR = True
  Later runs:  set RESET_GDIR = False  (reuses cached glacier dir)

REQUIREMENTS
  pip install oggm glacier3dviz pyvista tqdm
  conda install -c conda-forge ffmpeg
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPORTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os, glob, sys, threading, subprocess, tempfile, shutil, urllib.request
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation   # renamed to avoid clash

# tqdm gives a nice download progress bar — optional but recommended
try:
    from tqdm import tqdm
    _TQDM_OK = True
except ImportError:
    _TQDM_OK = False
    print("  (pip install tqdm for download progress bars)")

# ── PyVista MUST be configured before any pv or glacier3dviz import ──────────
# Setting PYVISTA_OFF_SCREEN=true tells PyVista to render to a buffer
# (no screen/display needed — works on headless servers and desktops alike).
os.environ["PYVISTA_OFF_SCREEN"] = "true"
import pyvista as pv
try:
    pv.start_xvfb()   # starts a virtual X framebuffer (Linux only, harmless on Windows)
except Exception:
    pass

import oggm
from oggm import cfg, utils, workflow, tasks
from oggm.shop import gcm_climate
from oggm.sandbox import distribute_2d

from glacier3dviz.tools import Glacier3DViz
from glacier3dviz.tools.map_annotations import (
    ArrowAnnotation,   # draws the north arrow
    PointAnnotation,   # pins a labelled point at lat/lon/height
    MaskAnnotation,    # overlays a binary mask (used for RGI outline)
    TextAnnotation,    # floating text label on the 3-D scene
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ①  CONFIGURATION  ← most things you need to change are here
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Glacier identity ─────────────────────────────────────────────────────────
RGI_ID        = "RGI60-14.18948"                    # Drang Drung glacier ID
OGGM_DATA_DIR = os.path.expanduser("~/OGGM")        # where OGGM stores raw data
WORKING_DIR   = os.path.expanduser("~/oggm_run")    # simulation output directory
ANIM_DIR      = os.path.join(WORKING_DIR, "animation")  # all video/image outputs
os.makedirs(ANIM_DIR, exist_ok=True)

# ── Run control flags ────────────────────────────────────────────────────────
RESET_GDIR       = False  # True = re-download glacier dir (use on very first run)
USE_BIAS_OFFSETS = True   # True = FAST (no extra downloads); False = rigorous
RENDER_SATELLITE = True   # True = also render satellite-texture versions

# ── Watermark settings ──────────────────────────────────────────────────────
# Stamped on every .png output and burned into every .mp4 video.
#
# To change the name text  : edit WATERMARK_NAME
# To change the logo image : edit WATERMARK_LOGO_PATH
# To move the watermark    : edit WATERMARK_POSITION
#                            choices: 'bottom-right' 'bottom-left'
#                                     'top-right'    'top-left'
# To resize the logo       : edit WATERMARK_LOGO_SIZE (width, height) px
# To change text font size : edit WATERMARK_FONTSIZE
# To disable watermark     : set  WATERMARK_ENABLED = False
WATERMARK_ENABLED   = True
WATERMARK_NAME      = "Pawan Singh  |  IIT Roorkee"
WATERMARK_LOGO_PATH = os.path.expanduser("/home/pawan/iitroorkee.jpg")
WATERMARK_POSITION  = "bottom-right"   # corner: bottom-right/left or top-right/left
WATERMARK_LOGO_SIZE = (100, 75)         # (width, height) in pixels
WATERMARK_FONTSIZE  = 9                # font size for name text on figures
WATERMARK_PADDING   = 10               # pixels gap from the frame edge

# ── Climate model ────────────────────────────────────────────────────────────
# Available GCMs in ISIMIP3b:
#   "mri-esm2-0_r1i1p1f1"       ← moderate warming (used here)
#   "gfdl-esm4_r1i1p1f1"        ← moderate warming
#   "ipsl-cm6a-lr_r1i1p1f1"     ← higher sensitivity
#   "ukesm1-0-ll_r1i1p1f2"      ← hottest model
ISIMIP_MEMBER = "mri-esm2-0_r1i1p1f1"
REFERENCE_SSP = "ssp370"   # this SSP's climate is actually downloaded

# ── Simulation period ────────────────────────────────────────────────────────
SIM_START = 2000
SIM_END   = 2100

# ── SSP scenarios ────────────────────────────────────────────────────────────
# Temperature bias (°C) applied relative to ssp370 at year 2100.
# Derived from IPCC AR6 Table SPM.1 best-estimate global mean warming:
#   ssp126 ≈ 1.8 °C,  ssp245 ≈ 2.7 °C,  ssp370 ≈ 3.6 °C,  ssp585 ≈ 4.4 °C
# → differences from ssp370: −1.8, −0.9, 0.0, +0.8  (rounded here)
SSP_TEMP_BIAS = {
    "ssp126": -1.5,   # significantly cooler than ssp370
    "ssp245": -0.7,   # moderately cooler
    "ssp370":  0.0,   # reference — real ISIMIP3b data
    "ssp585": +0.9,   # warmer
}
SSP_LABELS = {
    "ssp126": "SSP1-2.6  (Low emissions)",
    "ssp245": "SSP2-4.5  (Intermediate)",
    "ssp370": "SSP3-7.0  (High emissions)",
    "ssp585": "SSP5-8.5  (Very high emissions)",
}
# Panel border / line colours — one colour per SSP for visual distinction
SSP_COLORS = {
    "ssp126": "#1a9641",   # green  — best case
    "ssp245": "#fdae61",   # orange — intermediate
    "ssp370": "#d7191c",   # red    — high emissions
    "ssp585": "#7b2d8b",   # purple — worst case
}
# Order scenarios are rendered and appear in panels
SSP_SCENARIOS = ["ssp126", "ssp245", "ssp370", "ssp585"]

# ── OGGM pre-processing settings ─────────────────────────────────────────────
# Level 3 is REQUIRED — it is the only level that retains gridded_data.nc,
# which holds the 2-D topography and glacier mask needed for visualization.
# (Level 5 strips gridded_data.nc out, which breaks distribute_2d.)
PREPRO_BASE_URL = (
    "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/"
    "L3-L5_files/2023.3/elev_bands/W5E5/"
)
PREPRO_LEVEL  = 3
PREPRO_BORDER = 80   # grid border padding in metres; 80 is fine for 21st century

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ②  HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def nc_path(ssp):
    """Return the path of the output NetCDF file for a given SSP."""
    return os.path.join(WORKING_DIR,
                        f"{RGI_ID.replace('.','_')}_2000_2100_{ssp}.nc")

def ssp_dir(ssp):
    """Return (and create if needed) the per-SSP animation output directory."""
    d = os.path.join(ANIM_DIR, ssp)
    os.makedirs(d, exist_ok=True)
    return d

def out(ssp, filename):
    """Convenience: full path to a file inside the per-SSP animation directory."""
    return os.path.join(ssp_dir(ssp), filename)

def banner(msg):
    """Print a prominent section header to the console."""
    w = 68
    print(f"\n{'═'*w}\n  {msg}\n{'═'*w}")

def save_netcdf(ssp, run_suffix, gdir, gridded_path):
    """
    Convert OGGM's gridded simulation output to our standardised NetCDF format.

    OGGM writes a file called 'gridded_simulation_<suffix>.nc' with a variable
    named 'simulated_thickness'.  We rename it to 'ice_thickness', add the
    bedrock elevation (from gridded_data.nc), and write integer calendar years
    as a secondary coordinate so Glacier3DViz can display them as labels.

    Parameters
    ----------
    ssp         : SSP string, e.g. "ssp370"
    run_suffix  : OGGM file suffix for this projection run
    gdir        : OGGM GlacierDirectory object
    gridded_path: path to gridded_data.nc (holds topo / bedrock)

    Returns
    -------
    Path of the saved NetCDF file.
    """
    # Load the raw OGGM simulation output
    sim_path = gdir.get_filepath("gridded_simulation", filesuffix=run_suffix)
    with xr.open_dataset(sim_path) as s:
        s = s.load()

    thick = s["simulated_thickness"].values.astype(np.float32)
    years = s.time.values
    # Convert fractional OGGM years (e.g. 2050.083) to integer calendar years
    cal_y = np.floor(years).astype(np.int32)

    # Load bedrock elevation from gridded_data.nc
    # Prefer 'topo_bedrock' if available; otherwise estimate as topo − ice
    with xr.open_dataset(gridded_path) as t:
        t = t.load()
        if "topo_bedrock" in t:
            bedrock = t["topo_bedrock"].values.astype(np.float32)
        else:
            topo = t["topo_smoothed"].values
            cit  = (t["consensus_ice_thickness"].values
                    if "consensus_ice_thickness" in t
                    else np.zeros_like(topo))
            # Bedrock ≈ surface topo − ice thickness (rough estimate if no topo_bedrock)
            bedrock = (topo - np.where(np.isfinite(cit), cit, 0.0)).astype(np.float32)

    # Assemble the output dataset with clear variable names
    ds = xr.Dataset(
        {
            # 2-D static field — bedrock elevation does not change over time
            "bedrock": xr.DataArray(bedrock, dims=["y", "x"],
                attrs={"long_name": "Bedrock elevation", "units": "m a.s.l."}),
            # 3-D field — ice thickness changes each year
            "ice_thickness": xr.DataArray(thick, dims=["time", "y", "x"],
                attrs={"long_name": "Glacier ice thickness", "units": "m"}),
        },
        coords={
            "x":            xr.DataArray(s.x.values, dims=["x"],
                attrs={"long_name": "Easting", "units": "m"}),
            "y":            xr.DataArray(s.y.values, dims=["y"],
                attrs={"long_name": "Northing", "units": "m"}),
            "time":         xr.DataArray(years, dims=["time"],
                attrs={"long_name": "OGGM model time", "units": "years"}),
            # calendar_year is used by Glacier3DViz for the on-screen year label
            "calendar_year": xr.DataArray(cal_y, dims=["time"],
                attrs={"long_name": "Calendar year", "units": "year"}),
        },
    )
    # Store metadata as global attributes so we can identify the file later
    ds.attrs = {
        "rgi_id"               : RGI_ID,
        "glacier_name"         : gdir.name,
        "climate_scenario"     : f"ISIMIP3b {ISIMIP_MEMBER} {ssp}",
        "ssp"                  : ssp,
        "simulation_start"     : str(SIM_START),
        "simulation_end"       : str(SIM_END),
        "oggm_version"         : oggm.__version__,
        "projection_crs"       : str(gdir.grid.proj),  # used for satellite texture
        "topo_bedrock_var_name": "bedrock",
        "time_var_display"     : "calendar_year",       # tells Glacier3DViz which coord to label
    }
    # Save with lossless compression
    enc = {
        "ice_thickness" : {"zlib": True, "complevel": 4, "dtype": "float32"},
        "bedrock"       : {"zlib": True, "complevel": 4, "dtype": "float32"},
        "time"          : {"dtype": "float64"},
        "calendar_year" : {"dtype": "int32"},
    }
    dest = nc_path(ssp)
    ds.to_netcdf(dest, encoding=enc)
    return dest


def load_with_masks(ssp, gridded_path):
    """
    Load a saved SSP NetCDF and inject the glacier mask variables needed by
    MaskAnnotation (glacier_ext) and satellite texture (pyproj_srs).

    glacier_ext is a binary mask (1 inside RGI 2000 glacier outline, 0 outside).
    It lives in gridded_data.nc, NOT in our custom NetCDF, so we copy it in here.
    If gridded_data.nc is missing (shouldn't happen with Level 3), we fall back
    to deriving the mask from the year-2000 ice thickness footprint.

    pyproj_srs is the proj4 CRS string needed by Glacier3DViz's set_topo_texture()
    to geo-reference the satellite tile downloads.
    """
    with xr.open_dataset(nc_path(ssp)) as ds:
        ds = ds.load()

    # Inject glacier_ext from gridded_data.nc
    if os.path.exists(gridded_path):
        with xr.open_dataset(gridded_path) as gd:
            gd = gd.load()
        for v in ["glacier_ext", "glacier_mask"]:
            if v in gd:
                ds[v] = gd[v]   # copy 2-D mask into our dataset

    # Fallback: derive mask from 2000 ice thickness if gridded_data.nc missing
    if "glacier_ext" not in ds:
        ice0 = ds["ice_thickness"].sel(time=SIM_START, method="nearest")
        ds["glacier_ext"] = (ice0 > 0).astype("int8")
        print("  ⚠ glacier_ext derived from ice thickness (gridded_data.nc missing)")

    # Fix CRS string for satellite texture geo-referencing
    crs = ds.attrs.get("projection_crs", "")
    if crs and not crs.startswith("+"):
        crs = "+" + crs                  # OGGM sometimes omits the leading '+'
    if not crs:
        # Hardcoded fallback for Drang Drung: TM centred on 76.28°E
        crs = ("+proj=tmerc +lat_0=0 +lon_0=76.284934998 "
               "+k=0.9996 +x_0=0 +y_0=0 +datum=WGS84")
    ds.attrs["pyproj_srs"] = crs         # this key is required by Glacier3DViz
    return ds


def make_sat_dataset(ds_full, sat_x_slice, sat_y_slice):
    """
    Crop a full-grid dataset to the satellite tile region.

    Glacier3DViz can only download satellite tiles for small regions
    (the tile server rejects huge bounding boxes).  We crop to a ~27×41 km
    sub-region centred on the glacier before creating the satellite viz.

    Parameters
    ----------
    ds_full     : xarray Dataset from load_with_masks()
    sat_x_slice : Python slice for the x-axis crop, e.g. slice(60, 195)
    sat_y_slice : Python slice for the y-axis crop, e.g. slice(60, 265)

    Returns
    -------
    Cropped xarray Dataset with pyproj_srs and glacier_ext set.
    """
    ds_sat = ds_full.isel(x=sat_x_slice, y=sat_y_slice).copy()

    # glacier_ext must also be cropped
    if "glacier_ext" in ds_full:
        ds_sat["glacier_ext"] = ds_full["glacier_ext"].isel(
            x=sat_x_slice, y=sat_y_slice)

    # pyproj_srs is inherited from the parent dataset
    ds_sat.attrs["pyproj_srs"] = ds_full.attrs.get("pyproj_srs", "")
    return ds_sat


def extract_mp4_frames(mp4_path, tmpdir):
    """
    Use ffmpeg to decode an mp4 into individual PNG frames.

    Returns a sorted list of frame paths. Used when compositing the
    rotating panel video from per-SSP individual rotation movies.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp4_path,
         os.path.join(tmpdir, "frame_%04d.png")],
        capture_output=True, check=False,
    )
    return sorted(
        os.path.join(tmpdir, f)
        for f in os.listdir(tmpdir)
        if f.endswith(".png")
    )



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WATERMARK HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _wm_corners(W, H, lw, lh):
    # Returns (logo_x, logo_y, text_x, text_y) for the configured corner.
    # W/H = canvas size, lw/lh = logo size (all in pixels).
    PAD = WATERMARK_PADDING
    pos = WATERMARK_POSITION
    if pos == "bottom-right":
        lx, ly = W - lw - PAD,       H - lh - PAD - 22
        tx, ty = W - 300  - PAD,     H - 20  - PAD
    elif pos == "bottom-left":
        lx, ly = PAD,                  H - lh - PAD - 22
        tx, ty = PAD,                  H - 20  - PAD
    elif pos == "top-right":
        lx, ly = W - lw - PAD,       PAD
        tx, ty = W - 300  - PAD,     PAD + lh + 4
    else:   # top-left
        lx, ly = PAD,                  PAD
        tx, ty = PAD,                  PAD + lh + 4
    return lx, ly, tx, ty


def stamp_png(filepath):
    # Post-process a saved PNG to add the watermark logo + author name.
    # Glacier3DViz plot_year() writes directly to disk (not via plt.savefig),
    # so we open the PNG with PIL, composite the watermark, and save back.
    if not WATERMARK_ENABLED or not os.path.exists(filepath):
        return
    try:
        from PIL import Image as _PIL, ImageDraw as _Draw, ImageFont as _Font
        img     = _PIL.open(filepath).convert("RGBA")
        W, H    = img.size
        overlay = _PIL.new("RGBA", (W, H), (0, 0, 0, 0))
        draw    = _Draw.Draw(overlay)
        lw_cfg, lh_cfg = WATERMARK_LOGO_SIZE

        # ── Logo ──────────────────────────────────────────────────────
        if os.path.exists(WATERMARK_LOGO_PATH):
            logo = _PIL.open(WATERMARK_LOGO_PATH).convert("RGBA")
            logo.thumbnail((lw_cfg, lh_cfg), _PIL.LANCZOS)
            lw_a, lh_a = logo.size
            lx, ly, tx, ty = _wm_corners(W, H, lw_a, lh_a)
            r, g, b, a = logo.split()
            a = a.point(lambda v: int(v * 0.85))
            logo.putalpha(a)
            overlay.paste(logo, (lx, ly), logo)
        else:
            _, _, tx, ty = _wm_corners(W, H, lw_cfg, lh_cfg)

        # ── Name text ─────────────────────────────────────────────────
        try:
            font = _Font.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                size=WATERMARK_FONTSIZE + 3)
        except Exception:
            font = _Font.load_default()
        bb   = draw.textbbox((0, 0), WATERMARK_NAME, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.rectangle([tx-4, ty-2, tx+tw+4, ty+th+2],
                       fill=(0, 0, 0, 140))
        draw.text((tx, ty), WATERMARK_NAME,
                  font=font, fill=(255, 255, 255, 230))

        result = _PIL.alpha_composite(img, overlay).convert("RGB")
        result.save(filepath, "PNG")
    except Exception as _e:
        print(f"  watermark skipped ({os.path.basename(filepath)}): {_e}")


def add_watermark_to_figure(fig):
    # Stamp watermark onto a matplotlib Figure before plt.savefig().
    # Uses fig.figimage() for the logo and fig.text() for the name.
    if not WATERMARK_ENABLED:
        return
    from PIL import Image as _PIL
    import numpy as _np_wm
    W_px = int(fig.get_size_inches()[0] * fig.dpi)
    H_px = int(fig.get_size_inches()[1] * fig.dpi)
    lw_cfg, lh_cfg = WATERMARK_LOGO_SIZE

    # Logo
    if os.path.exists(WATERMARK_LOGO_PATH):
        try:
            logo = _PIL.open(WATERMARK_LOGO_PATH).convert("RGBA")
            logo.thumbnail((lw_cfg, lh_cfg), _PIL.LANCZOS)
            lw_a, lh_a = logo.size
            lx, ly, _, _ = _wm_corners(W_px, H_px, lw_a, lh_a)
            # figimage y=0 is the BOTTOM of the canvas
            fig.figimage(_np_wm.array(logo), xo=lx, yo=ly,
                         alpha=0.85, zorder=10)
        except Exception as _e:
            print(f"  logo watermark skipped: {_e}")

    # Name text (normalised 0-1 coordinates)
    pos = WATERMARK_POSITION
    tx  = 0.99 if "right" in pos else 0.01
    ha  = "right" if "right" in pos else "left"
    ty  = 0.01  if "bottom" in pos else 0.99
    va  = "bottom" if "bottom" in pos else "top"
    fig.text(tx, ty, WATERMARK_NAME,
             color="white", fontsize=WATERMARK_FONTSIZE,
             ha=ha, va=va, alpha=0.90, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2",
                       facecolor="#00000060", edgecolor="none"),
             zorder=11, transform=fig.transFigure)


def add_watermark_to_video(video_path):
    # Burn logo + name into every frame of an .mp4 using ffmpeg overlay +
    # drawtext filters. Overwrites the original file in-place.
    # Very fast — ffmpeg processes the whole video in one pass.
    if not WATERMARK_ENABLED or not os.path.exists(video_path):
        return True
    PAD     = WATERMARK_PADDING
    lw, lh  = WATERMARK_LOGO_SIZE
    pos     = WATERMARK_POSITION
    tmp_out = video_path.replace(".mp4", "_wm_tmp.mp4")

    # ffmpeg coordinate expressions (W/H = video dims, w/h = overlay dims)
    ov = {"bottom-right": f"W-w-{PAD}:H-h-30",
          "bottom-left":  f"{PAD}:H-h-30",
          "top-right":    f"W-w-{PAD}:{PAD+20}",
          "top-left":     f"{PAD}:{PAD+20}"}.get(pos, f"W-w-{PAD}:H-h-30")
    dt = {"bottom-right": f"x=W-tw-{PAD}:y=H-th-{PAD}",
          "bottom-left":  f"x={PAD}:y=H-th-{PAD}",
          "top-right":    f"x=W-tw-{PAD}:y={PAD}",
          "top-left":     f"x={PAD}:y={PAD}"}.get(pos, f"x=W-tw-{PAD}:y=H-th-{PAD}")

    safe = WATERMARK_NAME.replace("'", "\\'").replace(":", "\\:")

    if os.path.exists(WATERMARK_LOGO_PATH):
        fc = (f"[1:v]scale={lw}:{lh},format=rgba,"
              f"colorchannelmixer=aa=0.85[logo];"
              f"[0:v][logo]overlay={ov}[vl];"
              f"[vl]drawtext=text='{safe}':"
              f"fontsize=18:fontcolor=white@0.90:"
              f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
              f"box=1:boxcolor=black@0.4:boxborderw=4:{dt}[out]")
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", WATERMARK_LOGO_PATH,
               "-filter_complex", fc, "-map", "[out]", "-map", "0:a?",
               "-c:v", "libx264", "-crf", "18", "-preset", "fast",
               "-c:a", "copy", tmp_out]
    else:
        fg = (f"drawtext=text='{safe}':"
              f"fontsize=18:fontcolor=white@0.90:"
              f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
              f"box=1:boxcolor=black@0.4:boxborderw=4:{dt}")
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", fg,
               "-c:v", "libx264", "-crf", "18", "-preset", "fast",
               "-c:a", "copy", tmp_out]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(tmp_out):
        os.replace(tmp_out, video_path)
        print(f"    watermark burned -> {os.path.basename(video_path)}")
        return True
    for ln in res.stderr.strip().splitlines()[-3:]:
        print(f"    ffmpeg: {ln}")
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
    return False

def composite_4panel_video(rendered_dict, output_path, title_text,
                            fps=5, bitrate=4000, dpi=160,
                            globe_path=None, globe_label=""):
    """
    Composite 4 pre-rendered frame lists into a 2×2 panel animation and save.

    Parameters
    ----------
    rendered_dict : dict  {ssp_string: [numpy_array, ...]}
                    Each value is a list of RGB frames (from plt.imread).
    output_path   : str   Path to write the output .mp4 file.
    title_text    : str   Main title shown at the top of the video.
    fps           : int   Frames per second of the output video.
    bitrate       : int   Video bitrate in kbps (higher = better quality).
    dpi           : int   DPI for matplotlib figure rasterisation.
    globe_path    : str or None  If provided, add a globe inset with ffmpeg.
    globe_label   : str   Suffix for the globe-inset output filename.
    """
    ssps_ok      = [s for s in SSP_SCENARIOS if s in rendered_dict]
    n_frames     = min(len(rendered_dict[s]) for s in ssps_ok)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), facecolor="#1a1a2e")
    axes = axes.flatten()
    img_handles = []

    for ax, ssp in zip(axes, ssps_ok):
        ax.axis("off")
        h = ax.imshow(rendered_dict[ssp][0])
        img_handles.append((h, ssp))
        # Panel title: coloured border + SSP label — only one title source
        ax.set_title(SSP_LABELS[ssp], color=SSP_COLORS[ssp],
                     fontsize=12, fontweight="bold", pad=4)
        for sp in ax.spines.values():
            sp.set_edgecolor(SSP_COLORS[ssp])
            sp.set_linewidth(3)

    # Hide any unused panels (if fewer than 4 SSPs are available)
    for ax in axes[len(ssps_ok):]:
        ax.axis("off")

    title_obj = fig.suptitle(
        f"{title_text}",# | Year {int(rendered_dict[ssps_ok[0]][0].shape[0])}",
        color="white", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    def _update(frame_idx):
        for h, ssp in img_handles:
            h.set_data(rendered_dict[ssp][frame_idx])
        # Update the year shown in the main title if the caller embedded year info
        return [h for h, _ in img_handles] #+ [title_obj]

    ani = mpl_animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=1000 / fps, blit=False)

    writer = mpl_animation.FFMpegWriter(fps=fps, bitrate=bitrate)
    try:
        ani.save(output_path, writer=writer, dpi=dpi)
        print(f"  ✓ {output_path}")
        add_watermark_to_video(output_path)   # burn logo + name
    except Exception as e:
        print(f"  ✗ Video save failed: {e}")
        print("    → Install ffmpeg: conda install -c conda-forge ffmpeg")
    plt.close(fig)

    # Optionally add a globe inset to the composite video
    if globe_path and os.path.exists(globe_path) and os.path.exists(output_path):
        _globe_out = output_path.replace(".mp4", f"_with_globe{globe_label}.mp4")
        try:
            add_inset_with_ffmpeg(
                main_video=output_path, inset_image=globe_path,
                output_video=_globe_out,
                position="20:20",    # top-left corner, 20 px from each edge
                scale_factor=0.6,    # slightly smaller globe on wide 2×2 panel
            )
            print(f"  ✓ globe inset → {_globe_out}")
        except Exception as e:
            print(f"  Globe inset failed: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ③  DOWNLOAD PROGRESS HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_isimip_download_with_progress(gdirs, member, ssp, suffix,
                                        sim_start, sim_end):
    """
    Run OGGM's process_monthly_isimip_data() with a live download progress bar.

    Problem: OGGM downloads files silently with no progress feedback.
    Solution: A background thread watches the download cache directory,
    measures how fast the file grows, and feeds those increments to tqdm.

    The two files downloaded per SSP are:
      *_tasAdjust_global_monthly_flat_glaciers.nc  (~250 MB, temperature)
      *_prAdjust_global_monthly_flat_glaciers.nc   (~250 MB, precipitation)
    These are glacier-grid-point-only extracts, NOT global files.
    """
    _cache = os.path.join(
        os.path.expanduser("~/OGGM"), "download_cache",
        "cluster.klima.uni-bremen.de", "~oggm", "cmip6",
        "isimip3b", "flat", "2023.2", "monthly",
    )
    # The two file names OGGM will download
    _targets = [
        f"{member}_w5e5_{ssp}_tasAdjust_global_monthly_flat_glaciers.nc",
        f"{member}_w5e5_{ssp}_prAdjust_global_monthly_flat_glaciers.nc",
    ]
    EXPECTED_BYTES = 520 * 1024 * 1024   # ~520 MB conservative total

    # Create tqdm progress bar (falls back to plain print if tqdm not installed)
    if _TQDM_OK:
        bar = tqdm(
            total=EXPECTED_BYTES,
            unit="B", unit_scale=True, unit_divisor=1024,
            desc=f"  Downloading {ssp} climate",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            colour="cyan", dynamic_ncols=True,
        )
    else:
        bar = None

    _stop_flag = threading.Event()

    def _watcher():
        """Background thread: poll file sizes every 0.5 s and update bar."""
        prev = 0
        while not _stop_flag.is_set():
            total = 0
            for fn in _targets:
                fp = os.path.join(_cache, fn)
                # Check final path AND common partial-download extensions
                for candidate in [fp, fp + ".tmp", fp + ".part"]:
                    if os.path.exists(candidate):
                        try:
                            total += os.path.getsize(candidate)
                        except OSError:
                            pass
            if bar is not None and total > prev:
                bar.update(total - prev)
                prev = total
            elif bar is None and total > prev:
                mb = total / 1024 / 1024
                sys.stdout.write(f"  Downloaded: {mb:.1f} MB  \r")
                sys.stdout.flush()
                prev = total
            _stop_flag.wait(timeout=0.5)

    watcher_thread = threading.Thread(target=_watcher, daemon=True)
    watcher_thread.start()

    try:
        from oggm.shop import gcm_climate as _gcm
        from oggm import workflow as _wf
        _wf.execute_entity_task(
            _gcm.process_monthly_isimip_data, gdirs,
            member=member, ssp=ssp,
            output_filesuffix=suffix,
            y0=sim_start - 5, y1=sim_end + 5,
        )
    finally:
        _stop_flag.set()
        watcher_thread.join(timeout=2)
        if bar is not None:
            bar.n = bar.total   # snap to 100%
            bar.refresh()
            bar.close()
        else:
            print()   # newline after the \r progress line


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 1 — OGGM SIMULATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

banner("PHASE 1 — OGGM SIMULATION")

# Initialise OGGM global configuration
cfg.initialize(logging_level="WARNING")   # suppress verbose OGGM logs
cfg.PATHS["working_dir"]           = WORKING_DIR
cfg.PARAMS["use_multiprocessing"]  = False   # single glacier → no benefit
cfg.PARAMS["border"]               = PREPRO_BORDER
cfg.PARAMS["store_fl_diagnostics"] = True    # REQUIRED by distribute_2d

# Point OGGM at local data cache if available (speeds up repeat runs)
if os.path.isdir(OGGM_DATA_DIR):
    cfg.PATHS["dl_cache_dir"] = os.path.join(OGGM_DATA_DIR, "download_cache")

utils.mkdir(WORKING_DIR, reset=False)
print(f"OGGM {oggm.__version__}")
print(f"Mode: {'FAST — T-bias offsets' if USE_BIAS_OFFSETS else 'FULL — ISIMIP3b per SSP'}\n")

# ── Load RGI glacier entity ───────────────────────────────────────────────────
# We first try to find a local RGI shapefile (avoids a network request).
# If not found, OGGM downloads it automatically.
def find_rgi_shapefile(data_dir, region):
    """Search common locations for a local RGI60 shapefile."""
    roots = [data_dir,
             os.path.join(data_dir, "rgi"),
             os.path.join(data_dir, "download_cache"),
             os.path.join(data_dir, "download_cache", "RGI"),
             os.path.join(data_dir, "download_cache", "RGI", "rgi60")]
    patterns = [f"{region}_rgi60_*.shp",
                f"{int(region):02d}_rgi60_*.shp",
                f"*rgi60*{region}*.shp"]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for pat in patterns:
            for shp in glob.glob(os.path.join(root, "**", pat), recursive=True):
                return shp
    return None

region   = RGI_ID.split("-")[1].split(".")[0]   # "14" for Karakoram
shp_path = find_rgi_shapefile(OGGM_DATA_DIR, region)
if shp_path:
    gdf    = gpd.read_file(shp_path)
    id_col = "RGIId" if "RGIId" in gdf.columns else "rgi_id"
    rgi_df = gdf[gdf[id_col] == RGI_ID].reset_index(drop=True)
    print(f"RGI entity: {rgi_df.iloc[0].get('Name', '(unnamed)')}")
else:
    print("Local RGI shapefile not found — downloading …")
    rgi_df = utils.get_rgi_glacier_entities([RGI_ID])

# ── Glacier directory (download Level 3 once, reuse thereafter) ───────────────
# Level 3 = elevation band flowlines + W5E5 climate + gridded_data.nc
# The 'reset' flag controls whether to re-download (True) or reuse (False).
print("Initialising glacier directory …")
gdirs = workflow.init_glacier_directories(
    rgi_df,
    from_prepro_level=PREPRO_LEVEL,
    prepro_base_url=PREPRO_BASE_URL,
    prepro_border=PREPRO_BORDER,
    reset=RESET_GDIR, force=RESET_GDIR,
)
gdir         = gdirs[0]
gridded_path = gdir.get_filepath("gridded_data")

# Verify gridded_data.nc exists — create it if missing (shouldn't happen with L3)
if not os.path.exists(gridded_path):
    print("  gridded_data.nc missing — creating via glacier_masks …")
    workflow.execute_entity_task(tasks.glacier_masks, gdirs)
print(f"✓ Glacier dir ready: {gdir.rgi_id}")

# ── Prepare 2-D redistribution grid (once, shared by all SSPs) ───────────────
# These three tasks set up the band-to-grid mapping used by distribute_2d.
# They write auxiliary files into the glacier directory and only need
# to run once regardless of how many SSP projections you run.
print("Preparing 2-D redistribution grid …")
workflow.execute_entity_task(distribute_2d.add_smoothed_glacier_topo, gdirs)
workflow.execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs)
workflow.execute_entity_task(distribute_2d.assign_points_to_band, gdirs)

# ── Historical baseline (once, shared by all SSPs) ────────────────────────────
# Runs the glacier from its RGI date using W5E5 historical climate.
# This spin-up is required before any future projection can start.
print("Running historical baseline …")
workflow.execute_entity_task(
    tasks.run_from_climate_data, gdirs,
    output_filesuffix="_historical",
)
print("Done.\n")

# ── Download reference SSP climate data (ssp370 only in fast mode) ────────────
ref_climate_suffix = f"_monthly_ISIMIP3b_{ISIMIP_MEMBER}_{REFERENCE_SSP}"

# Guard: delete any corrupted partial downloads from previously interrupted runs.
# A truncated HDF5 file causes: OSError: [Errno -101] NetCDF: HDF error
_isimip_cache = os.path.join(
    OGGM_DATA_DIR, "download_cache",
    "cluster.klima.uni-bremen.de", "~oggm", "cmip6",
    "isimip3b", "flat", "2023.2", "monthly",
)
import netCDF4 as _nc4
for _fname in [
    f"{ISIMIP_MEMBER}_w5e5_{REFERENCE_SSP}_tasAdjust_global_monthly_flat_glaciers.nc",
    f"{ISIMIP_MEMBER}_w5e5_{REFERENCE_SSP}_prAdjust_global_monthly_flat_glaciers.nc",
    f"{ISIMIP_MEMBER}_w5e5_ssp126_tasAdjust_global_monthly_flat_glaciers.nc",
    f"{ISIMIP_MEMBER}_w5e5_ssp126_prAdjust_global_monthly_flat_glaciers.nc",
    f"{ISIMIP_MEMBER}_w5e5_ssp585_tasAdjust_global_monthly_flat_glaciers.nc",
    f"{ISIMIP_MEMBER}_w5e5_ssp585_prAdjust_global_monthly_flat_glaciers.nc",
]:
    _fp = os.path.join(_isimip_cache, _fname)
    if os.path.exists(_fp):
        try:
            _nc4.Dataset(_fp)   # will raise if the file is corrupt / truncated
        except Exception:
            print(f"  Removing corrupted cache file: {_fname}")
            os.remove(_fp)

# Skip download if the processed gcm_data file already exists in the glacier dir
_gcm_data_path = gdir.get_filepath("gcm_data", filesuffix=ref_climate_suffix)
if os.path.exists(_gcm_data_path):
    print("Reference climate already processed — skipping download.")
else:
    print(f"Downloading reference climate: {ISIMIP_MEMBER} {REFERENCE_SSP}")
    print("  (ISIMIP3b glacier-points-only file — ~500 MB)")
    _run_isimip_download_with_progress(
        gdirs, ISIMIP_MEMBER, REFERENCE_SSP,
        ref_climate_suffix, SIM_START, SIM_END)
print("✓ Reference climate ready.\n")

# ── Per-SSP projection loop ───────────────────────────────────────────────────
for ssp in SSP_SCENARIOS:
    banner(f"Simulating {SSP_LABELS[ssp]}  ({ssp})")
    dest = nc_path(ssp)

    # Skip if the NetCDF already exists from a previous run
    if os.path.exists(dest):
        print(f"  ✓ NetCDF exists — skipping simulation")
        continue

    run_suffix = f"_projection_{ssp}_{ISIMIP_MEMBER}"

    if USE_BIAS_OFFSETS and ssp != REFERENCE_SSP:
        # ── FAST PATH: apply a constant temperature offset to ssp370 climate ─
        # This avoids downloading a separate climate file for each SSP.
        # The bias is constant in time (not linearly ramped) which is a
        # simplification but gives a reasonable first-order scenario spread.
        bias = SSP_TEMP_BIAS[ssp]
        print(f"  T-bias vs {REFERENCE_SSP}: {bias:+.1f} °C")
        workflow.execute_entity_task(
            tasks.run_from_climate_data, gdirs,
            climate_filename="gcm_data",
            climate_input_filesuffix=ref_climate_suffix,
            temperature_bias=bias,   # °C offset applied to every time step
            output_filesuffix=run_suffix,
            ys=SIM_START, ye=SIM_END,
        )
    else:
        # ── FULL PATH: use independently downloaded ISIMIP3b climate ─────────
        if ssp != REFERENCE_SSP:
            climate_suffix = f"_monthly_ISIMIP3b_{ISIMIP_MEMBER}_{ssp}"
            print(f"  Downloading ISIMIP3b {ssp} (~500 MB) …")
            workflow.execute_entity_task(
                gcm_climate.process_monthly_isimip_data, gdirs,
                member=ISIMIP_MEMBER, ssp=ssp,
                output_filesuffix=climate_suffix,
                y0=SIM_START - 5, y1=SIM_END + 5,
            )
        else:
            climate_suffix = ref_climate_suffix   # ssp370 is already downloaded

        print(f"  Running projection {SIM_START}–{SIM_END} …")
        workflow.execute_entity_task(
            tasks.run_from_climate_data, gdirs,
            climate_filename="gcm_data",
            climate_input_filesuffix=climate_suffix,
            output_filesuffix=run_suffix,
            ys=SIM_START, ye=SIM_END,
        )

    # ── Distribute 1-D flowline output to 2-D grid ───────────────────────────
    # This step maps the simulated ice thickness from the flowline representation
    # back onto the 2-D glacier grid using the band assignments prepared earlier.
    print("  Distributing to 2-D grid …")
    workflow.execute_entity_task(
        distribute_2d.distribute_thickness_from_simulation, gdirs,
        input_filesuffix=run_suffix,
        output_filesuffix=run_suffix,
    )

    # ── Save standardised NetCDF ──────────────────────────────────────────────
    saved = save_netcdf(ssp, run_suffix, gdir, gridded_path)
    print(f"  ✓ Saved: {saved}")

    # Quick diagnostic: glacier area and max thickness at key years
    with xr.open_dataset(saved) as _d:
        _d = _d.load()
    dx   = float(_d.x[1] - _d.x[0])
    c_km = dx ** 2 / 1e6   # cell area in km²
    for yr in [2000, 2050, 2100]:
        t = _d["ice_thickness"].sel(time=yr, method="nearest")
        print(f"    {yr}: area≈{float((t>0).sum())*c_km:.1f} km²  "
              f"max≈{float(t.max()):.0f} m")

print("\n✓ All SSP simulations complete.\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 2 — 3-D VISUALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

banner("PHASE 2 — 3-D VISUALIZATION")

# ── Renderer settings ─────────────────────────────────────────────────────────
# window_size: [width, height] in pixels — increase for higher-resolution output
PLOTTER_ARGS = {"off_screen": True, "window_size": [1280, 720]}

# ── Ice thickness colour bar ──────────────────────────────────────────────────
# position_x / position_y: 0-1 fractional position in the render window
# height: fractional height of the colour bar
ICE_ARGS = {
    "scalar_bar_args": {
        "title"     : "Ice Thickness          ",
        "color"     : "white",
        "vertical"  : True,
        "position_x": 0.08,
        "position_y": 0.25,
        "height"    : 0.45,
    }
}

# ── Bedrock (terrain colourmap) colour bar ────────────────────────────────────
TOPO_ARGS = {
    "cmap": "terrain",   # 'terrain', 'viridis', 'gist_earth' all work well
    "scalar_bar_args": {
        "title"     : "Bedrock",
        "color"     : "white",
        "vertical"  : True,
        "fmt"       : "%.0f m",
        "position_x": 0.88,
        "position_y": 0.25,
        "height"    : 0.45,
    },
}

# ── Satellite texture: no colour bar (texture IS the colouring) ───────────────
TOPO_ARGS_SAT = {"show_scalar_bar": False}

# ── Time label (shown in upper-right of every frame) ─────────────────────────
TIME_LABEL = {
    "text"     : "Year: {:.0f}",   # {} is filled by the current calendar_year
    "color"    : "white",
    "position" : "upper_right",
    "font_size": 16,
}

# ── Satellite tile crop region ────────────────────────────────────────────────
# The satellite tile server rejects requests for large bounding boxes.
# We crop to a sub-region of the grid that covers the main glacier body.
# Adjust these slices if the glacier is off-centre in the frame.
#   x: east-west,  y: north-south
#   slice(start, stop) — python 0-based indexing
SAT_X_SLICE = slice(60, 195)   # 135 cells × 200 m/cell ≈ 27 km
SAT_Y_SLICE = slice(60, 265)   # 205 cells × 200 m/cell ≈ 41 km

# ── Shared Glacier3DViz constructor arguments ─────────────────────────────────
# These tell Glacier3DViz which NetCDF variable names to use.
# camera_args: azimuth (0=north, 90=east), elevation (degrees above horizon)
COMMON_VIZ = dict(
    x="x", y="y", time="time",
    topo_bedrock="bedrock",
    ice_thickness="ice_thickness",
    time_var_display="calendar_year",   # use integer years for the label
    camera_args={"azimuth": -135, "elevation": 40, "zoom": 1.2},
)

# ── Rotating camera trajectory settings ──────────────────────────────────────
# start_angle / end_angle: azimuth in degrees (0 = north)
# camera_height: height of the camera in scene units
# camera_radius: distance from glacier centre
# Change start/end to rotate through a different arc
ROTATE_CFG = dict(
    camera_trajectory="rotate",
    kwargs_camera_trajectory={
        "start_angle"           : 250,
        "end_angle"             : 350,
        "camera_height"         : 40,
        "camera_radius"         : 5,
        "camera_radius_ref_axis": "x",
    },
)

# ── Annotations ───────────────────────────────────────────────────────────────
# These are overlaid on every rendered frame.

# Doda Peak summit marker
# To move the label: change latitude, longitude, height
# To resize: change font_size and point_size
# To recolour: change text_color and point_color
doda_peak = PointAnnotation(
    latitude=33.673733,     # decimal degrees N
    longitude=76.306208,    # decimal degrees E
    height=6550,            # elevation in metres
    text="Doda Peak 6550m", # label text — keep short to avoid clutter
    font_size=8,            # small label so it doesn't dominate the frame
    text_color="white",
    point_color="white",
    point_size=3,           # small sphere marker
    shape=None,
    render_points_as_spheres=True,
    always_visible=True,    # show even if the point is behind terrain
)

# North arrow — shows orientation in the 3-D scene
# x_position / y_position: -1 to +1, relative to scene centre
north_arrow = ArrowAnnotation(
    text="N",
    y_position=0.5,
    x_position=-0.1,
    arrow_magnitude=0.2,
    text_position_offset=[-0.02, 0.02, 0],
)

# RGI 2000 glacier outline — drawn as a thin black border on the terrain
# mask_data must match a 2-D binary variable in the dataset (loaded from gridded_data.nc)
rgi_outline = MaskAnnotation(
    mask_data="glacier_ext",      # 1 = inside RGI outline, 0 = outside
    mask_color=[0, 0, 0, 255],    # RGBA: solid black border
    add_z=0.1,                    # raise slightly above terrain to avoid z-fighting
)

# ── Globe image (rendered once, reused by all videos) ────────────────────────
# plot_centered_globe() draws a small world map with a red dot on the glacier.
# add_inset_with_ffmpeg() composites it into the corner of the video.
globe_png = os.path.join(ANIM_DIR, "durung_drong_globe.png")
_globe_ok = False
try:
    from glacier3dviz.tools import plot_centered_globe, add_inset_with_ffmpeg
    if not os.path.exists(globe_png):
        with xr.open_dataset(nc_path(REFERENCE_SSP)) as _ds:
            plot_centered_globe(_ds.load(), filename=globe_png)
        print(f"  Globe image → {globe_png}")
    _globe_ok = True
    print("  ✓ Globe ready")
except Exception as _e:
    print(f"  Globe skipped ({_e})")
    print("  → Install ffmpeg: conda install -c conda-forge ffmpeg")

# ── Per-SSP render loop ───────────────────────────────────────────────────────
for ssp in SSP_SCENARIOS:
    if not os.path.exists(nc_path(ssp)):
        print(f"  {ssp}: NetCDF not found — skipping")
        continue
    banner(f"Rendering {SSP_LABELS[ssp]}")

    # Load dataset with glacier mask and CRS injected
    ds_ssp = load_with_masks(ssp, gridded_path)

    # Title annotation shown inside the 3-D frame (upper-edge = top centre)
    # This is the ONLY title source — do not add TextAnnotation elsewhere for
    # this SSP to avoid the label appearing twice.
    _ssp_title = TextAnnotation(
        text=(f"Drang Drung Glacier Evolution in Different SSP Scenarios\n"
              f"{SSP_LABELS[ssp]}"),
        position="upper_edge",
    )
    anns = [doda_peak, north_arrow, rgi_outline, _ssp_title]

    # Create the main Glacier3DViz object for this SSP
    viz = Glacier3DViz(ds_ssp, **COMMON_VIZ)

    # ── Static snapshots at 2000 / 2050 / 2100 ───────────────────────────────
    for yr in [2000, 2050, 2100]:
        fp = out(ssp, f"snapshot_{yr}.png")
        viz.plot_year(yr, filepath=fp, show_plot=False,
            plotter_args=PLOTTER_ARGS,
            add_mesh_ice_thick_args=ICE_ARGS,
            add_mesh_topo_args=TOPO_ARGS,
            text_time_args=TIME_LABEL,
            additional_annotations=anns)
        stamp_png(fp)    # add logo + name to snapshot
        print(f"  ✓ snapshot {yr}")

    # ── Fixed-camera animation ────────────────────────────────────────────────
    anim_path = out(ssp, "animation.mp4")
    viz.export_animation(
        filename=anim_path, framerate=20, quality=10,
        plotter_args=PLOTTER_ARGS,
        add_mesh_ice_thick_args=ICE_ARGS,
        add_mesh_topo_args=TOPO_ARGS,
        text_time_args=TIME_LABEL,
        additional_annotations=anns)
    print(f"  ✓ animation → {anim_path}")
    add_watermark_to_video(anim_path)   # burn logo + name into video

    # Add globe inset to the fixed-camera animation
    if _globe_ok:
        add_inset_with_ffmpeg(
            main_video=anim_path, inset_image=globe_png,
            output_video=out(ssp, "animation_with_globe.mp4"),
            position="20:20",   # pixels from top-left corner
            scale_factor=1)     # 1 = full size; 0.5 = half size
        print(f"  ✓ globe inset")

    # ── Satellite texture version ─────────────────────────────────────────────
    if RENDER_SATELLITE:
        print(f"  Building satellite texture version …")
        _ds_sat = make_sat_dataset(ds_ssp, SAT_X_SLICE, SAT_Y_SLICE)
        _sat_title = TextAnnotation(
            text=(f"Drang Drung — Satellite Texture\n{SSP_LABELS[ssp]}"),
            position="upper_edge")
        _anns_sat = [doda_peak, north_arrow, rgi_outline, _sat_title]

        try:
            # use_texture=True triggers satellite tile download + drape on bedrock
            viz_sat = Glacier3DViz(_ds_sat, use_texture=True, **COMMON_VIZ)

            # Satellite static snapshots
            for _yr in [2000, 2050, 2100]:
                _sat_snap = out(ssp, f"sat_snapshot_{_yr}.png")
                viz_sat.plot_year(
                    _yr, filepath=_sat_snap,
                    show_plot=False, plotter_args=PLOTTER_ARGS,
                    add_mesh_ice_thick_args=ICE_ARGS,
                    add_mesh_topo_args=TOPO_ARGS_SAT,
                    text_time_args=TIME_LABEL,
                    additional_annotations=_anns_sat)
                stamp_png(_sat_snap)   # add logo + name

            # Satellite fixed-camera animation
            _sat_mp4 = out(ssp, "animation_sat.mp4")
            viz_sat.export_animation(
                filename=_sat_mp4, framerate=20, quality=10,
                plotter_args=PLOTTER_ARGS,
                add_mesh_ice_thick_args=ICE_ARGS,
                add_mesh_topo_args=TOPO_ARGS_SAT,
                text_time_args=TIME_LABEL,
                additional_annotations=_anns_sat)
            print(f"  ✓ satellite animation → {_sat_mp4}")
            add_watermark_to_video(_sat_mp4)   # burn logo + name

            # Add globe inset to satellite animation
            if _globe_ok:
                add_inset_with_ffmpeg(
                    main_video=_sat_mp4, inset_image=globe_png,
                    output_video=out(ssp, "animation_sat_with_globe.mp4"),
                    position="20:20", scale_factor=1)
                print(f"  ✓ satellite + globe inset")

            del viz_sat
        except Exception as _e:
            print(f"  Satellite texture failed: {_e}")
        del _ds_sat

    del viz, ds_ssp

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 3 — 4-PANEL COMPARISON VIDEOS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

banner("PHASE 3 — 4-PANEL COMPARISON")

# Load all 4 SSP datasets (with masks already injected)
datasets = {
    ssp: load_with_masks(ssp, gridded_path)
    for ssp in SSP_SCENARIOS
    if os.path.exists(nc_path(ssp))
}

if len(datasets) < 2:
    print("Need ≥ 2 SSP datasets — skipping comparison.")
else:
    # Cell area in km² (used for computing glacier area from mask counts)
    dx   = float(list(datasets.values())[0].x[1] -
                 list(datasets.values())[0].x[0])
    c_km = dx ** 2 / 1e6

    # ── Static 2-D comparison maps ────────────────────────────────────────────
    # Bedrock (terrain colourmap) + ice thickness overlay for each SSP at each year
    for yr in [2000, 2050, 2100]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#1a1a2e")
        for ax, ssp in zip(axes.flatten(), SSP_SCENARIOS):
            if ssp not in datasets:
                ax.axis("off")
                continue
            ds = datasets[ssp]
            t  = ds["ice_thickness"].sel(time=yr, method="nearest")
            ds["bedrock"].plot(ax=ax, cmap="terrain",
                               add_colorbar=False, alpha=0.7)
            t.where(t > 0).plot(ax=ax, cmap="Blues", vmin=0,
                add_colorbar=True,
                cbar_kwargs={"label": "Ice thickness (m)", "shrink": 0.7})
            area = float((t > 0).sum()) * c_km
            ax.set_title(f"{SSP_LABELS[ssp]}\nArea ≈ {area:.1f} km²",
                         color="white", fontsize=11)
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            for sp in ax.spines.values():
                sp.set_edgecolor(SSP_COLORS[ssp])
                sp.set_linewidth(2)
        fig.suptitle(
            f"Drang Drung Glacier Evolution in Different SSP Scenarios ",
            color="white", fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        p = os.path.join(ANIM_DIR, f"comparison_4panel_{yr}.png")
        add_watermark_to_figure(fig)   # stamp logo + name
        plt.savefig(p, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  ✓ {p}")

    # ── Time-series comparison chart ──────────────────────────────────────────
    # Left panel: glacier area over time for all SSPs
    # Right panel: mean ice thickness over time for all SSPs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1a1a2e")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")

    for ssp in SSP_SCENARIOS:
        if ssp not in datasets:
            continue
        ds   = datasets[ssp]
        mask = ds["ice_thickness"] > 0
        yrs  = ds["calendar_year"].values
        ax1.plot(yrs, (mask.sum(dim=["x", "y"]) * c_km).values,
                 color=SSP_COLORS[ssp], lw=2, label=SSP_LABELS[ssp])
        ax2.plot(yrs, ds["ice_thickness"].where(mask).mean(dim=["x", "y"]).values,
                 color=SSP_COLORS[ssp], lw=2, label=SSP_LABELS[ssp])

    ax1.set(xlabel="Year", ylabel="Area (km²)", title="Glacier Area 2000–2100")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax1.grid(alpha=0.2, color="white")
    ax2.set(xlabel="Year", ylabel="Mean ice thickness (m)",
            title="Mean Ice Thickness 2000–2100")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax2.grid(alpha=0.2, color="white")
    fig.suptitle(
        "Drang Drung Glacier — SSP Comparison | ISIMIP3b MRI-ESM2-0",
        color="white", fontsize=13)
    plt.tight_layout()
    add_watermark_to_figure(fig)   # stamp logo + name
    ts_p = os.path.join(ANIM_DIR, "comparison_timeseries.png")
    plt.savefig(ts_p, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {ts_p}")

    # ── Render each SSP as individual per-frame PNG arrays ───────────────────
    # We render all 101 years for each SSP separately, store the frames in memory,
    # then composite them into the 2×2 panel figure.
    # The title annotation is OMITTED from 3D frames here because ax.set_title()
    # in the composite figure already labels each panel — having both would duplicate.
    all_years = list(datasets[REFERENCE_SSP].calendar_year.values)
    PANEL_TITLE = "Drang Drung Glacier Evolution in Different SSP Scenarios"

    # ── A) Fixed-camera terrain panel video ──────────────────────────────────
    print("\n  Rendering fixed-camera 4-panel video (terrain) …")
    rendered_fixed = {}
    for ssp in SSP_SCENARIOS:
        if ssp not in datasets:
            continue
        print(f"    {ssp} …")
        # No TextAnnotation — panel title comes from ax.set_title only
        _anns = [doda_peak, north_arrow, rgi_outline]
        _viz  = Glacier3DViz(datasets[ssp], **COMMON_VIZ)
        _tmp  = os.path.join(ANIM_DIR, f"_tmp_{ssp}.png")
        _frames = []
        for yr in all_years:
            _viz.plot_year(int(yr), filepath=_tmp, show_plot=False,
                plotter_args=PLOTTER_ARGS,
                add_mesh_ice_thick_args=ICE_ARGS,
                add_mesh_topo_args=TOPO_ARGS,
                text_time_args=TIME_LABEL,
                additional_annotations=_anns)
            _frames.append(plt.imread(_tmp).copy())
        rendered_fixed[ssp] = _frames
        del _viz
        if os.path.exists(_tmp):
            os.remove(_tmp)
        print(f"    ✓ {len(_frames)} frames")

    # Composite and save with year in title + optional globe inset
    _fixed_mp4 = os.path.join(ANIM_DIR, "comparison_4panel.mp4")
    composite_4panel_video(
        rendered_fixed, _fixed_mp4,
        title_text=PANEL_TITLE,
        fps=5, bitrate=4000, dpi=160,
        globe_path=globe_png if _globe_ok else None,
    )

    # ── B) Rotating-camera terrain panel video ────────────────────────────────
    # Each SSP is rendered with a rotating camera sweep (250°→350°),
    # decoded to individual PNG frames via ffmpeg, then composited.
    print("\n  Rendering rotating-camera 4-panel video (terrain) …")
    rendered_rot = {}
    for ssp in SSP_SCENARIOS:
        if ssp not in datasets:
            continue
        print(f"    {ssp} rotating …")
        _anns_r = [doda_peak, north_arrow, rgi_outline]
        _viz_r  = Glacier3DViz(datasets[ssp], **COMMON_VIZ)
        _rot_mp4_tmp = os.path.join(ANIM_DIR, f"_rot_{ssp}.mp4")

        # Export the rotating animation for this SSP to a temporary file
        _viz_r.export_animation(
            filename=_rot_mp4_tmp, framerate=20, quality=10,
            plotter_args=PLOTTER_ARGS,
            add_mesh_ice_thick_args=ICE_ARGS,
            add_mesh_topo_args=TOPO_ARGS,
            text_time_args=TIME_LABEL,
            additional_annotations=_anns_r,
            **ROTATE_CFG,
        )
        del _viz_r

        # Decode the mp4 to PNG frames using ffmpeg, then load into numpy
        _tmpdir = tempfile.mkdtemp()
        _flist  = extract_mp4_frames(_rot_mp4_tmp, _tmpdir)
        rendered_rot[ssp] = [plt.imread(fp).copy() for fp in _flist]
        shutil.rmtree(_tmpdir, ignore_errors=True)
        if os.path.exists(_rot_mp4_tmp):
            os.remove(_rot_mp4_tmp)
        print(f"    ✓ {len(rendered_rot[ssp])} frames")

    _rot_mp4 = os.path.join(ANIM_DIR, "comparison_4panel_rotating.mp4")
    composite_4panel_video(
        rendered_rot, _rot_mp4,
        title_text=f"{PANEL_TITLE}",
        fps=5, bitrate=4000, dpi=160,
        globe_path=globe_png if _globe_ok else None,
        globe_label="_rotating",
    )

    # ── C) Rotating-camera SATELLITE TEXTURE panel video ─────────────────────
    # Same as B) but each SSP uses satellite imagery on the bedrock.
    # Satellite datasets are cropped to SAT_X_SLICE/SAT_Y_SLICE.
    if RENDER_SATELLITE:
        print("\n  Rendering rotating-camera 4-panel video (satellite texture) …")
        rendered_sat_rot = {}
        for ssp in SSP_SCENARIOS:
            if ssp not in datasets:
                continue
            print(f"    {ssp} satellite rotating …")
            _ds_sat_r = make_sat_dataset(datasets[ssp], SAT_X_SLICE, SAT_Y_SLICE)
            _anns_sr  = [doda_peak, north_arrow, rgi_outline]
            _sat_rot_tmp = os.path.join(ANIM_DIR, f"_satrot_{ssp}.mp4")
            try:
                _viz_sr = Glacier3DViz(_ds_sat_r, use_texture=True, **COMMON_VIZ)
                _viz_sr.export_animation(
                    filename=_sat_rot_tmp, framerate=20, quality=10,
                    plotter_args=PLOTTER_ARGS,
                    add_mesh_ice_thick_args=ICE_ARGS,
                    add_mesh_topo_args=TOPO_ARGS_SAT,
                    text_time_args=TIME_LABEL,
                    additional_annotations=_anns_sr,
                    **ROTATE_CFG,
                )
                del _viz_sr

                _tmpdir_sr = tempfile.mkdtemp()
                _flist_sr  = extract_mp4_frames(_sat_rot_tmp, _tmpdir_sr)
                rendered_sat_rot[ssp] = [plt.imread(fp).copy() for fp in _flist_sr]
                shutil.rmtree(_tmpdir_sr, ignore_errors=True)
                if os.path.exists(_sat_rot_tmp):
                    os.remove(_sat_rot_tmp)
                print(f"    ✓ {len(rendered_sat_rot[ssp])} frames")
            except Exception as _e:
                print(f"    ✗ {ssp} satellite failed: {_e}")
            del _ds_sat_r

        if rendered_sat_rot:
            _sat_rot_mp4 = os.path.join(ANIM_DIR,
                                         "comparison_4panel_sat_rotating.mp4")
            composite_4panel_video(
                rendered_sat_rot, _sat_rot_mp4,
                title_text=f"{PANEL_TITLE}",
                fps=5, bitrate=4000, dpi=160,
                # Globe inset on the satellite rotating panel video
                globe_path=globe_png if _globe_ok else None,
                globe_label="_sat_rotating",
            )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

banner("ALL DONE")
print(f"Outputs directory: {WORKING_DIR}\n")

print("NetCDF simulation files:")
for ssp in SSP_SCENARIOS:
    p  = nc_path(ssp)
    sz = f"{os.path.getsize(p)/1e6:.0f} MB" if os.path.exists(p) else "MISSING"
    print(f"  [{sz:>6}]  {os.path.basename(p)}")

print("\nPer-SSP animations  (animation/<ssp>/):")
print("  snapshot_2000/2050/2100.png        ← terrain colourmap")
print("  animation.mp4                      ← fixed camera, terrain")
print("  animation_with_globe.mp4           ← fixed camera + globe inset")
print("  sat_snapshot_2000/2050/2100.png    ← satellite texture")
print("  animation_sat.mp4                  ← fixed camera, satellite")
print("  animation_sat_with_globe.mp4       ← satellite + globe inset")

print("\n4-panel comparison  (animation/):")
print("  comparison_4panel.mp4                       ← fixed camera, terrain")
print("  comparison_4panel_with_globe.mp4            ← fixed camera + globe")
print("  comparison_4panel_rotating.mp4              ← rotating camera, terrain")
print("  comparison_4panel_rotating_with_globe.mp4   ← rotating + globe")
print("  comparison_4panel_sat_rotating.mp4          ← rotating, satellite")
print("  comparison_4panel_sat_rotating_with_globe.mp4 ← sat + rotating + globe")
print("  comparison_4panel_2000/2050/2100.png        ← static 2-D maps")
print("  comparison_timeseries.png                   ← area + thickness chart")
