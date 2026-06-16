"""
Plant Calculation Script
------------------------

This script calculates temperature summary values for:

1. every individual plant point
2. species-dependent averages across all plant points of the same species
3. every named rockwall polygon section

If the summary raster matrices do not yet exist, they are automatically
computed from the aligned thermal TXT files and saved as CSV files first.

Workflow
--------
1. Load or compute raster summary matrices:
      - minimum temperature
      - mean temperature
      - maximum temperature
      - amplitude
2. Load the AOI mask
3. Load plant points
4. Load previously saved rockwall polygons
5. Calculate ring-based mean values for each plant
6. Calculate averages per species
7. Calculate the same parameters for each rockwall section
8. Export two CSV tables:
      - one for individual plant values
      - one for species averages and rockwall sections

Outputs
-------
- {STUDY_SITE}_species_values.csv
- {STUDY_SITE}_species_averages_rockwalls.csv
"""

##!/usr/bin/env python3
"""
Plants Calculation: plant stats + species means + rockwall stats
----------------------------------------------------------------

This version can work even if min/mean/max/amplitude matrix CSV files do not
exist yet.

Behavior
--------
1. If matrix CSVs are present, they are loaded directly.
2. If they are missing, the script reads aligned thermal TXT frames and
   computes:
      - min matrix
      - mean matrix
      - max matrix
      - amplitude matrix
3. Those matrices are then saved as CSV for future runs.
4. Plant values, species means, and rockwall polygon values are calculated.

By default this version uses already-saved rockwall polygons from ROCKWALL_POLY_CSV.

Final CSV contents
------------------
- one row per plant
- one row per species mean
- one row per rockwall section
- one overall rockwall mean row
"""

import os
import glob
import datetime
import re
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector

# ===================== USER CONFIG =====================

# Folder where matrix CSVs are expected / written
RUN_FOLDER = r"F:\Masterarbeit_Backup_2\Results\Plants\Lehesten_25-26.08.25"

# If matrix CSVs are missing, they are computed from these aligned TXT frames
INPUT_TXT_FOLDER = r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted_aligned\Lehesten_25-26.08.25"

# Matrix CSV paths
MIN_CSV  = os.path.join(RUN_FOLDER, "min_matrix.csv")
MEAN_CSV = os.path.join(RUN_FOLDER, "mean_matrix.csv")
MAX_CSV  = os.path.join(RUN_FOLDER, "max_matrix.csv")
AMP_CSV  = os.path.join(RUN_FOLDER, "amplitude_matrix.csv")

# Plant points
PLANTS_CSV = os.path.join(
    r"F:\Masterarbeit_Backup_2\Results\Plants\Lehesten_25-26.08.25",
    "plant_points_unmasked.csv"
)

# AOI mask
AOI_FOLDER = RUN_FOLDER
AOI_GLOB = "aoi_*.npz"
AOI_NPZ = None  # set explicit NPZ path if you prefer

# Rockwall polygons
ROCKWALL_POLY_CSV = os.path.join(
    r"F:\Masterarbeit_Backup_2\Results\Plants\Lehesten_25-26.08.25",
    "rockwall_polygons.csv"
)
REQUIRE_ROCKWALL_POLYGONS = True

# Interactive rockwall polygon drawing
# Set to False because rockwall polygons were already defined in Plant_overview_plot.py
INTERACTIVE_ROCKWALL = False

# Plant sampling geometry: RING (donut)
# 6.25=10 cm Paradiestal, 7.16~10 cm Lehesten
OUTER_R_PX = 14.32
INNER_R_PX = 7.16

# Optional top opening
OPENING_HALF_ANGLE_DEG = 45

# Study site name used for output filenames
STUDY_SITE = "Lehesten"

# Output
OUT_DIR = r"F:\Masterarbeit_Backup_2\Results\Plants\Lehesten_25-26.08.25"
OUT_CSV_PLANTS = os.path.join(OUT_DIR, f"{STUDY_SITE}_species_values.csv")
OUT_CSV_AVG = os.path.join(OUT_DIR, f"{STUDY_SITE}_species_averages_rockwalls.csv")
# =======================================================


# ======================================================
# GENERAL HELPER FUNCTIONS
# ======================================================
def natural_sort_key(path: str):
    """
    Generates a sorting key that sorts embedded numbers correctly.

    Example
    -------
    frame1.txt
    frame2.txt
    frame10.txt
    """
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


def list_txt_files(folder: str) -> list[str]:
    """
    Returns a sorted list of all aligned thermal TXT files in the input folder.
    """
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".txt")
    ]
    return sorted(files, key=natural_sort_key)


def read_thermal_txt(path: str) -> np.ndarray:
    """
    Reads one thermal camera TXT export.

    The function searches for the "[Data]" section and reads the following
    tab-separated temperature matrix into a NumPy array.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    try:
        i0 = lines.index("[Data]") + 1
    except ValueError as e:
        raise ValueError(f"Missing [Data] section in file: {path}") from e

    rows = []
    for ln in lines[i0:]:
        if not ln.strip():
            continue

        parts = ln.replace(",", ".").split("\t")
        try:
            rows.append([float(p) for p in parts if p != ""])
        except ValueError:
            break

    if not rows:
        raise ValueError(f"No numeric data found after [Data] in file: {path}")

    ncol = max(len(r) for r in rows)
    arr = np.full((len(rows), ncol), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = np.array(r, dtype=np.float32)
    return arr


def compute_and_save_matrices_if_missing() -> None:
    """
    Loads existing summary matrices if available.

    If one or more matrix CSV files are missing, the script computes:
        - min matrix
        - mean matrix
        - max matrix
        - amplitude matrix

    from the aligned thermal TXT files and saves them for future runs.
    """
    matrix_paths = [MIN_CSV, MEAN_CSV, MAX_CSV, AMP_CSV]
    if all(os.path.exists(p) for p in matrix_paths):
        return

    print("Matrix CSVs not found. Computing them from aligned thermal TXT files...")
    txt_files = list_txt_files(INPUT_TXT_FOLDER)
    if not txt_files:
        raise FileNotFoundError(f"No thermal .txt files found in:\n{INPUT_TXT_FOLDER}")

    print(f"Processing {len(txt_files)} thermal files...")
    mats = [read_thermal_txt(p) for p in txt_files]
    shapes = {m.shape for m in mats}
    if len(shapes) != 1:
        raise ValueError(f"Not all TXT rasters have the same shape: {sorted(shapes)}")

    stack = np.stack(mats)
    min_mat = np.nanmin(stack, axis=0)
    mean_mat = np.nanmean(stack, axis=0)
    max_mat = np.nanmax(stack, axis=0)
    amp_mat = max_mat - min_mat

    os.makedirs(RUN_FOLDER, exist_ok=True)
    pd.DataFrame(min_mat).to_csv(MIN_CSV, index=False)
    pd.DataFrame(mean_mat).to_csv(MEAN_CSV, index=False)
    pd.DataFrame(max_mat).to_csv(MAX_CSV, index=False)
    pd.DataFrame(amp_mat).to_csv(AMP_CSV, index=False)

    print("Saved computed matrix CSVs:")
    print(" ", MIN_CSV)
    print(" ", MEAN_CSV)
    print(" ", MAX_CSV)
    print(" ", AMP_CSV)


# ======================================================
# MATRIX + AOI LOADING FUNCTIONS
# ======================================================
def load_matrix(path: str) -> np.ndarray:
    """
    Loads one summary raster matrix from CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, header=0).values.astype(float)


def load_aoi_mask() -> np.ndarray:
    """
    Loads the AOI mask from an NPZ file.
    """
    if AOI_NPZ:
        path = AOI_NPZ
    else:
        matches = sorted(glob.glob(os.path.join(AOI_FOLDER, AOI_GLOB)))
        if not matches:
            raise FileNotFoundError(f"No AOI mask found in {AOI_FOLDER} matching {AOI_GLOB}")
        path = matches[0]

    data = np.load(path, allow_pickle=False)
    if "mask" not in data:
        raise KeyError(f"No 'mask' key in {path}")
    mask = data["mask"].astype(bool)
    if not mask.any():
        raise ValueError("AOI mask has 0 pixels.")
    return mask


def crop_to_common_shape(*arrays: np.ndarray) -> list[np.ndarray]:
    """
    Crops all input rasters to the smallest common shape.

    This prevents dimension mismatches if matrices or AOI masks differ slightly
    in width or height.
    """
    shapes = [a.shape for a in arrays]
    h = min(s[0] for s in shapes)
    w = min(s[1] for s in shapes)
    if len(set(shapes)) != 1:
        print("WARNING: Shape mismatch detected, cropping all to:", (h, w))
        for i, s in enumerate(shapes, start=1):
            print(f"  shape{i}: {s}")
    return [a[:h, :w] for a in arrays]


# ======================================================
# COORDINATE CONVERSION
# ======================================================
def display_to_original(x1, y1, nrow: int) -> tuple[int, int]:
    """
    Converts 1-based saved coordinates to 0-based raster indices.

    Important
    ---------
    The plant overview plot and picker already use the same raster-oriented
    coordinate convention, so no y-flip is applied here.
    """
    x1i = int(round(float(x1)))
    y1i = int(round(float(y1)))
    return x1i - 1, y1i - 1


# ======================================================
# PLANT RING SAMPLING FUNCTIONS
# ======================================================
def ring_with_optional_opening_mask(shape: tuple[int, int],
                                    xi: int, yi: int,
                                    r_outer: float, r_inner: float,
                                    opening_half_angle_deg: float):
    """
    Builds a local ring mask around one plant position.

    The ring can optionally include a top opening, matching the sampling
    geometry used in the plant analysis.
    """
    h, w = shape
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return None

    outer = int(round(max(r_outer, r_inner)))
    inner = int(round(min(r_outer, r_inner)))

    if not (0 <= inner < outer):
        raise ValueError("Require 0 <= INNER_R_PX < OUTER_R_PX after ordering radii.")

    x0 = max(0, xi - outer)
    x1 = min(w - 1, xi + outer)
    y0 = max(0, yi - outer)
    y1 = min(h - 1, yi + outer)

    yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
    dx = (xx - xi).astype(float)
    dy = (yy - yi).astype(float)

    rr2 = dx * dx + dy * dy
    ring = (rr2 <= outer * outer) & (rr2 > inner * inner)

    if opening_half_angle_deg and float(opening_half_angle_deg) > 0:
        ang = np.degrees(np.arctan2(dx, -dy))
        opening = np.abs(ang) <= float(opening_half_angle_deg)
        ring = ring & (~opening)

    return (y0, y1, x0, x1, ring)


def ring_nanmean(mat: np.ndarray, aoi: np.ndarray,
                 xi: int, yi: int,
                 r_outer: float, r_inner: float,
                 opening_half_angle_deg: float) -> float:
    """
    Calculates the mean value inside the valid AOI part of one plant ring.
    """
    res = ring_with_optional_opening_mask(mat.shape, xi, yi, r_outer, r_inner, opening_half_angle_deg)
    if res is None:
        return np.nan
    y0, y1, x0, x1, local = res
    sub = mat[y0:y1 + 1, x0:x1 + 1]
    sub_aoi = aoi[y0:y1 + 1, x0:x1 + 1]
    m = local & sub_aoi
    vals = sub[m]
    return float(np.nanmean(vals)) if vals.size else np.nan


# ======================================================
# ROCKWALL POLYGON FUNCTIONS
# ======================================================
def polygons_to_masks(poly_csv: str, nrow: int, ncol: int) -> list[tuple[str, np.ndarray]]:
    """
    Loads named rockwall polygons from CSV and rasterizes each one separately.

    Each polygon becomes an individual boolean mask that can later be used to
    calculate mean raster values for that rockwall section.
    """
    if not os.path.exists(poly_csv):
        raise FileNotFoundError(poly_csv)

    df = pd.read_csv(poly_csv)
    required = {"polygon", "vertex_order", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"{poly_csv} must contain columns: {sorted(required)}")

    yy, xx = np.mgrid[0:nrow, 0:ncol]
    pts = np.vstack([xx.ravel() + 0.5, yy.ravel() + 0.5]).T

    masks: list[tuple[str, np.ndarray]] = []

    for poly_name, sub in df.groupby("polygon"):
        sub = sub.sort_values("vertex_order")
        if len(sub) < 3:
            continue

        verts = []
        for _, r in sub.iterrows():
            xi, yi = display_to_original(r["x"], r["y"], nrow)
            verts.append((xi + 0.5, yi + 0.5))

        path = MplPath(np.array(verts, dtype=float), closed=True)
        inside = path.contains_points(pts).reshape((nrow, ncol))
        masks.append((str(poly_name), inside))

    return masks


# ======================================================
# INTERACTIVE ROCKWALL POLYGON PICKER
# ======================================================
def define_rockwall_polygons_interactive(background_mat: np.ndarray, aoi: np.ndarray) -> list[tuple[str, list[tuple[float, float]]]]:
    """
    Opens an interactive Matplotlib window to draw one or more rockwall polygons.

    Returned vertices use 1-based display coordinates so they stay consistent
    with the saved coordinate convention used elsewhere in the workflow.
    """
    nrow, ncol = background_mat.shape
    polys_1b: list[tuple[str, list[tuple[float, float]]]] = []

    fig, ax = plt.subplots()
    ax.set_title("Draw rockwall polygon(s): Enter=accept, Esc=cancel, q=finish")
    ax.imshow(np.flipud(background_mat), origin="lower")

    aoi_disp = np.flipud(aoi.astype(float))
    ax.imshow(np.where(aoi_disp > 0, np.nan, 1.0), origin="lower", alpha=0.25)

    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)

    current_verts: dict[str, list[tuple[float, float]] | None] = {"xy": None}

    def onselect(verts):
        current_verts["xy"] = verts

    try:
        selector = PolygonSelector(
            ax, onselect,
            useblit=True,
            lineprops=dict(linewidth=2),
            markerprops=dict(marker="o", markersize=4),
        )
    except TypeError:
        selector = PolygonSelector(
            ax, onselect,
            useblit=True,
            props=dict(linewidth=2),
            handle_props=dict(marker="o", markersize=4),
        )

    def _reset_selector():
        current_verts["xy"] = None
        for attr in ("_xs", "_ys"):
            if hasattr(selector, attr):
                setattr(selector, attr, [])
        if hasattr(selector, "_selection_completed"):
            selector._selection_completed = False
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "enter":
            verts = current_verts["xy"]
            if not verts or len(verts) < 3:
                print("Polygon not accepted: need at least 3 vertices.")
                return

            default_name = f"ROCKWALL_{len(polys_1b) + 1}"
            name = input(f"Name for this rockwall polygon [{default_name}]: ").strip()
            if name == "":
                name = default_name

            poly_1b = [(float(x) + 1.0, float(y) + 1.0) for x, y in verts]
            polys_1b.append((name, poly_1b))
            print(f"Accepted rockwall polygon '{name}' with {len(poly_1b)} vertices.")
            _reset_selector()

        elif event.key == "escape":
            print("Canceled current polygon.")
            _reset_selector()

        elif event.key == "q":
            if current_verts["xy"]:
                print("Note: current polygon not accepted (press Enter to accept).")
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    return polys_1b


def polygon1b_display_to_mask(poly_1b: list[tuple[float, float]], nrow: int, ncol: int) -> np.ndarray:
    """
    Rasterizes a single polygon given as 1-based display vertices into a mask.
    """
    yy, xx = np.mgrid[0:nrow, 0:ncol]
    pts = np.vstack([xx.ravel() + 0.5, yy.ravel() + 0.5]).T

    verts_orig = []
    for x1, y1 in poly_1b:
        xi, yi = display_to_original(x1, y1, nrow)
        verts_orig.append((xi + 0.5, yi + 0.5))

    path = MplPath(np.array(verts_orig, dtype=float), closed=True)
    inside = path.contains_points(pts).reshape((nrow, ncol))
    return inside


def build_metric_columns() -> tuple[str, str, str, str]:
    """
    Creates consistent metric column names used throughout the calculation.
    """
    col_min  = "min"
    col_mean = "mean"
    col_max  = "max"
    col_amp  = "amp"
    return col_min, col_mean, col_max, col_amp


def main():
    """
    Main processing workflow.

    Steps
    -----
    1. Load or compute summary raster matrices
    2. Load the AOI mask
    3. Load previously defined rockwall polygons
    4. Load plant points
    5. Calculate plant-level ring means
    6. Calculate species means
    7. Calculate rockwall section means
    8. Export two CSV tables
    """
    # --------------------------------------------------
    # Load or compute summary raster matrices
    # --------------------------------------------------
    compute_and_save_matrices_if_missing()

    print("Loading matrices...")
    min_mat  = load_matrix(MIN_CSV)
    mean_mat = load_matrix(MEAN_CSV)
    max_mat  = load_matrix(MAX_CSV)
    amp_mat  = load_matrix(AMP_CSV)

    # --------------------------------------------------
    # Load AOI mask
    # --------------------------------------------------
    print("Loading AOI mask...")
    aoi = load_aoi_mask()

    aoi, min_mat, mean_mat, max_mat, amp_mat = crop_to_common_shape(aoi, min_mat, mean_mat, max_mat, amp_mat)
    nrow, ncol = min_mat.shape
    print("Final raster shape:", (nrow, ncol))

    # --------------------------------------------------
    # Load previously defined rockwall polygons
    # --------------------------------------------------
    rockwall_sections: list[tuple[str, np.ndarray]] = []

    if INTERACTIVE_ROCKWALL:
        print("Interactive mode: define pure rockwall polygon section(s) in a pop-up window...")
        polys_named = define_rockwall_polygons_interactive(mean_mat, aoi)

        if not polys_named:
            raise ValueError("No rockwall polygons were defined interactively.")

        for poly_name, poly_1b in polys_named:
            poly_mask = polygon1b_display_to_mask(poly_1b, nrow=nrow, ncol=ncol)
            m = aoi & poly_mask
            if not m.any():
                print(f"WARNING: rockwall section '{poly_name}' has 0 pixels after intersecting with AOI (skipping).")
                continue
            rockwall_sections.append((poly_name, m))

        if not rockwall_sections:
            raise ValueError("All interactive rockwall sections resulted in empty masks after AOI intersection.")
    else:
        if os.path.exists(ROCKWALL_POLY_CSV):
            print("Using previously defined rockwall polygons from:")
            print(" ", ROCKWALL_POLY_CSV)
            masks = polygons_to_masks(ROCKWALL_POLY_CSV, nrow=nrow, ncol=ncol)
            for poly_name, poly_mask in masks:
                m = aoi & poly_mask
                if m.any():
                    rockwall_sections.append((poly_name, m))
                else:
                    print(f"WARNING: rockwall section '{poly_name}' has 0 pixels after intersecting with AOI (skipping).")
            if not rockwall_sections:
                raise ValueError("Rockwall polygons were loaded, but all resulted in empty masks after AOI intersection.")
        else:
            if REQUIRE_ROCKWALL_POLYGONS:
                raise FileNotFoundError(f"Rockwall polygon file not found: {ROCKWALL_POLY_CSV}")
            print("WARNING: Rockwall polygons missing; using AOI as rockwall reference.")
            rockwall_sections = [("ROCKWALL_ALL", aoi.copy())]

    # --------------------------------------------------
    # Load plant coordinate table
    # --------------------------------------------------
    print("Loading plant points:", PLANTS_CSV)
    plants = pd.read_csv(PLANTS_CSV)
    for col in ("x", "y", "species"):
        if col not in plants.columns:
            raise ValueError(f"Missing column '{col}' in {PLANTS_CSV}")

    col_min, col_mean, col_max, col_amp = build_metric_columns()

    # --------------------------------------------------
    # Calculate individual plant values
    # --------------------------------------------------
    plant_rows = []
    print(f"Computing plant ring means for {len(plants)} points...")
    for _, r in plants.iterrows():
        xi, yi = display_to_original(r["x"], r["y"], nrow)

        plant_rows.append({

            "species": r["species"],


            col_min:  ring_nanmean(min_mat,  aoi, xi, yi, OUTER_R_PX, INNER_R_PX, OPENING_HALF_ANGLE_DEG),
            col_mean: ring_nanmean(mean_mat, aoi, xi, yi, OUTER_R_PX, INNER_R_PX, OPENING_HALF_ANGLE_DEG),
            col_max:  ring_nanmean(max_mat,  aoi, xi, yi, OUTER_R_PX, INNER_R_PX, OPENING_HALF_ANGLE_DEG),
            col_amp:  ring_nanmean(amp_mat,  aoi, xi, yi, OUTER_R_PX, INNER_R_PX, OPENING_HALF_ANGLE_DEG),
        })

    plant_df = pd.DataFrame(plant_rows)

    # --------------------------------------------------
    # Calculate species-dependent averages
    # --------------------------------------------------
    species_df = (
        plant_df
        .groupby("species", dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )




    # --------------------------------------------------
    # Calculate rockwall section values
    # --------------------------------------------------
    rockwall_rows = []
    for section_name, m in rockwall_sections:
        rockwall_rows.append({

            "species": section_name,


            col_min:  float(np.nanmean(min_mat[m])),
            col_mean: float(np.nanmean(mean_mat[m])),
            col_max:  float(np.nanmean(max_mat[m])),
            col_amp:  float(np.nanmean(amp_mat[m])),
        })

    rockwall_df = pd.DataFrame(rockwall_rows)



    # --------------------------------------------------
    # Export final CSV tables
    # --------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)

    plants_out = OUT_CSV_PLANTS
    avg_out = OUT_CSV_AVG

    plants_only = plant_df.copy()
    averages = pd.concat([species_df, rockwall_df], ignore_index=True)

    try:
        plants_only.to_csv(plants_out, index=False)
    except PermissionError:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plants_out = os.path.join(OUT_DIR, f"{STUDY_SITE}_species_values_{stamp}.csv")
        plants_only.to_csv(plants_out, index=False)

    try:
        averages.to_csv(avg_out, index=False)
    except PermissionError:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        avg_out = os.path.join(OUT_DIR, f"{STUDY_SITE}_species_averages_rockwalls_{stamp}.csv")
        averages.to_csv(avg_out, index=False)

    print("DONE.")
    print("Plant values saved:", plants_out)
    print("Species averages + rockwalls saved:", avg_out)


if __name__ == "__main__":
    main()