"""
Thermal Image Simple Metrics Script
-----------------------------------

This script reads a series of thermal camera text files (.txt) that contain
temperature matrices exported from thermal imaging software, which are already preprocessed (image alignment).

It performs the following steps:

1. Loads all thermal matrices from the INPUT_FOLDER
2. Stacks them into a 3D array (time, y, x)
3. Calculates three statistics per pixel:
      - maximum temperature
      - minimum temperature
      - mean temperature
4. Defines an Area Of Interest (AOI) either by:
      - loading a previously saved AOI mask
      - interactively selecting a new AOI
5. Generates publication-ready temperature maps limited to the AOI
6. Exports:
      - three PNG images (max / min / mean)
      - one CSV file summarizing highest and lowest AOI temperatures
      - AOI mask files (.npz and .json) if not preexisting

The script is intended for thermal monitoring of rockwalls or similar study sites.
"""

from __future__ import annotations
import os
import json
import csv
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ======================================================
# USER CONFIGURATION
# ======================================================

# Folder containing thermal matrices (.txt)
INPUT_FOLDER  = r"F:/Masterarbeit_Backup_2/Data/Data_TXT_formatted_aligned/Karwendel_12-13.11.25"

# Folder where all results will be written
OUTPUT_FOLDER = r"F:/Masterarbeit_Backup_2/Results/Simple_metrics/Karwendel_12-13.11.25"

# Name of the study site (used in filenames)
STUDY_SITE = "Karwendel 2"

# Filename used for saving/loading AOI mask
AOI_BASENAME = "aoi_rockwall"

# If True the script loads an existing AOI instead of asking for a new one
USE_EXISTING_AOI = True

# Publication figure size
FIG_W_MM = 85
FIG_H_MM = 65
DPI_PNG  = 600

# Base font size used in plots
BASE_FONT_PT = 4

# AOI picker interface parameters
HUD_SCALE = 0.9
HUD_THICKNESS = 2
COARSE_STEP_C = 2.0
FINE_STEP_C = 0.2
GRID_OVERLAY = False
GRID_SPACING = 80

# Name of CSV summary file
SUMMARY_CSV_NAME = f"{STUDY_SITE}_simple_metrics.csv"


# ======================================================
# GENERAL HELPER FUNCTIONS
# ======================================================

def ensure_dir(path: str) -> None:
    """
    Ensures that a directory exists.
    Creates it if it does not yet exist.
    """
    os.makedirs(path, exist_ok=True)


def natural_sort_key(path: str):
    """
    Generates a sorting key that sorts numbers correctly.

    Example:
        frame1.txt
        frame2.txt
        frame10.txt

    instead of:
        frame1.txt
        frame10.txt
        frame2.txt
    """
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


def list_txt_files(folder: str) -> List[str]:
    """
    Returns a sorted list of all thermal .txt files in the input folder.
    """
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".txt")
    ]
    return sorted(files, key=natural_sort_key)


# ======================================================
# THERMAL DATA IMPORT
# ======================================================

def read_thermal_txt(path: str) -> np.ndarray:
    """
    Reads a thermal camera .txt export.

    The function searches for the "[Data]" section and reads
    the following tab-separated matrix into a NumPy array.

    Returns
    -------
    2D numpy array
        temperature matrix (float32)
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    try:
        i0 = lines.index("[Data]") + 1
    except ValueError as e:
        raise ValueError(f"Missing [Data] section in file: {path}") from e

    rows: List[List[float]] = []
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


# ======================================================
# AOI DATA STRUCTURE
# ======================================================

@dataclass
class AOI:
    """
    Data structure storing information about the Area Of Interest.

    Attributes
    ----------
    mask : bool array
        Pixel mask representing the AOI

    mode : str
        How the AOI was created ("threshold" or "polygon")

    cutoff : float
        Threshold temperature used (if threshold mode)

    polygon_xy : list
        Polygon vertices (if polygon mode)
    """

    mask: np.ndarray
    mode: str
    cutoff: Optional[float] = None
    keep_largest: Optional[bool] = None
    polygon_xy: Optional[List[Tuple[int, int]]] = None

    def save(self, folder: str, basename: str) -> None:
        """
        Saves the AOI mask and metadata to disk.
        """

        npz_path = os.path.join(folder, f"{basename}.npz")
        meta_path = os.path.join(folder, f"{basename}.json")

        np.savez_compressed(npz_path, mask=self.mask.astype(np.uint8))

        meta = {
            "mode": self.mode,
            "cutoff": self.cutoff,
            "keep_largest": self.keep_largest,
            "polygon_xy": self.polygon_xy
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(folder: str, basename: str) -> "AOI":
        """
        Loads a previously saved AOI mask.
        """

        npz_path = os.path.join(folder, f"{basename}.npz")
        meta_path = os.path.join(folder, f"{basename}.json")

        if not os.path.exists(npz_path):
            raise FileNotFoundError("AOI file not found")

        data = np.load(npz_path)
        mask = data["mask"].astype(bool)

        with open(meta_path) as f:
            meta = json.load(f)

        return AOI(mask=mask, **meta)


# ======================================================
# AOI HELPER FUNCTIONS + INTERACTIVE AOI PICKER
# ======================================================

def largest_connected_component_cv(mask: np.ndarray) -> np.ndarray:
    """
    Returns only the largest connected component of a boolean mask.

    This is useful in threshold mode when several separate regions exceed
    the threshold, but only the main rockwall body should be kept.
    """
    m = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)

    if num <= 1:
        return np.zeros_like(mask, dtype=bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    return labels == k


def normalize_to_u8(img: np.ndarray, mn: float, mx: float) -> np.ndarray:
    """
    Normalizes a float image to 8-bit grayscale (0-255) for display in OpenCV.
    """
    denom = (mx - mn) + 1e-12
    return (np.clip((img - mn) / denom, 0, 1) * 255.0).astype(np.uint8)


def draw_grid_bgr(
    img_bgr: np.ndarray,
    spacing: int = 80,
    color=(0, 255, 0),
    alpha: float = 0.25
) -> np.ndarray:
    """
    Draws an optional semi-transparent grid on the AOI picker preview.
    """
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()

    for x in range(0, w, spacing):
        cv2.line(overlay, (x, 0), (x, h - 1), color, 1)

    for y in range(0, h, spacing):
        cv2.line(overlay, (0, y), (w - 1, y), color, 1)

    return cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)


def overlay_mask_red(gray_u8: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    Overlays the AOI mask in red on top of a grayscale background image.
    """
    bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
    red = bgr.copy()
    red[..., 2] = 255.0

    m = mask.astype(np.float32)[..., None]
    out = (1 - alpha * m) * bgr + (alpha * m) * red
    return np.clip(out, 0, 255).astype(np.uint8)


def polygon_to_mask(shape_hw: Tuple[int, int], poly_xy: List[Tuple[int, int]]) -> np.ndarray:
    """
    Converts a list of polygon vertices into a boolean mask.
    """
    h, w = shape_hw

    if len(poly_xy) < 3:
        return np.zeros((h, w), dtype=bool)

    pts = np.array(poly_xy, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)

    return mask.astype(bool)


def pick_aoi_cv2(min_mat: np.ndarray, initial_aoi: Optional[AOI] = None) -> AOI:
    """
    Interactive AOI picker using OpenCV.

    Supports:
    - threshold AOI creation
    - polygon AOI creation
    - polygon AOI editing (move/add/remove vertices)
    - moving the entire polygon AOI with arrow keys

    Controls
    --------
    General:
        s       save AOI and continue
        q / ESC cancel
        m       switch between threshold mode and polygon mode

    Threshold mode:
        UP/DOWN arrows   coarse threshold change
        LEFT/RIGHT       fine threshold change
        t                type exact threshold value
        l                toggle keeping only the largest connected region

    Polygon mode:
        left click       add vertex OR select nearby vertex for dragging
        drag mouse       move selected vertex
        SHIFT+arrows     move entire polygon by 1 pixel
        SHIFT+UP/DOWN/LEFT/RIGHT
        u                undo last vertex
        c                close polygon
        x                clear polygon
        d                delete selected/last vertex
        a                add mode (always append vertex on click)
    """

    h, w = min_mat.shape

    finite = min_mat[np.isfinite(min_mat)]
    if finite.size == 0:
        raise ValueError("min_mat contains no finite values.")

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0

    qlo, qhi = np.nanquantile(min_mat, [0.02, 0.98])
    bg = np.nan_to_num(np.clip(min_mat, qlo, qhi), nan=qlo)
    bg_u8 = normalize_to_u8(bg, qlo, qhi)

    default_cut = float(np.nanquantile(min_mat, 0.05))
    cutoff = float(np.clip(default_cut, vmin, vmax))

    # Initialize from existing AOI if provided
    mode = "threshold"
    keep_largest = True
    polygon: List[Tuple[int, int]] = []
    poly_closed = False

    if initial_aoi is not None:
        if initial_aoi.mode == "threshold":
            mode = "threshold"
            if initial_aoi.cutoff is not None:
                cutoff = float(np.clip(initial_aoi.cutoff, vmin, vmax))
            if initial_aoi.keep_largest is not None:
                keep_largest = bool(initial_aoi.keep_largest)

        elif initial_aoi.mode == "polygon":
            mode = "polygon"
            polygon = [(int(x), int(y)) for x, y in (initial_aoi.polygon_xy or [])]
            poly_closed = len(polygon) >= 3

    win = "AOI Picker (edit enabled)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    typing = False
    typed = ""

    last_params = None
    cached_mask = np.zeros((h, w), dtype=bool)

    # Polygon editing state
    drag_idx: Optional[int] = None
    add_mode = False
    vertex_pick_radius = 10

    def compute_threshold_mask(cut: float, keep: bool) -> np.ndarray:
        m = (min_mat > cut) & np.isfinite(min_mat)
        if keep:
            m = largest_connected_component_cv(m)
        return m

    def clamp_polygon(poly: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Clamp polygon vertices so they stay inside the image.
        """
        out = []
        for x, y in poly:
            xx = int(np.clip(x, 0, w - 1))
            yy = int(np.clip(y, 0, h - 1))
            out.append((xx, yy))
        return out

    def move_polygon(dx: int, dy: int) -> None:
        """
        Move the whole polygon by (dx, dy) pixels.
        Keeps the polygon inside image bounds.
        """
        nonlocal polygon, last_params

        if not polygon:
            return

        # Compute allowed shift so no vertex leaves the image
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        dx = int(dx)
        dy = int(dy)

        if dx < 0:
            dx = max(dx, -min(xs))
        elif dx > 0:
            dx = min(dx, (w - 1) - max(xs))

        if dy < 0:
            dy = max(dy, -min(ys))
        elif dy > 0:
            dy = min(dy, (h - 1) - max(ys))

        if dx == 0 and dy == 0:
            return

        polygon = [(x + dx, y + dy) for x, y in polygon]
        polygon = clamp_polygon(polygon)
        last_params = None

    def compute_mask() -> np.ndarray:
        nonlocal cached_mask
        if mode == "threshold":
            cached_mask = compute_threshold_mask(cutoff, keep_largest)
        else:
            cached_mask = (
                polygon_to_mask((h, w), polygon)
                if poly_closed and len(polygon) >= 3
                else np.zeros((h, w), dtype=bool)
            )
        return cached_mask

    def nearest_vertex_index(x: int, y: int) -> Optional[int]:
        if not polygon:
            return None

        d2 = []
        for i, (px, py) in enumerate(polygon):
            d2.append((i, (px - x) ** 2 + (py - y) ** 2))

        idx, dist2 = min(d2, key=lambda t: t[1])
        if dist2 <= vertex_pick_radius ** 2:
            return idx
        return None

    def on_mouse(event, x, y, flags, param):
        nonlocal polygon, poly_closed, last_params, drag_idx

        if mode != "polygon" or typing:
            return

        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            idx = nearest_vertex_index(x, y)

            # If clicking near an existing vertex, start dragging it
            if idx is not None and not add_mode:
                drag_idx = idx
                return

            # Otherwise add a new vertex
            polygon.append((x, y))
            last_params = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if drag_idx is not None:
                polygon[drag_idx] = (x, y)
                polygon[:] = clamp_polygon(polygon)
                last_params = None

        elif event == cv2.EVENT_LBUTTONUP:
            if drag_idx is not None:
                polygon[drag_idx] = (x, y)
                polygon[:] = clamp_polygon(polygon)
                drag_idx = None
                last_params = None

    cv2.setMouseCallback(win, on_mouse)

    while True:
        params = (
            mode,
            round(cutoff, 6),
            keep_largest,
            tuple(polygon),
            poly_closed,
            add_mode,
            drag_idx,
        )

        if params != last_params:
            compute_mask()
            last_params = params

        view = overlay_mask_red(bg_u8, cached_mask, alpha=0.35)

        if mode == "polygon" and len(polygon) >= 1:
            for i in range(1, len(polygon)):
                cv2.line(view, polygon[i - 1], polygon[i], (0, 0, 0), 2)

            if poly_closed and len(polygon) >= 3:
                cv2.line(view, polygon[-1], polygon[0], (0, 0, 0), 2)

            for i, p in enumerate(polygon):
                color = (255, 255, 255) if i == drag_idx else (0, 0, 0)
                cv2.circle(view, p, 4, color, -1)

        if GRID_OVERLAY:
            view = draw_grid_bgr(view, spacing=GRID_SPACING)

        n_true = int(cached_mask.sum())
        n_all = int(cached_mask.size)
        pct = (100.0 * n_true / n_all) if n_all else 0.0

        if mode == "threshold":
            hud1 = f"mode=threshold | cutoff={cutoff:.2f} C | keep_largest={keep_largest} | AOI={n_true}/{n_all} ({pct:.2f}%)"
            hud2 = "Keys: arrows adjust | t type | l largest | m polygon | s save | q/ESC cancel"
        else:
            hud1 = f"mode=polygon | pts={len(polygon)} | closed={poly_closed} | add_mode={add_mode} | AOI={n_true}/{n_all} ({pct:.2f}%)"
            hud2 = "Keys: drag vertex | SHIFT+arrows move AOI | d delete | a add | u undo | c close | x clear | m threshold | s save"

        hud3 = f"TYPE cutoff (C): {typed}_   (ENTER apply, BACKSPACE edit, ESC cancel)" if typing else ""

        cv2.putText(view, hud1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, HUD_SCALE, (255, 255, 255), HUD_THICKNESS)
        cv2.putText(view, hud2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, HUD_SCALE * 0.9, (255, 255, 255), HUD_THICKNESS)

        if hud3:
            cv2.putText(view, hud3, (10, 88), cv2.FONT_HERSHEY_SIMPLEX, HUD_SCALE * 0.9, (255, 255, 255), HUD_THICKNESS)

        cv2.imshow(win, view)
        key = cv2.waitKeyEx(10)

        if key == -1:
            continue

        if key in (ord("q"), 27):
            cv2.destroyWindow(win)
            raise RuntimeError("AOI selection cancelled.")

        if key == ord("s"):
            if cached_mask.any():
                cv2.destroyWindow(win)
                return AOI(
                    mask=cached_mask.astype(bool),
                    mode=mode,
                    cutoff=float(cutoff) if mode == "threshold" else None,
                    keep_largest=bool(keep_largest) if mode == "threshold" else None,
                    polygon_xy=polygon.copy() if mode == "polygon" else None,
                )
            continue

        if key == ord("m") and not typing:
            mode = "polygon" if mode == "threshold" else "threshold"
            drag_idx = None
            last_params = None
            continue

        if key == ord("t") and (mode == "threshold") and not typing:
            typing = True
            typed = ""
            continue

        if typing:
            if key == 27:
                typing = False
                typed = ""
                continue

            if key in (13, 10):
                try:
                    val = float(typed)
                    cutoff = float(np.clip(val, vmin, vmax))
                    last_params = None
                except ValueError:
                    pass

                typing = False
                typed = ""
                continue

            if key in (8, 255):
                typed = typed[:-1]
                continue

            ch = None
            if 32 <= key <= 126:
                ch = chr(key)
            elif 0 <= key <= 255:
                ch = chr(key)

            if ch and (ch.isdigit() or ch in ".-"):
                typed += ch

            continue

        if mode == "threshold":
            if key == ord("l"):
                keep_largest = not keep_largest
                last_params = None
                continue

            if key == 2490368:  # UP
                cutoff = float(np.clip(cutoff + COARSE_STEP_C, vmin, vmax))
                last_params = None
                continue

            if key == 2621440:  # DOWN
                cutoff = float(np.clip(cutoff - COARSE_STEP_C, vmin, vmax))
                last_params = None
                continue

            if key == 2555904:  # RIGHT
                cutoff = float(np.clip(cutoff + FINE_STEP_C, vmin, vmax))
                last_params = None
                continue

            if key == 2424832:  # LEFT
                cutoff = float(np.clip(cutoff - FINE_STEP_C, vmin, vmax))
                last_params = None
                continue

        if mode == "polygon":
            # SHIFT + arrow keys in OpenCV/Windows
            if key == 2490368 + 1048576:  # SHIFT+UP
                move_polygon(0, -1)
                continue

            if key == 2621440 + 1048576:  # SHIFT+DOWN
                move_polygon(0, 1)
                continue

            if key == 2424832 + 1048576:  # SHIFT+LEFT
                move_polygon(-1, 0)
                continue

            if key == 2555904 + 1048576:  # SHIFT+RIGHT
                move_polygon(1, 0)
                continue

            # fallback: plain arrows also move polygon in polygon mode
            if key == 2490368:  # UP
                move_polygon(0, -1)
                continue

            if key == 2621440:  # DOWN
                move_polygon(0, 1)
                continue

            if key == 2424832:  # LEFT
                move_polygon(-1, 0)
                continue

            if key == 2555904:  # RIGHT
                move_polygon(1, 0)
                continue

            if key == ord("a"):
                add_mode = not add_mode
                continue

            if key == ord("u"):
                if polygon:
                    polygon.pop()
                    poly_closed = len(polygon) >= 3 and poly_closed
                    drag_idx = None
                    last_params = None
                continue

            if key == ord("x"):
                polygon = []
                poly_closed = False
                drag_idx = None
                last_params = None
                continue

            if key == ord("c"):
                poly_closed = len(polygon) >= 3
                drag_idx = None
                last_params = None
                continue

            if key == ord("d"):
                if drag_idx is not None and 0 <= drag_idx < len(polygon):
                    polygon.pop(drag_idx)
                    drag_idx = None
                elif polygon:
                    polygon.pop()

                if len(polygon) < 3:
                    poly_closed = False

                last_params = None
                continue


# ======================================================
# VISUALIZATION FUNCTIONS
# ======================================================

def mm_to_in(mm: float) -> float:
    """Converts millimeters to inches (needed by matplotlib)."""
    return mm / 25.4


def setup_pub_rcparams():
    """
    Applies matplotlib style parameters
    optimized for publication figures.
    """

    matplotlib.rcParams.update({
        "font.size": BASE_FONT_PT,
        "axes.titlesize": BASE_FONT_PT * 1.50,
        "axes.labelsize": BASE_FONT_PT * 1.05,
        "xtick.labelsize": BASE_FONT_PT * 0.95,
        "ytick.labelsize": BASE_FONT_PT * 0.95,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })



def save_pub_map(
    mat: np.ndarray,
    aoi_mask: np.ndarray,
    title: str,
    out_base: str,
    units: str,
    out_dir: str,
):
    """
    Creates a publication-ready temperature map.

    Only AOI pixels are shown; outside area is masked.

    Parameters
    ----------
    mat : temperature matrix
    aoi_mask : AOI boolean mask
    title : plot title
    out_base : output filename
    units : units for colorbar
    out_dir : output folder
    """

    setup_pub_rcparams()

    aoi_vals = mat[aoi_mask]

    vmin = float(np.nanmin(aoi_vals))
    vmax = float(np.nanmax(aoi_vals))

    fig = plt.figure(figsize=(mm_to_in(FIG_W_MM), mm_to_in(FIG_H_MM)), dpi=DPI_PNG)

    ax = fig.add_axes([0.10, 0.12, 0.72, 0.80])
    cax = fig.add_axes([0.85, 0.18, 0.04, 0.68])

    overlay = np.full_like(mat, np.nan)
    overlay[aoi_mask] = mat[aoi_mask]

    im = ax.imshow(
        overlay,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="nearest",
    )

    ax.set_xlim(-0.5, mat.shape[1] - 0.5)
    ax.set_ylim(mat.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")

    title_size = BASE_FONT_PT * 1.50
    label_size = BASE_FONT_PT * 1.05
    tick_size = BASE_FONT_PT * 0.95

    ax.set_xlabel("X [pixels]", fontsize=label_size, fontname="Arial")
    ax.set_ylabel("Y [pixels]", fontsize=label_size, fontname="Arial")
    ax.set_title(title, fontsize=title_size, fontname="Arial")

    ax.tick_params(axis="both", labelsize=tick_size)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Arial")

    cb = fig.colorbar(im, cax=cax)
    cb.formatter = FuncFormatter(lambda x, pos: f"{x:g} {units}")
    cb.update_ticks()
    cb.ax.tick_params(labelsize=tick_size)
    for tick in cb.ax.get_yticklabels():
        tick.set_fontname("Arial")

    ensure_dir(out_dir)
    fig.savefig(os.path.join(out_dir, f"{out_base}.png"), dpi=DPI_PNG)

    plt.close(fig)


# ======================================================
# METRIC SUMMARY FUNCTIONS
# ======================================================

def summarize_parameter(mat: np.ndarray, aoi_mask: np.ndarray, name: str) -> dict:
    """
    Computes minimum and maximum temperature within the AOI.
    """

    vals = mat[aoi_mask]

    return {
        "Parameter": name,
        "Highest value [C°]": float(np.nanmax(vals)),
        "Lowest value [C°]": float(np.nanmin(vals)),
    }


def write_summary_csv(rows: List[dict], out_path: str) -> None:
    """
    Writes a CSV table summarizing temperature statistics.
    """

    with open(out_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["Parameter", "Highest value [C°]", "Lowest value [C°]"]
        )

        writer.writeheader()
        writer.writerows(rows)


def ask_yes_no(prompt: str) -> bool:
    """
    Repeatedly asks the user a yes/no question in the console.
    Returns True for yes, False for no.
    """
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")


# ======================================================
# MAIN PROCESSING WORKFLOW
# ======================================================

def main():

    ensure_dir(OUTPUT_FOLDER)

    # --------------------------------------------------
    # Load thermal matrices
    # --------------------------------------------------

    txt_files = list_txt_files(INPUT_FOLDER)

    if not txt_files:
        raise FileNotFoundError("No thermal .txt files found")

    print(f"Processing {len(txt_files)} thermal files")

    mats = [read_thermal_txt(p) for p in txt_files]
    stack = np.stack(mats)

    # --------------------------------------------------
    # Calculate temperature statistics
    # --------------------------------------------------

    mean_mat = np.nanmean(stack, axis=0)
    min_mat = np.nanmin(stack, axis=0)
    max_mat = np.nanmax(stack, axis=0)

    # --------------------------------------------------
    # Load, edit, or create AOI
    # --------------------------------------------------

    # Ask every time whether the user wants to alter the AOI
    while True:
        alter_answer = input("Do you want to alter AOI vertices? (y/n): ").strip().lower()
        if alter_answer in {"y", "n"}:
            break
        print("Please enter 'y' or 'n'.")

    alter_aoi = (alter_answer == "y")

    existing_aoi = None

    if USE_EXISTING_AOI:
        try:
            # Try to load an AOI that was saved in a previous run
            existing_aoi = AOI.load(OUTPUT_FOLDER, AOI_BASENAME)

            # Safety check: make sure the loaded mask is not empty
            if (existing_aoi.mask is None) or (not existing_aoi.mask.any()):
                raise ValueError("Saved AOI mask is empty.")

        except Exception:
            existing_aoi = None

    if alter_aoi:
        print("Opening AOI editor...")

        # This will open the AOI window every time the user answers "y"
        # If an AOI exists, it is passed in for editing.
        # If none exists, a new AOI is created.
        aoi = pick_aoi_cv2(min_mat, initial_aoi=existing_aoi)
        aoi.save(OUTPUT_FOLDER, AOI_BASENAME)
        aoi_mask = aoi.mask.astype(bool)

    else:
        if existing_aoi is not None:
            print("Using saved AOI without editing.")
            aoi_mask = existing_aoi.mask.astype(bool)
        else:
            print("No valid AOI found. Opening AOI picker...")
            aoi = pick_aoi_cv2(min_mat, initial_aoi=None)
            aoi.save(OUTPUT_FOLDER, AOI_BASENAME)
            aoi_mask = aoi.mask.astype(bool)

    # --------------------------------------------------
    # Export temperature maps
    # --------------------------------------------------

    save_pub_map(
        mean_mat,
        aoi_mask,
        f"{STUDY_SITE} – Mean Temperature",
        f"{STUDY_SITE}_mean_temperature",
        "°C",
        OUTPUT_FOLDER
    )

    save_pub_map(
        min_mat,
        aoi_mask,
        f"{STUDY_SITE} – Minimum Temperature",
        f"{STUDY_SITE}_min_temperature",
        "°C",
        OUTPUT_FOLDER
    )

    save_pub_map(
        max_mat,
        aoi_mask,
        f"{STUDY_SITE} – Maximum Temperature",
        f"{STUDY_SITE}_max_temperature",
        "°C",
        OUTPUT_FOLDER
    )

    # --------------------------------------------------
    # Export CSV summary
    # --------------------------------------------------

    rows = [
        summarize_parameter(mean_mat, aoi_mask, f"{STUDY_SITE}_mean_temperature"),
        summarize_parameter(min_mat, aoi_mask, f"{STUDY_SITE}_min_temperature"),
        summarize_parameter(max_mat, aoi_mask, f"{STUDY_SITE}_max_temperature"),
    ]

    write_summary_csv(
        rows,
        os.path.join(OUTPUT_FOLDER, SUMMARY_CSV_NAME)
    )

    print("Processing complete.")


# ======================================================
# SCRIPT ENTRY POINT
# ======================================================

if __name__ == "__main__":
    main()