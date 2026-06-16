"""
All-in-one advanced temperature metrics for aligned thermal IR TXT frames.
(Partly unnecessary - Frost Paradiestal and Lehesten)

Computed metrics
----------------
1) Diurnal amplitude
2) Longest consecutive deep-frost duration (hours)
3) Freeze-thaw cycle completions per interval (timeseries PNG + CSV)
4) Max absolute change rate (°C/min)
"""

from __future__ import annotations

# Standard library imports
import csv
import glob
import os
from typing import List, Tuple

# Third-party imports
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


# -----------------------------------------------------------------------------
# GLOBAL MATPLOTLIB FONT SETUP
# -----------------------------------------------------------------------------
# These settings are applied once so that all plots use the same typography.
# Arial is requested as the main font. Math text is also mapped to Arial so that
# units like min^{-1} render correctly.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Arial"
mpl.rcParams["mathtext.it"] = "Arial:italic"
mpl.rcParams["mathtext.bf"] = "Arial:bold"


# =============================================================================
# 1. USER PARAMETERS
# =============================================================================
# Input folder containing aligned thermal TXT frames.
INPUT_FOLDER = r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted_aligned\Karwendel_12-13.11.25"

# Output folder where final PNG maps and one summary CSV will be saved.
OUTPUT_FOLDER = r"F:\Masterarbeit_Backup_2\Results\Advanced_metrics\Karwendel_12-13.11.25"

# Study-site name used in figure titles and the summary table filename.
# Example title: "Lehesten - Diurnal Amplitude"
STUDY_SITE = "Karwendel 2"

# Time spacing between two consecutive thermal frames, in minutes.
# This is needed for any time-based metric such as duration or change rate.
INTERVAL_MIN = 10.0

# Optional AOI (area of interest) mask exported by your Simple_Statistics script.
# If enabled, metrics are only evaluated inside the AOI.
APPLY_AOI_MASK = True
AOI_FOLDER = r"F:\Masterarbeit_Backup_2\Results\Simple_metrics\Karwendel_12-13.11.25"
AOI_BASENAME = "aoi_rockwall"  # expects AOI_BASENAME + ".npz" with key "mask"

# Temperature thresholds used by the metrics.
DEEP_FROST_THRESH_C = -3.0   # Deep frost means T < -3 °C
FREEZE_THAW_THRESH_C = 0.0   # Frozen: T < 0 °C ; Thawed: T >= 0 °C

# Figure export settings.
FIG_W_MM = 85
FIG_H_MM = 65
DPI_PNG = 600
BASE_FONT_PT = 4

# Name of the compact summary CSV.
SUMMARY_CSV_NAME = f"{STUDY_SITE}_summary_advanced_metrics.csv"


# =============================================================================
# 2. HELPERS (I/O)
# =============================================================================
def ensure_dir(path: str) -> None:
    """Create a folder if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def list_txt_files(folder: str) -> List[str]:
    """
    Return all TXT files in the input folder, sorted by filename.

    The order matters because the files are treated as a temporal sequence.
    """
    paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in: {folder}")
    return paths


def read_thermal_txt(path: str) -> np.ndarray:
    """
    Read one thermal IR TXT file into a 2D NumPy array.

    The script assumes the numeric raster starts after a line named [Data].
    Decimal commas are converted to decimal points.
    Non-numeric trailing text is ignored.
    Missing values are filled with NaN if rows have unequal lengths.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    try:
        i0 = lines.index("[Data]") + 1
    except ValueError as exc:
        raise ValueError(f"Missing [Data] section in file: {path}") from exc

    rows: List[List[float]] = []
    for ln in lines[i0:]:
        if not ln.strip():
            continue
        parts = ln.replace(",", ".").split("\t")
        try:
            rows.append([float(p) for p in parts if p != ""])
        except ValueError:
            # Stop reading once non-numeric content begins.
            break

    if not rows:
        raise ValueError(f"No numeric data found after [Data] in file: {path}")

    # If rows do not all have the same length, pad with NaN.
    ncol = max(len(r) for r in rows)
    arr = np.full((len(rows), ncol), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = np.array(r, dtype=np.float32)
    return arr


def load_aoi_mask(shape: Tuple[int, int]) -> np.ndarray:
    """
    Load the AOI mask from a .npz file and verify that it matches raster shape.

    If AOI masking is disabled, a full-True mask is returned so the entire image
    is processed.
    """
    if not APPLY_AOI_MASK:
        return np.ones(shape, dtype=bool)

    npz_path = os.path.join(AOI_FOLDER, f"{AOI_BASENAME}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"AOI mask .npz not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    if "mask" not in data:
        raise KeyError(f"AOI npz does not contain key 'mask': {npz_path}")

    mask = data["mask"].astype(bool)
    if mask.shape != shape:
        raise ValueError(
            f"AOI mask shape {mask.shape} does not match raster shape {shape}"
        )
    if not mask.any():
        raise ValueError("AOI mask contains 0 pixels.")
    return mask


def apply_aoi_nan(mat: np.ndarray, aoi: np.ndarray) -> np.ndarray:
    """
    Set all pixels outside the AOI to NaN.

    This keeps the matrix shape unchanged but ensures that only AOI pixels are
    used in plotting and summary statistics.
    """
    out = mat.astype(np.float32, copy=True)
    out[~aoi] = np.nan
    return out


# =============================================================================
# 3. IMAGE EXPORT (PNG ONLY)
# =============================================================================
def mm_to_in(mm: float) -> float:
    """Convert millimetres to inches for Matplotlib figure sizing."""
    return mm / 25.4


def setup_pub_rcparams() -> None:
    """
    Apply consistent publication-style plotting settings.

    This keeps font sizes, line widths, backgrounds, and general figure styling
    identical across all exported metric maps.
    """
    matplotlib.rcParams.update({
        "font.size": BASE_FONT_PT,
        "axes.titlesize": BASE_FONT_PT * 1.50,
        "axes.labelsize": BASE_FONT_PT * 1.05,
        "xtick.labelsize": BASE_FONT_PT * 0.95,
        "ytick.labelsize": BASE_FONT_PT * 0.95,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    })


def save_pub_map(
    mat: np.ndarray,
    aoi_mask: np.ndarray,
    title: str,
    out_base: str,
    units: str,
    out_dir: str,
) -> str:
    """
    Save one metric as a PNG map.

    Only AOI pixels are shown, matching the simple metrics script style.
    Outside the AOI, the background remains white.
    Inside the AOI, the metric is rendered with the inferno colormap.
    A colorbar is added on the right with the provided unit label.
    """
    setup_pub_rcparams()

    # Use only finite AOI values to determine the plotting range, matching
    # the simple metrics script scale behavior.
    vals = mat[aoi_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError(f"No finite AOI values to plot for {out_base}.")

    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if vmax <= vmin:
        vmax = vmin + 1e-6  # avoids zero color range if map is constant

    fig = plt.figure(figsize=(mm_to_in(FIG_W_MM), mm_to_in(FIG_H_MM)), dpi=DPI_PNG)

    # Main image axis and separate colorbar axis.
    ax = fig.add_axes([0.10, 0.12, 0.72, 0.80])
    cax = fig.add_axes([0.85, 0.18, 0.04, 0.68])

    # Overlay metric values only inside the AOI, leaving non-AOI areas white
    # like in the simple metrics script.
    overlay = np.full_like(mat, np.nan, dtype=float)
    overlay[aoi_mask] = mat[aoi_mask]
    im = ax.imshow(
        overlay,
        origin="upper",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )

    # Match image dimensions and invert y-axis so raster indexing is preserved.
    ax.set_xlim(-0.5, mat.shape[1] - 0.5)
    ax.set_ylim(mat.shape[0] - 0.5, -0.5)

    ax.set_aspect("equal")

    TITLE_SIZE = BASE_FONT_PT * 1.50
    LABEL_SIZE = BASE_FONT_PT * 1.05
    TICK_SIZE = BASE_FONT_PT * 0.95

    ax.set_xlabel("X [pixels]", fontsize=LABEL_SIZE, fontname="Arial")
    ax.set_ylabel("Y [pixels]", fontsize=LABEL_SIZE, fontname="Arial")
    ax.set_title(title, fontsize=TITLE_SIZE, fontname="Arial")

    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Arial")

    # Add colorbar and keep numeric ticks simple.
    cb = fig.colorbar(im, cax=cax)
    cb.formatter = FuncFormatter(lambda x, pos: f"{x:g}")
    cb.update_ticks()
    cb.ax.tick_params(labelsize=TICK_SIZE)
    for tick in cb.ax.get_yticklabels():
        tick.set_fontname("Arial")
    if units:
        cb.set_label(units, fontsize=LABEL_SIZE, fontname="Arial", rotation=90, labelpad=4)

    # Thin frame lines for a cleaner publication look.
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

    ensure_dir(out_dir)
    out_png = os.path.join(out_dir, f"{out_base}.png")
    fig.savefig(out_png, dpi=DPI_PNG, bbox_inches="tight")
    plt.close(fig)
    return out_png


# =============================================================================
# 4. SUMMARY TABLE
# =============================================================================
def matrix_min_max(mat: np.ndarray, aoi_mask: np.ndarray) -> Tuple[float, float]:
    """
    Return maximum and minimum values inside the AOI for one metric matrix.

    The output order is:
        (highest_value, lowest_value)
    """
    vals = mat[aoi_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmax(vals)), float(np.nanmin(vals))


def write_summary_table(rows: List[Tuple[str, float, float]], out_csv: str) -> str:
    """
    Write one compact summary CSV listing the highest and lowest value
    for each metric.
    """
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Parameter", "Highest value", "Lowest value"])
        w.writerows(rows)
    return out_csv


# =============================================================================
# 5. METRIC CALCULATIONS
# =============================================================================
def max_consecutive_true(bool_stack: np.ndarray) -> np.ndarray:
    """
    Compute the longest consecutive run of True values along the time axis.

    Input shape:
        (time, height, width)

    Output:
        2D array with the maximum run length per pixel.
    """
    _, h, w = bool_stack.shape
    run = np.zeros((h, w), dtype=np.int32)
    maxrun = np.zeros((h, w), dtype=np.int32)

    for t in range(bool_stack.shape[0]):
        run = np.where(bool_stack[t], run + 1, 0)
        maxrun = np.maximum(maxrun, run)

    return maxrun


def count_freeze_thaw_cycles(
    stack: np.ndarray,
    finite: np.ndarray,
    aoi_mask: np.ndarray,
    thresh_c: float,
) -> np.ndarray:
    """
    Count full freeze-thaw cycles per pixel.

    Definitions
    -----------
    Frozen : T < thresh_c
    Thawed : T >= thresh_c

    One full cycle means two threshold crossings in total, for example:
    thawed -> frozen -> thawed
    or
    frozen -> thawed -> frozen

    Therefore, this is not counting only freezing events or only thawing events.
    It counts completed cycles.
    """
    frozen = (stack < thresh_c) & finite

    # A flip happens wherever the frozen/not-frozen state changes between frames.
    flips = frozen[1:] ^ frozen[:-1]
    flips &= (finite[1:] & finite[:-1])

    # Two flips correspond to one full cycle.
    transitions_per_pixel = np.sum(flips, axis=0).astype(np.int32)
    cycles_map = (transitions_per_pixel // 2).astype(np.float32)

    # Remove values outside the AOI.
    cycles_map[~aoi_mask] = np.nan
    return cycles_map


# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
def main() -> None:
    """Run the complete workflow from input frames to final outputs."""
    ensure_dir(OUTPUT_FOLDER)

    # -------------------------------------------------------------------------
    # Load all thermal frames
    # -------------------------------------------------------------------------
    txt_paths = list_txt_files(INPUT_FOLDER)
    mats = [read_thermal_txt(p) for p in txt_paths]

    # Verify that all frames have identical raster dimensions.
    shapes = {m.shape for m in mats}
    if len(shapes) != 1:
        raise ValueError(
            f"Not all TXT rasters have same shape. Shapes: {sorted(shapes)}"
        )

    shape = mats[0].shape
    aoi_mask = load_aoi_mask(shape)

    # Build one 3D stack with dimensions (time, y, x).
    stack = np.stack([m.astype(np.float32) for m in mats], axis=0)

    # Finite pixels inside the AOI are valid for analysis.
    finite = np.isfinite(stack) & aoi_mask[None, :, :]

    if INTERVAL_MIN <= 0:
        raise ValueError("INTERVAL_MIN must be > 0")

    # -------------------------------------------------------------------------
    # 1) Diurnal amplitude
    # -------------------------------------------------------------------------
    # Difference between maximum and minimum temperature per pixel over time.
    amplitude = (np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)).astype(np.float32)
    amplitude = apply_aoi_nan(amplitude, aoi_mask)

    # -------------------------------------------------------------------------
    # 2) Longest consecutive deep frost duration (hours)
    # -------------------------------------------------------------------------
    # Mark each pixel and frame where temperature is below the deep-frost threshold.
    deep_frost = (stack < DEEP_FROST_THRESH_C) & finite

    # Count the longest uninterrupted sequence of deep-frost frames.
    maxrun_frames = max_consecutive_true(deep_frost).astype(np.float32)

    # Convert frame count to hours.
    longest_consecutive_deep_frost_h = (
        maxrun_frames * INTERVAL_MIN / 60.0
    ).astype(np.float32)
    longest_consecutive_deep_frost_h = apply_aoi_nan(
        longest_consecutive_deep_frost_h, aoi_mask
    )

    # -------------------------------------------------------------------------
    # 3) Freeze-thaw cycles
    # -------------------------------------------------------------------------
    # Count full cycles, not single freeze or thaw events.
    freeze_thaw_cycles = count_freeze_thaw_cycles(
        stack, finite, aoi_mask, FREEZE_THAW_THRESH_C
    )

    # -------------------------------------------------------------------------
    # 4) Maximum absolute change rate
    # -------------------------------------------------------------------------
    # Compute temperature change between each pair of consecutive frames and
    # divide by the frame interval to get °C per minute.
    pairwise_change_rate = (stack[1:] - stack[:-1]) / INTERVAL_MIN

    # Keep the largest absolute change rate observed for each pixel.
    max_abs_change_rate = np.nanmax(np.abs(pairwise_change_rate), axis=0).astype(np.float32)
    max_abs_change_rate = apply_aoi_nan(max_abs_change_rate, aoi_mask)

    # -------------------------------------------------------------------------
    # Export PNG maps
    # -------------------------------------------------------------------------
    save_pub_map(
        amplitude,
        aoi_mask,
        f"{STUDY_SITE} - Diurnal Amplitude",
        f"{STUDY_SITE}_diurnal_amplitude",
        "K",
        OUTPUT_FOLDER,
    )

    save_pub_map(
        longest_consecutive_deep_frost_h,
        aoi_mask,
        f"{STUDY_SITE} - Longest Consecutive Deep Frost Duration",
        f"{STUDY_SITE}_deep_frost",
        "h",
        OUTPUT_FOLDER,
    )

    save_pub_map(
        freeze_thaw_cycles,
        aoi_mask,
        f"{STUDY_SITE} - Freeze-thaw Cycles",
        f"{STUDY_SITE}_freeze_thaw_cycles",
        "cycles",
        OUTPUT_FOLDER,
    )

    save_pub_map(
        max_abs_change_rate,
        aoi_mask,
        f"{STUDY_SITE} - Maximum Change Rate",
        f"{STUDY_SITE}_maximum_change_rate",
        r"K min$^{-1}$",
        OUTPUT_FOLDER,
    )

    # -------------------------------------------------------------------------
    # Export one compact summary CSV
    # -------------------------------------------------------------------------
    summary_rows = [
        ("Diurnal amplitude [°C]", *matrix_min_max(amplitude, aoi_mask)),
        (
            "Longest consecutive deep frost duration [h]",
            *matrix_min_max(longest_consecutive_deep_frost_h, aoi_mask),
        ),
        ("Freeze-thaw cycles [cycles]", *matrix_min_max(freeze_thaw_cycles, aoi_mask)),
        (
            "Maximum change rate [°C min^-1]",
            *matrix_min_max(max_abs_change_rate, aoi_mask),
        ),
    ]
    write_summary_table(summary_rows, os.path.join(OUTPUT_FOLDER, SUMMARY_CSV_NAME))

    # -------------------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------------------
    print("Done.")
    print(f"Input frames: {len(txt_paths)}")
    print(f"Saved PNG maps to: {OUTPUT_FOLDER}")
    print(f"Saved summary table: {os.path.join(OUTPUT_FOLDER, SUMMARY_CSV_NAME)}")
    print(
        "Freeze-thaw cycles are counted as full cycles "
        "(two threshold crossings), not single freeze or thaw events."
    )


if __name__ == "__main__":
    main()