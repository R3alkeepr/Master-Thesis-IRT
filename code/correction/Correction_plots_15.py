"""
Plot divergence between uncorrected IRT frames and corrected outputs.

Supported corrected-output conventions
--------------------------------------
1) CORRECTED_MODE = "final_temperature"
   Corrected TXT already contains final corrected temperatures.
   Divergence is computed as:
       diff = T_corrected - T_uncorrected

2) CORRECTED_MODE = "correction_map"
   Corrected TXT contains only the correction magnitude:
       T_corr = |T_rad - T_sky| * k(theta)
   The final corrected temperature is reconstructed as:
       T_corrected = T_uncorrected - T_corr
   Divergence is then:
       diff = T_corrected - T_uncorrected = -T_corr

Outputs
-------
- mean divergence map
- highest absolute divergence map
- incidence-angle map

Units are written next to each colorbar tick value.
"""

import os
import re
import glob
import json
from io import StringIO

import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None


mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"]
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Arial"
mpl.rcParams["mathtext.it"] = "Arial:italic"
mpl.rcParams["mathtext.bf"] = "Arial:bold"


# ============================================================
# USER CONFIG
# ============================================================
UNCORRECTED_FOLDER = r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted_aligned\Paradiestal_12-13.08.25"
CORRECTED_FOLDER   = r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted_aligned_angled_final\Paradiestal_12-13.08.25"
THETA_CSV          = r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted_aligned_angled_final\Paradiestal_12-13.08.25\Paradiestal_pixelwise_theta.csv"
OUT_DIR            = r"F:\Masterarbeit_Backup_2\Results\Correction\Paradiestal_12-13.08.25"
AOI_JSON           = r"F:\Masterarbeit_Backup_2\Results\Correction\Paradiestal_12-13.08.25\aoi_rockwall.json"

SUFFIX             = "_anglecorr"
ANGLE_MIN          = 0.0
ANGLE_MAX          = 70.0
CMAP               = "inferno"
SAVE_NPY           = False

# "final_temperature" or "correction_map"
CORRECTED_MODE     = "correction_map"

# smoothing strength
SMOOTH_SIGMA_PX_MEAN = 1.0
SMOOTH_SIGMA_PX_HIGH = 1.0
SMOOTH_SIGMA_PX_ANGLE = 1.0

# if True, use the true max for non-symmetric maps; if False, use 98th percentile
USE_TRUE_MAX_FOR_NONSYMMETRIC = True

# publication figure export settings
FIG_W_MM           = 85
FIG_H_MM           = 65
DPI_PNG            = 600
BASE_FONT_PT       = 4
# ============================================================


def natural_sort_key(path):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from path."""
    parts = re.split(r"(\d+)", os.path.basename(path))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_txt(folder):
    """Return the available input files in deterministic processing order. Inputs are taken from folder."""
    files = sorted(glob.glob(os.path.join(folder, "*.txt")), key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {folder}")
    return files


def read_txt(path):
    """Read data from disk and return it in a parsed NumPy/Python structure. Inputs are taken from path."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    data_idx = next(i for i, line in enumerate(lines) if line.strip() == "[Data]")
    clean = "".join(line.replace(",", ".") for line in lines[data_idx + 1:])
    return np.loadtxt(StringIO(clean), delimiter="\t", dtype=np.float32)


def infer_site(folder):
    """Infer a project-specific label from the provided path or filename. Inputs are taken from folder."""
    base = os.path.basename(os.path.normpath(folder))
    return base.split("_")[0] if "_" in base else base


def build_lookup(corrected_files):
    """Build the helper structure required by the downstream workflow. Inputs are taken from corrected_files."""
    lookup = {}
    for path in corrected_files:
        name = os.path.splitext(os.path.basename(path))[0]
        if SUFFIX and name.endswith(SUFFIX):
            key = name[:-len(SUFFIX)]
        else:
            key = name
        lookup[key] = path
    return lookup


def load_aoi_mask(path, H, W):
    """Load and validate external data needed later in the workflow. Inputs are taken from path, H, W."""
    import cv2

    if not os.path.exists(path):
        raise FileNotFoundError(f"AOI JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "polygon_xy" not in data:
        raise ValueError("AOI JSON must contain 'polygon_xy'")

    polygon = np.asarray(data["polygon_xy"], dtype=np.int32)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("'polygon_xy' must be an Nx2 list of [x, y] coordinates")

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def load_theta_csv(path):
    """Load and validate external data needed later in the workflow. Inputs are taken from path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Theta CSV not found: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    names = set(data.dtype.names or [])

    if {"u", "v", "theta_deg"}.issubset(names):
        u = np.asarray(data["u"], dtype=np.float64)
        v = np.asarray(data["v"], dtype=np.float64)
        theta = np.asarray(data["theta_deg"], dtype=np.float64)
        return u, v, theta

    if {"u", "v", "theta_deg_mean"}.issubset(names):
        u = np.asarray(data["u"], dtype=np.float64)
        v = np.asarray(data["v"], dtype=np.float64)
        theta = np.asarray(data["theta_deg_mean"], dtype=np.float64)
        return u, v, theta

    raise ValueError(
        f"Theta CSV must contain either ['u','v','theta_deg'] or ['u','v','theta_deg_mean']; found {sorted(names)}"
    )


def rasterize_theta(u, v, theta, H, W):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from u, v, theta and related parameters."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    ui = np.rint(u).astype(int)
    vi = np.rint(v).astype(int)

    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & np.isfinite(theta)
    ui = ui[inside]
    vi = vi[inside]
    theta = theta[inside]

    buckets = {}
    for x, y, t in zip(ui, vi, theta):
        buckets.setdefault((y, x), []).append(float(t))

    img = np.full((H, W), np.nan, dtype=np.float64)
    for (y, x), vals in buckets.items():
        img[y, x] = float(np.mean(vals))
    return img


def masked_gaussian_interpolate(arr, aoi_mask, sigma_px=0.0):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from arr, aoi_mask, sigma_px."""
    arr = np.asarray(arr, dtype=np.float64)
    out = arr.copy()
    out[~aoi_mask] = np.nan

    if sigma_px is None or sigma_px <= 0:
        return out

    if gaussian_filter is None:
        print("[WARN] scipy not available; Gaussian interpolation skipped.")
        return out

    valid = np.isfinite(out) & aoi_mask
    if not np.any(valid):
        return out

    values = np.where(valid, out, 0.0)
    weights = valid.astype(np.float64)

    smoothed_values = gaussian_filter(values, sigma=sigma_px, mode="nearest")
    smoothed_weights = gaussian_filter(weights, sigma=sigma_px, mode="nearest")

    result = np.full_like(out, np.nan, dtype=np.float64)
    inside = aoi_mask & (smoothed_weights > 1e-8)
    result[inside] = smoothed_values[inside] / smoothed_weights[inside]
    result[~aoi_mask] = np.nan
    return result


def mm_to_in(mm: float) -> float:
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from mm."""
    return mm / 25.4


def setup_pub_rcparams() -> None:
    """Return the helper result used by the surrounding processing pipeline."""
    matplotlib.rcParams.update({
        "font.size": BASE_FONT_PT,
        "axes.titlesize": BASE_FONT_PT * 1.50,
        "axes.labelsize": BASE_FONT_PT * 1.05,
        "xtick.labelsize": BASE_FONT_PT * 0.95,
        "ytick.labelsize": BASE_FONT_PT * 0.95,
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    })


def make_unit_formatter(unit):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from unit."""
    unit = str(unit).strip()
    if not unit:
        return FuncFormatter(lambda x, pos: f"{x:g}")
    return FuncFormatter(lambda x, pos: f"{x:g} {unit}")


def save_pub_map(arr, path, title, tick_unit="", symmetric=False, angle_mode=False):
    """Save the current result to disk and preserve the expected schema. Inputs are taken from arr, path, title and related parameters."""
    setup_pub_rcparams()

    arr = np.asarray(arr, dtype=np.float64)
    vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        if angle_mode:
            vmin, vmax = ANGLE_MIN, ANGLE_MAX
        elif symmetric:
            vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0
    else:
        if angle_mode:
            vmin, vmax = ANGLE_MIN, ANGLE_MAX
        elif symmetric:
            m = float(np.nanmax(np.abs(vals))) if USE_TRUE_MAX_FOR_NONSYMMETRIC else float(np.nanpercentile(np.abs(vals), 98))
            if not np.isfinite(m) or m <= 0:
                m = 1.0
            vmin, vmax = -m, m
        else:
            vmin = 0.0
            vmax = float(np.nanmax(vals)) if USE_TRUE_MAX_FOR_NONSYMMETRIC else float(np.nanpercentile(vals, 98))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0

    fig = plt.figure(figsize=(mm_to_in(FIG_W_MM), mm_to_in(FIG_H_MM)), dpi=DPI_PNG)
    ax = fig.add_axes([0.10, 0.12, 0.72, 0.80])
    cax = fig.add_axes([0.85, 0.18, 0.04, 0.68])

    im = ax.imshow(
        arr,
        origin="upper",
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )

    ax.set_xlim(-0.5, arr.shape[1] - 0.5)
    ax.set_ylim(arr.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X [pixels]")
    ax.set_ylabel("Y [pixels]")
    ax.set_title(title)

    cb = fig.colorbar(im, cax=cax)
    cb.formatter = make_unit_formatter(tick_unit)
    cb.update_ticks()

    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

    fig.savefig(path, dpi=DPI_PNG, bbox_inches="tight")
    plt.close(fig)


def main():
    """Parse CLI/config inputs and run the complete workflow for this script."""
    os.makedirs(OUT_DIR, exist_ok=True)

    uncorrected_files = list_txt(UNCORRECTED_FOLDER)
    corrected_files = list_txt(CORRECTED_FOLDER)
    lookup = build_lookup(corrected_files)

    pairs = []
    for upath in uncorrected_files:
        key = os.path.splitext(os.path.basename(upath))[0]
        if key in lookup:
            pairs.append((upath, lookup[key]))

    if not pairs:
        raise RuntimeError(
            "No matching corrected/uncorrected TXT pairs found. "
            f'Check corrected suffix "{SUFFIX}" and folder paths.'
        )

    first = read_txt(pairs[0][0])
    H, W = first.shape
    aoi_mask = load_aoi_mask(AOI_JSON, H, W)

    sum_diff = np.zeros((H, W), dtype=np.float64)
    max_abs_diff = np.zeros((H, W), dtype=np.float64)

    for i, (upath, cpath) in enumerate(pairs, start=1):
        img_u = read_txt(upath).astype(np.float64)
        img_c = read_txt(cpath).astype(np.float64)

        if img_u.shape != (H, W) or img_c.shape != (H, W):
            raise RuntimeError(
                f"Frame shape mismatch:\n"
                f"  uncorrected: {upath} -> {img_u.shape}\n"
                f"  corrected  : {cpath} -> {img_c.shape}\n"
                f"  expected   : {(H, W)}"
            )

        if CORRECTED_MODE == "final_temperature":
            corrected_final = img_c
        elif CORRECTED_MODE == "correction_map":
            corrected_final = img_u - img_c
        else:
            raise ValueError(f"Unsupported CORRECTED_MODE: {CORRECTED_MODE}")

        diff = corrected_final - img_u

        # keep NaN unsupported pixels out of the stats
        diff[~np.isfinite(diff)] = np.nan

        sum_diff += np.nan_to_num(diff, nan=0.0)

        valid = np.isfinite(diff)
        max_abs_diff[valid] = np.maximum(max_abs_diff[valid], np.abs(diff[valid]))

        if i % 10 == 0 or i == len(pairs):
            print(f"Processed frame pairs: {i}/{len(pairs)}")

    # Mean only over pixels that had at least one valid value
    valid_counts = np.zeros((H, W), dtype=np.int32)
    for upath, cpath in pairs:
        img_u = read_txt(upath).astype(np.float64)
        img_c = read_txt(cpath).astype(np.float64)
        corrected_final = img_c if CORRECTED_MODE == "final_temperature" else (img_u - img_c)
        diff = corrected_final - img_u
        valid_counts += np.isfinite(diff)

    mean_diff = np.full((H, W), np.nan, dtype=np.float64)
    supported = valid_counts > 0
    mean_diff[supported] = sum_diff[supported] / valid_counts[supported]

    mean_diff[~aoi_mask] = np.nan
    max_abs_diff[~aoi_mask] = np.nan

    u, v, theta = load_theta_csv(THETA_CSV)
    theta_img = rasterize_theta(u, v, theta, H, W)
    theta_img[~aoi_mask] = np.nan

    mean_diff_plot = masked_gaussian_interpolate(mean_diff, aoi_mask, SMOOTH_SIGMA_PX_MEAN)
    max_abs_diff_plot = masked_gaussian_interpolate(max_abs_diff, aoi_mask, SMOOTH_SIGMA_PX_HIGH)
    theta_img_plot = masked_gaussian_interpolate(theta_img, aoi_mask, SMOOTH_SIGMA_PX_ANGLE)

    site = infer_site(UNCORRECTED_FOLDER)

    mean_png = os.path.join(OUT_DIR, f"{site}_mean_divergence.png")
    high_png = os.path.join(OUT_DIR, f"{site}_highest_divergence.png")
    ang_png = os.path.join(OUT_DIR, f"{site}_incidence_angle.png")

    save_pub_map(
        mean_diff_plot,
        mean_png,
        f"{site} - Mean Divergence",
        "K",
        symmetric=True,
        angle_mode=False,
    )

    save_pub_map(
        max_abs_diff_plot,
        high_png,
        f"{site} - Highest Divergence",
        "K",
        symmetric=False,
        angle_mode=False,
    )

    save_pub_map(
        theta_img_plot,
        ang_png,
        f"{site} - Incidence Angle",
        "°",
        symmetric=False,
        angle_mode=True,
    )

    if SAVE_NPY:
        np.save(os.path.join(OUT_DIR, f"{site}_mean_divergence.npy"), mean_diff)
        np.save(os.path.join(OUT_DIR, f"{site}_highest_divergence.npy"), max_abs_diff)
        np.save(os.path.join(OUT_DIR, f"{site}_incidence_angle.npy"), theta_img)

    raw_mean_vals = mean_diff[np.isfinite(mean_diff)]
    raw_high_vals = max_abs_diff[np.isfinite(max_abs_diff)]

    print("\nDone.")
    print(f"Matched frame pairs : {len(pairs)}")
    print(f"CORRECTED_MODE      : {CORRECTED_MODE}")
    print(f"Mean divergence min/max    : {np.nanmin(raw_mean_vals):.6f} .. {np.nanmax(raw_mean_vals):.6f}" if raw_mean_vals.size else "Mean divergence min/max    : n/a")
    print(f"Highest divergence min/max : {np.nanmin(raw_high_vals):.6f} .. {np.nanmax(raw_high_vals):.6f}" if raw_high_vals.size else "Highest divergence min/max : n/a")
    print(f"Mean divergence     : {mean_png}")
    print(f"Highest divergence  : {high_png}")
    print(f"Incidence angle     : {ang_png}")


if __name__ == "__main__":
    main()