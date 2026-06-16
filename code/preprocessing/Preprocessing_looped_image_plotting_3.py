import os
import re
import glob
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


# --- Function to read one IR .txt file (matches the R logic) ---
def read_thermal_txt(file_path: str) -> np.ndarray:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f]

    try:
        data_start = lines.index("[Data]") + 1
    except ValueError:
        raise ValueError(f'Could not find "[Data]" section in: {file_path}')

    dat_lines = lines[data_start:]
    # Convert decimal commas to dots, then parse tab-separated numeric matrix
    rows = []
    for line in dat_lines:
        if not line.strip():
            continue
        line = line.replace(",", ".")
        parts = line.split("\t")
        rows.append([float(x) if x.strip() != "" else np.nan for x in parts])

    return np.array(rows, dtype=float)


# --- Folder with your .txt files (kept identical) ---
folder = r"F:/Masterarbeit_Backup_2/Data/Data_TXT_formatted_aligned_angled/Paradiestal_12-13.08.25"

# Create folder for plots (new location; kept identical)
plot_folder = r"F:/Masterarbeit_Backup_2/Plots_aligned_angled/Paradiestal_12-13.08.25"
os.makedirs(plot_folder, exist_ok=True)

# Get all files for one day (same pattern as R: ^2025-11-.*\.txt$)
all_txt = glob.glob(os.path.join(folder, "*.txt"))
files = sorted([f for f in all_txt if re.search(r"^2025-08-.*\.txt$", os.path.basename(f))])

if not files:
    raise FileNotFoundError(f"No matching files found in {folder} for pattern ^2025-08-.*\\.txt$")

# --- Calculate global min and max temperatures (across all frames) ---
global_min = np.inf
global_max = -np.inf

for fpath in files:
    img = read_thermal_txt(fpath)
    finite_vals = img[np.isfinite(img)]
    if finite_vals.size:
        global_min = min(global_min, float(finite_vals.min()))
        global_max = max(global_max, float(finite_vals.max()))

if not np.isfinite(global_min) or not np.isfinite(global_max):
    raise ValueError("Could not compute global min/max (no finite numeric values found).")

# --- Loop through files and plot ---
for file_path in files:
    fname = os.path.basename(file_path)

    # Extract timestamp from filename (matches the R regex behavior)
    # date_label: first YYYY-MM-DD in filename
    m_date = re.match(r"^(\d{4}-\d{2}-\d{2}).*", fname)
    if not m_date:
        raise ValueError(f"Could not extract date from filename: {fname}")
    date_label = m_date.group(1)

    # time_label: 4 digits before .txt after underscore
    m_time = re.match(r".*_(\d{4})_.*\.txt$", fname)
    if not m_time:
        raise ValueError(f"Could not extract time (HHMM) from filename: {fname}")
    time_label = m_time.group(1)

    time_label_fmt = datetime.strptime(time_label, "%H%M").strftime("%H:%M")

    # Read data
    img = read_thermal_txt(file_path)

    # R code flips Y axis; replicate by flipping the array vertically
    img_plot = np.flipud(img)

    n_rows, n_cols = img_plot.shape

    # Grid spacing (same as R)
    x_spacing = n_cols / 12.0
    y_spacing = n_rows / 9.0

    # Generate chessboard labels (same intent as R)
    col_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[: int(np.ceil(n_cols / x_spacing))]
    row_labels = list(range(1, int(np.ceil(n_rows / y_spacing)) + 1))

    # Build plot
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    im = ax.imshow(
        img_plot,
        cmap="inferno",
        vmin=global_min,
        vmax=global_max,
        origin="lower",
        extent=(0, n_cols, 0, n_rows),
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_title(f"Paradiestal - Thermal Image {time_label_fmt}", fontweight="bold", fontsize=14)

    # Ticks centered in each grid cell band (matches R: seq(spacing/2, n, by=spacing))
    x_breaks = np.arange(x_spacing / 2.0, n_cols + 1e-9, x_spacing)
    y_breaks = np.arange(y_spacing / 2.0, n_rows + 1e-9, y_spacing)

    ax.set_xticks(x_breaks[: len(col_labels)])
    ax.set_xticklabels(col_labels, fontweight="bold", fontsize=10)

    ax.set_yticks(y_breaks[: len(row_labels)])
    ax.set_yticklabels([str(x) for x in row_labels], fontweight="bold", fontsize=10)

    # Grid lines (same positions as R: seq(0, n, by=spacing))
    x_lines = np.arange(0, n_cols + 1e-9, x_spacing)
    y_lines = np.arange(0, n_rows + 1e-9, y_spacing)

    ax.vlines(x_lines, ymin=0, ymax=n_rows, colors="grey", linewidth=0.3, alpha=0.5)
    ax.hlines(y_lines, xmin=0, xmax=n_cols, colors="grey", linewidth=0.3, alpha=0.5)

    # Remove axis labels (R: x=NULL, y=NULL)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Colorbar at bottom, horizontal (R: legend.position="bottom", direction="horizontal")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.10)
    cbar.set_label("Surface Temperature [°C]", fontweight="bold")
    cbar.ax.xaxis.set_label_position("top")

    # Tight layout to keep consistent sizing
    fig.tight_layout()

    # Save plot as PNG (same naming as R)
    plot_filename = os.path.join(plot_folder, f"{date_label}_{time_label}.png")
    fig.savefig(plot_filename, dpi=300)
    plt.close(fig)

    # Optional: print progress
    print(f"Saved: {plot_filename}")
