"""
Plant Overview Plot Script
--------------------------

This script visualizes plant locations on top of a thermal raster.

Workflow
--------
1. Loads aligned thermal camera matrices (.txt files)
2. Computes the minimum temperature raster
3. Loads plant coordinates from a CSV table
4. Optionally loads or interactively defines rockwall polygons
5. Plots the temperature raster
6. Overlays:
      - plant locations colored by species
      - rockwall polygons with labels
7. Exports a publication-ready figure

Outputs
-------
- One PNG figure showing:
      - minimum temperature raster
      - plant locations
      - species legend
      - optional rockwall polygons

Important
---------
If DEFINE_ROCKWALLS = True, the script first opens an interactive picker.
That picker is only for defining the two rockwall polygons.
After saving/closing the picker, the final matplotlib figure is created.
"""

import os
import re
import pickle

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.ticker import FuncFormatter


# ======================================================
# USER CONFIGURATION
# ======================================================

INPUT_FOLDER = r"F:/Masterarbeit_Backup_2/Data/Data_TXT_formatted_aligned/Lehesten_25-26.08.25"
OUTPUT_FOLDER = r"F:/Masterarbeit_Backup_2/Results/Plants/Lehesten_25-26.08.25"
PLANTS_CSV = os.path.join(OUTPUT_FOLDER, "plant_points_unmasked.csv")

ROCKWALL_CSV = os.path.join(OUTPUT_FOLDER, "rockwall_polygons.csv")
ROCKWALL_PKL = os.path.join(OUTPUT_FOLDER, "rockwall_polygons.pkl")

STUDY_SITE = "Lehesten"
OUT_FIG = os.path.join(OUTPUT_FOLDER, f"{STUDY_SITE}_plant_overview.png")

# Set False if you want to skip the polygon picker and just plot existing polygons
DEFINE_ROCKWALLS = True

FIG_W_MM = 110
FIG_H_MM = 85
DPI_PNG = 600
BASE_FONT_PT = 4

CMAP = "inferno"
POINT_SIZE = 8
POINT_EDGEWIDTH = 0.30

# Plant marker geometry
OUTER_R_PX = 12.5
INNER_R_PX = 6.25
OPENING_HALF_ANGLE_DEG = 45.0   # 90° total opening
OPENING_DIRECTION_DEG = 270.0   # opening points upward in image coordinates
RING_LINEWIDTH = 0.9
RING_ALPHA = 0.98

# Rockwall display
ROCKWALL_COLOR = "#00FFFF"
ROCKWALL_FILL_ALPHA = 0.18
ROCKWALL_LINEWIDTH = 1.2
ROCKWALL_LABEL_COLOR = "white"

# Picker display
WINDOW_NAME = "Rockwall polygon picker"
MAX_WINDOW_W = 1280
MAX_WINDOW_H = 960
START_ZOOM = 1.0
ZOOM_STEP = 1.20
MIN_ZOOM = 0.20
MAX_ZOOM = 30.0

COLOR_PALETTE = [
    "#ff4040", "#40dc40", "#40c8ff", "#ffdc40", "#dc40ff", "#ffa040",
    "#c8c8ff", "#ff80c8", "#78ff78", "#78a0ff", "#ffb478", "#ff6f91",
]


# ======================================================
# HELPERS
# ======================================================

def ensure_dir(path: str) -> None:
    """
    Ensures that a directory exists.

    Creates it if it does not already exist.
    """
    os.makedirs(path, exist_ok=True)


def natural_sort_key(path: str):
    name = os.path.basename(path)
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def list_txt_files(folder: str) -> list[str]:
    """
    Returns a sorted list of all thermal .txt files in the input folder.
    """
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".txt")
    ]
    return sorted(files, key=natural_sort_key)


def read_thermal_txt(path: str) -> np.ndarray:
    """
    Reads a thermal camera .txt export.

    The function searches for the "[Data]" section and reads the following
    tab-separated temperature matrix into a NumPy array.

    Returns
    -------
    ndarray
        2-D temperature matrix (float).
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    try:
        start = lines.index("[Data]") + 1
    except ValueError as e:
        raise ValueError(f"Missing [Data] section in file: {path}") from e

    rows = []
    for line in lines[start:]:
        if not line.strip():
            continue
        parts = line.replace(",", ".").split("\t")
        try:
            row = [float(p) for p in parts if p != ""]
        except ValueError as e:
            raise ValueError(f"Could not parse numeric row in file: {path}\nLine: {line[:200]}") from e
        if row:
            rows.append(row)

    if not rows:
        raise ValueError(f"No numeric data found after [Data] in file: {path}")

    ncol = max(len(r) for r in rows)
    arr = np.full((len(rows), ncol), np.nan, dtype=float)
    for i, row in enumerate(rows):
        arr[i, :len(row)] = row
    return arr


def common_valid_crop(frames: list[np.ndarray], min_valid_fraction: float = 0.01) -> tuple[int, int, int, int] | None:
    """
    Computes the common valid crop across all aligned frames.

    Only pixels that are finite in every frame are kept. This removes the
    alignment border artifacts introduced by warping.
    """
    if not frames:
        return None

    h, w = frames[0].shape
    valid_all = np.ones((h, w), dtype=bool)
    for im in frames:
        valid_all &= np.isfinite(im)

    if not np.any(valid_all):
        return None

    rows = np.where(np.any(valid_all, axis=1))[0]
    cols = np.where(np.any(valid_all, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None

    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    crop_area = (y1 - y0) * (x1 - x0)
    if crop_area < min_valid_fraction * (h * w):
        return None

    return y0, y1, x0, x1


def compute_min_temperature(folder: str) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """
    Computes the minimum temperature raster across all frames.

    In addition to the minimum raster, this also computes the common valid crop
    shared by all aligned frames so edge artifacts from image alignment can be
    removed from the final plot.

    Returns
    -------
    tuple
        - cropped 2-D minimum temperature matrix
        - crop tuple (y0, y1, x0, x1) in original raster coordinates
    """
    txt_files = list_txt_files(folder)
    if not txt_files:
        raise RuntimeError("No txt files found in input folder.")
    print(f"Loading {len(txt_files)} thermal frames...")
    mats = [read_thermal_txt(f) for f in txt_files]
    crop = common_valid_crop(mats)
    stack = np.stack(mats)
    with np.errstate(all="ignore"):
        min_temp = np.nanmin(stack, axis=0)

    if crop is not None:
        y0, y1, x0, x1 = crop
        min_temp = min_temp[y0:y1, x0:x1]

    return min_temp, crop


def mm_to_in(mm: float) -> float:
    """
    Converts millimeters to inches for matplotlib figure sizing.
    """
    return mm / 25.4


def setup_pub_rcparams() -> None:
    """
    Applies matplotlib style parameters optimized for compact publication
    figures.

    Styling is aligned with the simple metrics script so that title,
    label, and tick rendering match across figures.
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


def color_for_species(species: str, species_order: list[str]) -> str:
    """
    Returns a stable plotting color for one species name.

    The first time a species appears it is appended to species_order so the
    same species keeps the same color across the plot and legend.
    """
    if species not in species_order:
        species_order.append(species)
    idx = species_order.index(species)
    return COLOR_PALETTE[idx % len(COLOR_PALETTE)]


def load_plants(csv_path: str) -> pd.DataFrame:
    """
    Loads plant locations from CSV.

    Expected columns
    ----------------
    x : pixel coordinate
    y : pixel coordinate
    species : plant species name

    Coordinates are converted from 1-based indexing (saved format) to 0-based
    indexing used for plotting.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if not {"x", "y", "species"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: x, y, species")

    df = df.copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["species"] = df["species"].astype(str)
    df = df.dropna(subset=["x", "y", "species"]).reset_index(drop=True)
    df["x_plot"] = df["x"] - 1.0
    df["y_plot"] = df["y"] - 1.0
    return df


def load_rockwall_polygons(csv_path: str) -> pd.DataFrame:
    """
    Loads previously saved rockwall polygons from CSV.

    Expected columns
    ----------------
    polygon : polygon name
    vertex_order : order of vertices within a polygon
    x, y : raster coordinates (1-based)

    After loading, coordinates are also converted to 0-based plotting
    coordinates.
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["polygon", "vertex_order", "x", "y", "x_plot", "y_plot"])

    df = pd.read_csv(csv_path)
    required = {"polygon", "vertex_order", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"Rockwall CSV must contain columns: {sorted(required)}")

    df = df.copy()
    df["polygon"] = df["polygon"].astype(str)
    df["vertex_order"] = pd.to_numeric(df["vertex_order"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["polygon", "vertex_order", "x", "y"]).reset_index(drop=True)
    df["x_plot"] = df["x"] - 1.0
    df["y_plot"] = df["y"] - 1.0
    return df


def raster_to_bgr_image(arr: np.ndarray) -> np.ndarray:
    """
    Converts a temperature raster into an OpenCV BGR preview image.

    This preview is used by the interactive rockwall polygon picker.
    """
    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError("Raster contains no finite values.")

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    scaled = np.zeros(arr.shape, dtype=np.uint8)

    if vmax > vmin:
        scaled[finite] = np.clip(
            np.round((arr[finite] - vmin) / (vmax - vmin) * 255.0),
            0, 255
        ).astype(np.uint8)

    color = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    color[~finite] = (0, 0, 0)
    return color


def fit_window_dims(img_w: int, img_h: int) -> tuple[int, int]:
    """
    Chooses a window size that fits on screen while preserving aspect ratio.
    """
    scale = min(MAX_WINDOW_W / img_w, MAX_WINDOW_H / img_h)
    scale = max(scale, 0.1)
    return max(480, int(round(img_w * scale))), max(360, int(round(img_h * scale)))


# ======================================================
# ROCKWALL POLYGON PICKER
# ======================================================

class RockwallPolygonPickerState:
    """
    Interactive OpenCV interface for defining rockwall polygons.

    Features
    --------
    - zoom and pan navigation
    - vertex placement via mouse clicks
    - polygon closing via Enter
    - automatic naming of rockwall sections
    - saving polygons to CSV and PKL formats
    """
    def __init__(self, base_bgr: np.ndarray):
        self.base_bgr = base_bgr
        self.img_h, self.img_w = self.base_bgr.shape[:2]
        self.zoom = START_ZOOM
        self.window_w, self.window_h = fit_window_dims(self.img_w, self.img_h)

        self.view_w = max(1, min(self.img_w, int(round(self.img_w / self.zoom))))
        self.view_h = max(1, min(self.img_h, int(round(self.img_h / self.zoom))))
        self.offset_x = max(0, (self.img_w - self.view_w) // 2)
        self.offset_y = max(0, (self.img_h - self.view_h) // 2)

        self.dragging = False
        self.drag_start = (0, 0)
        self.offset_start = (self.offset_x, self.offset_y)

        self.polygons: list[dict] = []
        self.current_polygon: list[tuple[int, int]] = []
        self.last_disp = np.zeros((self.window_h, self.window_w, 3), dtype=np.uint8)

    def clamp_offsets(self):
        self.view_w = max(1, min(self.img_w, int(round(self.img_w / self.zoom))))
        self.view_h = max(1, min(self.img_h, int(round(self.img_h / self.zoom))))
        self.offset_x = max(0, min(self.offset_x, self.img_w - self.view_w))
        self.offset_y = max(0, min(self.offset_y, self.img_h - self.view_h))

    def window_to_image(self, xw: int, yw: int) -> tuple[int, int]:
        disp_h, disp_w = self.last_disp.shape[:2]
        xi = self.offset_x + int(np.clip(round(xw / max(disp_w, 1) * self.view_w), 0, self.view_w - 1))
        yi = self.offset_y + int(np.clip(round(yw / max(disp_h, 1) * self.view_h), 0, self.view_h - 1))
        return xi, yi

    def image_to_window(self, xi: int, yi: int) -> tuple[int, int]:
        xw = int(round((xi - self.offset_x) / max(self.view_w, 1) * self.window_w))
        yw = int(round((yi - self.offset_y) / max(self.view_h, 1) * self.window_h))
        return xw, yw

    def zoom_at(self, factor: float, xw: int, yw: int):
        xi_before, yi_before = self.window_to_image(xw, yw)
        self.zoom = float(np.clip(self.zoom * factor, MIN_ZOOM, MAX_ZOOM))
        self.clamp_offsets()
        self.view_w = max(1, min(self.img_w, int(round(self.img_w / self.zoom))))
        self.view_h = max(1, min(self.img_h, int(round(self.img_h / self.zoom))))
        self.offset_x = int(round(xi_before - (xw / max(self.window_w, 1)) * self.view_w))
        self.offset_y = int(round(yi_before - (yw / max(self.window_h, 1)) * self.view_h))
        self.clamp_offsets()

    def add_vertex(self, xi: int, yi: int):
        if len(self.polygons) >= 2:
            return
        self.current_polygon.append((xi, yi))

    def close_current_polygon(self):
        if len(self.current_polygon) < 3:
            print("Need at least 3 vertices to close a polygon.")
            return False

        default_name = f"rockwall_{len(self.polygons) + 1}"
        name = input(f"Name for this polygon [{default_name}]: ").strip()
        if name == "":
            name = default_name

        self.polygons.append({
            "name": name,
            "vertices": self.current_polygon.copy(),
        })
        print(f"Closed {name} with {len(self.current_polygon)} vertices.")
        self.current_polygon = []
        return True

    def redraw(self):
        self.clamp_offsets()
        crop = self.base_bgr[self.offset_y:self.offset_y + self.view_h,
                             self.offset_x:self.offset_x + self.view_w].copy()
        disp = cv2.resize(crop, (self.window_w, self.window_h), interpolation=cv2.INTER_NEAREST)

        for i, poly in enumerate(self.polygons, start=1):
            pts = np.array([self.image_to_window(x, y) for x, y in poly["vertices"]], dtype=np.int32)
            if len(pts) >= 3:
                overlay = disp.copy()
                cv2.fillPoly(overlay, [pts], color=(80, 255, 255))
                cv2.addWeighted(overlay, 0.15, disp, 0.85, 0, disp)
                cv2.polylines(disp, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
                x_lab, y_lab = pts[0]
                cv2.putText(disp, str(poly["name"]), (x_lab + 6, y_lab - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            for xw, yw in pts:
                cv2.circle(disp, (int(xw), int(yw)), 4, (255, 255, 255), -1)

        if self.current_polygon:
            pts = np.array([self.image_to_window(x, y) for x, y in self.current_polygon], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(disp, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
            for xw, yw in pts:
                cv2.circle(disp, (int(xw), int(yw)), 4, (0, 0, 255), -1)

        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (disp.shape[1] - 1, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.42, disp, 0.58, 0, disp)

        cv2.putText(disp, "Rockwall polygon picker | define exactly 2 polygons", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp,
                    f"Saved polygons: {len(self.polygons)} / 2 | Current vertices: {len(self.current_polygon)}",
                    (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp,
                    "Left click = add vertex | Enter = close current polygon | Backspace = remove last vertex",
                    (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp,
                    "Mouse wheel = zoom | Middle drag = pan | S = save + close | Q = close without saving",
                    (12, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        self.last_disp = disp
        return disp


def save_rockwall_polygons(polygons: list[dict]) -> tuple[str, str]:
    """
    Saves rockwall polygons to CSV and pickle format.

    Each row in the CSV corresponds to one polygon vertex stored in 1-based
    raster coordinates.
    """
    ensure_dir(OUTPUT_FOLDER)
    rows = []
    for i, poly in enumerate(polygons, start=1):
        poly_name = str(poly.get("name", f"rockwall_{i}"))
        for j, (x, y) in enumerate(poly["vertices"], start=1):
            rows.append({
                "polygon": poly_name,
                "vertex_order": j,
                "x": float(x + 1),
                "y": float(y + 1),
            })
    df = pd.DataFrame(rows, columns=["polygon", "vertex_order", "x", "y"])
    df.to_csv(ROCKWALL_CSV, index=False)
    with open(ROCKWALL_PKL, "wb") as f:
        pickle.dump(df, f)
    return ROCKWALL_CSV, ROCKWALL_PKL


def run_rockwall_polygon_picker(base_bgr: np.ndarray) -> pd.DataFrame:
    """
    Runs the interactive rockwall polygon picker.

    Returns
    -------
    DataFrame
        table of saved rockwall polygon vertices
    """
    state = RockwallPolygonPickerState(base_bgr)

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xi, yi = state.window_to_image(x, y)
            state.add_vertex(xi, yi)
            cv2.imshow(WINDOW_NAME, state.redraw())

        elif event == cv2.EVENT_MBUTTONDOWN:
            state.dragging = True
            state.drag_start = (x, y)
            state.offset_start = (state.offset_x, state.offset_y)

        elif event == cv2.EVENT_MBUTTONUP:
            state.dragging = False

        elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
            dx = x - state.drag_start[0]
            dy = y - state.drag_start[1]
            state.offset_x = state.offset_start[0] - int(round(dx * state.view_w / max(state.window_w, 1)))
            state.offset_y = state.offset_start[1] - int(round(dy * state.view_h / max(state.window_h, 1)))
            state.clamp_offsets()
            cv2.imshow(WINDOW_NAME, state.redraw())

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                state.zoom_at(ZOOM_STEP, x, y)
            else:
                state.zoom_at(1.0 / ZOOM_STEP, x, y)
            cv2.imshow(WINDOW_NAME, state.redraw())

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, state.window_w, state.window_h)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    cv2.imshow(WINDOW_NAME, state.redraw())

    print("\nRockwall polygon picker:")
    print("- left click to add vertices")
    print("- Enter closes the current polygon")
    print("- define exactly 2 polygons")
    print("- S = save + close")
    print("- Q = close without saving")

    saved = False
    while True:
        key = cv2.waitKeyEx(20)
        if key == -1:
            continue

        if key in (13, 10):
            state.close_current_polygon()
            cv2.imshow(WINDOW_NAME, state.redraw())
            continue

        if key in (8, 3014656, 2555904):
            if state.current_polygon:
                state.current_polygon.pop()
            cv2.imshow(WINDOW_NAME, state.redraw())
            continue

        if key in (ord('s'), ord('S')):
            if state.current_polygon:
                print("Current polygon is still open. Press Enter to close it before saving.")
                continue
            if len(state.polygons) != 2:
                print(f"Need exactly 2 polygons before saving. Currently: {len(state.polygons)}")
                continue
            csv_path, pkl_path = save_rockwall_polygons(state.polygons)
            print("Saved rockwall polygons:")
            print("  CSV:", csv_path)
            print("  PKL:", pkl_path)
            saved = True
            break

        if key in (27, ord('q'), ord('Q')):
            break

    cv2.destroyWindow(WINDOW_NAME)
    cv2.waitKey(1)

    return load_rockwall_polygons(ROCKWALL_CSV) if saved or os.path.exists(ROCKWALL_CSV) else load_rockwall_polygons(ROCKWALL_CSV)


# ======================================================
# PLOTTING
# ======================================================

def add_plant_dot(ax, x: float, y: float, color: str):
    """
    Draws one plant as a colored dot with a black edge.

    This helper keeps plant symbols visually consistent between map and legend.
    """
    return ax.scatter(
        [x],
        [y],
        s=POINT_SIZE,
        c=color,
        edgecolors="black",
        linewidths=POINT_EDGEWIDTH,
        zorder=4,
    )


def apply_crop_to_annotations(
    plants: pd.DataFrame,
    rockwalls: pd.DataFrame,
    crop: tuple[int, int, int, int] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Shifts plant and polygon coordinates into the cropped raster coordinate
    system used for plotting.
    """
    if crop is None:
        return plants.copy(), rockwalls.copy()

    y0, y1, x0, x1 = crop

    plants_out = plants.copy()
    plants_out["x_plot"] = plants_out["x_plot"] - x0
    plants_out["y_plot"] = plants_out["y_plot"] - y0

    rockwalls_out = rockwalls.copy()
    if len(rockwalls_out) > 0:
        rockwalls_out["x_plot"] = rockwalls_out["x_plot"] - x0
        rockwalls_out["y_plot"] = rockwalls_out["y_plot"] - y0

    return plants_out, rockwalls_out


def plot_overview(min_temp: np.ndarray, plants: pd.DataFrame, rockwalls: pd.DataFrame) -> None:
    """
    Generates the final plant overview figure.

    The figure contains:
        - minimum temperature raster
        - plant locations colored by species
        - species legend with counts
        - optional rockwall polygons
        - temperature colorbar

    The layout is optimized for publication figures.
    """
    setup_pub_rcparams()
    nrow, ncol = min_temp.shape

    plants = plants[
        (plants["x_plot"] >= 0) & (plants["x_plot"] < ncol) &
        (plants["y_plot"] >= 0) & (plants["y_plot"] < nrow)
    ].copy()

    finite = min_temp[np.isfinite(min_temp)]
    if finite.size == 0:
        raise ValueError("No finite temperature values available for plotting.")

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))

    fig = plt.figure(figsize=(mm_to_in(FIG_W_MM), mm_to_in(FIG_H_MM)), dpi=DPI_PNG)
    ax = fig.add_axes([0.10, 0.22, 0.70, 0.70])
    cax = fig.add_axes([0.815, 0.25, 0.030, 0.64])

    im = ax.imshow(
        min_temp,
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="nearest"
    )

    species_order = []
    legend_handles = []
    legend_labels = []

    for species in sorted(plants["species"].unique()):
        sub = plants.loc[plants["species"] == species]
        color = color_for_species(species, species_order)

        ax.scatter(
            sub["x_plot"],
            sub["y_plot"],
            s=POINT_SIZE,
            c=color,
            edgecolors="black",
            linewidths=POINT_EDGEWIDTH,
            zorder=4,
        )

        handle = ax.scatter([], [], s=POINT_SIZE, c=color, edgecolors="black", linewidths=POINT_EDGEWIDTH)
        legend_handles.append(handle)
        legend_labels.append(f"{species} (n={len(sub)})")

    if len(rockwalls) > 0:
        for poly_name, sub in rockwalls.groupby("polygon"):
            sub = sub.sort_values("vertex_order")
            xs = sub["x_plot"].to_numpy()
            ys = sub["y_plot"].to_numpy()
            ax.fill(xs, ys, facecolor=ROCKWALL_COLOR, alpha=ROCKWALL_FILL_ALPHA, zorder=5)
            ax.plot(np.r_[xs, xs[0]], np.r_[ys, ys[0]],
                    color=ROCKWALL_COLOR, linewidth=ROCKWALL_LINEWIDTH, zorder=6)
            ax.text(float(np.mean(xs)), float(np.mean(ys)), str(poly_name),
                    color=ROCKWALL_LABEL_COLOR, fontsize=BASE_FONT_PT,
                    ha="center", va="center", zorder=7)

    title_size = BASE_FONT_PT * 1.75
    label_size = BASE_FONT_PT * 1.05
    tick_size = BASE_FONT_PT * 0.95

    ax.set_title(f"{STUDY_SITE} - Plant Locations", fontsize=title_size, fontname="Arial")
    ax.set_xlabel("X [pixels]", fontsize=label_size, fontname="Arial")
    ax.set_ylabel("Y [pixels]", fontsize=label_size, fontname="Arial")

    ax.tick_params(axis="both", labelsize=tick_size)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Arial")
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)

    cb = fig.colorbar(im, cax=cax)
    cb.formatter = FuncFormatter(lambda x, pos: f"{x:g} °C")
    cb.update_ticks()
    cb.ax.tick_params(labelsize=tick_size)
    for tick in cb.ax.get_yticklabels():
        tick.set_fontname("Arial")

    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.44, -0.14),
            ncol=3,
            frameon=True,
            fontsize=BASE_FONT_PT,
            markerscale=0.9,
            handletextpad=0.4,
            columnspacing=0.8,
            borderpad=0.4,
        )

    ensure_dir(OUTPUT_FOLDER)
    fig.savefig(OUT_FIG, dpi=DPI_PNG, bbox_inches="tight")
    plt.show()

    print("Saved figure:", OUT_FIG)
    print(f"Plotted {len(plants)} plant points.")
    if len(rockwalls) > 0:
        print(f"Plotted {rockwalls['polygon'].nunique()} rockwall polygons.")


def main():
    """
    Main processing workflow.

    Steps
    -----
    1. Load the minimum temperature raster
    2. Load plant coordinates
    3. Optionally load or redefine rockwall polygons
    4. Generate the final overview plot
    """
    ensure_dir(OUTPUT_FOLDER)

    # --------------------------------------------------
    # Load minimum temperature raster
    # --------------------------------------------------
    min_temp, crop = compute_min_temperature(INPUT_FOLDER)

    # --------------------------------------------------
    # Load plant coordinate table
    # --------------------------------------------------
    plants = load_plants(PLANTS_CSV)
    base_bgr = raster_to_bgr_image(min_temp)

    # --------------------------------------------------
    # Load or redefine rockwall polygons
    # --------------------------------------------------
    use_existing = False
    if os.path.exists(ROCKWALL_CSV):
        resp = input("Existing rockwall polygons found. Use them? (y/n): ").strip().lower()
        if resp in ["y", "yes"]:
            use_existing = True

    if use_existing:
        rockwalls = load_rockwall_polygons(ROCKWALL_CSV)
    else:
        if DEFINE_ROCKWALLS:
            rockwalls = run_rockwall_polygon_picker(base_bgr)
        else:
            rockwalls = load_rockwall_polygons(ROCKWALL_CSV)

    # --------------------------------------------------
    # Shift annotations into cropped alignment extent
    # --------------------------------------------------
    plants, rockwalls = apply_crop_to_annotations(plants, rockwalls, crop)

    # --------------------------------------------------
    # Generate final overview plot
    # --------------------------------------------------
    plot_overview(min_temp, plants, rockwalls)


if __name__ == "__main__":
    main()