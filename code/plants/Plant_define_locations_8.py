"""
Plant Define Locations Script (Raster-Accurate)
-----------------------------------------------

This version fixes the coordinate-storage problem by NOT using an exported PNG
with possible borders, titles, axes, or colorbars as the clickable background.

Instead it:
1. loads all aligned thermal .txt files,
2. computes the minimum temperature raster directly,
3. converts that raster into a display image,
4. lets the user click directly on the raster itself,
5. saves plant positions as true raster coordinates.

Why this fixes the problem
--------------------------
In the previous script, clicks were taken from the full exported PNG size and
then rescaled to TARGET_COORD_W / TARGET_COORD_H. If the PNG contained margins,
axes, title, or a colorbar, the mapping was wrong because the full PNG was not
the same as the actual data area.

Here the displayed image IS the data raster, so:
    click on screen -> raster pixel -> saved x/y
No extra PNG-to-data conversion is needed.

Saved columns
-------------
x, y
    True raster coordinates (1-based)

species
    Species label entered by the user
"""

import os
import glob
import pickle
import re
from typing import Optional

import cv2
import numpy as np
import pandas as pd


# ===================== USER CONFIG =====================

# Folder containing aligned thermal .txt files
INPUT_FOLDER = (
    r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted_aligned"
    r"\Paradiestal_12-13.08.25"
)

# Folder where AOI masks (*.npz) are stored
AOI_FOLDER = (
    r"F:\Masterarbeit_Backup_2\Results\Simple_metrics"
    r"\Paradiestal_12-13.08.25"
)

# Output folder for plant point table
OUT_DIR = (
    r"F:\Masterarbeit_Backup_2\Results\Plants"
    r"\Paradiestal_12-13.08.25"
)

OUT_CSV = os.path.join(OUT_DIR, "plant_points_unmasked.csv")
OUT_PICKLE = os.path.join(OUT_DIR, "plant_points_unmasked.pkl")

# Behavior options
RESTRICT_CLICKS_TO_MASK = False
SHOW_MASK_FILL = True

START_ZOOM = 1.0
ZOOM_STEP = 1.20
MIN_ZOOM = 0.20
MAX_ZOOM = 30.0

WINDOW_NAME = "Plant point picker"
MAX_WINDOW_W = 1280
MAX_WINDOW_H = 960

# Temperature display settings
COLORMAP = cv2.COLORMAP_INFERNO
NAN_COLOR_BGR = (0, 0, 0)

# =======================================================

COLOR_PALETTE = [
    (255, 64, 64),
    (64, 220, 64),
    (64, 200, 255),
    (255, 220, 64),
    (220, 64, 255),
    (255, 160, 64),
    (64, 255, 220),
    (200, 200, 255),
    (255, 128, 200),
    (120, 255, 120),
    (120, 160, 255),
    (255, 180, 120),
]


def natural_sort_key(path: str):
    """Sort helper that orders embedded numbers naturally."""
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


def list_txt_files(folder: str) -> list[str]:
    """Return all thermal .txt files in natural order."""
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".txt")
    ]
    return sorted(files, key=natural_sort_key)


def read_thermal_txt(path: str) -> np.ndarray:
    """
    Read one thermal text export by parsing the [Data] section.
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
            raise ValueError(
                f"Could not parse numeric row in file: {path}\n"
                f"Line preview: {line[:200]}"
            ) from e

        if row:
            rows.append(row)

    if not rows:
        raise ValueError(f"No numeric data found after [Data] in file: {path}")

    ncol = max(len(r) for r in rows)
    arr = np.full((len(rows), ncol), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        arr[i, :len(row)] = row

    return arr


def load_min_temperature_raster(folder: str) -> np.ndarray:
    """
    Load all aligned thermal text files and compute the per-pixel minimum.
    """
    txt_files = list_txt_files(folder)
    if not txt_files:
        raise FileNotFoundError(f"No thermal .txt files found in:\n{folder}")

    print(f"Processing {len(txt_files)} thermal files for picker background...")
    mats = [read_thermal_txt(p) for p in txt_files]
    stack = np.stack(mats)
    return np.nanmin(stack, axis=0)


def raster_to_bgr_image(arr: np.ndarray) -> np.ndarray:
    """
    Convert the minimum-temperature raster into an OpenCV BGR image.

    Important:
    The output image has exactly the same width/height as the raster, so
    clicks correspond 1:1 to raster pixels.
    """
    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError("Raster contains no finite values.")

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))

    if vmax <= vmin:
        scaled = np.zeros(arr.shape, dtype=np.uint8)
    else:
        scaled = np.zeros(arr.shape, dtype=np.uint8)
        scaled[finite] = np.clip(
            np.round((arr[finite] - vmin) / (vmax - vmin) * 255.0),
            0, 255
        ).astype(np.uint8)

    color = cv2.applyColorMap(scaled, COLORMAP)
    color[~finite] = NAN_COLOR_BGR
    return color


def find_aoi_npz(folder: str) -> list[str]:
    """Returns all AOI mask files in the selected folder."""
    return sorted(glob.glob(os.path.join(folder, "*.npz")))


def load_mask_from_npz(npz_path: str) -> np.ndarray:
    """Loads a boolean AOI mask from an .npz file."""
    data = np.load(npz_path, allow_pickle=False)
    if "mask" not in data:
        raise KeyError(f'NPZ does not contain key "mask": {npz_path}')
    return data["mask"].astype(bool)


def prompt_select_mask(aoi_folder: str) -> tuple[str | None, np.ndarray | None]:
    """Lets the user choose an AOI mask from the console."""
    candidates = find_aoi_npz(aoi_folder)
    if not candidates:
        print(f"No AOI masks (*.npz) found in: {aoi_folder}")
        return None, None

    print("\nFound AOI mask files:")
    for i, p in enumerate(candidates, start=1):
        print(f"  {i}) {os.path.basename(p)}")
    print("  0) None (do not load a mask)")

    while True:
        s = input("Select mask number: ").strip() or "0"
        try:
            k = int(s)
        except ValueError:
            print("Please enter a number.")
            continue

        if k == 0:
            return None, None
        if 1 <= k <= len(candidates):
            path = candidates[k - 1]
            mask = load_mask_from_npz(path)
            print(f"Loaded mask: {os.path.basename(path)} | mask pixels = {int(mask.sum())}")
            return path, mask

        print("Selection out of range.")


def overlay_mask(base_bgr: np.ndarray, mask: Optional[np.ndarray], alpha: float = 0.22) -> np.ndarray:
    """Draw the AOI mask on top of the background image for visual guidance."""
    out = base_bgr.copy()
    if mask is None:
        return out

    if SHOW_MASK_FILL:
        fill = np.zeros_like(out)
        fill[mask] = (0, 255, 255)
        out = cv2.addWeighted(out, 1.0, fill, alpha, 0)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 0), 1)
    return out


def fit_window_dims(img_w: int, img_h: int) -> tuple[int, int]:
    """Choose a window size that fits on screen while preserving aspect ratio."""
    scale = min(MAX_WINDOW_W / img_w, MAX_WINDOW_H / img_h)
    scale = max(scale, 0.1)
    win_w = max(480, int(round(img_w * scale)))
    win_h = max(360, int(round(img_h * scale)))
    return win_w, win_h


def color_for_species(species: str, species_order: list[str]) -> tuple[int, int, int]:
    """Assign a stable display color to each species name."""
    if species not in species_order:
        species_order.append(species)
    idx = species_order.index(species)
    return COLOR_PALETTE[idx % len(COLOR_PALETTE)]


def draw_hud(view: np.ndarray,
             pending_point: Optional[tuple[int, int]],
             zoom: float,
             offset_x: int,
             offset_y: int,
             all_points: list[dict],
             species_order: list[str],
             mask_loaded: bool,
             raster_w: int,
             raster_h: int) -> np.ndarray:
    """Draw on-screen instructions and legend."""
    h, w = view.shape[:2]
    panel_h = min(190, max(130, 90 + 24 * ((len(species_order) + 3) // 4)))
    overlay = view.copy()
    cv2.rectangle(overlay, (0, 0), (w - 1, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.42, view, 0.58, 0, view)

    total_points = len(all_points)
    if pending_point is None:
        pending_text = "Pending point: none"
    else:
        pending_text = f"Pending XY=({pending_point[0] + 1}, {pending_point[1] + 1})"

    cv2.putText(view, "Plant picker | raster-accurate coordinates", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(view,
                f"Zoom: {zoom:.2f}x | Offset: ({offset_x}, {offset_y}) | Saved points: {total_points}",
                (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(view,
                f"Raster size: {raster_w}x{raster_h} | Saved coords: same raster coordinates",
                (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(view, pending_text,
                (12, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    restrict_text = "ON" if (mask_loaded and RESTRICT_CLICKS_TO_MASK) else "OFF"
    cv2.putText(view,
                "Left click select pending point | Enter assign species | Right click remove nearest saved point",
                (12, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(view,
                f"Mouse wheel zoom | Middle drag pan | Backspace/Delete clear pending | S save | Q/Esc finish | Mask restriction: {restrict_text}",
                (12, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(view, "Legend:", (12, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    counts = {}
    for p in all_points:
        counts[p["species"]] = counts.get(p["species"], 0) + 1

    cols = 4
    x_start = 78
    y_start = 162
    col_w = max(180, (w - x_start - 20) // max(cols, 1))
    row_h = 24

    for i, sp in enumerate(species_order):
        col = i % cols
        row = i // cols
        y = y_start + row * row_h
        x = x_start + col * col_w
        c = color_for_species(sp, species_order)
        cv2.circle(view, (x, y), 6, c, -1)
        label = f"{sp} ({counts.get(sp, 0)})"
        cv2.putText(view, label, (x + 12, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    return view


class PickerState:
    """Store the interactive picker state for zooming, panning, and saved points."""

    def __init__(self, base_bgr: np.ndarray, aoi_mask: Optional[np.ndarray]):
        self.base_bgr = overlay_mask(base_bgr, aoi_mask)
        self.aoi_mask = aoi_mask
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

        # Pending point is stored directly in raster pixel coordinates (0-based).
        self.pending_point: Optional[tuple[int, int]] = None

        # Saved points are stored directly in true raster coordinates (1-based).
        self.plant_points: list[dict] = []
        self.species_order: list[str] = []
        self.last_disp = np.zeros((self.window_h, self.window_w, 3), dtype=np.uint8)

    def clamp_offsets(self) -> None:
        """Keep the viewed crop inside the image boundaries."""
        self.view_w = max(1, min(self.img_w, int(round(self.img_w / self.zoom))))
        self.view_h = max(1, min(self.img_h, int(round(self.img_h / self.zoom))))
        self.offset_x = max(0, min(self.offset_x, self.img_w - self.view_w))
        self.offset_y = max(0, min(self.offset_y, self.img_h - self.view_h))

    def window_to_image(self, xw: int, yw: int) -> tuple[int, int]:
        """Convert mouse coordinates from the current window view into raster pixel coordinates."""
        disp_h, disp_w = self.last_disp.shape[:2]
        xi = self.offset_x + int(np.clip(round(xw / max(disp_w, 1) * self.view_w), 0, self.view_w - 1))
        yi = self.offset_y + int(np.clip(round(yw / max(disp_h, 1) * self.view_h), 0, self.view_h - 1))
        return xi, yi

    def image_to_window(self, xi: int, yi: int) -> tuple[int, int]:
        """Convert a raster pixel coordinate into the current displayed window coordinate."""
        xw = int(round((xi - self.offset_x) / max(self.view_w, 1) * self.window_w))
        yw = int(round((yi - self.offset_y) / max(self.view_h, 1) * self.window_h))
        return xw, yw

    def remove_nearest_saved_point(self, xi: int, yi: int) -> None:
        """Remove the saved point whose raster location is closest to the clicked position."""
        if not self.plant_points:
            return

        d2 = []
        for idx, p in enumerate(self.plant_points):
            px = int(round(p["x"] - 1))
            py = int(round(p["y"] - 1))
            d2.append(((px - xi) ** 2 + (py - yi) ** 2, idx))

        _, idx = min(d2, key=lambda t: t[0])
        removed = self.plant_points.pop(idx)
        print(
            "Removed point "
            f"xy=({int(removed['x'])}, {int(removed['y'])}) | "
            f"species='{removed['species']}'"
        )

    def add_pending_as_species(self, species: str) -> bool:
        """Commit the pending point directly as a true raster coordinate."""
        if self.pending_point is None:
            print("No pending point selected.")
            return False

        xi, yi = self.pending_point
        if self.aoi_mask is not None and RESTRICT_CLICKS_TO_MASK and not bool(self.aoi_mask[yi, xi]):
            print("Pending point is outside the AOI mask and was rejected.")
            self.pending_point = None
            return False

        color_for_species(species, self.species_order)
        self.plant_points.append({
            "x": float(xi + 1),
            "y": float(yi + 1),
            "species": species,
        })
        self.pending_point = None
        return True

    def redraw(self) -> np.ndarray:
        """Redraw the current zoomed/panned view including points and HUD."""
        self.clamp_offsets()
        crop = self.base_bgr[self.offset_y:self.offset_y + self.view_h,
                             self.offset_x:self.offset_x + self.view_w].copy()
        disp = cv2.resize(crop, (self.window_w, self.window_h), interpolation=cv2.INTER_NEAREST)

        for p in self.plant_points:
            xi = int(round(p["x"] - 1))
            yi = int(round(p["y"] - 1))
            if not (self.offset_x <= xi < self.offset_x + self.view_w and self.offset_y <= yi < self.offset_y + self.view_h):
                continue
            xw, yw = self.image_to_window(xi, yi)
            col = color_for_species(p["species"], self.species_order)
            cv2.circle(disp, (xw, yw), 6, col, -1)
            cv2.circle(disp, (xw, yw), 7, (0, 0, 0), 1)

        if self.pending_point is not None:
            xi, yi = self.pending_point
            if self.offset_x <= xi < self.offset_x + self.view_w and self.offset_y <= yi < self.offset_y + self.view_h:
                xw, yw = self.image_to_window(xi, yi)
                cv2.drawMarker(disp, (xw, yw), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
                cv2.circle(disp, (xw, yw), 9, (255, 255, 255), 1)

        disp = draw_hud(
            disp,
            pending_point=self.pending_point,
            zoom=self.zoom,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            all_points=self.plant_points,
            species_order=self.species_order,
            mask_loaded=self.aoi_mask is not None,
            raster_w=self.img_w,
            raster_h=self.img_h,
        )
        self.last_disp = disp
        return disp

    def zoom_at(self, factor: float, xw: int, yw: int) -> None:
        """Zoom around the current mouse position so the clicked feature stays centered."""
        xi_before, yi_before = self.window_to_image(xw, yw)
        self.zoom = float(np.clip(self.zoom * factor, MIN_ZOOM, MAX_ZOOM))
        self.clamp_offsets()
        self.view_w = max(1, min(self.img_w, int(round(self.img_w / self.zoom))))
        self.view_h = max(1, min(self.img_h, int(round(self.img_h / self.zoom))))
        self.offset_x = int(round(xi_before - (xw / max(self.window_w, 1)) * self.view_w))
        self.offset_y = int(round(yi_before - (yw / max(self.window_h, 1)) * self.view_h))
        self.clamp_offsets()


def run_picker(state: PickerState) -> None:
    """Start the OpenCV picker window and handle mouse/keyboard interaction."""

    def on_mouse(event: int, x: int, y: int, flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            xi, yi = state.window_to_image(x, y)
            state.pending_point = (xi, yi)
            cv2.imshow(WINDOW_NAME, state.redraw())

        elif event == cv2.EVENT_RBUTTONDOWN:
            xi, yi = state.window_to_image(x, y)
            state.remove_nearest_saved_point(xi, yi)
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

    print("\nInstructions:")
    print("- Left click: place/select one pending point")
    print("- Enter: assign species name to that pending point")
    print("- Right click: remove nearest saved point")
    print("- Mouse wheel: zoom")
    print("- Middle mouse drag: pan")
    print("- Backspace/Delete: clear pending point")
    print("- S: save now")
    print("- Q or Esc: finish and close")
    print("- Saved coordinates are written directly as true raster coordinates")
    if state.aoi_mask is not None and RESTRICT_CLICKS_TO_MASK:
        print("- NOTE: only points inside the AOI mask are accepted")

    while True:
        key = cv2.waitKeyEx(20)
        if key == -1:
            continue

        if key in (13, 10):  # Enter
            if state.pending_point is None:
                print("Select a point first with left click.")
                continue
            species = input("Species for pending point (blank = cancel point): ").strip()
            if species == "":
                print("Pending point cancelled.")
                state.pending_point = None
            else:
                ok = state.add_pending_as_species(species)
                if ok:
                    print(f"Added point for species: {species}")
            cv2.imshow(WINDOW_NAME, state.redraw())
            continue

        if key in (8, 3014656, 2555904):  # backspace/delete variants
            state.pending_point = None
            cv2.imshow(WINDOW_NAME, state.redraw())
            continue

        if key in (ord('s'), ord('S')):
            csv_path, pkl_path = save_points(state.plant_points)
            print(f"Saved current points. CSV: {csv_path}")
            print(f"Saved current points. PKL: {pkl_path}")
            print("Saving finished — closing picker.")
            break

        if key in (27, ord('q'), ord('Q')):
            break

        if key in (ord('+'), ord('=')):
            state.zoom_at(ZOOM_STEP, state.window_w // 2, state.window_h // 2)
            cv2.imshow(WINDOW_NAME, state.redraw())
            continue

        if key in (ord('-'), ord('_')):
            state.zoom_at(1.0 / ZOOM_STEP, state.window_w // 2, state.window_h // 2)
            cv2.imshow(WINDOW_NAME, state.redraw())
            continue

    cv2.destroyWindow(WINDOW_NAME)
    cv2.waitKey(1)


def save_points(plant_points: list[dict]) -> tuple[str, str]:
    """
    Write picked plant points to CSV and pickle.

    If the target file is currently open in Excel or another program and cannot
    be overwritten, a timestamped fallback filename is used instead.
    """
    import datetime

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(plant_points, columns=["x", "y", "species"])

    csv_path = OUT_CSV
    pkl_path = OUT_PICKLE

    try:
        df.to_csv(csv_path, index=False)
    except PermissionError:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(OUT_DIR, f"plant_points_unmasked_{stamp}.csv")
        print(f"Could not overwrite locked CSV. Saving instead to: {csv_path}")
        df.to_csv(csv_path, index=False)

    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(df, f)
    except PermissionError:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pkl_path = os.path.join(OUT_DIR, f"plant_points_unmasked_{stamp}.pkl")
        print(f"Could not overwrite locked pickle. Saving instead to: {pkl_path}")
        with open(pkl_path, "wb") as f:
            pickle.dump(df, f)

    return csv_path, pkl_path


def main():
    """Load raster background, optionally load AOI, and start the picker."""
    os.makedirs(OUT_DIR, exist_ok=True)

    min_temp = load_min_temperature_raster(INPUT_FOLDER)
    base_bgr = raster_to_bgr_image(min_temp)

    nrow, ncol = base_bgr.shape[:2]

    mask_path, aoi_mask = prompt_select_mask(AOI_FOLDER)

    if aoi_mask is not None and aoi_mask.shape != (nrow, ncol):
        raise ValueError(
            f"AOI mask shape {aoi_mask.shape} does not match raster shape {(nrow, ncol)}.\n"
            "This picker now uses the true thermal raster directly, so mask and raster must match exactly."
        )

    if mask_path:
        print(f"Loaded AOI mask: {os.path.basename(mask_path)}")

    print(f"Raster size used for picking: {ncol} x {nrow}")
    print("Saved x/y coordinates are direct raster coordinates (1-based).")

    state = PickerState(base_bgr, aoi_mask)
    run_picker(state)
    csv_path, pkl_path = save_points(state.plant_points)

    print("\nDone.")
    print("Saved CSV        :", csv_path)
    print("Saved pickle     :", pkl_path)


if __name__ == "__main__":
    main()