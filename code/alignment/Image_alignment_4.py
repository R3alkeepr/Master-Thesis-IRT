import os
import glob
import re
import csv
from io import StringIO
import numpy as np
import cv2
from PIL import Image

# ===================== USER CONFIG =====================
TXT_INPUT_FOLDER = r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted\Paradiestal_12-13.08.25"
TXT_OUTPUT_FOLDER = r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted_aligned\Paradiestal_12-13.08.25"
DECIMAL_COMMA_OUTPUT = True  # write decimals with ',' to match common thermal TXT conventions

ALIGN_ROOT = r"F:\Masterarbeit_Backup_2\Alignment\Paradiestal_12-13.08.25"

START_INDEX = 0
BASE_STEP_XY = 0.1      # pixels for translation
BASE_STEP_ROT = 0.01    # degrees for rotation
FAST_MULT = 10
RED_ALPHA = 0.95

# Grid overlay (visual aid)
GRID_ENABLED_DEFAULT = True
GRID_SPACING_PX = 80
GRID_COLOR = (0, 255, 0)
GRID_ALPHA = 0.25


# GIF export (to reduce palette/dither flicker and perceived jumps)
GIF_PALETTE_GLOBAL = True   # use a single palette for all frames
GIF_COLORS = 256            # GIF palette size (max 256)
GIF_DITHER = False          # disable dithering for pixel-stable playback
GIF_DISPOSAL = 2            # 2=restore to background between frames (safer)
# Apply policy
INTERP = cv2.INTER_LINEAR
BORDER_VALUE = np.nan
# =======================================================


# ----------------- IO helpers -----------------
def natural_sort_key(path: str) -> int:
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 10**12


def read_thermal_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    idx = next(i for i, ln in enumerate(lines) if ln.strip() == "[Data]")
    header = lines[:idx + 1]
    clean = "".join(ln.replace(",", ".") for ln in lines[idx + 1:])
    arr = np.loadtxt(StringIO(clean), delimiter="\t", dtype=np.float32)
    return header, arr


def load_txt_stack():
    files = sorted(glob.glob(os.path.join(TXT_INPUT_FOLDER, "*.txt")), key=natural_sort_key)
    if not files:
        raise SystemExit("No .txt files found in TXT_INPUT_FOLDER.")
    headers, frames = [], []
    for p in files:
        h, a = read_thermal_txt(p)
        headers.append(h)
        frames.append(a)
    return files, headers, frames




def save_aligned_txts(files, headers, frames, C_3x3, out_folder):
    """Apply the final per-frame affine transforms to the original TXT frames and write aligned TXT files.

    - Writes one TXT per input file name into out_folder.
    - Preserves the original header (up to and including [Data]).
    - Writes tab-separated floats. If DECIMAL_COMMA_OUTPUT is True, uses ',' as decimal separator.
    - Keeps NaNs (outside valid warp region) as 'nan'.
    """
    os.makedirs(out_folder, exist_ok=True)

    for i, (in_path, header, img) in enumerate(zip(files, headers, frames)):
        C = C_3x3[i]
        aligned = warp_affine_float(img, C[:2, :])

        out_path = os.path.join(out_folder, os.path.basename(in_path))

        # Write header exactly as read (already includes the [Data] line)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.writelines(header)

            # Write data (tab-separated). Use a stable float format.
            # Note: np.savetxt uses '.' decimals; we post-process to ',' if requested.
            from io import StringIO
            buf = StringIO()
            np.savetxt(buf, aligned, delimiter="\t", fmt="%.6f")
            data_txt = buf.getvalue()

            if DECIMAL_COMMA_OUTPUT:
                # Replace decimal points with commas; keep 'nan' unchanged.
                # This is safe because tabs/newlines separate fields.
                data_txt = data_txt.replace(".", ",")

            f.write(data_txt)

    print("Saved aligned TXT files to:", out_folder)
# ----------------- Visualization helpers -----------------
def normalize_to_u8(img, mn, mx):
    denom = (mx - mn) + 1e-6
    out = (np.clip((img - mn) / denom, 0, 1) * 255.0).astype(np.uint8)
    return out


def make_overlay(prev_u8, curr_u8, alpha_red=RED_ALPHA):
    prev = prev_u8.astype(np.float32)
    curr = curr_u8.astype(np.float32)

    rgb = np.zeros((prev.shape[0], prev.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = prev
    rgb[..., 1] = prev
    rgb[..., 2] = prev

    rgb[..., 2] = (1.0 - alpha_red) * rgb[..., 2] + alpha_red * curr
    return np.clip(rgb, 0, 255).astype(np.uint8)


def draw_grid(img_bgr, spacing=50, color=(0, 255, 0), alpha=0.25):
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    for x in range(0, w, spacing):
        cv2.line(overlay, (x, 0), (x, h - 1), color, 1)
    for y in range(0, h, spacing):
        cv2.line(overlay, (0, y), (w - 1, y), color, 1)
    return cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)


def overlay_alpha_blend(ref_u8, cur_u8, alpha=0.5):
    ref = ref_u8.astype(np.float32)
    cur = cur_u8.astype(np.float32)
    out = (1 - alpha) * ref + alpha * cur
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def overlay_absdiff(ref_u8, cur_u8):
    diff = cv2.absdiff(ref_u8, cur_u8)
    return cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)


def overlay_checkerboard(ref_u8, cur_u8, tile=20):
    h, w = ref_u8.shape
    yy, xx = np.indices((h, w))
    mask = ((xx // tile + yy // tile) % 2).astype(np.uint8)
    out = ref_u8.copy()
    out[mask == 1] = cur_u8[mask == 1]
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def overlay_edges(ref_u8, cur_u8, ksize=3, edge_thresh=35, overlap_tol=12):
    """
    ref edges   = blue
    cur edges   = red
    overlap     = purple (magenta) ONLY if edges match closely

    edge_thresh: suppress weak/noisy edges
    overlap_tol: how close ref & cur edge magnitudes must be to count as "perfect overlap"
                 (smaller = stricter)
    """
    def sobel_mag(u8):
        # Optional: uncomment if edges are too noisy
        # u8 = cv2.GaussianBlur(u8, (3, 3), 0)

        gx = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=ksize)
        mag = cv2.magnitude(gx, gy)

        # Normalize per-frame to 0..255
        mag = np.clip(mag / (float(np.max(mag)) + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        return mag

    e_ref = sobel_mag(ref_u8)
    e_cur = sobel_mag(cur_u8)

    # Suppress weak edges (noise control)
    ref_strong = e_ref >= edge_thresh
    cur_strong = e_cur >= edge_thresh

    # "Perfect overlap" criterion: both strong AND similar magnitude
    overlap = ref_strong & cur_strong & (cv2.absdiff(e_ref, e_cur) <= overlap_tol)

    out = np.zeros((ref_u8.shape[0], ref_u8.shape[1], 3), dtype=np.uint8)

    # Blue = ref edges (only where NOT overlapping)
    out[..., 0] = np.where(ref_strong & ~overlap, e_ref, 0).astype(np.uint8)

    # Red = current edges (only where NOT overlapping)
    out[..., 2] = np.where(cur_strong & ~overlap, e_cur, 0).astype(np.uint8)

    # Purple (magenta) = overlap (set BOTH red and blue)
    ov_val = np.maximum(e_ref, e_cur)
    out[..., 0] = np.where(overlap, ov_val, out[..., 0]).astype(np.uint8)  # blue
    out[..., 2] = np.where(overlap, ov_val, out[..., 2]).astype(np.uint8)  # red

    return out



def render_overlap(mode, ref_u8, cur_u8, alpha_red=RED_ALPHA, blend_alpha=0.5, checker_tile = 20):
    """
    mode:
      0 = red overlay
      1 = alpha blend
      2 = abs diff
      3 = checkerboard
      4 = edges
    """
    if mode == 0:
        return make_overlay(ref_u8, cur_u8, alpha_red=alpha_red)
    if mode == 1:
        return overlay_alpha_blend(ref_u8, cur_u8, alpha=blend_alpha)
    if mode == 2:
        return overlay_absdiff(ref_u8, cur_u8)
    if mode == 3:
        return overlay_checkerboard(ref_u8, cur_u8, tile=checker_tile)
    if mode == 4:
        return overlay_edges(ref_u8, cur_u8, ksize=3)
    return make_overlay(ref_u8, cur_u8, alpha_red=alpha_red)

def tile_frames_grid(frames_u8, cols, spacing_px=80, bg=0):
    """
    frames_u8 : list of HxW uint8 images
    cols      : number of columns in the grid
    spacing_px: spacing between tiles (pixels)
    bg        : background value (0=black)
    """
    n = len(frames_u8)
    h, w = frames_u8[0].shape
    rows = int(np.ceil(n / cols))

    out_h = rows * h + (rows - 1) * spacing_px
    out_w = cols * w + (cols - 1) * spacing_px

    canvas = np.full((out_h, out_w), bg, dtype=np.uint8)

    for idx, im in enumerate(frames_u8):
        r = idx // cols
        c = idx % cols
        y = r * (h + spacing_px)
        x = c * (w + spacing_px)
        canvas[y:y+h, x:x+w] = im

    return canvas


def common_valid_crop(frames_float, min_valid_fraction=0.01):
    """Compute a single crop (y0,y1,x0,x1) valid for all frames (intersection of non-NaN pixels).
    Returns None if no reasonable intersection exists."""
    if not frames_float:
        return None
    h, w = frames_float[0].shape
    valid_all = np.ones((h, w), dtype=bool)
    for im in frames_float:
        valid_all &= ~np.isnan(im)
    if not np.any(valid_all):
        return None

    rows = np.where(np.any(valid_all, axis=1))[0]
    cols = np.where(np.any(valid_all, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None

    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    # Sanity: ensure crop keeps a meaningful portion of the image
    crop_area = (y1 - y0) * (x1 - x0)
    if crop_area < min_valid_fraction * (h * w):
        return None
    return y0, y1, x0, x1


def apply_crop(frames_float, crop):
    if crop is None:
        return frames_float
    y0, y1, x0, x1 = crop
    return [im[y0:y1, x0:x1] for im in frames_float]

def save_gif_from_frames(frames_float, out_gif_path, duration_ms=70):
    os.makedirs(os.path.dirname(out_gif_path), exist_ok=True)

    # Crop consistently across all frames using the intersection of valid (non-NaN) pixels.
    crop = common_valid_crop(frames_float)
    frames_float = apply_crop(frames_float, crop)

    global_min = min(float(np.nanmin(im)) for im in frames_float)
    global_max = max(float(np.nanmax(im)) for im in frames_float)

    # Build RGB PIL frames (grid baked in, same as preview)
    pil_rgb = []
    for im in frames_float:
        u8 = normalize_to_u8(np.nan_to_num(im, nan=global_min), global_min, global_max)
        bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        bgr = draw_grid(bgr, spacing=GRID_SPACING_PX, color=GRID_COLOR, alpha=GRID_ALPHA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_rgb.append(Image.fromarray(rgb))

    if not pil_rgb:
        raise ValueError("No frames to write GIF.")

    # Convert to paletted GIF frames.
    # Using a single global palette and disabling dithering prevents frame-to-frame palette remapping flicker
    # (a common cause of 'jumps' that are not geometric misalignment).
    if GIF_DITHER:
        dither_mode = Image.Dither.FLOYDSTEINBERG
    else:
        dither_mode = Image.Dither.NONE

    if GIF_PALETTE_GLOBAL:
        # Derive a global palette from a representative composite to stabilize color mapping.
        # We use the median frame index as a simple, robust representative.
        rep = pil_rgb[len(pil_rgb) // 2]
        pal_img = rep.quantize(colors=GIF_COLORS, method=Image.Quantize.MEDIANCUT, dither=dither_mode)
        pil_frames = [im.quantize(palette=pal_img, dither=dither_mode) for im in pil_rgb]
    else:
        # Per-frame palettes (can flicker for near-uniform images)
        pil_frames = [im.quantize(colors=GIF_COLORS, method=Image.Quantize.MEDIANCUT, dither=dither_mode) for im in pil_rgb]

    pil_frames[0].save(
        out_gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=GIF_DISPOSAL,
        optimize=False
    )





# ----------------- Warps (float) -----------------
def warp_affine_float(img, M2x3):
    h, w = img.shape
    M = M2x3.astype(np.float32)

    warped = cv2.warpAffine(
        img.astype(np.float32), M, (w, h),
        flags=INTERP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0
    )

    mask = cv2.warpAffine(
        np.ones((h, w), np.uint8), M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    warped = warped.astype(np.float32)
    warped[mask == 0] = BORDER_VALUE
    return warped


def delta_translate(dx, dy):
    return np.array([
        [1.0, 0.0, float(dx)],
        [0.0, 1.0, float(dy)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


def delta_rotate(theta_deg, shape_hw):
    h, w = shape_hw
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    R = cv2.getRotationMatrix2D((cx, cy), float(theta_deg), 1.0).astype(np.float32)
    R3 = np.eye(3, dtype=np.float32)
    R3[:2, :] = R
    return R3


def save_transform_csv(path, per_step_rows, cumulative_mats_3x3):
    """
    Columns:
      frame_idx, dx_to_prev, dy_to_prev, dtheta_to_prev,
      a00,a01,a02,a10,a11,a12  (cumulative 2x3 affine)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_idx",
            "dx_to_prev", "dy_to_prev", "dtheta_to_prev",
            "a00", "a01", "a02", "a10", "a11", "a12"
        ])
        for (idx, dx, dy, th) in per_step_rows:
            C = cumulative_mats_3x3[idx]
            A = C[:2, :]
            w.writerow([
                idx,
                f"{float(dx):.6f}", f"{float(dy):.6f}", f"{float(th):.6f}",
                f"{float(A[0,0]):.9f}", f"{float(A[0,1]):.9f}", f"{float(A[0,2]):.9f}",
                f"{float(A[1,0]):.9f}", f"{float(A[1,1]):.9f}", f"{float(A[1,2]):.9f}",
            ])


def save_transform_csv_full(path, n_frames, accepted_dx, accepted_dy, accepted_th, cumulative_mats_3x3):
    """
    Save a full correction table for all frame indices 0..n_frames-1.

    This preserves untouched rows when resuming from an existing CSV and only
    updates the frames edited in the current session.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_idx",
            "dx_to_prev", "dy_to_prev", "dtheta_to_prev",
            "a00", "a01", "a02", "a10", "a11", "a12"
        ])
        for idx in range(n_frames):
            dx = float(accepted_dx.get(idx, 0.0))
            dy = float(accepted_dy.get(idx, 0.0))
            th = float(accepted_th.get(idx, 0.0))
            C = cumulative_mats_3x3[idx]
            A = C[:2, :]
            w.writerow([
                idx,
                f"{dx:.6f}", f"{dy:.6f}", f"{th:.6f}",
                f"{float(A[0,0]):.9f}", f"{float(A[0,1]):.9f}", f"{float(A[0,2]):.9f}",
                f"{float(A[1,0]):.9f}", f"{float(A[1,1]):.9f}", f"{float(A[1,2]):.9f}",
            ])


def load_transform_csv(path):
    """
    Reads the CSV written by save_transform_csv and returns:
      accepted_dx, accepted_dy, accepted_th as dicts keyed by frame_idx (int).

    Expected columns:
      frame_idx, dx_to_prev, dy_to_prev, dtheta_to_prev, ...

    Missing files return empty dicts.
    """
    accepted_dx = {}
    accepted_dy = {}
    accepted_th = {}

    if not path or (not os.path.exists(path)):
        return accepted_dx, accepted_dy, accepted_th

    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                idx = int(row.get("frame_idx"))
                accepted_dx[idx] = float(row.get("dx_to_prev", 0.0))
                accepted_dy[idx] = float(row.get("dy_to_prev", 0.0))
                accepted_th[idx] = float(row.get("dtheta_to_prev", 0.0))
            except Exception:
                # Skip malformed rows
                continue

    return accepted_dx, accepted_dy, accepted_th


def build_cumulative_from_deltas(frames, accepted_dx, accepted_dy, accepted_th):
    """
    Rebuilds the cumulative 3x3 transforms C[k] from per-frame deltas (to previous frame).
    C[k] = D[k] @ C[k-1], where D[k] is delta_rotate(th[k]) @ delta_translate(dx[k], dy[k]).
    """
    n = len(frames)
    C = [np.eye(3, dtype=np.float32) for _ in range(n)]
    for k in range(START_INDEX + 1, n):
        dx = float(accepted_dx.get(k, 0.0))
        dy = float(accepted_dy.get(k, 0.0))
        th = float(accepted_th.get(k, 0.0))

        T3 = delta_translate(dx, dy)
        R3 = delta_rotate(th, frames[k].shape)
        D3 = R3 @ T3
        C[k] = D3 @ C[k - 1]
    return C


# ----------------- Step: Manual XY + rotation -----------------
def manual_xytheta_alignment(frames, stage_folder, initial_dx=None, initial_dy=None, initial_th=None, edit_start_index=None):
    """
    Manual correction: dx, dy, theta (rotation around image center).
    Chaining uses previous accepted cumulative transform.

    Supports:
      - loading existing CSV corrections
      - resuming at any frame index safely
      - preserving all previously loaded rows when saving later
      - going back one frame with 'b'
    """
    n = len(frames)
    global_min = min(float(np.nanmin(im)) for im in frames)
    global_max = max(float(np.nanmax(im)) for im in frames)

    preview_folder = os.path.join(stage_folder, "previews_png")
    os.makedirs(preview_folder, exist_ok=True)

    win = "XY+ROT ALIGN (prev=GRAY, current=RED)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    show_grid = GRID_ENABLED_DEFAULT

    overlay_mode = 0
    overlay_names = ["red", "blend", "diff", "checker", "edges"]
    blend_alpha = 0.50
    checker_tile = 20

    C = [np.eye(3, dtype=np.float32) for _ in range(n)]

    # reference is the first frame (unwarped), to keep consistent display
    ref_aligned = frames[START_INDEX].astype(np.float32)

    # full-length preview list so resuming from any frame index is safe
    aligned_preview = [None] * n
    aligned_preview[START_INDEX] = ref_aligned.copy()

    # store per-frame accepted deltas (indexed by frame)
    accepted_dx = {START_INDEX: 0.0}
    accepted_dy = {START_INDEX: 0.0}
    accepted_th = {START_INDEX: 0.0}

    # If provided, preload an existing correction file (per-frame deltas)
    if initial_dx:
        accepted_dx.update({int(k): float(v) for k, v in initial_dx.items()})
    if initial_dy:
        accepted_dy.update({int(k): float(v) for k, v in initial_dy.items()})
    if initial_th:
        accepted_th.update({int(k): float(v) for k, v in initial_th.items()})

    # If we loaded deltas, rebuild cumulative transforms so editing can resume mid-sequence
    if initial_dx or initial_dy or initial_th:
        C[:] = build_cumulative_from_deltas(frames, accepted_dx, accepted_dy, accepted_th)

    use_first_ref = True

    Image.fromarray(normalize_to_u8(ref_aligned, global_min, global_max)).save(
        os.path.join(preview_folder, f"frame_{START_INDEX:04d}.png")
    )

    fast_mode = False
    step_xy = BASE_STEP_XY
    step_rot = BASE_STEP_ROT

    if edit_start_index is None:
        i = START_INDEX + 1
    else:
        i = max(START_INDEX + 1, int(edit_start_index))
        if i >= n:
            i = n - 1

    # Rebuild previews up to the resume point so aligned_preview[i-1] always exists.
    if i > START_INDEX + 1:
        for k in range(START_INDEX + 1, i):
            curr_k = frames[k].astype(np.float32)
            curr_warp_k = warp_affine_float(curr_k, C[k][:2, :])
            aligned_preview[k] = curr_warp_k

    last_dx = float(accepted_dx.get(i - 1, 0.0)) if i > START_INDEX else 0.0
    last_dy = float(accepted_dy.get(i - 1, 0.0)) if i > START_INDEX else 0.0
    last_theta = float(accepted_th.get(i - 1, 0.0)) if i > START_INDEX else 0.0

    while i < n:
        curr = frames[i].astype(np.float32)

        # If we already accepted this frame previously, start from stored values.
        dx = float(accepted_dx.get(i, last_dx))
        dy = float(accepted_dy.get(i, last_dy))
        theta = float(accepted_th.get(i, last_theta))

        # Base transform is always the previous frame's accepted transform.
        C_base = C[i - 1].copy() if i > START_INDEX else C[START_INDEX].copy()

        # Editable per-frame totals (loaded if previously accepted).
        dx_total = float(accepted_dx.get(i, dx))
        dy_total = float(accepted_dy.get(i, dy))
        th_total = float(accepted_th.get(i, theta))

        while True:
            # translate then rotate (about center)
            T3 = delta_translate(dx_total, dy_total)
            R3 = delta_rotate(th_total, curr.shape)
            D3 = R3 @ T3

            C_candidate = D3 @ C_base
            curr_warp = warp_affine_float(curr, C_candidate[:2, :])

            # AUTOSAVE: store current totals + transform for this frame on every redraw
            C[i] = C_candidate
            accepted_dx[i] = float(dx_total)
            accepted_dy[i] = float(dy_total)
            accepted_th[i] = float(th_total)
            aligned_preview[i] = curr_warp

            prev_aligned = aligned_preview[i - 1]
            if prev_aligned is None:
                prev_aligned = warp_affine_float(frames[i - 1].astype(np.float32), C[i - 1][:2, :])
                aligned_preview[i - 1] = prev_aligned

            ref_for_overlay = ref_aligned if use_first_ref else prev_aligned

            prev_u8 = normalize_to_u8(ref_for_overlay, global_min, global_max)
            curr_u8 = normalize_to_u8(np.nan_to_num(curr_warp, nan=global_min), global_min, global_max)

            overlay = render_overlap(
                overlay_mode, prev_u8, curr_u8,
                alpha_red=RED_ALPHA, blend_alpha=blend_alpha, checker_tile=checker_tile
            )

            hud = overlay.copy()
            if show_grid:
                hud = draw_grid(hud, GRID_SPACING_PX, GRID_COLOR, GRID_ALPHA)

            cv2.putText(
                hud,
                f"Frame {i}/{n-1} | ref={'FIRST' if use_first_ref else 'PREV'} | dx={dx_total:+.2f}px dy={dy_total:+.2f}px th={th_total:+.3f}deg | step_xy={step_xy}px | step_rot={step_rot}deg | fast={fast_mode}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                hud,
                "Arrows/WASD move | Q/E rotate | b back | r ref | f fast | g grid | m mode | [ ] tile | , . alpha | 0 reset | ENTER next | ESC quit+save",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )
            cv2.putText(
                hud,
                f"mode={overlay_names[overlay_mode]} | tile={checker_tile}px | alpha={blend_alpha:.2f}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )

            cv2.imshow(win, hud)
            key = cv2.waitKeyEx(0)

            # ---- Next frame ----
            if key in (13, 10):  # ENTER
                last_dx, last_dy, last_theta = float(dx_total), float(dy_total), float(th_total)

                Image.fromarray(
                    normalize_to_u8(np.nan_to_num(curr_warp, nan=global_min), global_min, global_max)
                ).save(os.path.join(preview_folder, f"frame_{i:04d}.png"))

                print(f"Saved frame {i}: dx={dx_total:.2f}, dy={dy_total:.2f}, theta={th_total:.3f}")
                i += 1
                break

            # ---- Abort ----
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                last_valid = max(k for k, v in enumerate(aligned_preview) if v is not None)
                valid_preview = aligned_preview[:last_valid + 1]
                return C, valid_preview, accepted_dx, accepted_dy, accepted_th

            # ---- Go back one frame ----
            if key == ord('b'):
                if i <= START_INDEX + 1:
                    continue
                i -= 1
                break

            # ---- Resets / toggles ----
            if key == ord('0'):
                dx_total, dy_total, th_total = 0.0, 0.0, 0.0
                continue
            if key == ord('f'):
                fast_mode = not fast_mode
                step_xy = BASE_STEP_XY * (FAST_MULT if fast_mode else 1)
                step_rot = BASE_STEP_ROT * (FAST_MULT if fast_mode else 1)
                continue
            if key == ord('r'):
                use_first_ref = not use_first_ref
                continue
            if key == ord('g'):
                show_grid = not show_grid
                continue
            if key == ord('m'):
                overlay_mode = (overlay_mode + 1) % len(overlay_names)
                continue
            if key == ord('['):
                checker_tile = max(4, checker_tile - 4)
                continue
            if key == ord(']'):
                checker_tile = min(256, checker_tile + 4)
                continue
            if key == ord(','):
                blend_alpha = max(0.0, blend_alpha - 0.05)
                continue
            if key == ord('.'):
                blend_alpha = min(1.0, blend_alpha + 0.05)
                continue

            # ---- Adjustments ----
            if key in (2424832, ord('a')):       # left
                dx_total -= step_xy
            elif key in (2555904, ord('d')):     # right
                dx_total += step_xy
            elif key in (2490368, ord('w')):     # up
                dy_total -= step_xy
            elif key in (2621440, ord('s')):     # down
                dy_total += step_xy
            elif key == ord('q'):                # rotate -
                th_total -= step_rot
            elif key == ord('e'):                # rotate +
                th_total += step_rot

    cv2.destroyAllWindows()
    last_valid = max(k for k, v in enumerate(aligned_preview) if v is not None)
    return C, aligned_preview[:last_valid + 1], accepted_dx, accepted_dy, accepted_th

def main():
    stage_folder = os.path.join(ALIGN_ROOT, "xytheta_correction")
    os.makedirs(stage_folder, exist_ok=True)

    files, headers, frames = load_txt_stack()
    n = len(frames)
    H, W = frames[0].shape
    print(f"Loaded {n} frames, size={W}x{H}")
    print("Stage folder:", stage_folder)

    print("TXT input folder:", TXT_INPUT_FOLDER)
    print("TXT output folder:", TXT_OUTPUT_FOLDER)
    default_csv_path = os.path.join(stage_folder, "xytheta_correction.csv")

    use_existing = input(f"Load existing correction CSV? (y/n) [default: {default_csv_path}]: ").strip().lower() == "y"
    initial_dx = initial_dy = initial_th = None
    edit_start_index = None

    if use_existing:
        path_in = input("Path to CSV (ENTER for default): ").strip()
        if not path_in:
            path_in = default_csv_path
        initial_dx, initial_dy, initial_th = load_transform_csv(path_in)

        s = input("Start editing from which frame index? (ENTER = start from 1): ").strip()
        if s:
            try:
                edit_start_index = int(s)
            except Exception:
                edit_start_index = None

    C_3x3, aligned_preview, accepted_dx, accepted_dy, accepted_th = manual_xytheta_alignment(
        frames,
        stage_folder,
        initial_dx=initial_dx,
        initial_dy=initial_dy,
        initial_th=initial_th,
        edit_start_index=edit_start_index,
    )

    csv_path = os.path.join(stage_folder, "xytheta_correction.csv")
    save_transform_csv_full(csv_path, len(frames), accepted_dx, accepted_dy, accepted_th, C_3x3)
    print("Saved CSV:", csv_path)

    gif_path = os.path.join(stage_folder, "xytheta_preview.gif")
    save_gif_from_frames(aligned_preview, gif_path, duration_ms=70)
    print("Saved GIF:", gif_path)


    # Export aligned TXT frames (data corrected using the final transforms)
    save_aligned_txts(files, headers, frames, C_3x3, TXT_OUTPUT_FOLDER)
    print("DONE.")
    print("If you still want an additional rotation-only refinement stage, update the rotation script to read xytheta_correction.csv.")
    print("Otherwise, proceed directly to crop/apply using the final transform CSV you choose.")


if __name__ == "__main__":
    main()
