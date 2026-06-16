"""Correction defining points.

Commented version with added helper docstrings to make the workflow easier to follow.
"""

import os
import json
import glob
import re
import shutil
import argparse
from datetime import datetime
from io import StringIO

import numpy as np
import cv2

try:
    import laspy
except Exception:
    laspy = None

try:
    import open3d as o3d
except Exception:
    o3d = None


# ==========================================================
# Defaults (can all be overridden by CLI)
# ==========================================================
PC_LAS_PATH = r"F:\Masterarbeit_Backup_2\Data\TLS\Paradiestal.las"
TXT_ALIGNED_FOLDER = r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted_aligned\Paradiestal_12-13.08.25"
OUT_JSON = r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted_aligned_angled\Paradiestal_12-13.08.25\picked_correspondences.json"

PREFER_NOON_FRAME = True
DEFAULT_DELETE_RADIUS_PX = 15
DEFAULT_ZOOM = 10
DEFAULT_PATCH_RADIUS = 15

SCRIPT_VERSION = "2026-03-17"


# ==========================================================
# Helpers
# ==========================================================
def ensure_dir(path: str) -> None:
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from path."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def natural_sort_key(path: str):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from path."""
    parts = re.split(r"(\d+)", os.path.basename(path))
    out = []
    for p in parts:
        out.append(int(p) if p.isdigit() else p.lower())
    return out


def list_txt(folder: str):
    """Return the available input files in deterministic processing order. Inputs are taken from folder."""
    files = sorted(glob.glob(os.path.join(folder, "*.txt")), key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {folder}")
    return files


def _noon_score_from_name(name: str) -> int:
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from name."""
    s = name.lower()
    score = 0
    if "noon" in s:
        score += 50
    patterns = [
        r"12[: ]00", r"12[-_]00", r"12[.]00", r"12h00", r"\b1200\b", r"12uhr", r"\b12h\b"
    ]
    for p in patterns:
        if re.search(p, s):
            score += 100
    m = re.search(r"\b(\d{1,2})[:h._-]?(\d{2})\b", s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        dist = abs((hh * 60 + mm) - 12 * 60)
        score += max(0, 60 - dist // 5)
    return score


def choose_frame(files, preferred_frame=None, prefer_noon=True):
    """Choose the most suitable input item based on the configured rules. Inputs are taken from files, preferred_frame, prefer_noon."""
    if preferred_frame:
        if os.path.isfile(preferred_frame):
            return preferred_frame, f"explicit file: {preferred_frame}"
        candidates = [f for f in files if preferred_frame.lower() in os.path.basename(f).lower()]
        if not candidates:
            raise FileNotFoundError(f"No TXT frame matches --frame={preferred_frame!r}")
        return candidates[0], f"matched --frame substring: {preferred_frame}"

    if prefer_noon:
        best = max(files, key=lambda p: _noon_score_from_name(os.path.basename(p)))
        score = _noon_score_from_name(os.path.basename(best))
        if score >= 100:
            return best, f"best noon-like frame (score={score})"
    return files[0], "fallback to first TXT file"


def read_thermal_txt(path: str) -> np.ndarray:
    """Read data from disk and return it in a parsed NumPy/Python structure. Inputs are taken from path."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    idx = next(i for i, ln in enumerate(lines) if ln.strip() == "[Data]")
    clean = "".join(ln.replace(",", ".") for ln in lines[idx + 1:])
    return np.loadtxt(StringIO(clean), delimiter="\t", dtype=np.float32)


def robust_u8(img_float: np.ndarray) -> np.ndarray:
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from img_float."""
    v = img_float.astype(np.float32)
    finite = np.isfinite(v)
    if not finite.any():
        return np.zeros(v.shape, dtype=np.uint8)
    v = np.nan_to_num(v, nan=np.nanmin(v[finite]))
    lo, hi = np.percentile(v[finite], 2), np.percentile(v[finite], 98)
    if hi <= lo:
        hi = lo + 1.0
    u8 = np.clip((v - lo) / (hi - lo), 0, 1)
    return (u8 * 255).astype(np.uint8)


def to_inferno_bgr(img_u8: np.ndarray) -> np.ndarray:
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from img_u8."""
    return cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)


def load_las_as_o3d_pointcloud(path: str):
    """Load and validate external data needed later in the workflow. Inputs are taken from path."""
    if laspy is None:
        raise RuntimeError("laspy is not installed. Run: pip install laspy")
    if o3d is None:
        raise RuntimeError("open3d is not installed. Run: pip install open3d")
    las = laspy.read(path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def pick_one_point_3d(pcd, title_suffix="", saved_view_params=None):
    """Open the interactive picker and return the selected element. Inputs are taken from pcd, title_suffix, saved_view_params."""
    if o3d is None:
        return None, None, None

    print("Pick ONE 3D point:")
    print("  Shift + Left click   -> pick point")
    print("  Shift + Right click  -> undo")
    print("  Q                    -> accept/close")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name=f"Pick ONE 3D point {title_suffix} (Shift+LMB, then Q)",
        width=1280,
        height=720,
    )
    pcd_vis = o3d.geometry.PointCloud(pcd)
    pcd_vis.paint_uniform_color([0.6, 0.6, 0.6])
    vis.add_geometry(pcd_vis)

    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.light_on = False

    if saved_view_params is not None:
        try:
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(saved_view_params, allow_arbitrary=True)
        except Exception as e:
            print(f"Warning: could not restore previous 3D view: {e}")

    vis.run()

    new_view_params = None
    try:
        ctr = vis.get_view_control()
        new_view_params = ctr.convert_to_pinhole_camera_parameters()
    except Exception as e:
        print(f"Warning: could not save current 3D view: {e}")

    idx = vis.get_picked_points()
    vis.destroy_window()

    if not idx:
        return None, None, new_view_params

    index = int(idx[-1])
    xyz = np.asarray(pcd.points)[index].astype(np.float64)
    return xyz, index, new_view_params


def timestamp_now():
    """Return the helper result used by the surrounding processing pipeline."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def backup_file(path: str):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from path."""
    if not os.path.exists(path):
        return None
    stem, ext = os.path.splitext(path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{stem}.backup_{ts}{ext or '.json'}"
    shutil.copy2(path, bak)
    return bak


def normalize_existing_json(d: dict):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from d."""
    pts3d = d.get("points_3d_xyz", [])
    pts2d = d.get("points_2d_xy", [])
    idx3d = d.get("picked_indices_3d", [])

    if idx3d:
        n = min(len(pts3d), len(pts2d), len(idx3d))
        idx3d = idx3d[:n]
    else:
        n = min(len(pts3d), len(pts2d))
        idx3d = [None] * n

    pts3d = [list(map(float, p)) for p in pts3d[:n]]
    pts2d = [list(map(float, p)) for p in pts2d[:n]]
    idx3d = [None if v is None else int(v) for v in idx3d]
    return pts3d, idx3d, pts2d


def load_existing_json(json_path: str):
    """Load and validate external data needed later in the workflow. Inputs are taken from json_path."""
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    pts3d, idx3d, pts2d = normalize_existing_json(d)
    return {
        "raw": d,
        "pts3d": pts3d,
        "idx3d": idx3d,
        "pts2d": pts2d,
        "pointcloud_path": d.get("pointcloud_path", ""),
        "thermal_txt_path": d.get("thermal_txt_path", ""),
    }


def build_output_json(existing_raw, pointcloud_path, thermal_txt_path, idx3d, pts3d, pts2d, args, frame_reason):
    """Build the helper structure required by the downstream workflow. Inputs are taken from existing_raw, pointcloud_path, thermal_txt_path and related parameters."""
    out = dict(existing_raw or {})
    out["schema_version"] = 2
    out["pointcloud_path"] = pointcloud_path
    out["thermal_txt_path"] = thermal_txt_path
    out["picked_indices_3d"] = [None if v is None else int(v) for v in idx3d]
    out["points_3d_xyz"] = [list(map(float, p)) for p in pts3d]
    out["points_2d_xy"] = [list(map(float, p)) for p in pts2d]
    out["note"] = "Paired point-by-point: points_3d_xyz[k] <-> points_2d_xy[k]"

    history = list(out.get("save_history", []))
    history.append({
        "timestamp": timestamp_now(),
        "script": os.path.basename(__file__),
        "script_version": SCRIPT_VERSION,
        "num_pairs": len(pts2d),
        "frame_selection_reason": frame_reason,
        "delete_radius_px": int(args.delete_radius),
        "autosave": bool(args.autosave),
    })
    out["save_history"] = history[-20:]

    meta = dict(out.get("picker_metadata", {}))
    meta.update({
        "script": os.path.basename(__file__),
        "script_version": SCRIPT_VERSION,
        "last_saved": timestamp_now(),
        "frame_selection_reason": frame_reason,
        "delete_radius_px": int(args.delete_radius),
        "zoom_factor": int(args.zoom),
        "zoom_patch_radius": int(args.zoom_patch_radius),
    })
    out["picker_metadata"] = meta
    return out


def save_json(json_path: str, existing_raw, pointcloud_path, thermal_txt_path, idx3d, pts3d, pts2d, args, frame_reason, make_backup=False):
    """Save the current result to disk and preserve the expected schema. Inputs are taken from json_path, existing_raw, pointcloud_path and related parameters."""
    ensure_dir(json_path)
    backup_path = backup_file(json_path) if make_backup else None
    out = build_output_json(existing_raw, pointcloud_path, thermal_txt_path, idx3d, pts3d, pts2d, args, frame_reason)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {json_path}")
    if backup_path:
        print(f"Backup: {backup_path}")
    return backup_path


def add_zoom_inset(vis, cursor, patch_radius=15, zoom=10):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from vis, cursor, patch_radius and related parameters."""
    if cursor is None:
        return vis
    x, y = int(cursor[0]), int(cursor[1])
    h, w = vis.shape[:2]
    r = max(4, int(patch_radius))
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    patch = vis[y0:y1, x0:x1]
    if patch.size == 0:
        return vis

    zoomed = cv2.resize(patch, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    zh, zw = zoomed.shape[:2]
    pad = 8
    tlx = max(0, w - zw - pad)
    tly = pad
    brx = min(w, tlx + zw)
    bry = min(h, tly + zh)
    zoomed = zoomed[:bry - tly, :brx - tlx]

    cv2.rectangle(vis, (tlx - 2, tly - 2), (brx + 2, bry + 2), (255, 255, 255), 1)
    vis[tly:bry, tlx:brx] = zoomed

    cx = tlx + (brx - tlx) // 2
    cy = tly + (bry - tly) // 2
    cv2.line(vis, (cx - 12, cy), (cx + 12, cy), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(vis, (cx, cy - 12), (cx, cy + 12), (255, 255, 255), 1, cv2.LINE_AA)
    label = f"zoom x{zoom} @ ({x},{y})"
    cv2.putText(vis, label, (max(0, tlx - 1), min(h - 4, bry + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_hud(img_bgr, num_pairs, pending_3d, delete_radius_px, frame_name, autosave_on, last_save_msg, pending_2d=None):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from img_bgr, num_pairs, pending_3d and related parameters."""
    overlay = img_bgr.copy()
    lines = [
        "INTERACTIVE POINT PAIR PICKER",
        f"Frame: {frame_name}",
        f"Pairs: {num_pairs}",
        f"Pending 3D: {'YES' if pending_3d else 'NO (press n to pick 3D)'}",
        f"Autosave: {'ON' if autosave_on else 'OFF'}",
    ]
    if pending_2d is not None:
        lines.append(f"Pending 2D: ({pending_2d[0]:.1f}, {pending_2d[1]:.1f})")
    lines += [
        "",
        "Mouse:",
        "  LMB: pick 2D for pending 3D",
        "  RMB: delete nearest pair",
        "",
        "Keys:",
        "  n: pick NEW 3D point (Open3D)",
        "  s: save JSON",
        "  u: undo last pair",
        "  r: remove by index (console prompt)",
        "  c: clear all (type YES)",
        "  h: toggle legend",
        "  arrows: nudge last 2D point by 1 px",
        "  Shift+arrows: nudge last 2D point by 5 px",
        "  q / ESC: quit",
        "",
        f"Delete radius: {delete_radius_px}px",
    ]
    if last_save_msg:
        lines += ["", last_save_msg]

    x0, y0 = 10, 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.48
    th = 1
    pad = 6
    line_h = 18
    box_w = 620
    box_h = pad * 2 + line_h * len(lines)

    cv2.rectangle(overlay, (x0 - 6, y0 - 6), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, img_bgr, 0.45, 0)

    y = y0 + pad + 12
    for ln in lines:
        cv2.putText(out, ln, (x0, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)
        y += line_h
    return out


def run_interactive_pairing(img_bgr, pcd, json_path, pointcloud_path, thermal_txt_path,
                            args, frame_reason, existing_raw=None,
                            initial_pts3d=None, initial_idx3d=None, initial_pts2d=None):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from img_bgr, pcd, json_path and related parameters."""
    pts3d = list(initial_pts3d) if initial_pts3d else []
    idx3d = list(initial_idx3d) if initial_idx3d else []
    pts2d = list(initial_pts2d) if initial_pts2d else []

    pending_3d = None
    pending_idx = None
    saved_view_params = None
    show_hud = True
    cursor = None
    last_save_msg = ""

    win = "IRT aligned (2D) - interactive"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    frame_name = os.path.basename(thermal_txt_path)

    def autosave(reason):
        """Return the helper result used by the surrounding processing pipeline. Inputs are taken from reason."""
        nonlocal last_save_msg
        backup = save_json(
            json_path=json_path,
            existing_raw=existing_raw,
            pointcloud_path=pointcloud_path,
            thermal_txt_path=thermal_txt_path,
            idx3d=idx3d,
            pts3d=pts3d,
            pts2d=pts2d,
            args=args,
            frame_reason=frame_reason,
            make_backup=True,
        )
        last_save_msg = f"Saved ({reason}) @ {datetime.now().strftime('%H:%M:%S')}"
        if backup:
            last_save_msg += " + backup"

    def redraw():
        """Return the helper result used by the surrounding processing pipeline."""
        vis = img_bgr.copy()

        for i, (x, y) in enumerate(pts2d):
            center = (int(round(x)), int(round(y)))
            cv2.circle(vis, center, 4, (0, 255, 0), -1)
            cv2.putText(vis, str(i), (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        if cursor is not None:
            cv2.drawMarker(vis, (int(cursor[0]), int(cursor[1])), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

        if pending_3d is not None:
            cv2.putText(vis, "Pending 3D selected -> click matching 2D point", (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        vis = add_zoom_inset(vis, cursor, patch_radius=args.zoom_patch_radius, zoom=args.zoom)

        if show_hud:
            vis = draw_hud(
                vis, len(pts2d), pending_3d is not None, args.delete_radius,
                frame_name, args.autosave, last_save_msg,
                pending_2d=pts2d[-1] if pts2d else None,
            )
        cv2.imshow(win, vis)

    def nearest_pair_index(x, y):
        """Return the helper result used by the surrounding processing pipeline. Inputs are taken from x, y."""
        if not pts2d:
            return None, None
        arr = np.asarray(pts2d, dtype=np.float64)
        d = np.sqrt((arr[:, 0] - x) ** 2 + (arr[:, 1] - y) ** 2)
        j = int(np.argmin(d))
        return j, float(d[j])

    def on_mouse(event, x, y, flags, param):
        """Return the helper result used by the surrounding processing pipeline. Inputs are taken from event, x, y and related parameters."""
        nonlocal pending_3d, pending_idx, cursor
        cursor = (x, y)
        if event == cv2.EVENT_MOUSEMOVE:
            redraw()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if pending_3d is None:
                print("No pending 3D point. Press 'n' first to pick a 3D point.")
                redraw()
                return
            pts3d.append(pending_3d.tolist())
            idx3d.append(None if pending_idx is None else int(pending_idx))
            pts2d.append([float(x), float(y)])
            k = len(pts2d) - 1
            print(f"Paired k={k}: 3D idx={pending_idx} xyz={pending_3d} <-> 2D uv=({x},{y})")
            pending_3d, pending_idx = None, None
            if args.autosave:
                autosave(reason=f"pair #{k} added")
            redraw()
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            j, dj = nearest_pair_index(float(x), float(y))
            if j is None:
                redraw()
                return
            if dj <= float(args.delete_radius):
                print(f"Deleted pair #{j} (distance {dj:.2f}px)")
                pts2d.pop(j)
                pts3d.pop(j)
                idx3d.pop(j)
                if args.autosave:
                    autosave(reason=f"pair #{j} deleted")
            else:
                print(f"Nearest pair is #{j} but distance {dj:.2f}px > {args.delete_radius}px (not deleted)")
            redraw()

    cv2.setMouseCallback(win, on_mouse)
    redraw()

    keymap = {
        2424832: (-1, 0),  # left
        2555904: (1, 0),   # right
        2490368: (0, -1),  # up
        2621440: (0, 1),   # down
    }
    shifted_keymap = {
        2162688: (-5, 0),
        2555904 + (1 << 16): (5, 0),
        2359296: (0, -5),
        2490368 + (1 << 16): (0, 5),
    }

    while True:
        key = cv2.waitKeyEx(20)
        if key == -1:
            continue

        if key in (ord('q'), 27):
            break
        if key == ord('h'):
            show_hud = not show_hud
            redraw()
            continue
        if key == ord('n'):
            if pcd is None:
                print("Point cloud not loaded; cannot pick 3D.")
                continue
            xyz, idx, saved_view_params = pick_one_point_3d(
                pcd,
                title_suffix=f"[next k={len(pts2d)}]",
                saved_view_params=saved_view_params,
            )
            if xyz is None:
                print("No 3D point picked.")
                continue
            pending_3d, pending_idx = xyz, idx
            print(f"Pending 3D set: idx={idx} xyz={xyz}. Now click the matching 2D point in the IRT window.")
            redraw()
            continue
        if key == ord('s'):
            autosave(reason="manual save")
            redraw()
            continue
        if key == ord('u'):
            if pts2d:
                pts2d.pop()
                pts3d.pop()
                idx3d.pop()
                pending_3d, pending_idx = None, None
                print("Undo last pair.")
                if args.autosave:
                    autosave(reason="undo")
                redraw()
            continue
        if key == ord('r'):
            if not pts2d:
                print("No points to remove.")
                continue
            s = input("Remove which index? ").strip()
            try:
                j = int(s)
                if j < 0 or j >= len(pts2d):
                    raise ValueError
                pts2d.pop(j)
                pts3d.pop(j)
                idx3d.pop(j)
                print(f"Removed pair #{j}.")
                if args.autosave:
                    autosave(reason=f"pair #{j} removed")
                redraw()
            except Exception:
                print("Invalid index.")
            continue
        if key == ord('c'):
            s = input("Type YES to clear ALL pairs: ").strip()
            if s == "YES":
                pts2d.clear()
                pts3d.clear()
                idx3d.clear()
                pending_3d, pending_idx = None, None
                print("Cleared all pairs.")
                if args.autosave:
                    autosave(reason="clear all")
                redraw()
            continue

        step = None
        if key in keymap:
            step = keymap[key]
        elif key in shifted_keymap:
            step = shifted_keymap[key]
        if step is not None and pts2d:
            pts2d[-1][0] += step[0]
            pts2d[-1][1] += step[1]
            print(f"Adjusted last point to ({pts2d[-1][0]:.1f}, {pts2d[-1][1]:.1f})")
            if args.autosave:
                autosave(reason="nudge")
            redraw()
            continue

    cv2.destroyWindow(win)
    return pts3d, idx3d, pts2d


def parse_args():
    """Parse the available metadata into the normalized internal representation."""
    ap = argparse.ArgumentParser(description="Interactive alignment picker: point cloud <-> aligned IRT image")
    ap.add_argument("--pointcloud", default=PC_LAS_PATH, help="LAS/LAZ point cloud path")
    ap.add_argument("--aligned_folder", default=TXT_ALIGNED_FOLDER, help="Folder with aligned thermal TXT frames")
    ap.add_argument("--out_json", default=OUT_JSON, help="Output JSON for correspondences")
    ap.add_argument("--frame", default=None, help="Specific TXT frame path or substring to use")
    ap.add_argument("--prefer_noon", action="store_true", default=PREFER_NOON_FRAME, help="Prefer a noon-like frame name")
    ap.add_argument("--no_prefer_noon", action="store_false", dest="prefer_noon", help="Disable noon preference")
    ap.add_argument("--delete_radius", type=int, default=DEFAULT_DELETE_RADIUS_PX, help="Delete radius in pixels for RMB")
    ap.add_argument("--autosave", action="store_true", default=True, help="Autosave after add/delete/nudge")
    ap.add_argument("--no_autosave", action="store_false", dest="autosave", help="Disable autosave")
    ap.add_argument("--zoom", type=int, default=DEFAULT_ZOOM, help="Inset zoom factor")
    ap.add_argument("--zoom_patch_radius", type=int, default=DEFAULT_PATCH_RADIUS, help="Half-size of zoom patch in px")
    return ap.parse_args()


def main():
    """Parse CLI/config inputs and run the complete workflow for this script."""
    args = parse_args()

    print("Loading point cloud...")
    pcd = load_las_as_o3d_pointcloud(args.pointcloud)

    files = list_txt(args.aligned_folder)
    txt0, frame_reason = choose_frame(files, preferred_frame=args.frame, prefer_noon=args.prefer_noon)
    print(f"Selected frame: {txt0}")
    print(f"Reason: {frame_reason}")

    thermal0 = read_thermal_txt(txt0)
    img_u8 = robust_u8(thermal0)
    img_bgr = to_inferno_bgr(img_u8)

    existing = load_existing_json(args.out_json)
    if existing is not None:
        print(f"Found existing JSON: {args.out_json}")
        if existing.get("pointcloud_path") and os.path.normpath(existing["pointcloud_path"]) != os.path.normpath(args.pointcloud):
            print("WARNING: JSON pointcloud_path differs from current --pointcloud")
            print("  JSON:", existing["pointcloud_path"])
            print("  NOW :", args.pointcloud)
        if existing.get("thermal_txt_path") and os.path.normpath(existing["thermal_txt_path"]) != os.path.normpath(txt0):
            print("NOTE: JSON thermal_txt_path differs from selected frame")
            print("  JSON:", existing["thermal_txt_path"])
            print("  NOW :", txt0)
        pts3d0 = existing["pts3d"]
        idx3d0 = existing["idx3d"]
        pts2d0 = existing["pts2d"]
        existing_raw = existing["raw"]
    else:
        print("No existing JSON found; starting fresh.")
        pts3d0, idx3d0, pts2d0, existing_raw = [], [], [], {}

    pts3d, idx3d, pts2d = run_interactive_pairing(
        img_bgr=img_bgr,
        pcd=pcd,
        json_path=args.out_json,
        pointcloud_path=args.pointcloud,
        thermal_txt_path=txt0,
        args=args,
        frame_reason=frame_reason,
        existing_raw=existing_raw,
        initial_pts3d=pts3d0,
        initial_idx3d=idx3d0,
        initial_pts2d=pts2d0,
    )

    if len(pts3d) == 0:
        raise SystemExit("No point pairs collected.")

    save_json(
        json_path=args.out_json,
        existing_raw=existing_raw,
        pointcloud_path=args.pointcloud,
        thermal_txt_path=txt0,
        idx3d=idx3d,
        pts3d=pts3d,
        pts2d=pts2d,
        args=args,
        frame_reason=frame_reason,
        make_backup=True,
    )
    print(f"Done. Final pair count: {len(pts2d)}")


if __name__ == "__main__":
    main()