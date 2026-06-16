"""
Angle-based IRT correction for ALL aligned frames.

This version keeps plotting separate and performs the correction inside
the correction script only.

Main correction used
--------------------
    T_corr = T_rad - |T_rad - T_sky| * k(theta)

where
- k(theta) comes from the geology CSV
- T_sky is derived from EBG data for each 10-minute interval

Sky-temperature handling
------------------------
Two modes are available:

1) effective_longwave
   T_sky is derived directly from downward longwave radiation Ld
   using Stefan-Boltzmann:
       T_sky = (Ld / sigma)**0.25 - 273.15

2) apparent_proxy   [DEFAULT]
   Uses the same EBG longwave data, but applies an empirical offset
   based on atmospheric emissivity / sky clarity to approximate the
   colder LWIR apparent sky temperatures typically seen by an IR camera
   under clear nighttime conditions.

   This is intended to make best use of the EBG variables you currently have.

Frame ↔ EBG matching
--------------------
- frame timestamps are parsed from header if possible
- otherwise from filenames like 2025-08-25_1120.txt
- EBG rows are cropped to the actual measurement time window
- nearest-timestamp matching is then used

Notes
-----
- Pixels without TLS support are left unchanged.
- Plotting is not performed here.
"""

import os
import re
import csv
import json
import glob
import argparse
from io import StringIO

import numpy as np
import cv2

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import laspy
except Exception:
    laspy = None

try:
    import pandas as pd
except Exception:
    pd = None


FULL_W = 1280
FULL_H = 960

DEFAULT_FOV_X = 30
DEFAULT_FOV_Y = 23

DEFAULT_RANSAC_REPROJ = 6.0
DEFAULT_RANSAC_ITERS = 5000
DEFAULT_RANSAC_CONF = 0.999

SIGMA_SB = 5.670374419e-8  # W m^-2 K^-4

# ============================================================
# USER-EDITABLE DEFAULT PATHS (optional; CLI args override)
# ============================================================
TLS_PATH = r"F:\Masterarbeit_Backup_2\Data\TLS\Lehesten.las"
ALIGNED_FOLDER_DEFAULT = r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted_aligned\Lehesten_25-26.08.25"
OUT_JSON_DEFAULT = r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted_aligned_angled\Lehesten_25-26.08.25\picked_correspondences.json"
OUT_DIR_DEFAULT = r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted_aligned_angled_final\Lehesten_25-26.08.25"
XYTHETA_CSV_DEFAULT = r"F:\Masterarbeit_Backup_2\Alignment\Lehesten_25-26.08.25\xytheta_correction\xytheta_correction.csv"
EBG_CSV_DEFAULT = r"F:\Masterarbeit_Backup_2\Data\Weather\EBG.csv"
GEOLOGY_CSV_DEFAULT = r"F:\Masterarbeit_Backup_2\Results\Correction\Geologic_properties.csv"
DEFAULT_LITHOLOGY = "Gneiss"

DEFAULT_EBG_TIME_COL = "Index"
DEFAULT_SKY_RAD_COL = "GEG_Wm2"
DEFAULT_AIR_TEMP_COL = "Temp_Psy-drybulb_degC"
DEFAULT_DAYNIGHT_COL = "daynight_indicator"
DEFAULT_GLB_COL = "GLBcorr_Wm2"
DEFAULT_SKY_MODE = "apparent_proxy"   # "apparent_proxy" or "effective_longwave"

def natural_sort_key(path: str):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from path."""
    parts = re.split(r"(\d+)", os.path.basename(path))
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def infer_study_site(path_or_folder):
    """Infer a project-specific label from the provided path or filename. Inputs are taken from path_or_folder."""
    base = os.path.basename(os.path.normpath(path_or_folder))
    return base.split("_")[0] if "_" in base else base

def write_pointwise_csv(out_csv, pts_in, uv_in, theta, T_rad0, T_sky0, T_corr0):
    """Write the processed result to disk using the expected project format. Inputs are taken from out_csv, pts_in, uv_in and related parameters."""
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "u", "v", "theta_deg", "T_rad", "T_sky", "T_corr"])
        for (x, y, z), (u, v), th, tr, ts, tc in zip(pts_in, uv_in, theta, T_rad0, T_sky0, T_corr0):
            w.writerow([
                f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
                f"{u:.3f}", f"{v:.3f}", f"{th:.3f}",
                f"{tr:.4f}", f"{ts:.4f}", f"{tc:.4f}"
            ])

def list_txt(folder):
    """Return the available input files in deterministic processing order. Inputs are taken from folder."""
    files = sorted(glob.glob(os.path.join(folder, "*.txt")), key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {folder}")
    return files

def read_txt_with_header(path):
    """Read data from disk and return it in a parsed NumPy/Python structure. Inputs are taken from path."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    data_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "[Data]")
    header = lines[:data_idx + 1]
    clean = "".join(ln.replace(",", ".") for ln in lines[data_idx + 1:])
    arr = np.loadtxt(StringIO(clean), delimiter="\t", dtype=np.float32)
    return header, arr

def print_header_preview(header_lines, label, max_lines=8):
    """Print a compact human-readable summary for quick inspection. Inputs are taken from header_lines, label, max_lines."""
    print(f"\n--- {label} (first {max_lines} lines) ---")
    for ln in header_lines[:max_lines]:
        print(ln.rstrip())
    print("--- end header preview ---\n")

def write_txt_with_header(path, header_lines, data_2d, fmt="%.4f"):
    """Write the processed result to disk using the expected project format. Inputs are taken from path, header_lines, data_2d and related parameters."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(header_lines)
        H, W = data_2d.shape
        for r in range(H):
            row = "\t".join(fmt % float(v) for v in data_2d[r, :])
            f.write(row + "\n")

def load_correspondences(json_path):
    """Load and validate external data needed later in the workflow. Inputs are taken from json_path."""
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    obj = np.asarray(d["points_3d_xyz"], dtype=np.float64)
    img_aligned = np.asarray(d["points_2d_xy"], dtype=np.float64)
    if obj.shape[0] < 6:
        raise ValueError("Need at least 6 correspondences for robust PnP.")
    if obj.shape[0] != img_aligned.shape[0]:
        raise ValueError("Mismatch between points_3d_xyz and points_2d_xy lengths.")
    thermal_txt_path = d.get("thermal_txt_path", None)
    return obj, img_aligned, thermal_txt_path

def resolve_xytheta_csv(path):
    """Resolve flexible user input into a concrete on-disk path. Inputs are taken from path."""
    if path is None:
        raise ValueError("xytheta CSV path is None")
    p = os.path.expanduser(path)
    if os.path.isdir(p):
        cand = os.path.join(p, "xytheta_correction.csv")
        if os.path.exists(cand):
            return cand
        matches = [f for f in glob.glob(os.path.join(p, "*.csv")) if "xytheta" in os.path.basename(f).lower()]
        if matches:
            return matches[0]
        raise FileNotFoundError(f"No xytheta CSV found in directory: {p}")
    if os.path.exists(p):
        return p
    if not p.lower().endswith(".csv") and os.path.exists(p + ".csv"):
        return p + ".csv"
    raise FileNotFoundError(f"xytheta CSV not found: {path}")

def read_affines_from_csv(csv_path):
    """Read data from disk and return it in a parsed NumPy/Python structure. Inputs are taken from csv_path."""
    mats = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        req = {"a00", "a01", "a02", "a10", "a11", "a12"}
        if not req.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV missing required columns {sorted(req)}. Found: {reader.fieldnames}")
        for r in reader:
            A = np.array([[float(r["a00"]), float(r["a01"]), float(r["a02"])],
                          [float(r["a10"]), float(r["a11"]), float(r["a12"])]], dtype=np.float64)
            mats.append(A)
    if not mats:
        raise ValueError("No rows read from xytheta CSV.")
    return mats

def invert_affine_2x3(A):
    """Invert the affine transform so coordinates can be mapped back. Inputs are taken from A."""
    A3 = np.eye(3, dtype=np.float64)
    A3[:2, :] = A
    A3i = np.linalg.inv(A3)
    return A3i[:2, :]

def compute_common_overlap_crop(affines, full_w, full_h):
    """Compute the derived quantity used by later processing stages. Inputs are taken from affines, full_w, full_h."""
    corners = np.array([[0, 0], [full_w, 0], [full_w, full_h], [0, full_h]], dtype=np.float64)
    corners_h = np.hstack([corners, np.ones((4, 1), dtype=np.float64)])
    mins, maxs = [], []
    for A in affines:
        warped = (A @ corners_h.T).T
        mins.append(warped.min(axis=0))
        maxs.append(warped.max(axis=0))
    mins = np.vstack(mins)
    maxs = np.vstack(maxs)
    x0 = int(np.ceil(np.max(mins[:, 0])))
    y0 = int(np.ceil(np.max(mins[:, 1])))
    x1 = int(np.floor(np.min(maxs[:, 0])))
    y1 = int(np.floor(np.min(maxs[:, 1])))
    w = max(0, x1 - x0)
    h = max(0, y1 - y0)
    return x0, y0, w, h

def apply_affine_to_points(A, pts):
    """Apply the requested transform to the provided values. Inputs are taken from A, pts."""
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    ph = np.hstack([pts, ones])
    out = (A @ ph.T).T
    return out

def intrinsics_from_fov(w, h, fovx_deg, fovy_deg):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from w, h, fovx_deg and related parameters."""
    fx = (w / 2.0) / np.tan(np.deg2rad(fovx_deg) / 2.0)
    fy = (h / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    return K, dist

def refine_pnp_lm(obj_pts, img_pts, K, dist, rvec, tvec, max_iters=80):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from obj_pts, img_pts, K and related parameters."""
    obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 1, 3)
    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)
    if hasattr(cv2, "solvePnPRefineLM"):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, int(max_iters), 1e-12)
        rvec2, tvec2 = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec, criteria=criteria)
        return rvec2, tvec2
    if hasattr(cv2, "solvePnPRefineVVS"):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, int(max_iters), 1e-12)
        rvec2, tvec2 = cv2.solvePnPRefineVVS(obj_pts, img_pts, K, dist, rvec, tvec, criteria=criteria)
        return rvec2, tvec2
    return rvec, tvec

def solve_pose_from_pairs(obj_xyz, img_aligned_uv, K, dist, A_ref, crop_x0, crop_y0,
                          ransac_reproj=DEFAULT_RANSAC_REPROJ,
                          ransac_iters=DEFAULT_RANSAC_ITERS,
                          ransac_conf=DEFAULT_RANSAC_CONF):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from obj_xyz, img_aligned_uv, K and related parameters."""
    A_ref_inv = invert_affine_2x3(A_ref)
    img_warped = img_aligned_uv + np.array([crop_x0, crop_y0], dtype=np.float64)
    img_full = apply_affine_to_points(A_ref_inv, img_warped)

    inside = (
        (img_full[:, 0] >= 0) & (img_full[:, 0] < FULL_W) &
        (img_full[:, 1] >= 0) & (img_full[:, 1] < FULL_H) &
        np.isfinite(img_full[:, 0]) & np.isfinite(img_full[:, 1])
    )
    if inside.sum() < 6:
        raise RuntimeError(f"Too few mapped correspondences inside full frame: {inside.sum()}/{len(img_full)}")

    obj_in = obj_xyz[inside]
    img_in = img_full[inside]

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_in.reshape(-1, 1, 3),
        img_in.reshape(-1, 1, 2),
        K, dist,
        reprojectionError=ransac_reproj,
        confidence=ransac_conf,
        iterationsCount=ransac_iters,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok or inliers is None or len(inliers) < 6:
        ok2, rvec, tvec = cv2.solvePnP(
            obj_in.reshape(-1, 1, 3),
            img_in.reshape(-1, 1, 2),
            K, dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok2:
            raise RuntimeError("solvePnPRansac and solvePnP failed.")
        inliers = np.arange(len(obj_in)).reshape(-1, 1)

    inlier_idx = inliers.reshape(-1).astype(int)
    rvec, tvec = refine_pnp_lm(obj_in[inlier_idx], img_in[inlier_idx], K, dist, rvec, tvec, max_iters=80)
    return rvec, tvec

def load_point_cloud(path):
    """Load and validate external data needed later in the workflow. Inputs are taken from path."""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".las", ".laz"]:
        if laspy is None:
            raise RuntimeError("laspy not available. Install with: pip install laspy")
        las = laspy.read(path)
        return np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    if ext in [".ply", ".pcd"]:
        if o3d is None:
            raise RuntimeError("open3d not available. Install with: pip install open3d")
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points, dtype=np.float64)
    if ext in [".xyz", ".txt", ".csv"]:
        pts = np.loadtxt(path, dtype=np.float64, delimiter=None)
        if pts.shape[1] >= 3:
            return pts[:, :3]
        raise ValueError("XYZ/TXT/CSV must have at least 3 columns x y z.")
    raise ValueError(f"Unsupported point cloud format: {ext}")

def estimate_normals(points_xyz, k_neighbors=30):
    """Estimate the required geometric model from the available data. Inputs are taken from points_xyz, k_neighbors."""
    if o3d is None:
        raise RuntimeError("open3d not available. Install with: pip install open3d")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=int(k_neighbors)))
    pcd.normalize_normals()
    return np.asarray(pcd.normals, dtype=np.float64)

def camera_center_world(rvec, tvec):
    """Compute the camera-space quantity needed for the projection workflow. Inputs are taken from rvec, tvec."""
    R, _ = cv2.Rodrigues(rvec)
    C = -R.T @ tvec.reshape(3, 1)
    return C.reshape(3)

def incidence_angle_deg(points_xyz, normals_xyz, C_world):
    """Compute incidence angles between the viewing direction and surface normals. Inputs are taken from points_xyz, normals_xyz, C_world."""
    v = points_xyz - C_world[None, :]
    v_hat = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    n_hat = normals_xyz / (np.linalg.norm(normals_xyz, axis=1, keepdims=True) + 1e-12)
    cosang = np.sum(n_hat * (-v_hat), axis=1)
    cosang = np.clip(np.abs(cosang), 0.0, 1.0)
    return np.degrees(np.arccos(cosang))

class CorrectionModel:
    def __init__(self, theta_knots_deg, k_knots):
        """Return the helper result used by the surrounding processing pipeline. Inputs are taken from self, theta_knots_deg, k_knots."""
        self.theta = np.asarray(theta_knots_deg, dtype=np.float64)
        self.kv = np.asarray(k_knots, dtype=np.float64)

    def k(self, theta_deg):
        """Return the helper result used by the surrounding processing pipeline. Inputs are taken from self, theta_deg."""
        theta_deg = np.asarray(theta_deg, dtype=np.float64)
        return np.interp(theta_deg, self.theta, self.kv, left=self.kv[0], right=self.kv[-1])

    def correct(self, T_rad, theta_deg, T_sky):
        """Return the helper result used by the surrounding processing pipeline. Inputs are taken from self, T_rad, theta_deg and related parameters."""
        T_rad = np.asarray(T_rad, dtype=np.float64)
        T_sky_arr = np.full_like(T_rad, float(T_sky), dtype=np.float64)
        k = self.k(theta_deg)
        return T_rad - np.abs(T_rad - T_sky_arr) * k

def load_geology_model(csv_path, lithology):
    """Load and validate external data needed later in the workflow. Inputs are taken from csv_path, lithology."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Geologic properties CSV not found: {csv_path}")

    theta_vals = []
    k_vals = []
    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if lithology not in fieldnames:
            raise ValueError(f'Lithology "{lithology}" not found in {csv_path}. Available: {fieldnames}')

        angle_col = fieldnames[0]
        for row in reader:
            angle_text = str(row.get(angle_col, "")).strip().replace(",", ".")
            m = re.search(r"[-+]?\d+(?:\.\d+)?", angle_text)
            if not m:
                continue
            theta_vals.append(float(m.group(0)))
            k_text = str(row.get(lithology, "")).strip().replace(",", ".")
            if k_text == "":
                raise ValueError(f'Missing value for lithology "{lithology}" at angle row "{angle_text}"')
            k_vals.append(float(k_text))

    if not theta_vals:
        raise ValueError(f"No angle rows read from geologic properties CSV: {csv_path}")

    theta = np.asarray(theta_vals, dtype=np.float64)
    k = np.asarray(k_vals, dtype=np.float64)

    order = np.argsort(theta)
    theta = theta[order]
    k = k[order]

    if theta[0] > 0.0:
        theta = np.insert(theta, 0, 0.0)
        k = np.insert(k, 0, k[0])

    return CorrectionModel(theta, k)

def bilinear_sample(img, uv):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from img, uv."""
    H, W = img.shape
    u = uv[:, 0]
    v = uv[:, 1]
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = np.clip(u0 + 1, 0, W - 1)
    v1 = np.clip(v0 + 1, 0, H - 1)
    u0 = np.clip(u0, 0, W - 1)
    v0 = np.clip(v0, 0, H - 1)
    du = (u - u0).astype(np.float64)
    dv = (v - v0).astype(np.float64)
    Ia = img[v0, u0]
    Ib = img[v0, u1]
    Ic = img[v1, u0]
    Id = img[v1, u1]
    return (Ia * (1 - du) * (1 - dv) + Ib * du * (1 - dv) + Ic * (1 - du) * dv + Id * du * dv)

def accumulate_median_per_pixel(H, W, uv, values):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from H, W, uv and related parameters."""
    u = np.floor(uv[:, 0]).astype(int)
    v = np.floor(uv[:, 1]).astype(int)
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    bins = {}
    for ui, vi, val in zip(u, v, values):
        key = (vi, ui)
        bins.setdefault(key, []).append(float(val))
    out = np.full((H, W), np.nan, dtype=np.float64)
    mask = np.zeros((H, W), dtype=bool)
    for (vi, ui), arr in bins.items():
        out[vi, ui] = float(np.median(arr))
        mask[vi, ui] = True
    return out, mask

def _try_parse_datetime_text(text):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from text."""
    if text is None:
        return None
    s = str(text).strip()

    patterns_and_formats = [
        (r"(\d{4}-\d{2}-\d{2}_\d{4})", "%Y-%m-%d_%H%M"),
        (r"(\d{4}-\d{2}-\d{2}_\d{2}:\d{2})", "%Y-%m-%d_%H:%M"),
        (r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})", "%Y-%m-%d %H:%M:%S"),
        (r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2})", "%Y-%m-%d %H:%M"),
        (r"(\d{2}\.\d{2}\.\d{4}[ T]\d{2}:\d{2}:\d{2})", "%d.%m.%Y %H:%M:%S"),
        (r"(\d{2}\.\d{2}\.\d{4}[ T]\d{2}:\d{2})", "%d.%m.%Y %H:%M"),
        (r"(\d{8}_\d{6})", "%Y%m%d_%H%M%S"),
        (r"(\d{8}_\d{4})", "%Y%m%d_%H%M"),
        (r"(\d{8}-\d{6})", "%Y%m%d-%H%M%S"),
        (r"(\d{8}-\d{4})", "%Y%m%d-%H%M"),
    ]

    import datetime as dt
    for pat, fmt in patterns_and_formats:
        m = re.search(pat, s)
        if m:
            try:
                return dt.datetime.strptime(m.group(1), fmt)
            except Exception:
                pass
    return None

def parse_frame_datetime(txt_path, header_lines=None):
    """Parse the available metadata into the normalized internal representation. Inputs are taken from txt_path, header_lines."""
    if header_lines is not None:
        for ln in header_lines:
            dtv = _try_parse_datetime_text(ln)
            if dtv is not None:
                return dtv
    return _try_parse_datetime_text(os.path.basename(txt_path))

def load_ebg_dataframe(csv_path, time_col):
    """Load and validate external data needed later in the workflow. Inputs are taken from csv_path, time_col."""
    if pd is None:
        raise RuntimeError("pandas is required for EBG handling. Install with: pip install pandas")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"EBG CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if time_col not in df.columns:
        raise ValueError(f'EBG CSV missing time column "{time_col}". Available: {list(df.columns)}')

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).copy()
    if df.empty:
        raise ValueError("No valid timestamps parsed from EBG CSV.")

    df = df.sort_values(time_col).reset_index(drop=True)
    return df

def crop_ebg_to_measurement_window(ebg_df, time_col, frame_times, pad_minutes=10):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from ebg_df, time_col, frame_times and related parameters."""
    valid_times = [t for t in frame_times if t is not None]
    if not valid_times:
        return ebg_df.copy(), None, None

    t0 = min(valid_times)
    t1 = max(valid_times)
    lo = pd.Timestamp(t0) - pd.Timedelta(minutes=pad_minutes)
    hi = pd.Timestamp(t1) + pd.Timedelta(minutes=pad_minutes)

    cropped = ebg_df[(ebg_df[time_col] >= lo) & (ebg_df[time_col] <= hi)].copy()
    if cropped.empty:
        raise ValueError(
            f"No EBG rows found in measurement window {lo} .. {hi}. "
            f"Full EBG time span is {ebg_df[time_col].min()} .. {ebg_df[time_col].max()}."
        )
    cropped = cropped.sort_values(time_col).reset_index(drop=True)
    return cropped, lo, hi

def derive_tsky_effective_longwave(df, sky_rad_col):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from df, sky_rad_col."""
    if sky_rad_col not in df.columns:
        raise ValueError(f'EBG CSV missing sky-radiation column "{sky_rad_col}". Available: {list(df.columns)}')

    Ld = pd.to_numeric(df[sky_rad_col], errors="coerce").to_numpy(dtype=np.float64)
    Ld = np.clip(Ld, 0.0, None)
    T_eff_C = np.power(Ld / SIGMA_SB, 0.25) - 273.15
    return T_eff_C

def derive_tsky_apparent_proxy(df, sky_rad_col, air_temp_col, daynight_col=None, glb_col=None):
    """
    Empirical proxy for apparent LWIR sky temperature from the available EBG fields.

    Steps:
    - derive effective longwave sky temperature from Ld
    - estimate atmospheric emissivity e_atm = Ld / (sigma * T_air^4)
    - use lower emissivity as a proxy for clearer / colder sky conditions
    - apply a stronger negative offset at night than in daytime

    This is intentionally pragmatic: it uses only the variables present in your EBG file
    and aims to produce a camera-like apparent sky temperature rather than a purely
    broadband effective-radiation temperature.
    """
    if air_temp_col not in df.columns:
        raise ValueError(f'EBG CSV missing air-temperature column "{air_temp_col}". Available: {list(df.columns)}')

    T_eff_C = derive_tsky_effective_longwave(df, sky_rad_col)
    T_air_C = pd.to_numeric(df[air_temp_col], errors="coerce").to_numpy(dtype=np.float64)
    T_air_K = T_air_C + 273.15

    Ld = pd.to_numeric(df[sky_rad_col], errors="coerce").to_numpy(dtype=np.float64)
    Ld = np.clip(Ld, 0.0, None)

    denom = SIGMA_SB * np.maximum(T_air_K, 1.0) ** 4
    emiss = np.divide(Ld, denom, out=np.full_like(Ld, np.nan, dtype=np.float64), where=denom > 0)

    # lower emissivity => clearer, colder sky
    clear_idx = np.clip((0.92 - emiss) / 0.18, 0.0, 1.0)

    is_night = np.zeros(len(df), dtype=bool)
    if daynight_col and daynight_col in df.columns:
        is_night |= (pd.to_numeric(df[daynight_col], errors="coerce").fillna(0).to_numpy() == 0)
    if glb_col and glb_col in df.columns:
        glb = pd.to_numeric(df[glb_col], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        is_night |= (glb <= 20.0)

    # Empirical offset chosen to better approximate apparent LWIR sky temperatures
    # seen by an IR camera under clear summer-night conditions.
    offset_C = np.where(
        is_night,
        10.0 + 18.0 * clear_idx,   # ~10 K cloudy night up to ~28 K clear night
        4.0 + 8.0 * clear_idx      # weaker offset in daytime
    )

    T_app_C = T_eff_C - offset_C
    return T_app_C

def prepare_frame_sky_lookup(txt_files, ebg_df, time_col, sky_temp_C):
    """Prepare the lookup data that links frame timestamps to auxiliary inputs. Inputs are taken from txt_files, ebg_df, time_col and related parameters."""
    ebg_times = np.array(pd.to_datetime(ebg_df[time_col]).dt.to_pydatetime())

    frame_dts = []
    parsed_count = 0
    for txt in txt_files:
        header, _ = read_txt_with_header(txt)
        dtv = parse_frame_datetime(txt, header)
        frame_dts.append(dtv)
        if dtv is not None:
            parsed_count += 1

    if parsed_count == 0:
        raise RuntimeError(
            "No frame timestamps could be parsed. "
            "Expected filenames like 2025-08-25_1120.txt or a matching timestamp in the TXT header."
        )

    Tsky_per_frame = []
    match_seconds = []
    for dtv in frame_dts:
        deltas = np.array([abs((t - dtv).total_seconds()) for t in ebg_times], dtype=np.float64)
        idx = int(np.argmin(deltas))
        Tsky_per_frame.append(float(sky_temp_C[idx]))
        match_seconds.append(float(deltas[idx]))

    return np.asarray(Tsky_per_frame, dtype=np.float64), np.asarray(match_seconds, dtype=np.float64), frame_dts

def main():
    """Parse CLI/config inputs and run the complete workflow for this script."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tls", required=not bool(TLS_PATH), default=TLS_PATH or None)
    ap.add_argument("--aligned_folder", required=not bool(ALIGNED_FOLDER_DEFAULT), default=ALIGNED_FOLDER_DEFAULT or None)
    ap.add_argument("--json", required=not bool(OUT_JSON_DEFAULT), default=OUT_JSON_DEFAULT or None)
    ap.add_argument("--xytheta_csv", required=not bool(XYTHETA_CSV_DEFAULT), default=XYTHETA_CSV_DEFAULT or None)
    ap.add_argument("--ref_index", type=int, default=0)
    ap.add_argument("--fovx", type=float, default=DEFAULT_FOV_X)
    ap.add_argument("--fovy", type=float, default=DEFAULT_FOV_Y)
    ap.add_argument("--knn", type=int, default=30)
    ap.add_argument("--max_points", type=int, default=0)
    ap.add_argument("--out_dir", required=not bool(OUT_DIR_DEFAULT), default=OUT_DIR_DEFAULT or None)
    ap.add_argument("--out_csv", default="", help="Optional pointwise CSV output path. If empty, auto-writes {study_site}_pointwise_corrected.csv in out_dir.")
    ap.add_argument("--suffix", default="_anglecorr")
    ap.add_argument("--geology_csv", required=not bool(GEOLOGY_CSV_DEFAULT), default=GEOLOGY_CSV_DEFAULT or None)
    ap.add_argument("--lithology", default=DEFAULT_LITHOLOGY)

    ap.add_argument("--ebg_csv", required=not bool(EBG_CSV_DEFAULT), default=EBG_CSV_DEFAULT or None)
    ap.add_argument("--ebg_time_col", default=DEFAULT_EBG_TIME_COL)
    ap.add_argument("--sky_rad_col", default=DEFAULT_SKY_RAD_COL)
    ap.add_argument("--air_temp_col", default=DEFAULT_AIR_TEMP_COL)
    ap.add_argument("--daynight_col", default=DEFAULT_DAYNIGHT_COL)
    ap.add_argument("--glb_col", default=DEFAULT_GLB_COL)
    ap.add_argument("--sky_mode", choices=["effective_longwave", "apparent_proxy"], default=DEFAULT_SKY_MODE)
    ap.add_argument("--match_pad_minutes", type=int, default=10)
    args = ap.parse_args()

    print("\n=== CONFIGURATION ===")
    print(f'TXT_ALIGNED_FOLDER = r"{args.aligned_folder}"')
    print(f'OUT_JSON          = r"{args.json}"')
    print(f'TLS_PATH          = r"{args.tls}"')
    print(f'XYTHETA_CSV       = r"{args.xytheta_csv}"')
    print(f'OUT_DIR           = r"{args.out_dir}"')
    print(f'GEOLOGY_CSV       = r"{args.geology_csv}"')
    print(f'EBG_CSV           = r"{args.ebg_csv}"')
    print(f'EBG_TIME_COL      = "{args.ebg_time_col}"')
    print(f'SKY_RAD_COL       = "{args.sky_rad_col}"')
    print(f'AIR_TEMP_COL      = "{args.air_temp_col}"')
    print(f'DAYNIGHT_COL      = "{args.daynight_col}"')
    print(f'GLB_COL           = "{args.glb_col}"')
    print(f'SKY_MODE          = "{args.sky_mode}"')
    print(f'LITHOLOGY         = "{args.lithology}"')
    print(f'FULL_FRAME_W      = {FULL_W}')
    print(f'FULL_FRAME_H      = {FULL_H}')
    print(f'FOV_X_DEG         = {args.fovx}')
    print(f'FOV_Y_DEG         = {args.fovy}')
    print(f'REF_INDEX         = {args.ref_index}')
    print(f'KNN normals       = {args.knn}')
    print(f'MAX_POINTS        = {args.max_points}')
    print(f'SUFFIX            = "{args.suffix}"')
    print("=== END CONFIGURATION ===\n")

    obj_xyz, img_aligned_uv, thermal_txt_path = load_correspondences(args.json)

    txt_files = list_txt(args.aligned_folder)
    txt0 = thermal_txt_path if (thermal_txt_path and os.path.exists(thermal_txt_path)) else txt_files[0]
    header0, img0 = read_txt_with_header(txt0)
    print_header_preview(header0, f"INPUT header from {os.path.basename(txt0)}")
    H_al, W_al = img0.shape

    # Parse frame timestamps early
    frame_times = []
    parsed_count = 0
    for txt in txt_files:
        header, _ = read_txt_with_header(txt)
        dtv = parse_frame_datetime(txt, header)
        frame_times.append(dtv)
        if dtv is not None:
            parsed_count += 1
    print(f"Parsed frame timestamps: {parsed_count}/{len(txt_files)}")

    csv_path = resolve_xytheta_csv(args.xytheta_csv)
    affines = read_affines_from_csv(csv_path)
    if args.ref_index < 0 or args.ref_index >= len(affines):
        raise IndexError(f"ref_index {args.ref_index} out of range (rows={len(affines)})")
    A_ref = affines[args.ref_index]
    crop_x0, crop_y0, crop_w, crop_h = compute_common_overlap_crop(affines, FULL_W, FULL_H)

    if abs(W_al - crop_w) > 2 or abs(H_al - crop_h) > 2:
        raise RuntimeError(
            f"Aligned frame is {W_al}x{H_al} but computed crop is {crop_w}x{crop_h}. "
            "This indicates extra processing (resize) beyond warp+crop."
        )

    K, dist = intrinsics_from_fov(FULL_W, FULL_H, args.fovx, args.fovy)
    rvec, tvec = solve_pose_from_pairs(obj_xyz, img_aligned_uv, K, dist, A_ref, crop_x0, crop_y0)

    pts = load_point_cloud(args.tls)
    if args.max_points and args.max_points > 0 and pts.shape[0] > args.max_points:
        rng = np.random.default_rng(123)
        idx = rng.choice(pts.shape[0], size=int(args.max_points), replace=False)
        pts = pts[idx]

    normals = estimate_normals(pts, k_neighbors=args.knn)

    proj_full, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, tvec, K, dist)
    proj_full = proj_full.reshape(-1, 2)
    proj_warped = apply_affine_to_points(A_ref, proj_full)
    uv_aligned = proj_warped - np.array([crop_x0, crop_y0], dtype=np.float64)

    inside = (
        (uv_aligned[:, 0] >= 0) & (uv_aligned[:, 0] < W_al - 1) &
        (uv_aligned[:, 1] >= 0) & (uv_aligned[:, 1] < H_al - 1) &
        np.isfinite(uv_aligned[:, 0]) & np.isfinite(uv_aligned[:, 1])
    )
    pts_in = pts[inside]
    n_in = normals[inside]
    uv_in = uv_aligned[inside]

    C = camera_center_world(rvec, tvec)
    theta = incidence_angle_deg(pts_in, n_in, C)

    model = load_geology_model(args.geology_csv, args.lithology)
    print(f"Loaded geologic correction model: {args.lithology}")
    print(f"Theta knots (deg): {np.round(model.theta, 3).tolist()}")
    print(f"k values        : {np.round(model.kv, 6).tolist()}")

    ebg_df_full = load_ebg_dataframe(args.ebg_csv, args.ebg_time_col)
    ebg_df, lo, hi = crop_ebg_to_measurement_window(
        ebg_df_full, args.ebg_time_col, frame_times, pad_minutes=args.match_pad_minutes
    )

    print(f"Loaded EBG rows (full): {len(ebg_df_full)}")
    if lo is not None:
        print(f"EBG crop window       : {lo} .. {hi}")
    print(f"Loaded EBG rows (crop): {len(ebg_df)}")

    if args.sky_mode == "effective_longwave":
        sky_temp_C = derive_tsky_effective_longwave(ebg_df, args.sky_rad_col)
    else:
        sky_temp_C = derive_tsky_apparent_proxy(
            ebg_df,
            args.sky_rad_col,
            args.air_temp_col,
            args.daynight_col,
            args.glb_col
        )

    finite_sky = sky_temp_C[np.isfinite(sky_temp_C)]
    if finite_sky.size == 0:
        raise RuntimeError("Derived T_sky contains no finite values.")
    print(f"Derived sky temperature mode: {args.sky_mode}")
    print(f"T_sky range (°C): {finite_sky.min():.3f} .. {finite_sky.max():.3f}")

    Tsky_per_frame, match_seconds, frame_dts = prepare_frame_sky_lookup(
        txt_files, ebg_df, args.ebg_time_col, sky_temp_C
    )
    print(f"Nearest-match offset (s): min={match_seconds.min():.1f}, median={np.median(match_seconds):.1f}, max={match_seconds.max():.1f}")

    os.makedirs(args.out_dir, exist_ok=True)

    if isinstance(args.out_csv, str) and args.out_csv.strip():
        pointwise_csv = args.out_csv
    else:
        study_site = infer_study_site(args.aligned_folder)
        pointwise_csv = os.path.join(args.out_dir, f"{study_site}_pointwise_corrected.csv")

    total = len(txt_files)
    first_frame_done = False
    for i, txt in enumerate(txt_files):
        header, img = read_txt_with_header(txt)
        if img.shape != (H_al, W_al):
            raise RuntimeError(f"Frame shape mismatch in {txt}: got {img.shape}, expected {(H_al, W_al)}")

        T_sky = float(Tsky_per_frame[i])

        T_rad = bilinear_sample(img.astype(np.float64), uv_in)
        T_corr = model.correct(T_rad, theta, T_sky)

        corr_pix, mask = accumulate_median_per_pixel(H_al, W_al, uv_in, T_corr)
        out_img = img.astype(np.float64).copy()
        out_img[mask] = corr_pix[mask]

        base = os.path.splitext(os.path.basename(txt))[0]
        out_name = base + args.suffix + ".txt"
        out_path = os.path.join(args.out_dir, out_name)
        write_txt_with_header(out_path, header, out_img, fmt="%.4f")

        if not first_frame_done:
            T_sky0 = np.full_like(T_rad, T_sky, dtype=np.float64)
            write_pointwise_csv(pointwise_csv, pts_in, uv_in, theta, T_rad, T_sky0, T_corr)
            print(f"Wrote per-point CSV (first frame): {pointwise_csv}")
            print_header_preview(header, f"OUTPUT header example -> {out_name}")
            first_frame_done = True

        if (i + 1) % 10 == 0 or (i + 1) == total:
            ts = frame_dts[i].strftime("%Y-%m-%d %H:%M") if frame_dts[i] is not None else "n/a"
            print(f"Corrected frames: {i+1}/{total} | frame={ts} | T_sky={T_sky:.3f} °C | match={match_seconds[i]:.0f}s")

    print(f"Done. Corrected TXT frames saved to: {args.out_dir}")
    print(f"Applied geologic correction model from: {args.geology_csv}")
    print(f"Lithology used: {args.lithology}")
    print(f"Sky temperature source CSV: {args.ebg_csv}")
    print(f"Sky mode used: {args.sky_mode}")
    print("Correction formula used:")
    print("  T_corr = T_rad - |T_rad - T_sky| * k(theta)")

if __name__ == "__main__":
    main()