"""Correction summary.

Commented version with added helper docstrings to make the workflow easier to follow.
"""

import os
import json
import argparse
import glob
import re
import csv
from io import StringIO

import numpy as np
import cv2


# ============================================================
# CONFIG (edit as needed)
# ============================================================

TXT_ALIGNED_FOLDER = r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted_aligned\Paradiestal_12-13.08.25"
OUT_JSON = r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted_aligned_angled\Paradiestal_12-13.08.25\picked_correspondences.json"

# VarioCAM HR (ResEnhance) native geometry
FULL_FRAME_W = 1280
FULL_FRAME_H = 960

# Standard fixed lens (30 mm)
FOV_X_DEG = 30
FOV_Y_DEG = 23

# Alignment CSV (contains per-frame affine matrices a00..a12)
XYTHETA_CSV = r"F:\Masterarbeit_Backup_2\Alignment\Paradiestal_12-13.08.25\xytheta_correction\xytheta_correction.csv"

# PnP settings
RANSAC_REPROJ_ERR_PX = 6.0
RANSAC_ITERS = 5000
RANSAC_CONF = 0.999


# ============================================================
# Helpers: thermal TXT
# ============================================================

def natural_sort_key(path: str) -> int:
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from path."""
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 10**12

def first_txt(folder):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from folder."""
    files = sorted(glob.glob(os.path.join(folder, "*.txt")), key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {folder}")
    return files[0]

def read_thermal_txt(path):
    """Read data from disk and return it in a parsed NumPy/Python structure. Inputs are taken from path."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    idx = next(i for i, ln in enumerate(lines) if ln.strip() == "[Data]")
    clean = "".join(ln.replace(",", ".") for ln in lines[idx + 1:])
    return np.loadtxt(StringIO(clean), delimiter="\t", dtype=np.float32)

def robust_u8(img):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from img."""
    v = np.nan_to_num(img, nan=np.nanmin(img))
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((v - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


# ============================================================
# Helpers: JSON + CSV
# ============================================================

def load_existing_pairs(json_path):
    """Load and validate external data needed later in the workflow. Inputs are taken from json_path."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if "points_3d_xyz" not in d or "points_2d_xy" not in d:
        raise ValueError("JSON missing points_3d_xyz / points_2d_xy.")
    if len(d["points_3d_xyz"]) != len(d["points_2d_xy"]):
        raise ValueError("Count mismatch: points_3d_xyz vs points_2d_xy.")
    return d

def resolve_xytheta_csv(path):
    """
    Accepts:
      - direct .csv file path
      - directory containing xytheta_correction.csv
      - path without extension (adds .csv if needed)
    """
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

    if not p.lower().endswith(".csv"):
        p2 = p + ".csv"
        if os.path.exists(p2):
            return p2

    raise FileNotFoundError(f"xytheta CSV not found: {path}")

def read_affines_from_csv(csv_path):
    """
    Returns a list of 2x3 affine matrices in CSV order.
    Requires columns: a00 a01 a02 a10 a11 a12
    """
    mats = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        required = {"a00","a01","a02","a10","a11","a12"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV missing required columns. Found: {reader.fieldnames}")
        for r in reader:
            A = np.array([[float(r["a00"]), float(r["a01"]), float(r["a02"])],
                          [float(r["a10"]), float(r["a11"]), float(r["a12"])]] , dtype=np.float64)
            mats.append(A)
    if not mats:
        raise ValueError("No rows read from xytheta CSV.")
    return mats

def compute_common_overlap_crop(affines, full_w, full_h):
    """
    Compute intersection crop box after applying each affine to the full-frame corners.
    Returns (x0,y0,w,h) in the WARPED coordinate system (i.e., after warpAffine).
    """
    corners = np.array([[0,0],[full_w,0],[full_w,full_h],[0,full_h]], dtype=np.float64)
    corners_h = np.hstack([corners, np.ones((4,1), dtype=np.float64)])  # (4,3)

    mins, maxs = [], []
    for A in affines:
        warped = (A @ corners_h.T).T  # (4,2)
        mins.append(warped.min(axis=0))
        maxs.append(warped.max(axis=0))
    mins = np.vstack(mins)
    maxs = np.vstack(maxs)

    x0 = int(np.ceil(np.max(mins[:,0])))
    y0 = int(np.ceil(np.max(mins[:,1])))
    x1 = int(np.floor(np.min(maxs[:,0])))
    y1 = int(np.floor(np.min(maxs[:,1])))

    w = max(0, x1 - x0)
    h = max(0, y1 - y0)
    return x0, y0, w, h

def invert_affine_2x3(A):
    """Invert the affine transform so coordinates can be mapped back. Inputs are taken from A."""
    A3 = np.eye(3, dtype=np.float64)
    A3[:2,:] = A
    A3_inv = np.linalg.inv(A3)
    return A3_inv[:2,:]


# ============================================================
# Camera model + PnP
# ============================================================

def estimate_intrinsics_from_fov(w, h, fovx, fovy):
    """Estimate the required geometric model from the available data. Inputs are taken from w, h, fovx and related parameters."""
    fx = (w/2.0) / np.tan(np.deg2rad(fovx)/2.0)
    fy = (h/2.0) / np.tan(np.deg2rad(fovy)/2.0)
    cx, cy = w/2.0, h/2.0
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    dist = np.zeros((5,1), dtype=np.float64)
    return K, dist


def refine_pnp_lm(obj_pts, img_pts, K, dist, rvec, tvec, max_iters=80):
    '''
    Refine an initial PnP solution with Levenberg–Marquardt on inlier correspondences.
    Uses cv2.solvePnPRefineLM when available; falls back to VVS; otherwise returns inputs unchanged.
    '''
    obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 1, 3)
    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)

    if hasattr(cv2, "solvePnPRefineLM"):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, int(max_iters), 1e-12)
        rvec2, tvec2 = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec, criteria=criteria)
        return rvec2, tvec2, "solvePnPRefineLM"

    if hasattr(cv2, "solvePnPRefineVVS"):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, int(max_iters), 1e-12)
        rvec2, tvec2 = cv2.solvePnPRefineVVS(obj_pts, img_pts, K, dist, rvec, tvec, criteria=criteria)
        return rvec2, tvec2, "solvePnPRefineVVS"

    return rvec, tvec, "no_refine_available"

def apply_affine_to_points(A, pts):
    """Apply the requested transform to the provided values. Inputs are taken from A, pts."""
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts.shape[0],1), dtype=np.float64)
    ph = np.hstack([pts, ones])  # (N,3)
    out = (A @ ph.T).T
    return out

def build_error_table_rows(obj_xyz, img_aligned_uv, proj_aligned_uv, err_aligned, inlier_mask_full=None):
    """Build the helper structure required by the downstream workflow. Inputs are taken from obj_xyz, img_aligned_uv, proj_aligned_uv and related parameters."""
    obj_xyz = np.asarray(obj_xyz, dtype=np.float64)
    img_aligned_uv = np.asarray(img_aligned_uv, dtype=np.float64)
    proj_aligned_uv = np.asarray(proj_aligned_uv, dtype=np.float64)
    err_aligned = np.asarray(err_aligned, dtype=np.float64)

    n = len(err_aligned)
    if inlier_mask_full is None:
        inlier_mask_full = np.zeros(n, dtype=bool)
    else:
        inlier_mask_full = np.asarray(inlier_mask_full, dtype=bool)
        if len(inlier_mask_full) != n:
            raise ValueError("inlier_mask_full length mismatch")

    rows = []
    for i in range(n):
        du = float(proj_aligned_uv[i, 0] - img_aligned_uv[i, 0])
        dv = float(proj_aligned_uv[i, 1] - img_aligned_uv[i, 1])
        rows.append({
            "point_id": int(i),
            "inlier_for_pose": bool(inlier_mask_full[i]),
            "error_px": float(err_aligned[i]),
            "picked_u": float(img_aligned_uv[i, 0]),
            "picked_v": float(img_aligned_uv[i, 1]),
            "projected_u": float(proj_aligned_uv[i, 0]),
            "projected_v": float(proj_aligned_uv[i, 1]),
            "du_px": du,
            "dv_px": dv,
            "X": float(obj_xyz[i, 0]),
            "Y": float(obj_xyz[i, 1]),
            "Z": float(obj_xyz[i, 2]),
        })
    rows.sort(key=lambda r: (-r["error_px"], r["point_id"]))
    return rows


def print_error_table(rows, top_n=None):
    """Print a compact human-readable summary for quick inspection. Inputs are taken from rows, top_n."""
    rows_to_show = rows if top_n is None else rows[:int(top_n)]
    print("\nPer-point reprojection summary (sorted by error):")
    header = f'{"id":>3} | {"inlier":>6} | {"err_px":>8} | {"du_px":>8} | {"dv_px":>8} | {"picked(u,v)":>23} | {"projected(u,v)":>23}'
    print(header)
    print("-" * len(header))
    for r in rows_to_show:
        print(
            f'{r["point_id"]:>3d} | '
            f'{("yes" if r["inlier_for_pose"] else "no"):>6} | '
            f'{r["error_px"]:>8.3f} | '
            f'{r["du_px"]:>8.3f} | '
            f'{r["dv_px"]:>8.3f} | '
            f'({r["picked_u"]:7.1f},{r["picked_v"]:7.1f}) | '
            f'({r["projected_u"]:7.1f},{r["projected_v"]:7.1f})'
        )


def write_error_table_csv(csv_path, rows):
    """Write the processed result to disk using the expected project format. Inputs are taken from csv_path, rows."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["ID", "error_px"])
        for r in rows:
            w.writerow([r["point_id"], r["error_px"]])

def run_check_overlay(json_path, fovx, fovy, xytheta_csv_path, ref_index=0, write_table=True, table_csv=None, top_n=None):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from json_path, fovx, fovy and related parameters."""
    d = load_existing_pairs(json_path)
    obj = np.array(d["points_3d_xyz"], dtype=np.float64)
    img_aligned = np.array(d["points_2d_xy"], dtype=np.float64)

    txt = d.get("thermal_txt_path", None)
    if not txt or not os.path.exists(txt):
        txt = first_txt(TXT_ALIGNED_FOLDER)

    img_u8 = robust_u8(read_thermal_txt(txt))
    H_aligned, W_aligned = img_u8.shape[:2]

    csv_resolved = resolve_xytheta_csv(xytheta_csv_path)
    affines = read_affines_from_csv(csv_resolved)
    if ref_index < 0 or ref_index >= len(affines):
        raise IndexError(f"ref_index {ref_index} out of range for {len(affines)} rows")

    A_ref = affines[ref_index]            # FULL -> WARPED(ref)
    A_ref_inv = invert_affine_2x3(A_ref)  # WARPED -> FULL

    crop_x0, crop_y0, crop_w, crop_h = compute_common_overlap_crop(affines, FULL_FRAME_W, FULL_FRAME_H)
    print(f"Auto crop (common overlap): x0={crop_x0}, y0={crop_y0}, w={crop_w}, h={crop_h}")
    if crop_w > 0 and crop_h > 0 and (W_aligned != crop_w or H_aligned != crop_h):
        print(f"[WARN] Aligned TXT size is {W_aligned}x{H_aligned}, but computed crop is {crop_w}x{crop_h}.")
        print("       This indicates extra processing (e.g., resize) beyond warp+crop.")

    # aligned -> warped (add crop) -> full (inverse affine)
    img_warped = img_aligned + np.array([crop_x0, crop_y0], dtype=np.float64)
    img_full = apply_affine_to_points(A_ref_inv, img_warped)

    K, dist = estimate_intrinsics_from_fov(FULL_FRAME_W, FULL_FRAME_H, fovx, fovy)

    # Sanity-check: keep only points that map into the full-frame image
    inside = (
        (img_full[:, 0] >= 0) & (img_full[:, 0] < FULL_FRAME_W) &
        (img_full[:, 1] >= 0) & (img_full[:, 1] < FULL_FRAME_H) &
        np.isfinite(img_full[:, 0]) & np.isfinite(img_full[:, 1])
    )
    if inside.sum() < 6:
        raise RuntimeError(
            f"Too few valid 2D points inside full frame after mapping: {inside.sum()} / {len(img_full)}. "
            f"Check ref_index and xytheta CSV."
        )

    obj_in = obj[inside]
    img_full_in = img_full[inside]

    print(f"Mapped-to-full points inside frame: {len(img_full_in)} / {len(img_full)}")
    print(f"Full-frame 2D range u:[{img_full_in[:,0].min():.1f},{img_full_in[:,0].max():.1f}] "
          f"v:[{img_full_in[:,1].min():.1f},{img_full_in[:,1].max():.1f}]")

    # Robust PnP: try RANSAC first, fall back to non-RANSAC if needed
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_in.reshape(-1,1,3),
        img_full_in.reshape(-1,1,2),
        K, dist,
        reprojectionError=RANSAC_REPROJ_ERR_PX,
        confidence=RANSAC_CONF,
        iterationsCount=RANSAC_ITERS,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if (not ok) or (inliers is None) or (len(inliers) < 6):
        print("[WARN] solvePnPRansac failed or returned too few inliers; falling back to solvePnP (no RANSAC).")
        ok2, rvec, tvec = cv2.solvePnP(
            obj_in.reshape(-1,1,3),
            img_full_in.reshape(-1,1,2),
            K, dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok2:
            raise RuntimeError("Both solvePnPRansac and solvePnP failed. Check intrinsics / correspondences.")
        inliers = np.arange(len(obj_in)).reshape(-1,1)  # treat all as inliers for reporting

    # --- Refinement on inliers (LM/VVS) ---
    inlier_idx = inliers.reshape(-1).astype(int)
    obj_ref = obj_in[inlier_idx]
    img_ref = img_full_in[inlier_idx]

    # Initial inlier RMSE in FULL-frame pixel coordinates
    proj0, _ = cv2.projectPoints(obj_ref.reshape(-1,1,3), rvec, tvec, K, dist)
    proj0 = proj0.reshape(-1,2)
    err0 = np.linalg.norm(proj0 - img_ref, axis=1)
    rmse0 = float(np.sqrt(np.mean(err0**2))) if len(err0) else float("nan")
    print(f"Inlier RMSE before refine (full px): {rmse0:.2f} (n={len(obj_ref)})")

    rvec, tvec, refine_method = refine_pnp_lm(obj_ref, img_ref, K, dist, rvec, tvec, max_iters=80)

    proj1, _ = cv2.projectPoints(obj_ref.reshape(-1,1,3), rvec, tvec, K, dist)
    proj1 = proj1.reshape(-1,2)
    err1 = np.linalg.norm(proj1 - img_ref, axis=1)
    rmse1 = float(np.sqrt(np.mean(err1**2))) if len(err1) else float("nan")
    print(f"Inlier RMSE after  refine ({refine_method}) (full px): {rmse1:.2f}")

    # full -> warped (affine) -> aligned (subtract crop)
    proj_full, _ = cv2.projectPoints(obj.reshape(-1,1,3), rvec, tvec, K, dist)
    proj_full = proj_full.reshape(-1,2)
    proj_warped = apply_affine_to_points(A_ref, proj_full)
    proj_aligned = proj_warped - np.array([crop_x0, crop_y0], dtype=np.float64)

    err = np.linalg.norm(proj_aligned - img_aligned, axis=1)
    rmse = float(np.sqrt(np.mean(err**2)))

    print(f"Ref affine row index: {ref_index}")
    print(f"Used correspondences after inside-frame filter: {len(obj_in)} / {len(obj)}")
    print(f"Inliers: {len(inliers)} / {len(obj)}")
    print(f"RMSE (aligned px): {rmse:.2f}")
    print("Per-point errors (aligned px):", np.round(err, 2).tolist())

    inlier_mask_full = np.zeros(len(obj), dtype=bool)
    inside_idx = np.where(inside)[0]
    inlier_mask_full[inside_idx[inlier_idx]] = True

    rows = build_error_table_rows(
        obj_xyz=obj,
        img_aligned_uv=img_aligned,
        proj_aligned_uv=proj_aligned,
        err_aligned=err,
        inlier_mask_full=inlier_mask_full,
    )
    print_error_table(rows, top_n=top_n)

    if write_table:
        if not isinstance(table_csv, str) or not table_csv.strip():
            json_dir = os.path.dirname(json_path)
            folder_name = os.path.basename(json_dir)
            study_site = folder_name.split("_")[0] if "_" in folder_name else folder_name
            table_csv = os.path.join(json_dir, f"{study_site}_point_error.csv")
        write_error_table_csv(table_csv, rows)
        print(f"Error table written: {table_csv}")

    vis = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    for i, ((u,v),(up,vp)) in enumerate(zip(img_aligned, proj_aligned)):
        cv2.circle(vis, (int(round(u)), int(round(v))), 5, (0,255,0), -1)
        cv2.circle(vis, (int(round(up)), int(round(vp))), 5, (0,0,255), 2)
        cv2.line(vis, (int(round(u)), int(round(v))), (int(round(up)), int(round(vp))), (255,255,255), 1)
        cv2.putText(vis, f"{i} ({err[i]:.1f}px)",
                    (int(round(u))+6, int(round(v))-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.putText(vis, f"PnP overlay | RMSE={rmse:.1f}px", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow("PnP overlay", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Parse CLI/config inputs and run the complete workflow for this script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=OUT_JSON, help="picked_correspondences.json")
    parser.add_argument("--fovx", type=float, default=FOV_X_DEG)
    parser.add_argument("--fovy", type=float, default=FOV_Y_DEG)
    parser.add_argument("--xytheta_csv", default=XYTHETA_CSV,
                        help="Path to xytheta_correction.csv OR its folder OR path without .csv")
    parser.add_argument("--ref_index", type=int, default=0,
                        help="Row index in xytheta CSV for the reference frame (0 if you used index frame).")
    parser.add_argument("--write_table", action="store_true",
                        help="Write a per-point reprojection summary CSV.")
    parser.add_argument("--table_csv", default=None,
                        help="Optional output CSV path for the per-point reprojection table.")
    parser.add_argument("--top_n", type=int, default=None,
                        help="Show only the top N worst points in the console table (default: show all).")
    args = parser.parse_args()

    run_check_overlay(
        json_path=args.json,
        fovx=args.fovx,
        fovy=args.fovy,
        xytheta_csv_path=args.xytheta_csv,
        ref_index=args.ref_index,
        write_table=True,
        table_csv=args.table_csv,
        top_n=args.top_n
    )

if __name__ == "__main__":
    main()