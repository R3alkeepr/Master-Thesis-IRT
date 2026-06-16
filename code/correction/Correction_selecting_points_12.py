"""
Refine + filter correspondences (NO point-picking) by reprojection error in ALIGNED pixel space.

Fixes vs previous version:
- No required CLI args. If you run without parameters, it uses DEFAULT paths below.
- CLI args still override defaults if provided.

What it does:
1) Loads your JSON correspondences: points_3d_xyz + points_2d_xy (aligned coords)
2) Uses xytheta_correction.csv (affine rows) + ref_index to map aligned->full
3) Solves pose with PnP RANSAC + LM refinement
4) Reprojects ALL points back to ALIGNED coords and computes per-point error (px)
5) Deletes pairs with error >= thresh (default 1.0 px)
6) Saves the cleaned correspondences back into the SAME JSON (optionally with a .bak backup)
7) Optionally shows an overlay image (picked vs projected) before/after filtering

Notes:
- If filtering would leave <6 points, the script aborts WITHOUT saving.
"""

import os
import json
import glob
import re
import csv
import argparse
from io import StringIO

import numpy as np
import cv2


# ==========================================================
# DEFAULT PATHS (EDIT THESE ONCE PER DATASET)
# ==========================================================
JSON_DEFAULT = r"F:\Masterarbeit_Backup_2\Data\Data_TXT_formatted_aligned_angled\Paradiestal_12-13.08.25\picked_correspondences.json"
XYTHETA_CSV_DEFAULT = r"F:\Masterarbeit_Backup_2\Alignment\Paradiestal_12-13.08.25\xytheta_correction\xytheta_correction.csv"
# Optional: aligned thermal TXT used only for --show overlay (if JSON doesn't contain thermal_txt_path)
TXT_DEFAULT = r""


# ---------------------------
# Camera defaults (override via CLI)
# ---------------------------
FULL_W = 1280
FULL_H = 960
DEFAULT_FOV_X = 30.0
DEFAULT_FOV_Y = 23.0

DEFAULT_RANSAC_REPROJ = 6.0
DEFAULT_RANSAC_ITERS = 5000
DEFAULT_RANSAC_CONF = 0.999


# ============================================================
# Helpers: thermal TXT
# ============================================================
def read_txt_with_header(path):
    """Read data from disk and return it in a parsed NumPy/Python structure. Inputs are taken from path."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    data_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "[Data]")
    header = lines[:data_idx + 1]
    clean = "".join(ln.replace(",", ".") for ln in lines[data_idx + 1:])
    arr = np.loadtxt(StringIO(clean), delimiter="\t", dtype=np.float32)
    return header, arr

def robust_u8(img):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from img."""
    v = img.astype(np.float64)
    finite = np.isfinite(v)
    if not finite.any():
        return np.zeros(v.shape, dtype=np.uint8)
    v = np.nan_to_num(v, nan=np.nanmin(v[finite]))
    lo, hi = np.percentile(v[finite], 2), np.percentile(v[finite], 98)
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((v - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


# ============================================================
# JSON correspondences (keep your schema)
# ============================================================
def load_correspondences(json_path):
    """Load and validate external data needed later in the workflow. Inputs are taken from json_path."""
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    obj = np.asarray(d.get("points_3d_xyz", []), dtype=np.float64)
    img_aligned = np.asarray(d.get("points_2d_xy", []), dtype=np.float64)
    thermal_txt_path = d.get("thermal_txt_path", None)
    return d, obj, img_aligned, thermal_txt_path

def save_correspondences(json_path, points_3d_xyz, points_2d_xy, thermal_txt_path=None, keep_other_keys=None):
    """Save the current result to disk and preserve the expected schema. Inputs are taken from json_path, points_3d_xyz, points_2d_xy and related parameters."""
    out = dict(keep_other_keys or {})
    out["points_3d_xyz"] = np.asarray(points_3d_xyz, dtype=float).tolist()
    out["points_2d_xy"] = np.asarray(points_2d_xy, dtype=float).tolist()
    if thermal_txt_path:
        out["thermal_txt_path"] = thermal_txt_path
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


# ============================================================
# Alignment CSV + affine helpers
# ============================================================
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
        req = {"a00","a01","a02","a10","a11","a12"}
        if not req.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV missing required columns {sorted(req)}. Found: {reader.fieldnames}")
        for r in reader:
            A = np.array([[float(r["a00"]), float(r["a01"]), float(r["a02"])],
                          [float(r["a10"]), float(r["a11"]), float(r["a12"])]] , dtype=np.float64)
            mats.append(A)
    if not mats:
        raise ValueError("No rows read from xytheta CSV.")
    return mats

def invert_affine_2x3(A):
    """Invert the affine transform so coordinates can be mapped back. Inputs are taken from A."""
    A3 = np.eye(3, dtype=np.float64)
    A3[:2,:] = A
    A3i = np.linalg.inv(A3)
    return A3i[:2,:]

def apply_affine_to_points(A, pts):
    """Apply the requested transform to the provided values. Inputs are taken from A, pts."""
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts.shape[0],1), dtype=np.float64)
    ph = np.hstack([pts, ones])
    return (A @ ph.T).T

def compute_common_overlap_crop(affines, full_w, full_h):
    """Compute the derived quantity used by later processing stages. Inputs are taken from affines, full_w, full_h."""
    corners = np.array([[0,0],[full_w,0],[full_w,full_h],[0,full_h]], dtype=np.float64)
    corners_h = np.hstack([corners, np.ones((4,1), dtype=np.float64)])
    mins, maxs = [], []
    for A in affines:
        warped = (A @ corners_h.T).T
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


# ============================================================
# Camera intrinsics + pose
# ============================================================
def intrinsics_from_fov(full_w, full_h, fovx_deg, fovy_deg):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from full_w, full_h, fovx_deg and related parameters."""
    fx = (full_w/2.0) / np.tan(np.deg2rad(fovx_deg)/2.0)
    fy = (full_h/2.0) / np.tan(np.deg2rad(fovy_deg)/2.0)
    cx, cy = full_w/2.0, full_h/2.0
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    dist = np.zeros((5,1), dtype=np.float64)
    return K, dist

def refine_pnp_lm(obj_pts, img_pts, K, dist, rvec, tvec, max_iters=80):
    """Return the helper result used by the surrounding processing pipeline. Inputs are taken from obj_pts, img_pts, K and related parameters."""
    obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1,1,3)
    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1,1,2)
    if hasattr(cv2, "solvePnPRefineLM"):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, int(max_iters), 1e-12)
        rvec2, tvec2 = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec, criteria=criteria)
        return rvec2, tvec2, "LM"
    if hasattr(cv2, "solvePnPRefineVVS"):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, int(max_iters), 1e-12)
        rvec2, tvec2 = cv2.solvePnPRefineVVS(obj_pts, img_pts, K, dist, rvec, tvec, criteria=criteria)
        return rvec2, tvec2, "VVS"
    return rvec, tvec, "NONE"


# ============================================================
# Core: compute aligned reprojection and filter
# ============================================================
def compute_aligned_reprojection(obj_xyz, img_aligned_uv, xytheta_csv_path, ref_index, fovx, fovy,
                                 ransac_reproj, ransac_iters, ransac_conf):
    """Compute the derived quantity used by later processing stages. Inputs are taken from obj_xyz, img_aligned_uv, xytheta_csv_path and related parameters."""
    csv_path = resolve_xytheta_csv(xytheta_csv_path)
    affines = read_affines_from_csv(csv_path)
    if ref_index < 0 or ref_index >= len(affines):
        raise IndexError(f"ref_index {ref_index} out of range (rows={len(affines)})")
    A_ref = affines[ref_index]
    A_ref_inv = invert_affine_2x3(A_ref)
    crop_x0, crop_y0, crop_w, crop_h = compute_common_overlap_crop(affines, FULL_W, FULL_H)

    img_warped = img_aligned_uv + np.array([crop_x0, crop_y0], dtype=np.float64)
    img_full = apply_affine_to_points(A_ref_inv, img_warped)

    K, dist = intrinsics_from_fov(FULL_W, FULL_H, fovx, fovy)

    inside = (
        (img_full[:,0] >= 0) & (img_full[:,0] < FULL_W) &
        (img_full[:,1] >= 0) & (img_full[:,1] < FULL_H) &
        np.isfinite(img_full[:,0]) & np.isfinite(img_full[:,1])
    )
    if inside.sum() < 6:
        raise RuntimeError(f"Too few points map into full frame: {inside.sum()}/{len(img_full)}")

    obj_in = obj_xyz[inside]
    img_full_in = img_full[inside]

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_in.reshape(-1,1,3),
        img_full_in.reshape(-1,1,2),
        K, dist,
        reprojectionError=float(ransac_reproj),
        confidence=float(ransac_conf),
        iterationsCount=int(ransac_iters),
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if (not ok) or (inliers is None) or (len(inliers) < 6):
        ok2, rvec, tvec = cv2.solvePnP(
            obj_in.reshape(-1,1,3),
            img_full_in.reshape(-1,1,2),
            K, dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok2:
            raise RuntimeError("solvePnPRansac and solvePnP failed.")
        inliers = np.arange(len(obj_in)).reshape(-1,1)

    inlier_idx = inliers.reshape(-1).astype(int)
    rvec, tvec, refine_method = refine_pnp_lm(obj_in[inlier_idx], img_full_in[inlier_idx], K, dist, rvec, tvec, max_iters=80)

    proj_full, _ = cv2.projectPoints(obj_xyz.reshape(-1,1,3), rvec, tvec, K, dist)
    proj_full = proj_full.reshape(-1,2)
    proj_warped = apply_affine_to_points(A_ref, proj_full)
    proj_aligned = proj_warped - np.array([crop_x0, crop_y0], dtype=np.float64)

    err_aligned = np.linalg.norm(proj_aligned - img_aligned_uv, axis=1)
    rmse_aligned = float(np.sqrt(np.mean(err_aligned**2))) if len(err_aligned) else float("nan")

    return proj_aligned, err_aligned, rmse_aligned, refine_method, int(inside.sum()), int(len(inliers))


def show_overlay(img_u8, img_aligned_uv, proj_aligned, err_aligned, rmse, title="PnP overlay"):
    """Visualize the current result for manual QA. Inputs are taken from img_u8, img_aligned_uv, proj_aligned and related parameters."""
    vis = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    for i, ((u,v),(up,vp)) in enumerate(zip(img_aligned_uv, proj_aligned)):
        cv2.circle(vis, (int(round(u)), int(round(v))), 5, (0,255,0), -1)      # picked
        cv2.circle(vis, (int(round(up)), int(round(vp))), 5, (0,0,255), 2)     # projected
        cv2.line(vis, (int(round(u)), int(round(v))), (int(round(up)), int(round(vp))), (255,255,255), 1)
        cv2.putText(vis, f"{i}:{err_aligned[i]:.1f}px",
                    (int(round(u))+6, int(round(v))-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    cv2.putText(vis, f"RMSE={rmse:.1f}px", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow(title, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Parse CLI/config inputs and run the complete workflow for this script."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="picked_correspondences.json (will be updated)")
    ap.add_argument("--xytheta_csv", default=None, help="xytheta_correction.csv (or folder, or path without .csv)")
    ap.add_argument("--ref_index", type=int, default=0)
    ap.add_argument("--fovx", type=float, default=DEFAULT_FOV_X)
    ap.add_argument("--fovy", type=float, default=DEFAULT_FOV_Y)
    ap.add_argument("--thresh", type=float, default=1.0, help="Delete pairs with reprojection error >= thresh (aligned px)")
    ap.add_argument("--backup", action="store_true", help="Write a .bak copy of JSON before modifying")
    ap.add_argument("--show", action="store_true", help="Show overlay before and after filtering")
    ap.add_argument("--txt", default=None, help="Aligned thermal TXT for overlay (optional)")
    ap.add_argument("--ransac_reproj", type=float, default=DEFAULT_RANSAC_REPROJ)
    ap.add_argument("--ransac_iters", type=int, default=DEFAULT_RANSAC_ITERS)
    ap.add_argument("--ransac_conf", type=float, default=DEFAULT_RANSAC_CONF)
    args = ap.parse_args()

    json_path = args.json if args.json else JSON_DEFAULT
    xytheta_path = args.xytheta_csv if args.xytheta_csv else XYTHETA_CSV_DEFAULT
    txt_hint = args.txt if args.txt else TXT_DEFAULT

    if not json_path or not os.path.exists(json_path):
        raise SystemExit(f"JSON not found: {json_path}")
    # xytheta_csv is resolved later (can be folder or file); just ensure not empty
    if not xytheta_path:
        raise SystemExit("xytheta_csv path is empty. Set XYTHETA_CSV_DEFAULT or pass --xytheta_csv.")

    d, obj_xyz, img_aligned_uv, thermal_txt_path = load_correspondences(json_path)
    if obj_xyz.shape[0] < 6 or img_aligned_uv.shape[0] < 6:
        raise SystemExit("Need at least 6 correspondences.")
    if obj_xyz.shape[0] != img_aligned_uv.shape[0]:
        raise SystemExit("Mismatch between points_3d_xyz and points_2d_xy length.")

    # choose txt for overlay
    txt_for_overlay = ""
    if txt_hint and os.path.exists(txt_hint):
        txt_for_overlay = txt_hint
    elif thermal_txt_path and os.path.exists(thermal_txt_path):
        txt_for_overlay = thermal_txt_path

    img_u8 = None
    if args.show and txt_for_overlay:
        _, img0 = read_txt_with_header(txt_for_overlay)
        img_u8 = robust_u8(img0)
    elif args.show and not txt_for_overlay:
        print("[WARN] --show requested but no usable TXT path. Overlay disabled.")

    proj_aligned, err_aligned, rmse, refine_method, used_inside, inliers = compute_aligned_reprojection(
        obj_xyz=obj_xyz,
        img_aligned_uv=img_aligned_uv,
        xytheta_csv_path=xytheta_path,
        ref_index=args.ref_index,
        fovx=args.fovx,
        fovy=args.fovy,
        ransac_reproj=args.ransac_reproj,
        ransac_iters=args.ransac_iters,
        ransac_conf=args.ransac_conf
    )

    print(f"Refine method: {refine_method}")
    print(f"Used after inside-frame filter: {used_inside} / {len(obj_xyz)}")
    print(f"Inliers (RANSAC): {inliers} / {len(obj_xyz)}")
    print(f"RMSE (aligned px): {rmse:.2f}")

    if args.show and img_u8 is not None:
        show_overlay(img_u8, img_aligned_uv, proj_aligned, err_aligned, rmse, title="PnP overlay (BEFORE filter)")

    keep = (err_aligned < float(args.thresh)) & np.isfinite(err_aligned)
    removed = int((~keep).sum())
    if removed == 0:
        print(f"No points with error >= {args.thresh:.2f}px. JSON unchanged.")
        return
    if int(keep.sum()) < 6:
        print(f"[ERROR] Filtering would leave only {int(keep.sum())} points (<6). Aborting without saving.")
        return

    if args.backup:
        bak = json_path + ".bak"
        with open(bak, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        print("Backup written:", bak)

    obj2 = obj_xyz[keep]
    img2 = img_aligned_uv[keep]

    keep_other = dict(d)
    keep_other.pop("points_3d_xyz", None)
    keep_other.pop("points_2d_xy", None)

    save_correspondences(json_path, obj2, img2, thermal_txt_path=thermal_txt_path, keep_other_keys=keep_other)
    print(f"Filtered JSON saved: {json_path}")
    print(f"Removed {removed} points with error >= {args.thresh:.2f}px. Remaining: {len(obj2)}")

    if args.show and img_u8 is not None:
        proj2, err2, rmse2, _, _, _ = compute_aligned_reprojection(
            obj_xyz=obj2,
            img_aligned_uv=img2,
            xytheta_csv_path=xytheta_path,
            ref_index=args.ref_index,
            fovx=args.fovx,
            fovy=args.fovy,
            ransac_reproj=args.ransac_reproj,
            ransac_iters=args.ransac_iters,
            ransac_conf=args.ransac_conf
        )
        show_overlay(img_u8, img2, proj2, err2, rmse2, title="PnP overlay (AFTER filter)")

if __name__ == "__main__":
    main()