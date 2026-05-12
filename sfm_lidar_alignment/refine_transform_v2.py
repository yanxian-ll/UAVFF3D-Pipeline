# -*- coding: utf-8 -*-
"""Refine LiDAR-to-SfM alignment from rendered image/depth pairs.

The script matches same-stem rendered views, lifts reliable 2D matches through
their depth maps, estimates a global transform, and can optionally refine it
with ICP over the lifted point clouds.
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"]
DEPTH_EXTS = [".npy", ".exr", ".pfm", ".png", ".tiff", ".tif"]


# =========================================================
# 基础工具
# =========================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_image_for_stem(img_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    valid_exts = {e.lower() for e in IMG_EXTS}
    for p in img_dir.glob(f"{stem}.*"):
        if p.suffix.lower() in valid_exts:
            return p
    return None


def find_depth_dir(root: Path):
    for p in [root / "depth_npy", root / "depth"]:
        if p.is_dir():
            return p
    return None


def find_depth_for_stem(depth_dir: Path, stem: str):
    for ext in DEPTH_EXTS:
        p = depth_dir / f"{stem}{ext}"
        if p.exists():
            return p
    valid_exts = {e.lower() for e in DEPTH_EXTS}
    for p in depth_dir.glob(f"{stem}.*"):
        if p.suffix.lower() in valid_exts:
            return p
    return None


def _try_parse_float_line(line: str):
    vals = line.strip().split()
    if len(vals) == 0:
        return None
    try:
        return [float(v) for v in vals]
    except Exception:
        return None


def read_cam_blendedmvs_txt(path_txt: str):
    with open(path_txt, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]

    ex_idx = None
    in_idx = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if low.startswith("extrinsic"):
            ex_idx = i
        elif low.startswith("intrinsic"):
            in_idx = i

    if ex_idx is None or in_idx is None:
        raise ValueError(f"Invalid cam txt format: {path_txt}")

    RT = []
    for j in range(ex_idx + 1, min(ex_idx + 8, len(lines))):
        vals = _try_parse_float_line(lines[j])
        if vals is not None and len(vals) >= 4:
            RT.append(vals[:4])
        if len(RT) == 4:
            break
    if len(RT) != 4:
        raise ValueError(f"Failed to parse extrinsic from: {path_txt}")
    RT = np.asarray(RT, dtype=np.float64)

    K = []
    for j in range(in_idx + 1, min(in_idx + 8, len(lines))):
        vals = _try_parse_float_line(lines[j])
        if vals is not None and len(vals) >= 3:
            K.append(vals[:3])
        if len(K) == 3:
            break
    if len(K) != 3:
        raise ValueError(f"Failed to parse intrinsic from: {path_txt}")
    K = np.asarray(K, dtype=np.float64)

    cam_h, cam_w, misc = 0, 0, 0.0
    for j in range(in_idx + 4, len(lines)):
        vals = _try_parse_float_line(lines[j])
        if vals is not None and len(vals) >= 3:
            cam_h = int(round(vals[0]))
            cam_w = int(round(vals[1]))
            misc = float(vals[2])
            break

    cam2world = np.linalg.inv(RT)
    return K, cam2world, cam_h, cam_w, misc


def read_depth_any(path: Path):
    suf = path.suffix.lower()
    if suf == ".npy":
        d = np.load(path)
        if d is None:
            return None
        if d.ndim == 3:
            d = d[..., 0]
        return d.astype(np.float32)

    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    if d.ndim == 3:
        d = d[..., 0]
    return d.astype(np.float32)


def read_image_bgr(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def scale_intrinsics(K, src_w, src_h, dst_w, dst_h):
    K2 = K.copy().astype(np.float64)
    if src_w <= 0 or src_h <= 0:
        return K2
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def collect_entries(root: Path):
    cam_dir = root / "cams"
    img_dir = root / "images"
    depth_dir = find_depth_dir(root)

    assert cam_dir.is_dir(), f"cams not found: {cam_dir}"
    assert img_dir.is_dir(), f"images not found: {img_dir}"
    assert depth_dir is not None and depth_dir.is_dir(), f"depth not found under: {root}"

    data = {}
    for cam_path in sorted(cam_dir.glob("*.txt")):
        stem = cam_path.stem
        img_path = find_image_for_stem(img_dir, stem)
        depth_path = find_depth_for_stem(depth_dir, stem)
        if img_path is None or depth_path is None:
            continue
        data[stem] = {"cam": cam_path, "img": img_path, "depth": depth_path}
    return data


def load_view(entry):
    K, cam2world, cam_h, cam_w, _ = read_cam_blendedmvs_txt(str(entry["cam"]))
    img = read_image_bgr(entry["img"])
    depth = read_depth_any(entry["depth"])
    if img is None or depth is None:
        return None

    h_img, w_img = img.shape[:2]
    if depth.shape[:2] != (h_img, w_img):
        depth = cv2.resize(depth, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

    K_scaled = scale_intrinsics(K, cam_w, cam_h, w_img, h_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "img": img,
        "gray": gray,
        "depth": depth,
        "K": K_scaled,
        "cam2world": cam2world,
    }


# =========================================================
# 几何工具
# =========================================================
def backproject_pixel_to_cam(u, v, z, K):
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = (u - cx) * z / max(fx, 1e-12)
    y = (v - cy) * z / max(fy, 1e-12)
    return np.array([x, y, z], dtype=np.float64)


def cam_to_world(Xc, cam2world):
    Xh = np.array([Xc[0], Xc[1], Xc[2], 1.0], dtype=np.float64)
    return (cam2world @ Xh)[:3]


def transform_points(pts, T):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(-1, 3)
    return pts @ T[:3, :3].T + T[:3, 3]


def compose_transforms(T2, T1):
    return np.asarray(T2, dtype=np.float64) @ np.asarray(T1, dtype=np.float64)


def sim3_to_matrix(s, R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def decompose_sim3(T):
    M = T[:3, :3]
    detM = np.linalg.det(M)
    s = abs(detM) ** (1.0 / 3.0)
    s = max(s, 1e-12)
    R = M / s
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = T[:3, 3].copy()
    return s, R, t


def simple_umeyama_sim3(src, dst, estimate_scale=True):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src/dst should be Nx3 and same shape")
    n = src.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for similarity transform")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    X = src - mu_src
    Y = dst - mu_dst

    cov = (Y.T @ X) / float(n)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0
    R = U @ D @ Vt

    if estimate_scale:
        var_src = float(np.mean(np.sum(X * X, axis=1)))
        if var_src <= 1e-15:
            raise ValueError("Degenerate source variance")
        s = float(np.trace(np.diag(S) @ D) / var_src)
    else:
        s = 1.0

    t = mu_dst - s * (R @ mu_src)
    return s, R, t


# =========================================================
# 特征匹配与缓存
# =========================================================
def _file_sig(path: Path):
    st = Path(path).stat()
    return {
        "path": str(Path(path).resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def create_feature_detector():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(), cv2.NORM_L2, "SIFT"
    return cv2.ORB_create(nfeatures=10000), cv2.NORM_HAMMING, "ORB"


def build_feature_match_cache_key(entry_a, entry_b, ratio, min_kpts, min_good, keep_top_for_ransac):
    _, _, det_name = create_feature_detector()
    meta = {
        "img_a": _file_sig(entry_a["img"]),
        "img_b": _file_sig(entry_b["img"]),
        "ratio": float(ratio),
        "min_kpts": int(min_kpts),
        "min_good": int(min_good),
        "keep_top_for_ransac": int(keep_top_for_ransac),
        "detector": det_name,
    }
    key = hashlib.sha1(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()
    return key, meta


def load_cached_feature_match(cache_path: Path):
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        ptsA = data["ptsA"].astype(np.float32)
        ptsB = data["ptsB"].astype(np.float32)
        dists = data["dists"].astype(np.float32)
        return ptsA, ptsB, dists
    except Exception:
        return None


def save_cached_feature_match(cache_path: Path, ptsA, ptsB, dists, meta):
    ensure_dir(cache_path.parent)
    if ptsA is None or ptsB is None or dists is None:
        ptsA = np.zeros((0, 2), dtype=np.float32)
        ptsB = np.zeros((0, 2), dtype=np.float32)
        dists = np.zeros((0,), dtype=np.float32)
    np.savez_compressed(
        cache_path,
        ptsA=np.asarray(ptsA, dtype=np.float32),
        ptsB=np.asarray(ptsB, dtype=np.float32),
        dists=np.asarray(dists, dtype=np.float32),
        meta_json=np.array(json.dumps(meta, ensure_ascii=False), dtype=object),
    )


def feature_match(gray_a, gray_b, ratio=0.75, min_kpts=100, min_good=30, keep_top_for_ransac=200):
    detector, norm_type, _ = create_feature_detector()
    kps_a, des_a = detector.detectAndCompute(gray_a, None)
    kps_b, des_b = detector.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None:
        return None, None, None
    if len(kps_a) < min_kpts or len(kps_b) < min_kpts:
        return None, None, None

    matcher = cv2.BFMatcher(norm_type)
    knn = matcher.knnMatch(des_a, des_b, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < min_good:
        return None, None, None

    good = sorted(good, key=lambda x: float(x.distance))
    if keep_top_for_ransac > 0:
        good = good[:keep_top_for_ransac]

    ptsA = np.float32([kps_a[m.queryIdx].pt for m in good])
    ptsB = np.float32([kps_b[m.trainIdx].pt for m in good])
    dists = np.float32([float(m.distance) for m in good])
    return ptsA, ptsB, dists


def feature_match_cached(entry_a, entry_b, gray_a, gray_b, cache_dir: Path, ratio=0.75,
                         min_kpts=100, min_good=30, keep_top_for_ransac=200,
                         force_recompute=False):
    key, meta = build_feature_match_cache_key(
        entry_a, entry_b, ratio, min_kpts, min_good, keep_top_for_ransac
    )
    stem = entry_a["cam"].stem
    cache_path = cache_dir / f"{stem}__{key[:16]}.npz"

    if not force_recompute:
        cached = load_cached_feature_match(cache_path)
        if cached is not None:
            ptsA, ptsB, dists = cached
            if ptsA.shape[0] == 0:
                return None, None, None, str(cache_path), True
            return ptsA, ptsB, dists, str(cache_path), True

    ptsA, ptsB, dists = feature_match(
        gray_a, gray_b,
        ratio=ratio,
        min_kpts=min_kpts,
        min_good=min_good,
        keep_top_for_ransac=keep_top_for_ransac,
    )
    save_cached_feature_match(cache_path, ptsA, ptsB, dists, meta)
    return ptsA, ptsB, dists, str(cache_path), False


def ransac_filter_2d(ptsA, ptsB, dists, thr_px=4.0, min_inliers=12, min_inlier_ratio=0.2):
    if ptsA is None or ptsB is None or dists is None or len(ptsA) < 8:
        return None, None, None, None

    F, mask = cv2.findFundamentalMat(
        ptsA,
        ptsB,
        cv2.FM_RANSAC,
        ransacReprojThreshold=float(thr_px),
        confidence=0.999,
    )
    if F is None or mask is None:
        return None, None, None, None

    mask = mask.ravel().astype(bool)
    ptsA_in = ptsA[mask]
    ptsB_in = ptsB[mask]
    dists_in = dists[mask]
    num_in = int(mask.sum())
    ratio_in = float(num_in) / max(float(len(mask)), 1.0)
    if num_in < int(min_inliers) or ratio_in < float(min_inlier_ratio):
        return None, None, None, None
    return ptsA_in, ptsB_in, dists_in, mask


def filter_by_2d_translation(ptsA, ptsB, dists, thr_px=10.0, min_keep=5):
    if ptsA is None or ptsB is None or dists is None:
        return None, None, None, None
    if len(ptsA) < min_keep:
        return None, None, None, None

    disp = ptsB - ptsA
    med = np.median(disp, axis=0)
    err = np.linalg.norm(disp - med[None, :], axis=1)
    mask = err <= float(thr_px)
    if int(mask.sum()) < int(min_keep):
        return None, None, None, None
    return ptsA[mask], ptsB[mask], dists[mask], mask


# =========================================================
# 3D lifting / 点云
# =========================================================
def lift_matches_to_3d_two_views(KA, cam2worldA, depthA, KB, cam2worldB, depthB,
                                 ptsA, ptsB, dists, min_depth=1e-3):
    hA, wA = depthA.shape[:2]
    hB, wB = depthB.shape[:2]

    kept_ptsA, kept_ptsB, XA_w, XB_w, kept_dists = [], [], [], [], []
    for (uA, vA), (uB, vB), dist in zip(ptsA, ptsB, dists):
        xA = int(round(float(uA)))
        yA = int(round(float(vA)))
        xB = int(round(float(uB)))
        yB = int(round(float(vB)))

        if not (0 <= xA < wA and 0 <= yA < hA and 0 <= xB < wB and 0 <= yB < hB):
            continue

        zA = float(depthA[yA, xA])
        zB = float(depthB[yB, xB])
        if not np.isfinite(zA) or not np.isfinite(zB):
            continue
        if zA <= min_depth or zB <= min_depth:
            continue

        XA_c = backproject_pixel_to_cam(uA, vA, zA, KA)
        XB_c = backproject_pixel_to_cam(uB, vB, zB, KB)
        XA_w.append(cam_to_world(XA_c, cam2worldA))
        XB_w.append(cam_to_world(XB_c, cam2worldB))
        kept_ptsA.append([uA, vA])
        kept_ptsB.append([uB, vB])
        kept_dists.append(float(dist))

    if len(XA_w) == 0:
        return None
    return {
        "ptsA": np.asarray(kept_ptsA, dtype=np.float32),
        "ptsB": np.asarray(kept_ptsB, dtype=np.float32),
        "XA_w": np.asarray(XA_w, dtype=np.float64),
        "XB_w": np.asarray(XB_w, dtype=np.float64),
        "dists": np.asarray(kept_dists, dtype=np.float32),
    }


def depth_to_world_points(depth, K, cam2world, stride=8, min_depth=1e-3, max_depth=None):
    h, w = depth.shape[:2]
    ys = np.arange(0, h, max(1, int(stride)), dtype=np.int32)
    xs = np.arange(0, w, max(1, int(stride)), dtype=np.int32)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    uu, vv = np.meshgrid(xs, ys)
    z = depth[vv, uu].astype(np.float64).reshape(-1)
    u = uu.reshape(-1).astype(np.float64)
    v = vv.reshape(-1).astype(np.float64)

    mask = np.isfinite(z) & (z > min_depth)
    if max_depth is not None and max_depth > 0:
        mask &= (z <= max_depth)
    if not np.any(mask):
        return np.zeros((0, 3), dtype=np.float64)

    z = z[mask]
    u = u[mask]
    v = v[mask]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    x = (u - cx) * z / max(fx, 1e-12)
    y = (v - cy) * z / max(fy, 1e-12)
    pts_cam = np.stack([x, y, z], axis=1)
    R = cam2world[:3, :3]
    t = cam2world[:3, 3]
    return (pts_cam @ R.T + t).astype(np.float64)


def voxel_downsample(points, voxel_size):
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0 or voxel_size is None or voxel_size <= 0:
        return points
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def maybe_subsample_points(points, max_points, seed=0):
    points = np.asarray(points)
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def build_global_cloud_from_views(data, stems_used, args):
    rng = np.random.default_rng(args.seed)
    clouds = []
    for stem in tqdm(stems_used, desc="Build cloud"):
        view = load_view(data[stem])
        if view is None:
            continue
        pts = depth_to_world_points(
            view["depth"], view["K"], view["cam2world"],
            stride=args.cloud_stride,
            min_depth=args.min_depth,
            max_depth=args.cloud_max_depth,
        )
        if pts.shape[0] == 0:
            continue
        if args.cloud_voxel_per_frame > 0:
            pts = voxel_downsample(pts, args.cloud_voxel_per_frame)
        if args.cloud_max_points_per_frame > 0 and pts.shape[0] > args.cloud_max_points_per_frame:
            idx = rng.choice(pts.shape[0], size=args.cloud_max_points_per_frame, replace=False)
            pts = pts[idx]
        clouds.append(pts)

    if len(clouds) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    cloud = np.concatenate(clouds, axis=0)
    print(f"{cloud.shape[0]}")

    # if args.cloud_voxel_global > 0:
    #     cloud = voxel_downsample(cloud, args.cloud_voxel_global)
    # if args.cloud_max_points_total > 0 and cloud.shape[0] > args.cloud_max_points_total:
    #     idx = rng.choice(cloud.shape[0], size=args.cloud_max_points_total, replace=False)
    #     cloud = cloud[idx]
    return cloud


# =========================================================
# 可视化 / 输出
# =========================================================
def visualize_matches_points(imgA_bgr, imgB_bgr, ptsA, ptsB, out_path, title_text=None):
    if ptsA is None or ptsB is None or len(ptsA) == 0:
        return

    hA, wA = imgA_bgr.shape[:2]
    hB, wB = imgB_bgr.shape[:2]
    H = max(hA, hB)
    W = wA + wB
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:hA, :wA] = imgA_bgr
    canvas[:hB, wA:wA + wB] = imgB_bgr

    rng = np.random.default_rng(0)
    for (x1, y1), (x2, y2) in zip(ptsA, ptsB):
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        p1 = (int(round(float(x1))), int(round(float(y1))))
        p2 = (int(round(float(x2))) + wA, int(round(float(y2))))
        cv2.circle(canvas, p1, 4, color, -1)
        cv2.circle(canvas, p2, 4, color, -1)
        cv2.line(canvas, p1, p2, color, 2)

    if title_text:
        cv2.putText(canvas, title_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)

    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), canvas)


def save_point_cloud_ply(path: Path, points, colors=None):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points should be Nx3")

    if colors is None:
        colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)
    else:
        colors = np.asarray(colors)
        if colors.ndim == 1:
            colors = np.tile(colors[None, :], (points.shape[0], 1))
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.10f} {p[1]:.10f} {p[2]:.10f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def save_xyz_txt(path: Path, pts):
    ensure_dir(path.parent)
    np.savetxt(str(path), np.asarray(pts, dtype=np.float64), fmt="%.10f")


def save_transform_txt(path: Path, T):
    ensure_dir(path.parent)
    np.savetxt(str(path), np.asarray(T, dtype=np.float64), fmt="%.10f")


def save_json(path: Path, obj):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_points_stage(out_dir: Path, src_pts, dst_pts, src_color=(255, 0, 0), dst_color=(0, 255, 0), save_merged=True):
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)

    if src_pts.shape[0] > 0:
        save_point_cloud_ply(out_dir / "src_world.ply", src_pts, np.array(src_color, dtype=np.uint8))
    if dst_pts.shape[0] > 0:
        save_point_cloud_ply(out_dir / "dst_world.ply", dst_pts, np.array(dst_color, dtype=np.uint8))
    if (src_pts.shape[0] > 0 or dst_pts.shape[0] > 0) and save_merged:
        merged_pts = np.concatenate([src_pts, dst_pts], axis=0)
        merged_col = np.concatenate([
            np.tile(np.array(src_color, dtype=np.uint8)[None, :], (src_pts.shape[0], 1)),
            np.tile(np.array(dst_color, dtype=np.uint8)[None, :], (dst_pts.shape[0], 1)),
        ], axis=0)
        save_point_cloud_ply(out_dir / "merged.ply", merged_pts, merged_col)


def make_o3d_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def evaluate_point_cloud_alignment(src_pts, dst_pts, out_dir: Path, stage_name="",
                                   inlier_thr=1.0):
    """
    简单计算点云对齐指标，并保存到对应目录下。
    """
    src_pts = np.asarray(src_pts, dtype=np.float64).reshape(-1, 3)
    dst_pts = np.asarray(dst_pts, dtype=np.float64).reshape(-1, 3)

    if src_pts.shape[0] == 0 or dst_pts.shape[0] == 0:
        metrics = {
            "stage_name": stage_name,
            "num_src": int(src_pts.shape[0]),
            "num_dst": int(dst_pts.shape[0]),
            "inlier_threshold": float(inlier_thr),
            "valid": False,
            "message": "src 或 dst 点云为空，无法计算对齐指标",
        }
        save_json(out_dir / "alignment_metrics.json", metrics)
        with open(out_dir / "alignment_metrics.txt", "w", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False, indent=2))
        return metrics

    pcd_src = make_o3d_pcd(src_pts)
    pcd_dst = make_o3d_pcd(dst_pts)

    dist_src_to_dst = np.asarray(pcd_src.compute_point_cloud_distance(pcd_dst), dtype=np.float64)
    dist_dst_to_src = np.asarray(pcd_dst.compute_point_cloud_distance(pcd_src), dtype=np.float64)

    def _safe_stats(d):
        if d.size == 0:
            return {
                "mean": None,
                "median": None,
                "rmse": None,
                "max": None,
                "min": None,
                "inlier_ratio": None,
            }
        return {
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "max": float(np.max(d)),
            "min": float(np.min(d)),
            "inlier_ratio": float(np.mean(d <= float(inlier_thr))),
        }

    s2d = _safe_stats(dist_src_to_dst)
    d2s = _safe_stats(dist_dst_to_src)
    all_d = np.concatenate([dist_src_to_dst, dist_dst_to_src], axis=0)

    metrics = {
        "stage_name": stage_name,
        "valid": True,
        "num_src": int(src_pts.shape[0]),
        "num_dst": int(dst_pts.shape[0]),
        "inlier_threshold": float(inlier_thr),

        "src_to_dst_mean": s2d["mean"],
        "src_to_dst_median": s2d["median"],
        "src_to_dst_rmse": s2d["rmse"],
        "src_to_dst_min": s2d["min"],
        "src_to_dst_max": s2d["max"],
        "inlier_ratio_src_to_dst": s2d["inlier_ratio"],

        "dst_to_src_mean": d2s["mean"],
        "dst_to_src_median": d2s["median"],
        "dst_to_src_rmse": d2s["rmse"],
        "dst_to_src_min": d2s["min"],
        "dst_to_src_max": d2s["max"],
        "inlier_ratio_dst_to_src": d2s["inlier_ratio"],

        "symmetric_mean": float(np.mean(all_d)),
        "symmetric_median": float(np.median(all_d)),
        "symmetric_rmse": float(np.sqrt(np.mean(all_d ** 2))),
        "symmetric_max": float(np.max(all_d)),
    }

    save_json(out_dir / "alignment_metrics.json", metrics)

    with open(out_dir / "alignment_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"stage_name: {metrics['stage_name']}\n")
        f.write(f"num_src: {metrics['num_src']}\n")
        f.write(f"num_dst: {metrics['num_dst']}\n")
        f.write(f"inlier_threshold: {metrics['inlier_threshold']}\n\n")

        f.write("[src -> dst]\n")
        f.write(f"mean: {metrics['src_to_dst_mean']:.6f}\n")
        f.write(f"median: {metrics['src_to_dst_median']:.6f}\n")
        f.write(f"rmse: {metrics['src_to_dst_rmse']:.6f}\n")
        f.write(f"min: {metrics['src_to_dst_min']:.6f}\n")
        f.write(f"max: {metrics['src_to_dst_max']:.6f}\n")
        f.write(f"inlier_ratio: {metrics['inlier_ratio_src_to_dst']:.6f}\n\n")

        f.write("[dst -> src]\n")
        f.write(f"mean: {metrics['dst_to_src_mean']:.6f}\n")
        f.write(f"median: {metrics['dst_to_src_median']:.6f}\n")
        f.write(f"rmse: {metrics['dst_to_src_rmse']:.6f}\n")
        f.write(f"min: {metrics['dst_to_src_min']:.6f}\n")
        f.write(f"max: {metrics['dst_to_src_max']:.6f}\n")
        f.write(f"inlier_ratio: {metrics['inlier_ratio_dst_to_src']:.6f}\n\n")

        f.write("[symmetric]\n")
        f.write(f"mean: {metrics['symmetric_mean']:.6f}\n")
        f.write(f"median: {metrics['symmetric_median']:.6f}\n")
        f.write(f"rmse: {metrics['symmetric_rmse']:.6f}\n")
        f.write(f"max: {metrics['symmetric_max']:.6f}\n")

    return metrics


# =========================================================
# 核心流程：匹配
# =========================================================
def run_match_pipeline(data_a, data_b, stems_used, args, match_out_dir: Path, init_transform=None):
    init_transform = np.eye(4, dtype=np.float64) if init_transform is None else np.asarray(init_transform, dtype=np.float64)

    vis_dir = ensure_dir(match_out_dir / "match_vis")
    cache_dir = ensure_dir(Path(args.feature_match_cache_dir)) if args.feature_match_cache_dir else ensure_dir(match_out_dir / "match_cache")
    points_7p_before = ensure_dir(match_out_dir / "points_7p_before")
    points_7p_after = ensure_dir(match_out_dir / "points_7p_after")

    records = []
    quality_records = []

    pbar = tqdm(stems_used, desc="Match frames")
    for stem in pbar:
        view_a = load_view(data_a[stem])
        view_b = load_view(data_b[stem])
        if view_a is None or view_b is None:
            continue

        ptsA, ptsB, dists, cache_path, cache_hit = feature_match_cached(
            data_a[stem], data_b[stem],
            view_a["gray"], view_b["gray"],
            cache_dir=cache_dir,
            ratio=args.sift_ratio,
            min_kpts=args.min_kpts,
            min_good=args.min_good_matches,
            keep_top_for_ransac=args.keep_top_for_ransac,
            force_recompute=args.recompute_feature_match_cache,
        )
        if ptsA is None:
            continue

        ptsA, ptsB, dists, ransac_mask = ransac_filter_2d(
            ptsA, ptsB, dists,
            thr_px=args.fm_thr,
            min_inliers=args.min_inliers_2d,
            min_inlier_ratio=args.min_inlier_ratio_2d,
        )
        if ptsA is None:
            continue

        if args.enable_translation_filter:
            ptsA, ptsB, dists, _ = filter_by_2d_translation(
                ptsA, ptsB, dists,
                thr_px=args.translation_thr_px,
                min_keep=max(args.topk_per_image, 3),
            )
            if ptsA is None:
                continue

        lifted = lift_matches_to_3d_two_views(
            view_a["K"], view_a["cam2world"], view_a["depth"],
            view_b["K"], view_b["cam2world"], view_b["depth"],
            ptsA, ptsB, dists,
            min_depth=args.min_depth,
        )
        if lifted is None:
            continue

        src_raw = lifted["XA_w"]
        src_solver = transform_points(src_raw, init_transform)
        dst = lifted["XB_w"]
        ptsA3 = lifted["ptsA"]
        ptsB3 = lifted["ptsB"]
        dists3 = lifted["dists"]

        if args.prealign_3d_thr > 0:
            err3d = np.linalg.norm(src_solver - dst, axis=1)
            keep = err3d <= float(args.prealign_3d_thr)
            if int(keep.sum()) < max(args.topk_per_image, 3):
                continue
            src_raw = src_raw[keep]
            src_solver = src_solver[keep]
            dst = dst[keep]
            ptsA3 = ptsA3[keep]
            ptsB3 = ptsB3[keep]
            dists3 = dists3[keep]
            err3d = err3d[keep]
        else:
            err3d = np.linalg.norm(src_solver - dst, axis=1)

        order = np.argsort(dists3)
        if args.topk_per_image > 0:
            order = order[:args.topk_per_image]
        src_raw = src_raw[order]
        src_solver = src_solver[order]
        dst = dst[order]
        ptsA3 = ptsA3[order]
        ptsB3 = ptsB3[order]
        dists3 = dists3[order]
        err3d = err3d[order]

        if src_raw.shape[0] < max(args.min_pairs_per_image, 3):
            continue

        mean_desc = float(np.mean(dists3))
        mean_err3d = float(np.mean(err3d))
        if args.max_mean_match_distance > 0 and mean_desc > float(args.max_mean_match_distance):
            continue
        if args.max_mean_3d_error > 0 and mean_err3d > float(args.max_mean_3d_error):
            continue

        visualize_matches_points(
            view_a["img"], view_b["img"],
            ptsA3, ptsB3,
            vis_dir / f"{stem}.png",
            title_text=(
                f"{stem} | keep={len(ptsA3)} | mean_desc={mean_desc:.3f} | mean_3d={mean_err3d:.3f}m"
            ),
        )

        records.append({
            "stem": stem,
            "src_raw": src_raw,
            "src_solver": src_solver,
            "dst": dst,
            "ptsA": ptsA3,
            "ptsB": ptsB3,
            "dists": dists3,
            "mean_desc": mean_desc,
            "mean_err3d": mean_err3d,
            "feature_cache_path": cache_path,
            "feature_cache_hit": bool(cache_hit),
        })
        quality_records.append((stem, mean_desc, mean_err3d, int(src_raw.shape[0])))
        pbar.set_postfix({
            "accepted": len(records),
            "topk": int(src_raw.shape[0]),
            "desc": f"{mean_desc:.2f}",
            "3d": f"{mean_err3d:.2f}",
            "cache": "hit" if cache_hit else "new",
        })

    if len(records) == 0:
        raise RuntimeError("没有保留下有效匹配帧，请放宽阈值或检查数据")

    src_raw_all = np.concatenate([r["src_raw"] for r in records], axis=0)
    src_solver_all = np.concatenate([r["src_solver"] for r in records], axis=0)
    dst_all = np.concatenate([r["dst"] for r in records], axis=0)

    if args.match_max_points_total > 0 and src_raw_all.shape[0] > args.match_max_points_total:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(src_raw_all.shape[0], size=args.match_max_points_total, replace=False)
        src_raw_all = src_raw_all[idx]
        src_solver_all = src_solver_all[idx]
        dst_all = dst_all[idx]

    save_points_stage(points_7p_before, src_raw_all, dst_all)

    s7, R7, t7 = simple_umeyama_sim3(
        src_solver_all,
        dst_all,
        estimate_scale=args.estimate_scale,
    )
    T_delta = sim3_to_matrix(s7, R7, t7)
    T_match = compose_transforms(T_delta, init_transform)

    src_after_7p = transform_points(src_raw_all, T_match)
    save_points_stage(points_7p_after, src_after_7p, dst_all, src_color=(0, 0, 255), dst_color=(0, 255, 0))

    residual_7p = np.linalg.norm(src_after_7p - dst_all, axis=1)
    save_transform_txt(match_out_dir / "transform_match.txt", T_match)

    summary = {
        "accepted_frames": int(len(records)),
        "num_pairs": int(src_raw_all.shape[0]),
        "topk_per_image": int(args.topk_per_image),
        "estimate_scale": bool(args.estimate_scale),
        "transform_match": T_match.tolist(),
        "scale": float(decompose_sim3(T_match)[0]),
        "mean_residual_7p": float(np.mean(residual_7p)),
        "median_residual_7p": float(np.median(residual_7p)),
        "per_frame": [
            {
                "stem": r["stem"],
                "num_pairs": int(r["src_raw"].shape[0]),
                "mean_descriptor_distance": float(r["mean_desc"]),
                "mean_3d_error_before_7p": float(r["mean_err3d"]),
                "feature_cache_hit": bool(r["feature_cache_hit"]),
                "feature_cache_path": r["feature_cache_path"],
            }
            for r in records
        ],
    }
    with open(match_out_dir / "summary_match.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "T": T_match,
        "summary": summary,
        "src_after_7p": src_after_7p,
        "dst": dst_all,
        "residual_7p": residual_7p,
    }


# =========================================================
# 核心流程：ICP（Open3D 库实现）
# =========================================================
def run_icp_pipeline(data_a, data_b, stems_used, args, icp_out_dir: Path, init_transform=None):
    points_coarse = ensure_dir(icp_out_dir / "points_caorse")
    points_icp_before = ensure_dir(icp_out_dir / "points_icp_before")
    points_icp_after = ensure_dir(icp_out_dir / "points_icp_after")

    cloud_a = build_global_cloud_from_views(data_a, stems_used, args)
    cloud_b = build_global_cloud_from_views(data_b, stems_used, args)
    if cloud_a.shape[0] < 3 or cloud_b.shape[0] < 3:
        raise RuntimeError("ICP 点云不足，无法执行")

    print(f"save ply: A: {cloud_a.shape[0]}, B: {cloud_b.shape[0]}")

    save_points_stage(points_coarse, cloud_a, cloud_b,
                      src_color=(255, 255, 0), dst_color=(0, 255, 0), save_merged=False)
    metrics_coarse = evaluate_point_cloud_alignment(
        cloud_a, cloud_b,
        out_dir=points_coarse,
        stage_name="coarse",
        inlier_thr=args.icp_corr_thr,
    )

    init_transform = np.eye(4, dtype=np.float64) if init_transform is None else np.asarray(init_transform, dtype=np.float64)
    cloud_a_before = transform_points(cloud_a, init_transform)
    save_points_stage(points_icp_before, cloud_a_before, cloud_b,
                      src_color=(255, 255, 0), dst_color=(0, 255, 0), save_merged=False)
    metrics_icp_before = evaluate_point_cloud_alignment(
        cloud_a_before, cloud_b,
        out_dir=points_icp_before,
        stage_name="icp_before",
        inlier_thr=args.icp_corr_thr,
    )

    pcd_a = make_o3d_pcd(cloud_a)
    pcd_b = make_o3d_pcd(cloud_b)

    if args.icp_method == "point_to_plane":
        pcd_a.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=args.icp_normal_k))
        pcd_b.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=args.icp_normal_k))
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=int(args.icp_iters),
        relative_fitness=float(args.icp_relative_fitness),
        relative_rmse=float(args.icp_relative_rmse),
    )

    result = o3d.pipelines.registration.registration_icp(
        pcd_a,
        pcd_b,
        max_correspondence_distance=float(args.icp_corr_thr),
        init=init_transform.astype(np.float64),
        estimation_method=estimation,
        criteria=criteria,
    )

    T_icp = np.asarray(result.transformation, dtype=np.float64)
    src_after_icp = transform_points(cloud_a, T_icp)

    save_points_stage(points_icp_after, src_after_icp, cloud_b,
                      src_color=(0, 0, 255), dst_color=(0, 255, 0), save_merged=False)
    metrics_icp_after = evaluate_point_cloud_alignment(
        src_after_icp, cloud_b,
        out_dir=points_icp_after,
        stage_name="icp_after",
        inlier_thr=args.icp_corr_thr,
    )

    save_transform_txt(icp_out_dir / "transform_icp.txt", T_icp)

    summary = {
        "num_cloud_a": int(cloud_a.shape[0]),
        "num_cloud_b": int(cloud_b.shape[0]),
        "transform_icp": T_icp.tolist(),
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "correspondence_count": int(len(result.correspondence_set)),
        "icp_method": args.icp_method,
        "max_correspondence_distance": float(args.icp_corr_thr),
        "metrics_coarse": metrics_coarse,
        "metrics_icp_before": metrics_icp_before,
        "metrics_icp_after": metrics_icp_after,
    }
    with open(icp_out_dir / "summary_icp.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {"T": T_icp, "summary": summary}


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_a", type=str, required=True, help="路径A：包含 cams/images/depth")
    parser.add_argument("--root_b", type=str, required=True, help="路径B：包含 cams/images/depth")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument(
        "--mode",
        type=str,
        default="match_then_icp",
        choices=["only_match", "only_icp", "icp_then_match", "match_then_icp"],
        help="配准模式",
    )
    parser.add_argument("--num_images", type=int, default=50, help="最多使用多少个公共视角")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature_match_cache_dir", type=str, default="", help="特征匹配缓存目录")
    parser.add_argument("--recompute_feature_match_cache", action="store_true", help="强制重算特征匹配缓存")

    # 匹配参数
    parser.add_argument("--sift_ratio", type=float, default=0.85)
    parser.add_argument("--min_kpts", type=int, default=20)
    parser.add_argument("--min_good_matches", type=int, default=10)
    parser.add_argument("--keep_top_for_ransac", type=int, default=50, help="RANSAC 前最多保留多少个最好匹配")
    parser.add_argument("--fm_thr", type=float, default=2.0, help="基础矩阵RANSAC阈值(像素)")
    parser.add_argument("--min_inliers_2d", type=int, default=5)
    parser.add_argument("--min_inlier_ratio_2d", type=float, default=0.10)
    parser.add_argument("--enable_translation_filter", action="store_true")
    parser.add_argument("--translation_thr_px", type=float, default=10.0)
    parser.add_argument("--prealign_3d_thr", type=float, default=10.0, help="初始变换后3D一致性阈值(米)，<=0表示不启用")
    parser.add_argument("--topk_per_image", type=int, default=5, help="每帧最终最多保留多少个最优匹配")
    parser.add_argument("--min_pairs_per_image", type=int, default=1, help="每帧最少保留多少个匹配")
    parser.add_argument("--max_mean_match_distance", type=float, default=0.0, help="每帧最大平均描述子距离，<=0表示不限制")
    parser.add_argument("--max_mean_3d_error", type=float, default=0.0, help="每帧最大平均3D误差(米)，<=0表示不限制")
    parser.add_argument("--match_max_points_total", type=int, default=50000, help="七参数求解前最多保留多少全局点")
    parser.add_argument("--estimate_scale", action="store_true", help="七参数求解时估计尺度")

    # 点云 / ICP 参数
    parser.add_argument("--cloud_stride", type=int, default=4)
    parser.add_argument("--cloud_max_depth", type=float, default=0.0)
    parser.add_argument("--cloud_voxel_per_frame", type=float, default=0.0)
    parser.add_argument("--cloud_voxel_global", type=float, default=0.10)
    parser.add_argument("--cloud_max_points_per_frame", type=int, default=0)
    parser.add_argument("--cloud_max_points_total", type=int, default=100_000_000)

    parser.add_argument("--icp_method", type=str, default="point_to_point", choices=["point_to_plane", "point_to_point"])
    parser.add_argument("--icp_corr_thr", type=float, default=1.0)
    parser.add_argument("--icp_iters", type=int, default=500)
    parser.add_argument("--icp_sample_size", type=int, default=80_000, help="ICP 源点最大点数")
    parser.add_argument("--icp_target_sample_size", type=int, default=120_000, help="ICP 目标点最大点数")
    parser.add_argument("--icp_normal_k", type=int, default=20)
    parser.add_argument("--icp_relative_fitness", type=float, default=1e-6)
    parser.add_argument("--icp_relative_rmse", type=float, default=1e-6)

    parser.add_argument("--min_depth", type=float, default=1e-3)
    args = parser.parse_args()

    root_a = Path(args.root_a)
    root_b = Path(args.root_b)
    out_dir = ensure_dir(Path(args.out_dir))
    feature_cache_dir = Path(args.feature_match_cache_dir) if args.feature_match_cache_dir else (out_dir / "match_cache")

    data_a = collect_entries(root_a)
    data_b = collect_entries(root_b)
    common_stems = sorted(set(data_a.keys()) & set(data_b.keys()))
    if len(common_stems) == 0:
        raise RuntimeError("两个路径下没有找到同名视角（cams/images/depth 都齐全）")

    rng = random.Random(args.seed)
    stems_used = rng.sample(common_stems, args.num_images) if len(common_stems) > args.num_images else common_stems
    print(f"[INFO] common stems: {len(common_stems)} | use stems: {len(stems_used)}")

    match_result = None
    icp_result = None
    T_final = np.eye(4, dtype=np.float64)

    if args.mode == "only_match":
        match_result = run_match_pipeline(
            data_a, data_b, stems_used, args,
            ensure_dir(out_dir / "stage_match"),
            init_transform=None,
        )
        T_final = match_result["T"]

    elif args.mode == "only_icp":
        icp_result = run_icp_pipeline(
            data_a, data_b, stems_used, args,
            ensure_dir(out_dir / "stage_icp"),
            init_transform=None,
        )
        T_final = icp_result["T"]

    elif args.mode == "icp_then_match":
        icp_result = run_icp_pipeline(
            data_a, data_b, stems_used, args,
            ensure_dir(out_dir / "stage_icp"),
            init_transform=None,
        )
        match_result = run_match_pipeline(
            data_a, data_b, stems_used, args,
            ensure_dir(out_dir / "stage_match"),
            init_transform=icp_result["T"],
        )
        T_final = match_result["T"]

    elif args.mode == "match_then_icp":
        match_result = run_match_pipeline(
            data_a, data_b, stems_used, args,
            ensure_dir(out_dir / "stage_match"),
            init_transform=None,
        )
        icp_result = run_icp_pipeline(
            data_a, data_b, stems_used, args,
            ensure_dir(out_dir / "stage_icp"),
            init_transform=match_result["T"],
        )
        T_final = icp_result["T"]

    save_transform_txt(out_dir / "transform_refine.txt", T_final)
    final_summary = {
        "mode": args.mode,
        "final_transform": T_final.tolist(),
        "final_scale": float(decompose_sim3(T_final)[0]),
        "stage_match": match_result["summary"] if match_result is not None else None,
        "stage_icp": icp_result["summary"] if icp_result is not None else None,
        "saved_items": [
            "match_vis/",
            "match_cache/",
            "points_7p_before/",
            "points_7p_after/",
            "points_icp_before/",
            "points_icp_after/",
            "transform_match.txt",
            "transform_icp.txt",
            "transform_refine.txt",
            "summary*.json",
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] mode: {args.mode}")
    print(f"[INFO] final transform saved to: {out_dir / 'transform_refine.txt'}")


if __name__ == "__main__":
    main()



"""
# 根据选取的视角渲染的激光rgb和深度图，配准基于OBJ采样点云渲染的深度图和真实RGB

python refine_transform_v2.py \
    --root_a ./nanfang \
    --root_b ../recon/nanfang_render \
    --out_dir ./nanfang/transform \
    --mode match_then_icp \
    --estimate_scale --enable_translation_filter

python refine_transform_v2.py \
    --root_a ./yanghaitang \
    --root_b ../recon/yanghaitang_render \
    --out_dir ./yanghaitang/transform \
    --mode match_then_icp \
    --estimate_scale --enable_translation_filter

python refine_transform_v2.py \
    --root_a ./xiaoxiang \
    --root_b ../recon/xiaoxiang_render \
    --out_dir ./xiaoxiang/transform \
    --mode match_then_icp \
    --estimate_scale --enable_translation_filter
"""


"""
# 根据选取的视角渲染的激光rgb和深度图，配准基于OBJ采样点云渲染的深度图和真实RGB
# 只使用 ICP

python refine_transform_v2.py \
    --root_a ./nanfang \
    --root_b ../recon/nanfang_render \
    --out_dir ./nanfang/transform_only_icp \
    --mode only_icp \
    --estimate_scale --enable_translation_filter

python refine_transform_v2.py \
    --root_a ./yanghaitang \
    --root_b ../recon/yanghaitang_render \
    --out_dir ./yanghaitang/transform_only_icp \
    --mode only_icp \
    --estimate_scale --enable_translation_filter

python refine_transform_v2.py \
    --root_a ./xiaoxiang \
    --root_b ../recon/xiaoxiang_render \
    --out_dir ./xiaoxiang/transform_only_icp \
    --mode only_icp \
    --estimate_scale --enable_translation_filter
"""
