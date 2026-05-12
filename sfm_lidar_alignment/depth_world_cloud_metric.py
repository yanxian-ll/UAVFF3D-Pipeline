#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from scipy.spatial import cKDTree  # type: ignore
    HAVE_CKDTREE = True
except Exception:
    cKDTree = None
    HAVE_CKDTREE = False


DEPTH_EXTS = [".npy", ".exr", ".pfm", ".png", ".tiff", ".tif"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"]


# =========================================================
# utils
# =========================================================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def try_parse_float_line(line: str):
    vals = line.strip().split()
    if not vals:
        return None
    try:
        return [float(v) for v in vals]
    except Exception:
        return None


def find_file_for_stem(folder: Path, stem: str, exts: List[str]) -> Optional[Path]:
    for ext in exts:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    valid_exts = {e.lower() for e in exts}
    for p in folder.glob(f"{stem}.*"):
        if p.suffix.lower() in valid_exts:
            return p
    return None


def read_mask_any(path: Path) -> Optional[np.ndarray]:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return m


# =========================================================
# IO
# =========================================================
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
        vals = try_parse_float_line(lines[j])
        if vals is not None and len(vals) >= 4:
            RT.append(vals[:4])
        if len(RT) == 4:
            break
    if len(RT) != 4:
        raise ValueError(f"Failed to parse extrinsic from: {path_txt}")
    RT = np.asarray(RT, dtype=np.float64)

    K = []
    for j in range(in_idx + 1, min(in_idx + 8, len(lines))):
        vals = try_parse_float_line(lines[j])
        if vals is not None and len(vals) >= 3:
            K.append(vals[:3])
        if len(K) == 3:
            break
    if len(K) != 3:
        raise ValueError(f"Failed to parse intrinsic from: {path_txt}")
    K = np.asarray(K, dtype=np.float64)

    cam_h, cam_w = 0, 0
    for j in range(in_idx + 4, len(lines)):
        vals = try_parse_float_line(lines[j])
        if vals is not None and len(vals) >= 3:
            cam_h = int(round(vals[0]))
            cam_w = int(round(vals[1]))
            break

    cam2world = np.linalg.inv(RT)
    return K, cam2world, cam_h, cam_w


def read_depth_any(path: Path) -> Optional[np.ndarray]:
    path = Path(path)
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


def scale_intrinsics(K: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
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


# =========================================================
# geometry
# =========================================================
def depth_to_world_points_masked(
    depth: np.ndarray,
    K_depth: np.ndarray,
    cam2world: np.ndarray,
    exclude_mask: np.ndarray,
    stride: int = 4,
    min_depth: float = 1e-3,
    max_depth: float = 0.0,
) -> np.ndarray:
    h, w = depth.shape[:2]
    ys = np.arange(0, h, max(1, int(stride)), dtype=np.int32)
    xs = np.arange(0, w, max(1, int(stride)), dtype=np.int32)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    uu, vv = np.meshgrid(xs, ys)
    z = depth[vv, uu].astype(np.float64).reshape(-1)
    u = uu.reshape(-1).astype(np.float64)
    v = vv.reshape(-1).astype(np.float64)
    m = exclude_mask[vv, uu].reshape(-1)

    valid = np.isfinite(z) & (z > min_depth) & (m == 0)
    if max_depth is not None and max_depth > 0:
        valid &= (z <= max_depth)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)

    z = z[valid]
    u = u[valid]
    v = v[valid]

    fx = float(K_depth[0, 0])
    fy = float(K_depth[1, 1])
    cx = float(K_depth[0, 2])
    cy = float(K_depth[1, 2])

    x = (u - cx) * z / max(fx, 1e-12)
    y = (v - cy) * z / max(fy, 1e-12)
    pts_cam = np.stack([x, y, z], axis=1)
    R = cam2world[:3, :3]
    t = cam2world[:3, 3]
    pts_world = pts_cam @ R.T + t
    return pts_world.astype(np.float64)


# =========================================================
# metrics
# =========================================================
def nearest_neighbor_dists(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    if HAVE_CKDTREE:
        tree = cKDTree(dst)
        try:
            dists, _ = tree.query(src, k=1, workers=-1)
        except TypeError:
            dists, _ = tree.query(src, k=1)
        return np.asarray(dists, dtype=np.float64)

    block = 2000
    out = []
    dst2 = np.sum(dst * dst, axis=1)[None, :]
    for st in range(0, src.shape[0], block):
        x = src[st:st + block]
        x2 = np.sum(x * x, axis=1, keepdims=True)
        dist2 = np.maximum(x2 + dst2 - 2.0 * (x @ dst.T), 0.0)
        out.append(np.sqrt(np.min(dist2, axis=1)))
    return np.concatenate(out, axis=0)


def compute_distance_stats(d: np.ndarray, thresholds: List[float]) -> Dict[str, Optional[float]]:
    d = np.asarray(d, dtype=np.float64)
    if d.size == 0:
        out: Dict[str, Optional[float]] = {
            "count": 0,
            "mean": None,
            "median": None,
            "rmse": None,
            "p95": None,
            "max": None,
        }
        for thr in thresholds:
            out[f"inlier@{thr:g}"] = None
        return out

    out = {
        "count": int(d.size),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
    }
    for thr in thresholds:
        out[f"inlier@{thr:g}"] = float(np.mean(d <= float(thr)))
    return out


def symmetric_cloud_metrics(cloud_a: np.ndarray, cloud_b: np.ndarray, thresholds: List[float]) -> Dict[str, object]:
    d_ab = nearest_neighbor_dists(cloud_a, cloud_b)
    d_ba = nearest_neighbor_dists(cloud_b, cloud_a)
    d_sym = np.concatenate([d_ab, d_ba], axis=0) if (d_ab.size + d_ba.size) > 0 else np.zeros((0,), dtype=np.float64)

    return {
        "A_to_B": compute_distance_stats(d_ab, thresholds),
        "B_to_A": compute_distance_stats(d_ba, thresholds),
        "symmetric": compute_distance_stats(d_sym, thresholds),
        "chamfer_L1": (float(d_ab.mean() + d_ba.mean()) if d_ab.size > 0 and d_ba.size > 0 else None),
        "chamfer_L2": (float(np.mean(d_ab ** 2) + np.mean(d_ba ** 2)) if d_ab.size > 0 and d_ba.size > 0 else None),
    }


# =========================================================
# output
# =========================================================
def save_point_cloud_ply(path: Path, points: np.ndarray, color=(200, 200, 200)):
    points = np.asarray(points, dtype=np.float64)
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
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        for p in points:
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {r} {g} {b}\n")


def flatten_metric_dict(prefix: str, obj: Dict[str, object], out: Dict[str, object]):
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flatten_metric_dict(key, v, out)
        else:
            out[key] = v


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="输入两套 depth_npy、cams、exclude_masks，转世界坐标系后计算点云对齐指标"
    )
    parser.add_argument("--depth_dir_a", type=str, required=True, help="深度图目录 A")
    parser.add_argument("--depth_dir_b", type=str, required=True, help="深度图目录 B")
    parser.add_argument("--cam_dir", type=str, required=True, help="cams 目录")
    parser.add_argument("--exclude_mask_dir", type=str, required=True, help="exclude_masks 目录，255=剔除")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")

    parser.add_argument("--stride", type=int, default=4, help="深度转点云采样步长")
    parser.add_argument("--min_depth", type=float, default=1e-3)
    parser.add_argument("--max_depth", type=float, default=0.0, help="最大深度，<=0 表示不限制")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5], help="内点阈值（米）")
    parser.add_argument("--save_ply", action="store_true", help="保存两组世界点云")
    args = parser.parse_args()

    depth_dir_a = Path(args.depth_dir_a)
    depth_dir_b = Path(args.depth_dir_b)
    cam_dir = Path(args.cam_dir)
    exclude_mask_dir = Path(args.exclude_mask_dir)
    out_dir = ensure_dir(Path(args.out_dir))

    cam_stems = {p.stem for p in cam_dir.glob("*.txt")}
    depth_a_stems = {p.stem for p in depth_dir_a.iterdir() if p.is_file() and p.suffix.lower() in {e.lower() for e in DEPTH_EXTS}}
    depth_b_stems = {p.stem for p in depth_dir_b.iterdir() if p.is_file() and p.suffix.lower() in {e.lower() for e in DEPTH_EXTS}}
    mask_stems = {p.stem for p in exclude_mask_dir.iterdir() if p.is_file() and p.suffix.lower() in {e.lower() for e in MASK_EXTS}}

    stems = sorted(cam_stems & depth_a_stems & depth_b_stems & mask_stems)
    if len(stems) == 0:
        raise RuntimeError("没有找到公共视角，请检查 depth_dir_a / depth_dir_b / cam_dir / exclude_mask_dir")

    per_view_rows: List[Dict[str, object]] = []
    global_a: List[np.ndarray] = []
    global_b: List[np.ndarray] = []

    for stem in tqdm(stems, desc="Build world clouds", dynamic_ncols=True):
        cam_path = cam_dir / f"{stem}.txt"
        depth_path_a = find_file_for_stem(depth_dir_a, stem, DEPTH_EXTS)
        depth_path_b = find_file_for_stem(depth_dir_b, stem, DEPTH_EXTS)
        mask_path = find_file_for_stem(exclude_mask_dir, stem, MASK_EXTS)
        if depth_path_a is None or depth_path_b is None or mask_path is None:
            continue

        K_nom, cam2world, cam_h, cam_w = read_cam_blendedmvs_txt(str(cam_path))
        depth_a = read_depth_any(depth_path_a)
        depth_b = read_depth_any(depth_path_b)
        mask = read_mask_any(mask_path)
        if depth_a is None or depth_b is None or mask is None:
            continue

        h_a, w_a = depth_a.shape[:2]
        h_b, w_b = depth_b.shape[:2]
        K_a = scale_intrinsics(K_nom, cam_w, cam_h, w_a, h_a)
        K_b = scale_intrinsics(K_nom, cam_w, cam_h, w_b, h_b)

        mask_a = cv2.resize(mask, (w_a, h_a), interpolation=cv2.INTER_NEAREST)
        mask_b = cv2.resize(mask, (w_b, h_b), interpolation=cv2.INTER_NEAREST)

        pts_a = depth_to_world_points_masked(
            depth_a, K_a, cam2world, mask_a,
            stride=args.stride,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        pts_b = depth_to_world_points_masked(
            depth_b, K_b, cam2world, mask_b,
            stride=args.stride,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )

        m = symmetric_cloud_metrics(pts_a, pts_b, args.thresholds)
        row: Dict[str, object] = {
            "stem": stem,
            "num_points_a": int(pts_a.shape[0]),
            "num_points_b": int(pts_b.shape[0]),
            "mask_exclude_ratio_a": float((mask_a > 0).mean()),
            "mask_exclude_ratio_b": float((mask_b > 0).mean()),
        }
        flatten_metric_dict("", m, row)
        per_view_rows.append(row)

        if pts_a.shape[0] > 0:
            global_a.append(pts_a)
        if pts_b.shape[0] > 0:
            global_b.append(pts_b)

    if len(global_a) == 0 or len(global_b) == 0:
        raise RuntimeError("有效点云为空，无法计算全局指标。")

    cloud_a = np.concatenate(global_a, axis=0)
    cloud_b = np.concatenate(global_b, axis=0)
    global_metrics = symmetric_cloud_metrics(cloud_a, cloud_b, args.thresholds)

    csv_path = out_dir / "metrics_per_view.csv"
    if len(per_view_rows) > 0:
        fieldnames = list(per_view_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_view_rows:
                writer.writerow(row)

    with open(out_dir / "metrics_per_view.json", "w", encoding="utf-8") as f:
        json.dump(per_view_rows, f, ensure_ascii=False, indent=2)

    summary = {
        "num_valid_views": int(len(per_view_rows)),
        "depth_dir_a": str(depth_dir_a),
        "depth_dir_b": str(depth_dir_b),
        "cam_dir": str(cam_dir),
        "exclude_mask_dir": str(exclude_mask_dir),
        "args": vars(args),
        "global_num_points_a": int(cloud_a.shape[0]),
        "global_num_points_b": int(cloud_b.shape[0]),
        "global_metrics": global_metrics,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("========== World-Cloud Registration Metrics ==========" + "\n")
        f.write(f"num_valid_views: {summary['num_valid_views']}\n")
        f.write(f"global_num_points_a: {summary['global_num_points_a']}\n")
        f.write(f"global_num_points_b: {summary['global_num_points_b']}\n")
        f.write("\n[global_metrics]\n")
        flat = {}
        flatten_metric_dict("", global_metrics, flat)
        for k, v in flat.items():
            f.write(f"{k}: {v}\n")

    if args.save_ply:
        save_point_cloud_ply(out_dir / "cloud_a_world.ply", cloud_a, color=(255, 80, 80))
        save_point_cloud_ply(out_dir / "cloud_b_world.ply", cloud_b, color=(80, 255, 80))

    print(f"[INFO] 有效视角数: {len(per_view_rows)}")
    print(f"[INFO] 全局点数 A: {cloud_a.shape[0]} | B: {cloud_b.shape[0]}")
    print(f"[INFO] 输出目录: {out_dir}")
    flat = {}
    flatten_metric_dict("", global_metrics, flat)
    print("[INFO] 全局关键指标:")
    for k in [
        "A_to_B.mean", "B_to_A.mean", "symmetric.mean",
        "symmetric.rmse", "symmetric.p95", "chamfer_L1", "chamfer_L2"
    ]:
        if k in flat:
            print(f"  {k}: {flat[k]}")


if __name__ == "__main__":
    main()


"""
# 粗对齐后评估
python depth_world_cloud_metric.py \
    --depth_dir_a lidar/nanfang/depth_npy \
    --depth_dir_b recon/nanfang_render/depth_npy \
    --cam_dir recon/nanfang_render/cams \
    --exclude_mask_dir ./eval_mask/nanfang_eval_masks/binary_masks \
    --out_dir ./eval/nanfang_world_metrics_masked_coarse \
    --stride 8 \
    --thresholds 0.05 0.1 0.2 0.5

    
python depth_world_cloud_metric.py \
    --depth_dir_a lidar/yanghaitang/depth_npy \
    --depth_dir_b recon/yanghaitang_render/depth_npy \
    --cam_dir recon/yanghaitang_render/cams \
    --exclude_mask_dir ./eval_mask/yanghaitang_eval_masks/binary_masks \
    --out_dir ./eval/yanghaitang_world_metrics_masked_coarse \
    --stride 8 \
    --thresholds 0.05 0.1 0.2 0.5
    
"""



"""
# 精对齐后评估
python depth_world_cloud_metric.py \
    --depth_dir_a lidar/nanfang_final/depth_npy \
    --depth_dir_b recon/nanfang_render/depth_npy \
    --cam_dir recon/nanfang_render/cams \
    --exclude_mask_dir ./eval_mask/nanfang_eval_masks/binary_masks \
    --out_dir ./eval/nanfang_world_metrics_masked \
    --stride 8 \
    --thresholds 0.05 0.1 0.2 0.5 \

"""