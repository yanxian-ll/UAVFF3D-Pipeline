#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
以激光点云为主，将 OBJ 采样点云中“激光未覆盖”或“激光局部缺失”的区域补到最终结果中。

设计目标：
1) 不做简单拼接，而是优先保留 LiDAR；
2) 仅在 LiDAR 缺失/稀疏区域引入 OBJ 采样点；
3) 尽量抑制重叠区域的重复点；
4) 两个输入点云已大体对齐，但可选做一次小范围 ICP 微调。

依赖：
    pip install open3d scipy numpy tqdm

示例：
    python fuse_lidar_obj_completion.py \
        --lidar lidar.ply \
        --obj obj_sampled.ply \
        --out_dir fused_out \
        --refine_icp \
        --voxel_size -1

说明：
- 默认参数大量采用“auto”策略，会根据 LiDAR 的中位点间距自动推断阈值。
- 若你的数据单位不是“米”，阈值也会随点云尺度自适应。
- 如果输入点数极大，建议开启 voxel 下采样（默认 auto）；若你想保留最高密度，可设 --voxel_size 0。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm


# =========================
# 基础工具
# =========================
def log(msg: str) -> None:
    print(msg, flush=True)


def as_numpy_colors(pcd: o3d.geometry.PointCloud) -> Optional[np.ndarray]:
    if pcd.has_colors() and len(pcd.colors) == len(pcd.points):
        return np.asarray(pcd.colors).copy()
    return None


def numpy_to_pcd(points: np.ndarray, colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and len(colors) == len(points):
        colors = np.clip(colors, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def load_point_cloud(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError(f"点云为空: {path}")
    cols = as_numpy_colors(pcd)
    return pts.copy(), cols


def remove_non_finite(points: np.ndarray, colors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    if colors is not None:
        colors = colors[mask]
    return points, colors


def voxel_downsample_np(points: np.ndarray,
                        colors: Optional[np.ndarray],
                        voxel_size: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if voxel_size <= 0:
        return points, colors
    pcd = numpy_to_pcd(points, colors)
    pcd = pcd.voxel_down_sample(voxel_size)
    out_pts = np.asarray(pcd.points).copy()
    out_cols = as_numpy_colors(pcd)
    return out_pts, out_cols


def remove_statistical_outlier_np(points: np.ndarray,
                                  colors: Optional[np.ndarray],
                                  nb_neighbors: int,
                                  std_ratio: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(points) == 0:
        return points, colors
    pcd = numpy_to_pcd(points, colors)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    out_pts = np.asarray(pcd.points).copy()
    out_cols = as_numpy_colors(pcd)
    return out_pts, out_cols


def estimate_spacing(points: np.ndarray, sample_size: int = 50000, seed: int = 42) -> float:
    """估计 LiDAR 的中位最近邻间距，用于自动阈值。"""
    if len(points) < 2:
        raise ValueError("点数不足，无法估计点间距")

    rng = np.random.default_rng(seed)
    if len(points) > sample_size:
        idx = rng.choice(len(points), size=sample_size, replace=False)
        pts = points[idx]
    else:
        pts = points

    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2, workers=-1)
    # 第0列是自己到自己距离=0，第1列是最近邻
    spacing = float(np.median(d[:, 1]))
    if not np.isfinite(spacing) or spacing <= 0:
        # 极端情况下回退到 bbox 尺度的一个很小比例
        bbox = pts.max(axis=0) - pts.min(axis=0)
        spacing = max(float(np.linalg.norm(bbox)) * 1e-5, 1e-6)
    return spacing


def estimate_normals_inplace(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int = 30) -> None:
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))


# =========================
# voxel key / 覆盖区判定
# =========================
def voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0:
        raise ValueError("voxel 必须 > 0")
    return np.floor(points / voxel).astype(np.int64)


def unique_rows_int(arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr)
    view = arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    unique = np.unique(view)
    return unique.view(arr.dtype).reshape(-1, arr.shape[1])


def row_membership(query: np.ndarray, ref_unique: np.ndarray) -> np.ndarray:
    query = np.ascontiguousarray(query)
    ref_unique = np.ascontiguousarray(ref_unique)
    qv = query.view(np.dtype((np.void, query.dtype.itemsize * query.shape[1])))
    rv = ref_unique.view(np.dtype((np.void, ref_unique.dtype.itemsize * ref_unique.shape[1])))
    return np.isin(qv, rv)


def dilate_voxel_keys(keys_unique: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0:
        return keys_unique
    offsets = np.array(np.meshgrid(
        np.arange(-margin, margin + 1),
        np.arange(-margin, margin + 1),
        np.arange(-margin, margin + 1),
        indexing="ij"
    )).reshape(3, -1).T.astype(np.int64)
    expanded = keys_unique[:, None, :] + offsets[None, :, :]
    expanded = expanded.reshape(-1, 3)
    return unique_rows_int(expanded)


# =========================
# ICP 微调（可选）
# =========================
def refine_alignment_icp(obj_points: np.ndarray,
                         obj_colors: Optional[np.ndarray],
                         lidar_points: np.ndarray,
                         lidar_colors: Optional[np.ndarray],
                         normal_radius: float,
                         max_corr: float,
                         max_iter: int) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Dict]:
    src = numpy_to_pcd(obj_points, obj_colors)
    tgt = numpy_to_pcd(lidar_points, lidar_colors)

    estimate_normals_inplace(src, radius=normal_radius)
    estimate_normals_inplace(tgt, radius=normal_radius)

    reg = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_correspondence_distance=max_corr,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )

    src.transform(reg.transformation)
    out_pts = np.asarray(src.points).copy()
    out_cols = as_numpy_colors(src)
    info = {
        "fitness": float(reg.fitness),
        "inlier_rmse": float(reg.inlier_rmse),
        "transformation": np.asarray(reg.transformation).tolist(),
    }
    return out_pts, out_cols, np.asarray(reg.transformation), info


# =========================
# 补点逻辑
# =========================
def fit_plane_svd(neighbors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = neighbors.mean(axis=0)
    centered = neighbors - centroid
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    return centroid, normal, s


def decide_supplement_points(lidar_points: np.ndarray,
                             obj_points: np.ndarray,
                             coverage_voxel: float,
                             coverage_margin: int,
                             duplicate_dist: float,
                             support_radius: float,
                             plane_tol: float,
                             k_fit: int,
                             min_support: int,
                             max_hole_nn_dist: float,
                             verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    返回：
        keep_mask: 哪些 obj 点应加入最终补点
        stats: 统计信息
    """
    if len(lidar_points) == 0 or len(obj_points) == 0:
        return np.zeros((len(obj_points),), dtype=bool), {}

    # 1) 近邻距离：快速剔除重复点
    log("[1/4] 构建 LiDAR KDTree ...") if verbose else None
    lidar_tree = cKDTree(lidar_points)
    d_nn, _ = lidar_tree.query(obj_points, k=1, workers=-1)

    # 2) 覆盖区判定：落在 LiDAR 未覆盖体素中的点，优先作为候选补点
    log("[2/4] 计算 LiDAR 覆盖体素 ...") if verbose else None
    lidar_keys = voxel_keys(lidar_points, coverage_voxel)
    lidar_keys_unique = unique_rows_int(lidar_keys)
    lidar_keys_unique = dilate_voxel_keys(lidar_keys_unique, coverage_margin)

    obj_keys = voxel_keys(obj_points, coverage_voxel)
    in_covered_voxel = row_membership(obj_keys, lidar_keys_unique)

    uncovered_keep = (~in_covered_voxel) & (d_nn > duplicate_dist)

    # 3) 对“在覆盖体素里，但离 LiDAR 最近点不算太近”的 OBJ 点，
    #    做局部平面一致性检测，用于补建筑立面/墙面等局部空洞。
    log("[3/4] 检测 LiDAR 局部孔洞并补点 ...") if verbose else None
    hole_candidate_mask = (
        in_covered_voxel
        & (d_nn > duplicate_dist)
        & (d_nn <= max_hole_nn_dist)
    )
    hole_candidate_idx = np.flatnonzero(hole_candidate_mask)
    hole_keep = np.zeros(len(obj_points), dtype=bool)

    if len(hole_candidate_idx) > 0:
        cand_points = obj_points[hole_candidate_idx]
        dk, ik = lidar_tree.query(cand_points, k=k_fit, distance_upper_bound=support_radius, workers=-1)
        if k_fit == 1:
            dk = dk[:, None]
            ik = ik[:, None]

        iterator = range(len(hole_candidate_idx))
        if verbose and len(hole_candidate_idx) > 5000:
            iterator = tqdm(iterator, total=len(hole_candidate_idx), desc="局部平面筛选")

        n_lidar = len(lidar_points)
        for j in iterator:
            valid = np.isfinite(dk[j]) & (ik[j] < n_lidar)
            if int(valid.sum()) < min_support:
                continue

            nbrs = lidar_points[ik[j][valid]]
            centroid, normal, svals = fit_plane_svd(nbrs)

            # 退化邻域：跳过
            if not np.isfinite(svals).all() or svals[0] <= 1e-12:
                continue

            # 简单平面性约束：邻域必须更像平面，而不是各向同性团块
            planarity = (svals[1] - svals[2]) / (svals[0] + 1e-12)
            if planarity < 0.15:
                continue

            p = cand_points[j]
            plane_dist = abs(np.dot(p - centroid, normal))

            # 满足“与 LiDAR 局部表面共面，但最近点距离又偏大” => 视作 LiDAR 空洞补点
            if plane_dist <= plane_tol:
                hole_keep[hole_candidate_idx[j]] = True

    keep_mask = uncovered_keep | hole_keep

    stats = {
        "obj_total": int(len(obj_points)),
        "uncovered_keep": int(uncovered_keep.sum()),
        "hole_candidates": int(len(hole_candidate_idx)),
        "hole_keep": int(hole_keep.sum()),
        "keep_total": int(keep_mask.sum()),
        "discard_total": int((~keep_mask).sum()),
        "median_obj_to_lidar_nn": float(np.median(d_nn)),
        "p95_obj_to_lidar_nn": float(np.percentile(d_nn, 95)),
    }
    return keep_mask, stats


# =========================
# 主流程
# =========================
def auto_or_value(user_value: float, auto_value: float) -> float:
    return auto_value if user_value < 0 else user_value


def main() -> None:
    parser = argparse.ArgumentParser(description="融合 LiDAR 与 OBJ 采样点云：以 LiDAR 为主，对缺失区域做受控补点")
    parser.add_argument("--lidar", required=True, help="LiDAR 点云 PLY")
    parser.add_argument("--obj", required=True, help="OBJ 采样点云 PLY（范围更广）")
    parser.add_argument("--out_dir", required=True, help="输出目录")

    # 预处理
    parser.add_argument("--voxel_size", type=float, default=-1.0,
                        help="体素下采样尺寸；<0 表示自动(约 1.5x LiDAR 中位点间距)，0 表示不下采样")
    parser.add_argument("--clean_outlier", action="store_true", help="是否做统计离群点剔除")
    parser.add_argument("--sor_nb_neighbors", type=int, default=20)
    parser.add_argument("--sor_std_ratio", type=float, default=2.0)

    # 对齐微调
    parser.add_argument("--refine_icp", action="store_true", help="已大致对齐的情况下，是否再做一次小范围 ICP 微调")
    parser.add_argument("--icp_max_corr", type=float, default=-1.0,
                        help="ICP 最大对应距离；<0 自动(约 8x 点间距)")
    parser.add_argument("--icp_max_iter", type=int, default=50)

    # 融合参数（大量支持 auto）
    parser.add_argument("--coverage_voxel", type=float, default=-1.0,
                        help="LiDAR 覆盖判定体素；<0 自动(约 3x 点间距)")
    parser.add_argument("--coverage_margin", type=int, default=1,
                        help="对 LiDAR 覆盖体素做膨胀，1 表示 26 邻域，0 表示不膨胀")
    parser.add_argument("--duplicate_dist", type=float, default=-1.0,
                        help="与 LiDAR 最近点距离小于该值时视为重复；<0 自动(约 1.2x 点间距)")
    parser.add_argument("--support_radius", type=float, default=-1.0,
                        help="局部平面拟合的 LiDAR 邻域半径；<0 自动(约 6x 点间距)")
    parser.add_argument("--plane_tol", type=float, default=-1.0,
                        help="点到 LiDAR 局部平面的距离阈值；<0 自动(约 2x 点间距)")
    parser.add_argument("--k_fit", type=int, default=20, help="局部平面拟合最多取多少个 LiDAR 邻点")
    parser.add_argument("--min_support", type=int, default=6, help="局部平面拟合最少需要多少个 LiDAR 邻点")
    parser.add_argument("--max_hole_nn_dist", type=float, default=-1.0,
                        help="仅对最近 LiDAR 距离不超过该值的 covered 点执行“空洞补点”判定；<0 自动(约 8x 点间距)")

    # 输出控制
    parser.add_argument("--save_debug_source_color", action="store_true",
                        help="额外保存一个按来源着色的调试点云（LiDAR=灰，补点=红）")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 读入 ----------
    log("读取点云 ...")
    lidar_pts_raw, lidar_cols_raw = load_point_cloud(args.lidar)
    obj_pts_raw, obj_cols_raw = load_point_cloud(args.obj)

    lidar_pts_raw, lidar_cols_raw = remove_non_finite(lidar_pts_raw, lidar_cols_raw)
    obj_pts_raw, obj_cols_raw = remove_non_finite(obj_pts_raw, obj_cols_raw)

    log(f"LiDAR 原始点数: {len(lidar_pts_raw):,}")
    log(f"OBJ采样原始点数: {len(obj_pts_raw):,}")

    spacing = estimate_spacing(lidar_pts_raw)
    log(f"估计 LiDAR 中位点间距: {spacing:.6f}")

    voxel_size = auto_or_value(args.voxel_size, 1.5 * spacing)
    coverage_voxel = auto_or_value(args.coverage_voxel, 3.0 * spacing)
    duplicate_dist = auto_or_value(args.duplicate_dist, 1.2 * spacing)
    support_radius = auto_or_value(args.support_radius, 6.0 * spacing)
    plane_tol = auto_or_value(args.plane_tol, 2.0 * spacing)
    icp_max_corr = auto_or_value(args.icp_max_corr, 8.0 * spacing)
    max_hole_nn_dist = auto_or_value(args.max_hole_nn_dist, 8.0 * spacing)

    param_report = {
        "estimated_spacing": spacing,
        "voxel_size": voxel_size,
        "coverage_voxel": coverage_voxel,
        "coverage_margin": args.coverage_margin,
        "duplicate_dist": duplicate_dist,
        "support_radius": support_radius,
        "plane_tol": plane_tol,
        "k_fit": args.k_fit,
        "min_support": args.min_support,
        "icp_max_corr": icp_max_corr,
        "icp_max_iter": args.icp_max_iter,
        "max_hole_nn_dist": max_hole_nn_dist,
        "clean_outlier": bool(args.clean_outlier),
        "sor_nb_neighbors": args.sor_nb_neighbors,
        "sor_std_ratio": args.sor_std_ratio,
        "refine_icp": bool(args.refine_icp),
    }

    log("实际参数：")
    for k, v in param_report.items():
        log(f"  - {k}: {v}")

    # ---------- 预处理 ----------
    lidar_pts, lidar_cols = lidar_pts_raw, lidar_cols_raw
    obj_pts, obj_cols = obj_pts_raw, obj_cols_raw

    if voxel_size > 0:
        log(f"体素下采样中 ... voxel_size={voxel_size:.6f}")
        lidar_pts, lidar_cols = voxel_downsample_np(lidar_pts, lidar_cols, voxel_size)
        obj_pts, obj_cols = voxel_downsample_np(obj_pts, obj_cols, voxel_size)
        log(f"LiDAR 下采样后点数: {len(lidar_pts):,}")
        log(f"OBJ采样下采样后点数: {len(obj_pts):,}")

    if args.clean_outlier:
        log("执行统计离群点剔除 ...")
        lidar_pts, lidar_cols = remove_statistical_outlier_np(
            lidar_pts, lidar_cols, args.sor_nb_neighbors, args.sor_std_ratio
        )
        obj_pts, obj_cols = remove_statistical_outlier_np(
            obj_pts, obj_cols, args.sor_nb_neighbors, args.sor_std_ratio
        )
        log(f"LiDAR 去噪后点数: {len(lidar_pts):,}")
        log(f"OBJ采样去噪后点数: {len(obj_pts):,}")

    # ---------- ICP 微调（可选） ----------
    icp_info = None
    icp_transform = np.eye(4)
    if args.refine_icp:
        log("执行 ICP 微调对齐（OBJ -> LiDAR）...")
        obj_pts, obj_cols, icp_transform, icp_info = refine_alignment_icp(
            obj_pts, obj_cols,
            lidar_pts, lidar_cols,
            normal_radius=max(2.0 * spacing, support_radius),
            max_corr=icp_max_corr,
            max_iter=args.icp_max_iter,
        )
        log(f"ICP fitness={icp_info['fitness']:.6f}, inlier_rmse={icp_info['inlier_rmse']:.6f}")

    # ---------- 融合判定 ----------
    keep_mask, fusion_stats = decide_supplement_points(
        lidar_points=lidar_pts,
        obj_points=obj_pts,
        coverage_voxel=coverage_voxel,
        coverage_margin=args.coverage_margin,
        duplicate_dist=duplicate_dist,
        support_radius=support_radius,
        plane_tol=plane_tol,
        k_fit=args.k_fit,
        min_support=args.min_support,
        max_hole_nn_dist=max_hole_nn_dist,
        verbose=True,
    )

    supplement_pts = obj_pts[keep_mask]
    supplement_cols = obj_cols[keep_mask] if obj_cols is not None else None

    # 补点再做一次轻量去噪，避免边缘单点残留
    if args.clean_outlier and len(supplement_pts) > 0:
        log("对补点结果做一次轻量去噪 ...")
        supplement_pts, supplement_cols = remove_statistical_outlier_np(
            supplement_pts, supplement_cols,
            nb_neighbors=max(10, min(args.sor_nb_neighbors, 20)),
            std_ratio=max(1.5, args.sor_std_ratio),
        )

    # ---------- 合并输出 ----------
    log("生成输出点云 ...")
    merged_pts = np.concatenate([lidar_pts, supplement_pts], axis=0)

    # 只有双方都带颜色时才直接保留原颜色；否则输出几何点云
    merged_cols = None
    if lidar_cols is not None and supplement_cols is not None:
        merged_cols = np.concatenate([lidar_cols, supplement_cols], axis=0)

    merged_pcd = numpy_to_pcd(merged_pts, merged_cols)
    supplement_pcd = numpy_to_pcd(supplement_pts, supplement_cols)
    lidar_pcd = numpy_to_pcd(lidar_pts, lidar_cols)

    lidar_path = out_dir / "lidar_processed.ply"
    supplement_path = out_dir / "supplement_from_obj.ply"
    merged_path = out_dir / "merged_fused.ply"

    o3d.io.write_point_cloud(str(lidar_path), lidar_pcd, write_ascii=False)
    o3d.io.write_point_cloud(str(supplement_path), supplement_pcd, write_ascii=False)
    o3d.io.write_point_cloud(str(merged_path), merged_pcd, write_ascii=False)

    debug_path = None
    if args.save_debug_source_color:
        lidar_debug_cols = np.tile(np.array([[0.70, 0.70, 0.70]], dtype=np.float64), (len(lidar_pts), 1))
        supplement_debug_cols = np.tile(np.array([[0.85, 0.15, 0.15]], dtype=np.float64), (len(supplement_pts), 1))
        debug_pts = np.concatenate([lidar_pts, supplement_pts], axis=0)
        debug_cols = np.concatenate([lidar_debug_cols, supplement_debug_cols], axis=0)
        debug_pcd = numpy_to_pcd(debug_pts, debug_cols)
        debug_path = out_dir / "merged_debug_sourcecolor.ply"
        o3d.io.write_point_cloud(str(debug_path), debug_pcd, write_ascii=False)

    # ---------- 统计 ----------
    summary = {
        "input": {
            "lidar_raw_points": int(len(lidar_pts_raw)),
            "obj_raw_points": int(len(obj_pts_raw)),
        },
        "processed": {
            "lidar_points": int(len(lidar_pts)),
            "obj_points": int(len(obj_pts)),
            "supplement_points": int(len(supplement_pts)),
            "merged_points": int(len(merged_pts)),
        },
        "params": param_report,
        "fusion_stats": fusion_stats,
        "icp": icp_info,
        "outputs": {
            "lidar_processed": str(lidar_path),
            "supplement_from_obj": str(supplement_path),
            "merged_fused": str(merged_path),
            "merged_debug_sourcecolor": str(debug_path) if debug_path is not None else None,
        }
    }

    with open(out_dir / "fusion_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("完成。")
    log(f"输出：{merged_path}")
    log(f"补点数量：{len(supplement_pts):,}")


if __name__ == "__main__":
    main()

