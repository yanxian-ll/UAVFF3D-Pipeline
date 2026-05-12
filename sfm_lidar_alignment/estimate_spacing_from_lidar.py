import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import open3d as o3d

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def load_ply_points(ply_path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd is None or pcd.is_empty():
        raise RuntimeError(f"无法读取点云或点云为空: {ply_path}")

    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise RuntimeError(f"点云格式异常: {pts.shape}")
    return pts


def build_xy_grid(points: np.ndarray, nx: int, ny: int):
    """
    按 XY 将整云划成 nx * ny 的规则网格。
    """
    xy = points[:, :2]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    extent = maxs - mins

    if extent[0] <= 0 or extent[1] <= 0:
        raise RuntimeError("点云 XY 范围异常，无法划分网格")

    dx = extent[0] / nx
    dy = extent[1] / ny

    ix = np.floor((xy[:, 0] - mins[0]) / dx).astype(np.int64)
    iy = np.floor((xy[:, 1] - mins[1]) / dy).astype(np.int64)

    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    cell_to_indices = defaultdict(list)
    for idx, (cx, cy) in enumerate(zip(ix, iy)):
        cell_to_indices[(int(cx), int(cy))].append(idx)

    return {
        "mins": mins,
        "maxs": maxs,
        "dx": dx,
        "dy": dy,
        "ix": ix,
        "iy": iy,
        "cell_to_indices": cell_to_indices,
        "nx": nx,
        "ny": ny,
    }


def choose_cells(cell_to_indices, max_cells: int, min_points_per_cell: int, seed: int = 0):
    """
    在非空网格中挑一些点数足够的格子。
    """
    valid_cells = []
    for cell, inds in cell_to_indices.items():
        if len(inds) >= min_points_per_cell:
            valid_cells.append(cell)

    if len(valid_cells) == 0:
        raise RuntimeError(
            f"没有找到点数 >= {min_points_per_cell} 的网格，请减小 min_points_per_cell 或调整网格大小"
        )

    rng = np.random.default_rng(seed)
    chosen_n = min(max_cells, len(valid_cells))
    chosen_idx = rng.choice(len(valid_cells), size=chosen_n, replace=False)
    chosen_cells = [valid_cells[i] for i in chosen_idx]
    return chosen_cells, valid_cells


def gather_neighbor_cell_indices(cell_to_indices, cx: int, cy: int, nx: int, ny: int, halo_cells: int):
    """
    取 (cx, cy) 周围 halo_cells 圈网格中的全部点索引。
    """
    out = []
    x0 = max(0, cx - halo_cells)
    x1 = min(nx - 1, cx + halo_cells)
    y0 = max(0, cy - halo_cells)
    y1 = min(ny - 1, cy + halo_cells)

    for xx in range(x0, x1 + 1):
        for yy in range(y0, y1 + 1):
            inds = cell_to_indices.get((xx, yy), [])
            if len(inds) > 0:
                out.extend(inds)

    if len(out) == 0:
        return np.empty((0,), dtype=np.int64)

    return np.asarray(out, dtype=np.int64)


def random_subsample_indices(indices: np.ndarray, max_num: int, seed: int = 0):
    if indices.size <= max_num:
        return indices

    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=max_num, replace=False)
    return np.asarray(chosen, dtype=np.int64)


def knn_spacing_ckdtree(search_points: np.ndarray, query_points: np.ndarray, k: int = 6) -> np.ndarray:
    tree = cKDTree(search_points)
    dists, _ = tree.query(query_points, k=k + 1, workers=-1)
    spacing = np.mean(dists[:, 1:k + 1], axis=1)
    return spacing


def knn_spacing_open3d(search_points: np.ndarray, query_points: np.ndarray, k: int = 6) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(search_points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    out = np.zeros(query_points.shape[0], dtype=np.float64)
    for i, p in enumerate(query_points):
        kk, idx, dist2 = kdtree.search_knn_vector_3d(p, k + 1)
        if kk <= 1:
            out[i] = np.nan
        else:
            cur = np.sqrt(np.asarray(dist2[1:kk], dtype=np.float64))
            out[i] = np.mean(cur)
    return out


def robust_filter(x: np.ndarray, low_q=5, high_q=95) -> np.ndarray:
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size == 0:
        raise RuntimeError("没有有效的局部间距数据")

    lo = np.percentile(x, low_q)
    hi = np.percentile(x, high_q)
    x2 = x[(x >= lo) & (x <= hi)]
    if x2.size == 0:
        return x
    return x2


def calc_stats(x: np.ndarray) -> dict:
    return {
        "count": int(x.size),
        "min": float(np.min(x)),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p60": float(np.percentile(x, 60)),
        "p70": float(np.percentile(x, 70)),
        "p75": float(np.percentile(x, 75)),
        "p85": float(np.percentile(x, 85)),
        "p90": float(np.percentile(x, 90)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "max": float(np.max(x)),
    }


def suggest_spacing(stats: dict):
    """
    方案A：只保留采样点，不并入 OBJ 原始顶点
    """
    return {
        "dense": stats["p50"],
        "balanced": stats["p60"],
        "sparse": stats["p70"],
        "per_obj_voxel": 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="按整云规则网格挑若干格子估计激光点云局部间距，输出方案A对应的 spacing。"
    )
    parser.add_argument("ply_path", type=str, help="输入激光点云 PLY 文件路径")

    parser.add_argument("--grid-nx", type=int, default=30, help="X方向网格数，默认30")
    parser.add_argument("--grid-ny", type=int, default=30, help="Y方向网格数，默认30")
    parser.add_argument("--max-cells", type=int, default=20, help="最多抽多少个网格做估计，默认20")
    parser.add_argument("--halo-cells", type=int, default=1, help="每个查询网格向外扩几圈邻接网格作为搜索范围，默认1")
    parser.add_argument("--min-points-per-cell", type=int, default=200, help="参与候选的网格至少要有多少点，默认200")
    parser.add_argument("--max-query-per-cell", type=int, default=10000, help="每个网格最多取多少查询点，默认10000")
    parser.add_argument("--k", type=int, default=6, help="局部点间距估计时使用的近邻数，默认6")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--trim-low", type=float, default=5.0, help="低分位裁剪，默认5")
    parser.add_argument("--trim-high", type=float, default=95.0, help="高分位裁剪，默认95")

    args = parser.parse_args()

    ply_path = Path(args.ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"文件不存在: {ply_path}")

    points = load_ply_points(ply_path)
    print(f"[INFO] 读取点云: {ply_path}")
    print(f"[INFO] 点云总数: {points.shape[0]}")

    grid = build_xy_grid(points, nx=args.grid_nx, ny=args.grid_ny)
    cell_to_indices = grid["cell_to_indices"]

    chosen_cells, valid_cells = choose_cells(
        cell_to_indices=cell_to_indices,
        max_cells=args.max_cells,
        min_points_per_cell=args.min_points_per_cell,
        seed=args.seed,
    )

    print(f"[INFO] 网格划分: {args.grid_nx} x {args.grid_ny}")
    print(f"[INFO] 有效候选网格数: {len(valid_cells)}")
    print(f"[INFO] 实际选中网格数: {len(chosen_cells)}")
    print(f"[INFO] halo_cells = {args.halo_cells}")
    print(f"[INFO] k = {args.k}")

    all_spacing = []
    rng_seed_base = args.seed * 1000003 + 17

    for cell_idx, (cx, cy) in enumerate(chosen_cells, 1):
        query_idx = np.asarray(cell_to_indices[(cx, cy)], dtype=np.int64)
        query_idx = random_subsample_indices(
            query_idx,
            max_num=args.max_query_per_cell,
            seed=rng_seed_base + cell_idx,
        )

        search_idx = gather_neighbor_cell_indices(
            cell_to_indices=cell_to_indices,
            cx=cx,
            cy=cy,
            nx=args.grid_nx,
            ny=args.grid_ny,
            halo_cells=args.halo_cells,
        )

        if query_idx.size == 0 or search_idx.size <= args.k:
            print(f"[WARN] 网格 ({cx},{cy}) 点太少，跳过")
            continue

        query_points = points[query_idx]
        search_points = points[search_idx]

        print(
            f"[INFO] Cell {cell_idx}/{len(chosen_cells)} ({cx},{cy}) | "
            f"query={query_points.shape[0]} | search={search_points.shape[0]}"
        )

        if HAS_SCIPY:
            spacing = knn_spacing_ckdtree(
                search_points=search_points,
                query_points=query_points,
                k=args.k,
            )
        else:
            spacing = knn_spacing_open3d(
                search_points=search_points,
                query_points=query_points,
                k=args.k,
            )

        spacing = spacing[np.isfinite(spacing)]
        spacing = spacing[spacing > 0]

        if spacing.size == 0:
            print(f"[WARN] 网格 ({cx},{cy}) 无有效间距结果，跳过")
            continue

        all_spacing.append(spacing)

    if len(all_spacing) == 0:
        raise RuntimeError("没有成功得到任何局部间距结果，请增大 max-cells 或减小 min-points-per-cell")

    all_spacing = np.concatenate(all_spacing, axis=0)
    all_spacing = robust_filter(all_spacing, low_q=args.trim_low, high_q=args.trim_high)
    stats = calc_stats(all_spacing)
    sugg = suggest_spacing(stats)

    print("\n" + "=" * 96)
    print("[局部点间距统计（仅在抽中的网格区域上估计）]")
    for k in ["min", "p10", "p25", "p50", "p60", "p70", "p75", "p85", "p90", "mean", "std", "max"]:
        print(f"{k:>6}: {stats[k]:.6f}")

    print("\n[方案A推荐参数]")
    print(f"密一些：  --spacing {sugg['dense']:.6f}")
    print(f"平衡些：  --spacing {sugg['balanced']:.6f}")
    print(f"稀一些：  --spacing {sugg['sparse']:.6f}")
    print("per-obj-voxel 固定为 0，不再使用。")

    print("\n[默认建议]")
    print(f"先试：--spacing {sugg['balanced']:.6f}")


if __name__ == "__main__":
    main()



"""
python estimate_spacing_from_lidar.py ../lidar/nanfang/transform/lidar_full.ply

[局部点间距统计（仅在抽中的网格区域上估计）]
   min: 0.064063
   p10: 0.079480
   p25: 0.096852
   p50: 0.114191
   p60: 0.121226
   p70: 0.130836
   p75: 0.137557
   p85: 0.158340
   p90: 0.175669
  mean: 0.121772
   std: 0.037927
   max: 0.255187

密一些：  --spacing 0.114191
平衡些：  --spacing 0.121226
稀一些：  --spacing 0.130836


python estimate_spacing_from_lidar.py ../lidar/yanghaitang/transform/lidar_full.ply

[局部点间距统计（仅在抽中的网格区域上估计）]
   min: 0.060427
   p10: 0.073752
   p25: 0.089882
   p50: 0.109626
   p60: 0.116180
   p70: 0.123897
   p75: 0.128814
   p85: 0.143262
   p90: 0.155460
  mean: 0.112612
   std: 0.031537
   max: 0.215485

密一些：  --spacing 0.109626
平衡些：  --spacing 0.116180
稀一些：  --spacing 0.123897


python estimate_spacing_from_lidar.py ../lidar/xiaoxiang/transform/lidar_full.ply

[局部点间距统计（仅在抽中的网格区域上估计）]
   min: 0.062123
   p10: 0.073865
   p25: 0.088765
   p50: 0.108940
   p60: 0.115020
   p70: 0.121791
   p75: 0.125969
   p85: 0.137686
   p90: 0.146740
  mean: 0.109888
   std: 0.027608
   max: 0.193037

密一些：  --spacing 0.108940
平衡些：  --spacing 0.115020
稀一些：  --spacing 0.121791



"""
