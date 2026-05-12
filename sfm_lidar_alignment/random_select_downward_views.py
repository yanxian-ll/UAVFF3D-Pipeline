# -*- coding: utf-8 -*-

import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


def resolve_cams_dir(cam_root: Path):
    """
    用户输入可能是:
    1) 直接是 cams/
    2) 上一级目录，里面有 cams/
    """
    if cam_root.name == "cams" and cam_root.is_dir():
        return cam_root

    cams_dir = cam_root / "cams"
    if cams_dir.is_dir():
        return cams_dir

    raise FileNotFoundError(f"未找到 cams 目录: {cam_root} 或 {cams_dir}")


def find_cam_txts(cams_dir: Path):
    txts = sorted(cams_dir.glob("*.txt"))
    if len(txts) == 0:
        raise FileNotFoundError(f"cams 下没有找到 txt 文件: {cams_dir}")
    return txts


def parse_cam_txt(cam_txt_path: Path):
    """
    解析格式：

    extrinsic opencv(x Right, y Down, z Forward) world2camera
    4x4

    intrinsic: fx fy cx cy (pixel)
    3x3

    h w fov
    h w fov
    """
    with open(cam_txt_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f.readlines()]

    lines = [x for x in raw_lines if x != ""]

    def find_line_idx(prefix):
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                return i
        return -1

    ext_idx = find_line_idx("extrinsic")
    intr_idx = find_line_idx("intrinsic")
    hwfov_idx = find_line_idx("h w fov")

    if ext_idx < 0 or intr_idx < 0 or hwfov_idx < 0:
        raise ValueError(f"相机文件格式错误: {cam_txt_path}")

    ext_lines = lines[ext_idx + 1 : ext_idx + 5]
    intr_lines = lines[intr_idx + 1 : intr_idx + 4]
    hwfov_line = lines[hwfov_idx + 1]

    if len(ext_lines) != 4:
        raise ValueError(f"extrinsic 行数不正确: {cam_txt_path}")
    if len(intr_lines) != 3:
        raise ValueError(f"intrinsic 行数不正确: {cam_txt_path}")

    extrinsic = np.array(
        [[float(v) for v in line.split()] for line in ext_lines],
        dtype=np.float64,
    )

    intrinsic = np.array(
        [[float(v) for v in line.split()] for line in intr_lines],
        dtype=np.float64,
    )

    parts = hwfov_line.split()
    if len(parts) != 3:
        raise ValueError(f"h w fov 行格式错误: {cam_txt_path}")

    h = int(float(parts[0]))
    w = int(float(parts[1]))
    fov = float(parts[2])

    return {
        "name": cam_txt_path.stem,
        "path": str(cam_txt_path),
        "extrinsic": extrinsic,   # world -> camera
        "intrinsic": intrinsic,
        "h": h,
        "w": w,
        "fov": fov,
    }


def camera_center_from_world2cam(world2cam: np.ndarray):
    """
    已知 world2camera:
        Xc = R * Xw + t

    相机中心:
        C = -R^T * t
    """
    R = world2cam[:3, :3]
    t = world2cam[:3, 3]
    C = -R.T @ t
    return C


def camera_forward_world_from_world2cam(world2cam: np.ndarray):
    """
    相机坐标系中 forward = [0, 0, 1]
    转到世界坐标系:
        f_world = R^T @ [0,0,1]
    """
    R = world2cam[:3, :3]
    f_world = R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm = np.linalg.norm(f_world)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return f_world / norm


def compute_downward_angle_deg(forward_world: np.ndarray, down_axis=np.array([0.0, 0.0, -1.0])):
    """
    计算相机光轴与“向下方向”的夹角（单位：度）
    默认世界坐标 Z 向上，因此 down_axis = [0,0,-1]
    """
    down_axis = np.asarray(down_axis, dtype=np.float64)
    down_axis = down_axis / (np.linalg.norm(down_axis) + 1e-12)

    cos_val = float(np.dot(forward_world, down_axis))
    cos_val = np.clip(cos_val, -1.0, 1.0)
    angle_deg = math.degrees(math.acos(cos_val))
    return angle_deg, cos_val


def select_downward_cameras(cam_txts, max_off_nadir_deg=30.0):
    """
    筛选近似下视相机
    """
    selected = []

    for cam_txt in cam_txts:
        cam = parse_cam_txt(cam_txt)
        ext = cam["extrinsic"]

        center = camera_center_from_world2cam(ext)
        forward_world = camera_forward_world_from_world2cam(ext)
        angle_deg, cos_down = compute_downward_angle_deg(forward_world)

        if angle_deg <= max_off_nadir_deg:
            selected.append({
                "name": cam["name"],
                "path": cam["path"],
                "center": center,
                "forward_world": forward_world,
                "downward_angle_deg": angle_deg,
                "cos_down": cos_down,
            })

    return selected


def grid_random_sample(cameras, grid_size=50.0, seed=42, samples_per_cell=1):
    """
    按 XY 平面网格随机抽样
    每个格子默认选 1 个
    """
    if len(cameras) == 0:
        return []

    rng = np.random.default_rng(seed)

    centers = np.array([x["center"] for x in cameras], dtype=np.float64)
    xs = centers[:, 0]
    ys = centers[:, 1]

    xmin = xs.min()
    ymin = ys.min()

    buckets = defaultdict(list)

    for cam in cameras:
        x, y = cam["center"][0], cam["center"][1]
        gx = int(np.floor((x - xmin) / grid_size))
        gy = int(np.floor((y - ymin) / grid_size))
        buckets[(gx, gy)].append(cam)

    selected = []
    for cell, items in buckets.items():
        if len(items) <= samples_per_cell:
            chosen = items
        else:
            idx = rng.choice(len(items), size=samples_per_cell, replace=False)
            chosen = [items[i] for i in idx]
        selected.extend(chosen)

    # 最终按名字排序，便于稳定使用
    selected = sorted(selected, key=lambda x: x["name"])
    return selected


def save_selected_names(selected, output_txt: Path):
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(item["name"] + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="从 cams/*.txt 中筛选近似下视视角，并按网格随机选取，输出名字列表 txt"
    )
    parser.add_argument("cam_root", type=str, help="带 cams 的路径，或直接是 cams/ 目录")
    parser.add_argument("output_txt", type=str, help="输出名字列表 txt，一行一个")

    parser.add_argument(
        "--max-off-nadir-deg",
        type=float,
        default=20.0,
        help="近似下视阈值：相机光轴与世界负Z方向夹角上限（度）",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=50.0,
        help="XY 网格大小，单位与相机坐标一致",
    )
    parser.add_argument(
        "--samples-per-cell",
        type=int,
        default=1,
        help="每个网格随机选几个视角，默认 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    args = parser.parse_args()

    cam_root = Path(args.cam_root)
    output_txt = Path(args.output_txt)

    cams_dir = resolve_cams_dir(cam_root)
    cam_txts = find_cam_txts(cams_dir)

    print(f"[INFO] cams dir: {cams_dir}")
    print(f"[INFO] total camera files: {len(cam_txts)}")

    downward_cams = select_downward_cameras(
        cam_txts,
        max_off_nadir_deg=args.max_off_nadir_deg,
    )
    print(f"[INFO] downward cameras: {len(downward_cams)}")

    selected = grid_random_sample(
        downward_cams,
        grid_size=args.grid_size,
        seed=args.seed,
        samples_per_cell=args.samples_per_cell,
    )
    print(f"[INFO] grid-random selected cameras: {len(selected)}")

    save_selected_names(selected, output_txt)
    print(f"[INFO] saved to: {output_txt}")


if __name__ == "__main__":
    main()

"""
python random_select_downward_views.py ../recon/nanfang ./nanfang/selected_views.txt

python random_select_downward_views.py ../recon/yanghaitang ./yanghaitang/selected_views.txt

python random_select_downward_views.py ../recon/xiaoxiang ./xiaoxiang/selected_views.txt

"""