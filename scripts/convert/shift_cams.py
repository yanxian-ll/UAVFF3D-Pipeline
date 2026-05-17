import argparse
from pathlib import Path
import shutil
import numpy as np


def is_float_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except Exception:
        return False


def parse_cam_file(cam_path: Path):
    """
    读取 cam txt，返回：
        lines: 原始文本行
        extrinsic: 4x4 numpy array
        extrinsic_line_ids: 外参4行在原始文件中的行号
    兼容格式示例：
        extrinsic opencv(x Right, y Down, z Forward) world2camera
        r11 r12 r13 t1
        r21 r22 r23 t2
        r31 r32 r33 t3
        0   0   0   1
    """
    lines = cam_path.read_text(encoding="utf-8").splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if "extrinsic" in line.lower():
            start_idx = i + 1
            break

    candidate_ids = []
    if start_idx is not None:
        for i in range(start_idx, len(lines)):
            s = lines[i].strip()
            if not s:
                if len(candidate_ids) >= 4:
                    break
                continue
            parts = s.split()
            if len(parts) == 4 and all(is_float_token(x) for x in parts):
                candidate_ids.append(i)
                if len(candidate_ids) == 4:
                    break

    if len(candidate_ids) != 4:
        # 回退：找文件里前4行“恰好4个浮点数”的行
        candidate_ids = []
        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) == 4 and all(is_float_token(x) for x in parts):
                candidate_ids.append(i)
                if len(candidate_ids) == 4:
                    break

    if len(candidate_ids) != 4:
        raise RuntimeError(f"无法在文件中找到 4x4 extrinsic: {cam_path}")

    extrinsic = []
    for idx in candidate_ids:
        extrinsic.append([float(x) for x in lines[idx].split()])
    extrinsic = np.asarray(extrinsic, dtype=np.float64)

    if extrinsic.shape != (4, 4):
        raise RuntimeError(f"extrinsic 维度异常: {cam_path}, got {extrinsic.shape}")

    return lines, extrinsic, candidate_ids


def camera_center_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    C = -R.T @ t
    return C


def apply_global_shift_to_extrinsic(extrinsic: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """
    原始模型：p_cam = R * p_world + t

    令新世界坐标系满足：p_world_new = p_world - offset
    则：
        p_cam = R * (p_world_new + offset) + t
              = R * p_world_new + (R*offset + t)

    所以：
        t_new = t + R @ offset
    旋转 R 不变。
    """
    out = extrinsic.copy()
    R = out[:3, :3]
    t = out[:3, 3]
    t_new = t + R @ offset.reshape(3)
    out[:3, 3] = t_new
    return out


def format_row(row: np.ndarray) -> str:
    return " ".join(f"{v:.12f}" for v in row)


def save_cam_file(lines, extrinsic_line_ids, new_extrinsic, out_path: Path):
    new_lines = list(lines)
    for k, idx in enumerate(extrinsic_line_ids):
        new_lines[idx] = format_row(new_extrinsic[k])
    out_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def collect_cam_info(cam_paths):
    infos = []
    centers = []
    for cam_path in cam_paths:
        lines, extrinsic, extrinsic_line_ids = parse_cam_file(cam_path)
        center = camera_center_from_extrinsic(extrinsic)
        infos.append({
            "path": cam_path,
            "lines": lines,
            "extrinsic": extrinsic,
            "extrinsic_line_ids": extrinsic_line_ids,
            "center": center,
        })
        centers.append(center)
    centers = np.asarray(centers, dtype=np.float64)
    return infos, centers


def build_offset(centers: np.ndarray, args) -> np.ndarray:
    if args.shift_mode == "none":
        return np.zeros(3, dtype=np.float64)
    if args.shift_mode == "mean":
        return centers.mean(axis=0)
    if args.shift_mode == "first":
        return centers[0]
    if args.shift_mode == "manual":
        return np.array([args.shift_tx, args.shift_ty, args.shift_tz], dtype=np.float64)
    raise ValueError(f"未知 shift_mode: {args.shift_mode}")


def main():
    parser = argparse.ArgumentParser(description="只平移现有 cams/*.txt 的外参，不重跑图像/depth。")
    parser.add_argument("--cams_dir", required=True, help="cams 文件夹路径")
    parser.add_argument(
        "--shift_mode",
        default="mean",
        choices=["mean", "first", "manual", "none"],
        help="整体平移方式：mean=减去所有相机中心均值；first=减去第一张相机中心；manual=手动指定；none=不平移",
    )
    parser.add_argument("--shift_tx", type=float, default=0.0, help="manual 模式下的 X 平移")
    parser.add_argument("--shift_ty", type=float, default=0.0, help="manual 模式下的 Y 平移")
    parser.add_argument("--shift_tz", type=float, default=0.0, help="manual 模式下的 Z 平移")
    parser.add_argument("--backup_dir", default="", help="可选：保存原始 cam 备份目录；为空则不备份")
    parser.add_argument("--summary_name", default="cam_global_shift.txt", help="平移记录文件名")
    parser.add_argument("--dry_run", action="store_true", help="只计算不写回")
    parser.add_argument("--verbose", action="store_true", help="打印部分平移前后中心")
    args = parser.parse_args()

    cams_dir = Path(args.cams_dir)
    if not cams_dir.exists():
        raise FileNotFoundError(f"cams_dir 不存在: {cams_dir}")

    cam_paths = sorted(cams_dir.glob("*.txt"))
    cam_paths = [p for p in cam_paths if p.name != args.summary_name]
    if len(cam_paths) == 0:
        raise RuntimeError(f"在 {cams_dir} 下没有找到 txt cam 文件")

    infos, centers = collect_cam_info(cam_paths)
    offset = build_offset(centers, args)

    backup_dir = None
    if args.backup_dir:
        backup_dir = Path(args.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

    new_centers = []
    for info in infos:
        src_path = info["path"]
        lines = info["lines"]
        extrinsic = info["extrinsic"]
        extrinsic_line_ids = info["extrinsic_line_ids"]

        new_extrinsic = apply_global_shift_to_extrinsic(extrinsic, offset)
        new_center = camera_center_from_extrinsic(new_extrinsic)
        new_centers.append(new_center)

        if args.verbose:
            print(f"{src_path.name}")
            print(f"  old center: {info['center']}")
            print(f"  new center: {new_center}")

        if not args.dry_run:
            if backup_dir is not None:
                shutil.copy2(src_path, backup_dir / src_path.name)
            save_cam_file(lines, extrinsic_line_ids, new_extrinsic, src_path)

    new_centers = np.asarray(new_centers, dtype=np.float64)

    summary_path = cams_dir / args.summary_name
    summary_lines = [
        f"num_cams: {len(cam_paths)}",
        f"shift_mode: {args.shift_mode}",
        f"global_offset_xyz: {offset[0]:.12f} {offset[1]:.12f} {offset[2]:.12f}",
        f"old_center_mean_xyz: {centers.mean(axis=0)[0]:.12f} {centers.mean(axis=0)[1]:.12f} {centers.mean(axis=0)[2]:.12f}",
        f"new_center_mean_xyz: {new_centers.mean(axis=0)[0]:.12f} {new_centers.mean(axis=0)[1]:.12f} {new_centers.mean(axis=0)[2]:.12f}",
        f"old_center_min_xyz: {centers.min(axis=0)[0]:.12f} {centers.min(axis=0)[1]:.12f} {centers.min(axis=0)[2]:.12f}",
        f"old_center_max_xyz: {centers.max(axis=0)[0]:.12f} {centers.max(axis=0)[1]:.12f} {centers.max(axis=0)[2]:.12f}",
        f"new_center_min_xyz: {new_centers.min(axis=0)[0]:.12f} {new_centers.min(axis=0)[1]:.12f} {new_centers.min(axis=0)[2]:.12f}",
        f"new_center_max_xyz: {new_centers.max(axis=0)[0]:.12f} {new_centers.max(axis=0)[1]:.12f} {new_centers.max(axis=0)[2]:.12f}",
    ]
    if not args.dry_run:
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("=" * 80)
    print(f"总 cam 数量: {len(cam_paths)}")
    print(f"shift_mode   : {args.shift_mode}")
    print(f"global_offset: {offset}")
    print(f"old mean     : {centers.mean(axis=0)}")
    print(f"new mean     : {new_centers.mean(axis=0)}")
    if args.dry_run:
        print("dry_run=True，本次未写回文件。")
    else:
        print(f"已原地覆盖保存到: {cams_dir}")
        if backup_dir is not None:
            print(f"原文件备份到: {backup_dir}")
        print(f"平移记录文件: {summary_path}")


if __name__ == "__main__":
    main()


"""

python shift_cams.py --cams_dir /opt/data/private/dataset/data/usegeo/dataset1/cams
python shift_cams.py --cams_dir /opt/data/private/dataset/data/usegeo/dataset2/cams
python shift_cams.py --cams_dir /opt/data/private/dataset/data/usegeo/dataset3/cams

"""