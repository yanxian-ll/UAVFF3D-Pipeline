#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import errno
import math
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# -----------------------------
# 文件扫描 / 解析
# -----------------------------
def list_files_by_stem(dir_path: Path, suffix: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    with os.scandir(dir_path) as it:
        for entry in it:
            if not entry.is_file():
                continue
            name = entry.name
            if name.endswith(suffix):
                result[Path(name).stem] = Path(entry.path)
    return result


def collect_common_stems(scene_dir: Path) -> dict[str, dict[str, Path]]:
    subdirs = {
        "images": scene_dir / "images",
        "depth": scene_dir / "depth",
        # "mask": scene_dir / "mask",
        "cams": scene_dir / "cams",
    }

    for name, p in subdirs.items():
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"缺少目录: {p}")

    files_map = {
        "images": list_files_by_stem(subdirs["images"], ".png"),
        "depth": list_files_by_stem(subdirs["depth"], ".exr"),
        # "mask": list_files_by_stem(subdirs["mask"], ".png"),
        "cams": list_files_by_stem(subdirs["cams"], ".txt"),
    }

    common_stems = (
        set(files_map["images"])
        & set(files_map["depth"])
        # & set(files_map["mask"])
        & set(files_map["cams"])
    )
    if not common_stems:
        raise FileNotFoundError("四个子目录之间没有公共 stem")

    result = {
        stem: {key: files_map[key][stem] for key in ("images", "depth", "cams")}
        for stem in sorted(common_stems)
    }

    for key, mp in files_map.items():
        missing = sorted(set(mp) - common_stems)
        if missing:
            print(f"[WARN] {key} 中有 {len(missing)} 个 stem 不完整，已忽略。")

    return result


def parse_extrinsic_fast(txt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    从 cams/*.txt 中只解析 extrinsic 4x4 的前三行：
        Xc = R * Xw + t

    返回:
        R: (3, 3) world->camera
        t: (3,)    translation
    """
    extr_lines = []
    found = False

    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if not found:
                if line.lower().startswith("extrinsic"):
                    found = True
                continue

            extr_lines.append(line)
            if len(extr_lines) == 4:
                break

    if len(extr_lines) != 4:
        raise ValueError(f"extrinsic 段格式错误，应为 4 行: {txt_path}")

    row0 = np.fromstring(extr_lines[0], sep=" ", dtype=np.float64)
    row1 = np.fromstring(extr_lines[1], sep=" ", dtype=np.float64)
    row2 = np.fromstring(extr_lines[2], sep=" ", dtype=np.float64)

    if row0.size != 4 or row1.size != 4 or row2.size != 4:
        raise ValueError(f"extrinsic 行解析失败: {txt_path}")

    R = np.stack((row0[:3], row1[:3], row2[:3]), axis=0)
    t = np.array((row0[3], row1[3], row2[3]), dtype=np.float64)
    return R, t


def get_camera_center_and_forward(
    txt_path: Path,
    camera_forward_axis: str = "pos_z",
) -> tuple[np.ndarray, np.ndarray]:
    """
    根据 extrinsic 解析：
      - 相机中心 Cw = -R^T t
      - 相机光轴 forward_world

    约定：Xc = R Xw + t，R 是 world->camera。
    在这个约定下，camera 坐标系某一轴在 world 中的方向可写为 R^T @ axis_cam。

    参数:
        camera_forward_axis:
            - 'pos_z'：认为相机前向是相机坐标系 +Z（多数 CV pinhole 约定）
            - 'neg_z'：认为相机前向是相机坐标系 -Z（部分图形学/OpenGL 约定）
    """
    R, t = parse_extrinsic_fast(txt_path)

    cam_center = -R.T @ t

    if camera_forward_axis == "pos_z":
        axis_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    elif camera_forward_axis == "neg_z":
        axis_cam = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        raise ValueError(f"不支持的 camera_forward_axis: {camera_forward_axis}")

    forward_world = R.T @ axis_cam
    norm = np.linalg.norm(forward_world)
    if norm < 1e-12:
        raise ValueError(f"光轴方向退化: {txt_path}")
    forward_world = forward_world / norm

    return cam_center, forward_world


def get_world_up_vector(up_axis: str = "z", up_sign: int = 1) -> np.ndarray:
    axis_map = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    if up_axis not in axis_map:
        raise ValueError(f"不支持的 up_axis: {up_axis}")
    if up_sign not in (-1, 1):
        raise ValueError("up_sign 只能是 1 或 -1")
    return axis_map[up_axis] * float(up_sign)


def classify_view_from_forward(
    forward_world: np.ndarray,
    world_up: np.ndarray,
    down_max_angle_deg: float = 35.0,
    use_abs_dot: bool = False,
) -> tuple[str, float, float, float]:
    """
    基于相机光轴与世界竖直方向关系做判断。

    记 world_up 为“向上”方向。
    若 forward_world 更接近 -world_up，则说明相机更偏向“向下看”。

    返回:
        label: 'down' or 'side'
        angle_to_down_deg: 与“向下方向(-up)”的夹角，越小越像下视
        down_score: 与下方向的对齐程度，越大越像下视
        abs_verticality: 与竖直轴的绝对对齐程度，越大越接近竖直拍摄
    """
    forward_world = np.asarray(forward_world, dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)

    forward_world = forward_world / max(np.linalg.norm(forward_world), 1e-12)
    world_up = world_up / max(np.linalg.norm(world_up), 1e-12)

    dot_up = float(np.dot(forward_world, world_up))
    dot_up = max(-1.0, min(1.0, dot_up))

    # 与“向下方向(-up)”越一致，down_score 越大。
    down_score = abs(dot_up) if use_abs_dot else (-dot_up)
    down_score = max(-1.0, min(1.0, down_score))

    angle_to_down_deg = math.degrees(math.acos(max(-1.0, min(1.0, down_score))))
    abs_verticality = abs(dot_up)

    label = "down" if angle_to_down_deg <= down_max_angle_deg else "side"
    return label, angle_to_down_deg, down_score, abs_verticality


def _read_pose_task(item: tuple[str, dict[str, Path]], camera_forward_axis: str):
    stem, paths = item
    center, forward = get_camera_center_and_forward(
        paths["cams"],
        camera_forward_axis=camera_forward_axis,
    )
    return stem, center, forward


def load_camera_poses_parallel(
    common_files: dict[str, dict[str, Path]],
    workers: int = 16,
    camera_forward_axis: str = "pos_z",
) -> tuple[list[str], np.ndarray, np.ndarray]:
    items = list(common_files.items())
    if not items:
        return [], np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    stems = [None] * len(items)
    centers = np.empty((len(items), 3), dtype=np.float64)
    forwards = np.empty((len(items), 3), dtype=np.float64)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_read_pose_task, item, camera_forward_axis): idx
            for idx, item in enumerate(items)
        }
        for future in as_completed(futures):
            idx = futures[future]
            stem, center, forward = future.result()
            stems[idx] = stem
            centers[idx] = center
            forwards[idx] = forward

    return stems, centers, forwards


# -----------------------------
# 输出目录 / 文件移动
# -----------------------------
def build_output_scene_dirs(scene_dir: Path, down_suffix: str, side_suffix: str) -> tuple[Path, Path]:
    parent = scene_dir.parent
    name = scene_dir.name

    out_down = parent / f"{name}{down_suffix}"
    out_side = parent / f"{name}{side_suffix}"

    for out_scene in (out_down, out_side):
        for sub in ("images", "depth", "cams"):
            (out_scene / sub).mkdir(parents=True, exist_ok=True)

    return out_down, out_side


def fast_move(src: Path, dst: Path) -> None:
    try:
        os.replace(src, dst)
    except OSError as e:
        if e.errno == errno.EXDEV:
            shutil.move(str(src), str(dst))
        else:
            raise


def fast_copy(src: Path, dst: Path, preserve_metadata: bool) -> None:
    if preserve_metadata:
        shutil.copy2(src, dst)
    else:
        shutil.copyfile(src, dst)


def move_one_stem_group(
    stem: str,
    file_group: dict[str, Path],
    out_scene: Path,
    use_copy: bool = False,
    preserve_metadata: bool = False,
) -> str:
    for sub in ("images", "depth", "cams"):
        src = file_group[sub]
        dst = out_scene / sub / src.name

        if use_copy:
            fast_copy(src, dst, preserve_metadata=preserve_metadata)
        else:
            fast_move(src, dst)

    return stem


def maybe_remove_empty_dirs(scene_dir: Path) -> None:
    for sub in ("images", "depth",  "cams"):
        p = scene_dir / sub
        try:
            if p.exists() and p.is_dir() and not any(p.iterdir()):
                p.rmdir()
        except OSError:
            pass

    try:
        if scene_dir.exists() and scene_dir.is_dir() and not any(scene_dir.iterdir()):
            scene_dir.rmdir()
    except OSError:
        pass


class ProgressPrinter:
    def __init__(self, total: int, log_every: int = 100):
        self.total = total
        self.log_every = max(1, int(log_every))
        self.done = 0
        self.errors = 0
        self.lock = threading.Lock()

    def ok(self, stem: str) -> None:
        with self.lock:
            self.done += 1
            cur = self.done + self.errors
            if cur % self.log_every == 0 or cur == self.total:
                print(f"[PROGRESS] {cur}/{self.total} | success={self.done} | errors={self.errors}")

    def fail(self, stem: str, exc: Exception) -> None:
        with self.lock:
            self.errors += 1
            cur = self.done + self.errors
            print(f"[ERROR] stem={stem} | {exc}")
            if cur % self.log_every == 0 or cur == self.total:
                print(f"[PROGRESS] {cur}/{self.total} | success={self.done} | errors={self.errors}")


def write_manifest(
    manifest_path: Path,
    stems: list[str],
    centers: np.ndarray,
    forwards: np.ndarray,
    labels: list[str],
    angles: list[float],
    down_scores: list[float],
    abs_verticalities: list[float],
):
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(
            "stem,label,angle_to_down_deg,down_score,abs_verticality,"
            "center_x,center_y,center_z,forward_x,forward_y,forward_z\n"
        )
        for stem, c, fw, lb, ang, ds, av in zip(
            stems, centers, forwards, labels, angles, down_scores, abs_verticalities
        ):
            f.write(
                f"{stem},{lb},{ang:.6f},{ds:.6f},{av:.6f},"
                f"{c[0]:.6f},{c[1]:.6f},{c[2]:.6f},"
                f"{fw[0]:.6f},{fw[1]:.6f},{fw[2]:.6f}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="根据 cams 中的相机姿态，把一个场景划分为下视场景和侧视场景。"
    )
    parser.add_argument("--scene", type=str, required=True, help="输入场景路径")
    parser.add_argument("--down_suffix", type=str, default="_ndir", help="下视子场景后缀")
    parser.add_argument("--side_suffix", type=str, default="_oblique", help="侧视子场景后缀")

    parser.add_argument("--parse_workers", type=int, default=32, help="并行解析 cam 文件线程数")
    parser.add_argument("--move_workers", type=int, default=32, help="并行移动/复制线程数")
    parser.add_argument("--log_every", type=int, default=500, help="每处理多少个 stem 输出一次进度")

    # 姿态判断相关参数
    parser.add_argument(
        "--camera_forward_axis",
        type=str,
        default="pos_z",
        choices=["pos_z", "neg_z"],
        help="相机前向在相机坐标系中是 +Z 还是 -Z",
    )
    parser.add_argument(
        "--up_axis",
        type=str,
        default="z",
        choices=["x", "y", "z"],
        help="世界坐标中的竖直向上轴",
    )
    parser.add_argument(
        "--up_sign",
        type=int,
        default=1,
        choices=[-1, 1],
        help="竖直向上的方向符号，1 表示 +axis 是向上，-1 表示 -axis 是向上",
    )
    parser.add_argument(
        "--down_max_angle_deg",
        type=float,
        default=15.0,
        help="与向下方向夹角小于等于该阈值时，判为下视；否则判为侧视",
    )
    parser.add_argument(
        "--use_abs_dot",
        action="store_true",
        help="忽略上下符号，只看是否接近竖直。若数据里没有朝上的相机，可打开这个选项提高鲁棒性。",
    )

    parser.add_argument("--copy", action="store_true", help="复制而不是移动")
    parser.add_argument(
        "--preserve_metadata",
        action="store_true",
        help="复制时保留元数据（更稳但更慢；默认关闭以获得更快复制速度）",
    )
    parser.add_argument("--keep_empty_scene", action="store_true", help="保留原始空场景目录")
    parser.add_argument(
        "--write_manifest",
        action="store_true",
        help="在原场景同级目录输出一个 CSV 清单，方便排查分类是否正确",
    )

    args = parser.parse_args()

    scene_dir = Path(args.scene)
    if not scene_dir.exists() or not scene_dir.is_dir():
        raise FileNotFoundError(f"场景路径不存在: {scene_dir}")

    print(f"[START] scene={scene_dir}")
    print(
        f"[INFO] camera_forward_axis={args.camera_forward_axis}, "
        f"up_axis={args.up_axis}, up_sign={args.up_sign}, "
        f"down_max_angle_deg={args.down_max_angle_deg}, use_abs_dot={args.use_abs_dot}"
    )

    common_files = collect_common_stems(scene_dir)
    if len(common_files) < 1:
        raise ValueError("没有可用样本")

    print(f"[INFO] 有效公共 stem 数量: {len(common_files)}")
    print("[INFO] 并行读取相机中心与光轴方向...")
    stems, centers, forwards = load_camera_poses_parallel(
        common_files,
        workers=args.parse_workers,
        camera_forward_axis=args.camera_forward_axis,
    )

    world_up = get_world_up_vector(args.up_axis, args.up_sign)

    labels = []
    angles = []
    down_scores = []
    abs_verticalities = []

    for fw in forwards:
        label, angle_deg, down_score, abs_verticality = classify_view_from_forward(
            fw,
            world_up=world_up,
            down_max_angle_deg=args.down_max_angle_deg,
            use_abs_dot=args.use_abs_dot,
        )
        labels.append(label)
        angles.append(angle_deg)
        down_scores.append(down_score)
        abs_verticalities.append(abs_verticality)

    num_down = sum(lb == "down" for lb in labels)
    num_side = sum(lb == "side" for lb in labels)

    out_down, out_side = build_output_scene_dirs(scene_dir, args.down_suffix, args.side_suffix)
    print(f"[INFO] 输出下视子场景: {out_down}")
    print(f"[INFO] 输出侧视子场景: {out_side}")
    print(f"[INFO] 统计: down={num_down}, side={num_side}, total={len(labels)}")

    if args.write_manifest:
        manifest_path = scene_dir.parent / f"{scene_dir.name}_view_split_manifest.csv"
        write_manifest(
            manifest_path=manifest_path,
            stems=stems,
            centers=centers,
            forwards=forwards,
            labels=labels,
            angles=angles,
            down_scores=down_scores,
            abs_verticalities=abs_verticalities,
        )
        print(f"[INFO] 已写出清单: {manifest_path}")

    tasks = []
    for stem, label in zip(stems, labels):
        out_scene = out_down if label == "down" else out_side
        tasks.append((stem, common_files[stem], out_scene))

    progress = ProgressPrinter(total=len(tasks), log_every=args.log_every)

    with ThreadPoolExecutor(max_workers=max(1, int(args.move_workers))) as executor:
        future_map = {
            executor.submit(
                move_one_stem_group,
                stem,
                file_group,
                out_scene,
                args.copy,
                args.preserve_metadata,
            ): stem
            for stem, file_group, out_scene in tasks
        }

        for future in as_completed(future_map):
            stem = future_map[future]
            try:
                future.result()
                progress.ok(stem)
            except Exception as e:
                progress.fail(stem, e)

    print(f"[DONE] success={progress.done}, errors={progress.errors}, total={len(tasks)}")

    if not args.copy and not args.keep_empty_scene:
        maybe_remove_empty_dirs(scene_dir)

    if progress.errors > 0:
        raise RuntimeError(f"存在 {progress.errors} 个 stem 处理失败，请检查日志。")


if __name__ == "__main__":
    main()


"""

python split_to_ndir_oblique.py --scene UAVFF3D-Real/nanfang_part0 --copy
python split_to_ndir_oblique.py --scene UAVFF3D-Real/nanfang_part1 --copy

python split_to_ndir_oblique.py --scene UAVFF3D-Real/yanghaitang_part0 --copy
python split_to_ndir_oblique.py --scene UAVFF3D-Real/yanghaitang_part1 --copy

python split_to_ndir_oblique.py --scene UAVFF3D-Real/xiaoxiang_part0 --copy
python split_to_ndir_oblique.py --scene UAVFF3D-Real/xiaoxiang_part1 --copy
python split_to_ndir_oblique.py --scene UAVFF3D-Real/xiaoxiang_part2 --copy
python split_to_ndir_oblique.py --scene UAVFF3D-Real/xiaoxiang_part3 --copy


python split_to_ndir_oblique.py --scene urbanscene3d/artsci --copy
python split_to_ndir_oblique.py --scene urbanscene3d/polytech --copy
"""
