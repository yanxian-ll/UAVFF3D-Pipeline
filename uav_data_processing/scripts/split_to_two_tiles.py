#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import errno
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
    """
    比 Path.glob 更轻量的目录扫描。
    """
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
    """
    收集 images/depth/mask/cams 四类文件的公共 stem。
    """
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
        # stem: {key: files_map[key][stem] for key in ("images", "depth", "mask", "cams")}
        stem: {key: files_map[key][stem] for key in ("images", "depth", "cams")}
        for stem in sorted(common_stems)
    }

    for key, mp in files_map.items():
        missing = sorted(set(mp) - common_stems)
        if missing:
            print(f"[WARN] {key} 中有 {len(missing)} 个 stem 不完整，已忽略。")

    return result


def parse_camera_center_fast(txt_path: Path) -> np.ndarray:
    """
    仅解析 extrinsic 4x4，并直接计算相机中心：
        Xc = R * Xw + t
        Cw = -R^T t

    相比原实现，这里：
    1. 不再解析 intrinsic / h w hfov
    2. 不构造无用返回 dict
    3. 只做一次最小化文本扫描
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

    # np.fromstring 比逐个 float(...) 更快
    row0 = np.fromstring(extr_lines[0], sep=" ", dtype=np.float64)
    row1 = np.fromstring(extr_lines[1], sep=" ", dtype=np.float64)
    row2 = np.fromstring(extr_lines[2], sep=" ", dtype=np.float64)

    if row0.size != 4 or row1.size != 4 or row2.size != 4:
        raise ValueError(f"extrinsic 行解析失败: {txt_path}")

    R = np.stack((row0[:3], row1[:3], row2[:3]), axis=0)
    t = np.array((row0[3], row1[3], row2[3]), dtype=np.float64)

    cam_center = -R.T @ t
    return cam_center


def _read_center_task(item: tuple[str, dict[str, Path]]) -> tuple[str, np.ndarray]:
    stem, paths = item
    center = parse_camera_center_fast(paths["cams"])
    return stem, center


def load_camera_centers_parallel(
    common_files: dict[str, dict[str, Path]],
    workers: int = 16,
) -> tuple[list[str], np.ndarray]:
    """
    并行读取所有 cam，得到相机中心。
    这里是典型的小文件 I/O + 少量数值解析，线程并行通常有效。
    """
    items = list(common_files.items())
    if not items:
        return [], np.empty((0, 3), dtype=np.float64)

    stems = [None] * len(items)
    centers = np.empty((len(items), 3), dtype=np.float64)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_read_center_task, item): idx for idx, item in enumerate(items)}
        for future in as_completed(futures):
            idx = futures[future]
            stem, center = future.result()
            stems[idx] = stem
            centers[idx] = center

    return stems, centers


# -----------------------------
# 切分逻辑
# -----------------------------
def split_by_xy_long_axis(points_xyz: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    只使用 xy 坐标，按长轴方向切分。
    步骤：
      1. 取 points_xy
      2. PCA 求第一主轴
      3. 投影到主轴
      4. 以投影中位数为阈值切分成两部分

    返回:
        labels: (N,), 值为 0 或 1
        info: 诊断信息
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz 必须是 (N, 3)")
    if points_xyz.shape[0] < 2:
        raise ValueError("至少需要 2 个相机位置才能划分")

    points_xy = np.ascontiguousarray(points_xyz[:, :2], dtype=np.float64)
    center_xy = points_xy.mean(axis=0, keepdims=True)
    centered = points_xy - center_xy

    cov = centered.T @ centered / max(points_xy.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    major_idx = int(np.argmax(eigvals))
    major_axis = eigvecs[:, major_idx]
    norm = np.linalg.norm(major_axis)
    if norm < 1e-12:
        # 完全退化，直接按 x 排序切一半
        proj = centered[:, 0]
        split_value = float(np.median(proj))
        order = np.argsort(proj)
        labels = np.zeros(points_xy.shape[0], dtype=np.int32)
        labels[order[len(order) // 2 :]] = 1
        info = {
            "center_xy": center_xy.reshape(-1).tolist(),
            "major_axis_xy": [1.0, 0.0],
            "eigenvalues": eigvals.tolist(),
            "split_value": split_value,
            "num_part0": int(np.sum(labels == 0)),
            "num_part1": int(np.sum(labels == 1)),
            "degenerate": True,
        }
        return labels, info

    major_axis = major_axis / norm

    # 固定方向，便于稳定输出
    if (major_axis[0] < 0) or (major_axis[0] == 0 and major_axis[1] < 0):
        major_axis = -major_axis

    proj = centered @ major_axis
    split_value = float(np.median(proj))
    labels = (proj > split_value).astype(np.int32)

    # 防止退化：如果全到一边，则按排序硬切一半
    if np.all(labels == 0) or np.all(labels == 1):
        order = np.argsort(proj)
        labels = np.zeros(points_xy.shape[0], dtype=np.int32)
        labels[order[len(order) // 2 :]] = 1

    info = {
        "center_xy": center_xy.reshape(-1).tolist(),
        "major_axis_xy": major_axis.tolist(),
        "eigenvalues": eigvals.tolist(),
        "split_value": split_value,
        "num_part0": int(np.sum(labels == 0)),
        "num_part1": int(np.sum(labels == 1)),
        "degenerate": False,
    }
    return labels, info


# -----------------------------
# 输出目录 / 文件移动
# -----------------------------
def build_output_scene_dirs(scene_dir: Path, suffix0: str, suffix1: str) -> tuple[Path, Path]:
    """
    在输入场景同级目录下创建两个子场景。
    """
    parent = scene_dir.parent
    name = scene_dir.name

    out0 = parent / f"{name}{suffix0}"
    out1 = parent / f"{name}{suffix1}"

    for out_scene in (out0, out1):
        # for sub in ("images", "depth", "mask", "cams"):
        for sub in ("images", "depth", "cams"):
            (out_scene / sub).mkdir(parents=True, exist_ok=True)

    return out0, out1


def fast_move(src: Path, dst: Path) -> None:
    """
    同文件系统下优先使用 os.replace / rename，通常比 shutil.move 更快。
    若跨设备则回退到 shutil.move。
    """
    try:
        os.replace(src, dst)
    except OSError as e:
        if e.errno == errno.EXDEV:
            shutil.move(str(src), str(dst))
        else:
            raise


def fast_copy(src: Path, dst: Path, preserve_metadata: bool) -> None:
    """
    preserve_metadata=False 时使用 copyfile，通常比 copy2 更快。
    """
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
    """
    把一个 stem 对应的四类文件移动或复制到目标子场景。
    """
    # for sub in ("images", "depth", "mask", "cams"):
    for sub in ("images", "depth", "cams"):
        src = file_group[sub]
        dst = out_scene / sub / src.name

        if use_copy:
            fast_copy(src, dst, preserve_metadata=preserve_metadata)
        else:
            fast_move(src, dst)

    return stem


def maybe_remove_empty_dirs(scene_dir: Path) -> None:
    """
    尝试删除空的原始子目录；删除失败则忽略。
    """
    # for sub in ("images", "depth", "mask", "cams"):
    for sub in ("images", "depth", "cams"):
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


def main():
    parser = argparse.ArgumentParser(
        description="根据相机中心的 xy 平面长轴，把一个场景沿长轴切分成两个子场景，并移动 images/depth/mask/cams 四类同名文件"
    )
    parser.add_argument("--scene", type=str, required=True, help="输入场景路径")
    parser.add_argument("--suffix0", type=str, default="_part0", help="第 1 个子场景后缀")
    parser.add_argument("--suffix1", type=str, default="_part1", help="第 2 个子场景后缀")

    parser.add_argument("--parse_workers", type=int, default=32, help="并行解析相机文件线程数")
    parser.add_argument("--move_workers", type=int, default=32, help="并行移动/复制线程数")
    parser.add_argument("--log_every", type=int, default=500, help="每处理多少个 stem 输出一次进度")

    parser.add_argument("--copy", action="store_true", help="复制而不是移动")
    parser.add_argument(
        "--preserve_metadata",
        action="store_true",
        help="复制时保留元数据（更稳但更慢；默认关闭以获得更快复制速度）",
    )
    parser.add_argument("--keep_empty_scene", action="store_true", help="保留原始空场景目录")

    args = parser.parse_args()

    scene_dir = Path(args.scene)
    if not scene_dir.exists() or not scene_dir.is_dir():
        raise FileNotFoundError(f"场景路径不存在: {scene_dir}")

    print(f"[START] scene={scene_dir}")

    common_files = collect_common_stems(scene_dir)
    if len(common_files) < 2:
        raise ValueError(f"可用样本数不足 2，无法划分两部分。当前只有 {len(common_files)} 个。")

    print(f"[INFO] 有效公共 stem 数量: {len(common_files)}")
    print("[INFO] 并行读取相机中心...")
    stems, centers = load_camera_centers_parallel(common_files, workers=args.parse_workers)

    print("[INFO] 正在根据相机中心的 xy 长轴进行切分...")
    labels, split_info = split_by_xy_long_axis(centers)

    out0, out1 = build_output_scene_dirs(scene_dir, args.suffix0, args.suffix1)
    print(f"[INFO] 输出子场景 0: {out0}")
    print(f"[INFO] 输出子场景 1: {out1}")
    print(f"[INFO] 长轴切分信息: {split_info}")

    tasks = []
    for stem, label in zip(stems, labels.tolist()):
        out_scene = out0 if label == 0 else out1
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
python split_to_two_tiles.py --scene A3D-Real/nanfang --copy --suffix0 _part0 --suffix1 _part1
python split_to_two_tiles.py --scene A3D-Real/yanghaitang --copy --suffix0 _part0 --suffix1 _part1
python split_to_two_tiles.py --scene A3D-Real/xiaoxiang --copy --suffix0 _tmp0 --suffix1 _tmp1

python split_to_two_tiles.py --scene A3D-Real/xiaoxiang_tmp0 --suffix0 _part0 --suffix1 _part1
python split_to_two_tiles.py --scene A3D-Real/xiaoxiang_tmp1 --suffix0 _part2 --suffix1 _part3

"""
