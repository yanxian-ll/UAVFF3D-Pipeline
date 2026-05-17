#!/usr/bin/env python3
"""
Fast ENRICH-Aerial converter.

Compared with the serial version, this one speeds up conversion mainly by:
1) parallel per-sample processing with ThreadPoolExecutor;
2) lower PNG compression by default;
3) configurable EXR compression;
4) precomputing camera parameters in the main thread.

Input layout (under --src_root):
  cameras.csv
  images/*.jpg
  depth/exr/*_depth.exr

Output layout (under --dst_root):
  images/*.png
  depth/*.exr
  cams/*.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402


FOCAL_PX = {
    "nadir": 5882.0,
    "nadir-2": 5882.0,
    "ndir-2": 5882.0,
    "forward": 11764.0,
    "backward": 11764.0,
    "left": 11764.0,
    "right": 11764.0,
}

GL_TO_CV = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
CAMERA_SUFFIXES = ("nadir-2", "ndir-2", "nadir", "forward", "backward", "left", "right")

EXR_COMPRESSION_MAP = {
    "none": cv2.IMWRITE_EXR_COMPRESSION_NO,
    "rle": cv2.IMWRITE_EXR_COMPRESSION_RLE,
    "zips": cv2.IMWRITE_EXR_COMPRESSION_ZIPS,
    "zip": cv2.IMWRITE_EXR_COMPRESSION_ZIP,
    "piz": cv2.IMWRITE_EXR_COMPRESSION_PIZ,
    "pxr24": cv2.IMWRITE_EXR_COMPRESSION_PXR24,
    "b44": cv2.IMWRITE_EXR_COMPRESSION_B44,
    "b44a": cv2.IMWRITE_EXR_COMPRESSION_B44A,
    "dwaa": cv2.IMWRITE_EXR_COMPRESSION_DWAA,
    "dwab": cv2.IMWRITE_EXR_COMPRESSION_DWAB,
}


@dataclass(frozen=True)
class Task:
    label: str
    stem: str
    camera_name: str
    img_path: Path
    depth_path: Path
    png_out: Path
    exr_out: Path
    cam_out: Path
    position: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    lookat: tuple[float, float, float] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast convert ENRICH-Aerial to images/depth/cams format.")
    parser.add_argument("--src_root", type=Path, required=True)
    parser.add_argument("--dst_root", type=Path, required=True)
    parser.add_argument("--max_size", type=int, default=1024, help="Max output side. <=0 keeps original size.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--depth_interp", choices=["nearest", "linear"], default="nearest")
    parser.add_argument("--check_lookat", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 8))),
        help="Parallel workers. For SSD, 4-8 is usually a good start.",
    )
    parser.add_argument(
        "--png_compression",
        type=int,
        default=1,
        choices=list(range(10)),
        help="0 is fastest and largest; 9 is slowest and smallest.",
    )
    parser.add_argument(
        "--exr_compression",
        type=str,
        default="zip",
        choices=sorted(EXR_COMPRESSION_MAP.keys()),
        help="Use 'none' for fastest EXR writing.",
    )
    parser.add_argument(
        "--depth_half",
        action="store_true",
        help="Write depth as float16 EXR to reduce IO and disk usage.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def quat_wxyz_to_rotmat(w: float, x: float, y: float, z: float) -> np.ndarray:
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n == 0:
        raise ValueError(f"Invalid quaternion: {q.tolist()}")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def infer_camera_name(stem: str) -> str:
    for cam_name in CAMERA_SUFFIXES:
        if stem.endswith(f"_{cam_name}"):
            return cam_name
    raise ValueError(f"Cannot infer camera name from stem: {stem}")


def make_depth_name(stem: str) -> str:
    return f"{stem}_depth.exr"


def compute_scale(h: int, w: int, max_size: int) -> tuple[int, int, float]:
    if max_size <= 0 or max(h, w) <= max_size:
        return h, w, 1.0
    scale = max_size / float(max(h, w))
    return max(1, int(round(h * scale))), max(1, int(round(w * scale))), scale


def resize_array(img: np.ndarray, new_h: int, new_w: int, is_depth: bool, depth_interp: str) -> np.ndarray:
    if img.shape[0] == new_h and img.shape[1] == new_w:
        return img
    if is_depth:
        interp = cv2.INTER_NEAREST if depth_interp == "nearest" else cv2.INTER_LINEAR
    else:
        interp = cv2.INTER_AREA if new_h < img.shape[0] or new_w < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def get_intrinsic_matrix(camera_name: str, w: int, h: int, scale: float) -> np.ndarray:
    fx = FOCAL_PX[camera_name] * scale
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_pose_to_w2c_opencv(position_xyz: Iterable[float], quat_wxyz: Iterable[float]) -> np.ndarray:
    C = np.asarray(tuple(position_xyz), dtype=np.float64).reshape(3)
    R_c2w_gl = quat_wxyz_to_rotmat(*quat_wxyz)
    R_w2c_cv = GL_TO_CV @ R_c2w_gl.T
    t_w2c_cv = -R_w2c_cv @ C
    ext = np.eye(4, dtype=np.float64)
    ext[:3, :3] = R_w2c_cv
    ext[:3, 3] = t_w2c_cv
    return ext


def predicted_lookat_from_quat(quat_wxyz: Iterable[float]) -> np.ndarray:
    R_c2w_gl = quat_wxyz_to_rotmat(*quat_wxyz)
    v = R_c2w_gl @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def write_cam_file(path: Path, ext: np.ndarray, K: np.ndarray, h: int, w: int) -> None:
    hfov = math.degrees(2.0 * math.atan(w / (2.0 * K[0, 0])))
    with open(path, "w", encoding="utf-8") as f:
        f.write("extrinsic opencv(x Right, y Down, z Forward) world2camera\n")
        f.writelines(" ".join(f"{v:.12f}" for v in row) + "\n" for row in ext)
        f.write("\n")
        f.write("intrinsic: fx fy cx cy (pixel)\n")
        f.writelines(" ".join(f"{v:.12f}" for v in row) + "\n" for row in K)
        f.write("\n")
        f.write("h w hfov\n")
        f.write(f"{h} {w} {hfov:.12f}\n")


def build_tasks(src_root: Path, dst_root: Path, overwrite: bool, check_lookat: bool) -> list[Task]:
    cameras_csv = src_root / "cameras.csv"
    images_dir = src_root / "images"
    depth_dir = src_root / "depth" / "exr"

    if not cameras_csv.exists():
        raise FileNotFoundError(cameras_csv)
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)
    if not depth_dir.exists():
        raise FileNotFoundError(depth_dir)

    out_images = dst_root / "images"
    out_depth = dst_root / "depth"
    out_cams = dst_root / "cams"
    ensure_dir(out_images)
    ensure_dir(out_depth)
    ensure_dir(out_cams)

    tasks: list[Task] = []
    with open(cameras_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip()
            stem = Path(label).stem
            camera_name = infer_camera_name(stem)
            img_path = images_dir / label
            depth_path = depth_dir / make_depth_name(stem)
            png_out = out_images / f"{stem}.png"
            exr_out = out_depth / f"{stem}.exr"
            cam_out = out_cams / f"{stem}.txt"

            if not img_path.exists():
                raise FileNotFoundError(img_path)
            if not depth_path.exists():
                raise FileNotFoundError(depth_path)
            if not overwrite and png_out.exists() and exr_out.exists() and cam_out.exists():
                continue

            lookat = None
            if check_lookat:
                lookat = (
                    float(row["lookat_x"]),
                    float(row["lookat_y"]),
                    float(row["lookat_z"]),
                )

            tasks.append(
                Task(
                    label=label,
                    stem=stem,
                    camera_name=camera_name,
                    img_path=img_path,
                    depth_path=depth_path,
                    png_out=png_out,
                    exr_out=exr_out,
                    cam_out=cam_out,
                    position=(
                        float(row["position_x"]),
                        float(row["position_y"]),
                        float(row["position_z"]),
                    ),
                    quat=(
                        float(row["rotation_w"]),
                        float(row["rotation_x"]),
                        float(row["rotation_y"]),
                        float(row["rotation_z"]),
                    ),
                    lookat=lookat,
                )
            )
    return tasks


def convert_one(task: Task, max_size: int, depth_interp: str, png_compression: int, exr_compression: str, depth_half: bool, check_lookat: bool) -> str:
    img = cv2.imread(str(task.img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {task.img_path}")
    h0, w0 = img.shape[:2]

    depth = cv2.imread(str(task.depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read EXR depth: {task.depth_path}")
    if depth.shape[:2] != (h0, w0):
        raise RuntimeError(f"Shape mismatch for {task.stem}: image={img.shape}, depth={depth.shape}")

    new_h, new_w, scale = compute_scale(h0, w0, max_size)
    img_out = resize_array(img, new_h, new_w, is_depth=False, depth_interp=depth_interp)
    depth_out = resize_array(depth, new_h, new_w, is_depth=True, depth_interp=depth_interp)

    if depth_half and depth_out.dtype == np.float32:
        depth_out = depth_out.astype(np.float16, copy=False)

    K = get_intrinsic_matrix(task.camera_name, new_w, new_h, scale)
    ext = camera_pose_to_w2c_opencv(task.position, task.quat)

    if check_lookat and task.lookat is not None:
        lookat = np.asarray(task.lookat, dtype=np.float64)
        ln = np.linalg.norm(lookat)
        if ln > 0:
            lookat = lookat / ln
            pred = predicted_lookat_from_quat(task.quat)
            cos_sim = float(np.clip(np.dot(lookat, pred), -1.0, 1.0))
            if cos_sim < 0.999:
                print(f"[WARN] look-at mismatch for {task.label}: cos={cos_sim:.6f}")

    png_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
    if not cv2.imwrite(str(task.png_out), img_out, png_params):
        raise RuntimeError(f"Failed to write PNG: {task.png_out}")

    exr_params = [cv2.IMWRITE_EXR_COMPRESSION, EXR_COMPRESSION_MAP[exr_compression]]
    if depth_half:
        exr_params += [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]
    else:
        exr_params += [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
    if not cv2.imwrite(str(task.exr_out), depth_out, exr_params):
        raise RuntimeError(f"Failed to write EXR: {task.exr_out}")

    write_cam_file(task.cam_out, ext, K, new_h, new_w)
    return f"[OK] {task.stem} -> {new_w}x{new_h}"


def main() -> None:
    args = parse_args()
    tasks = build_tasks(args.src_root, args.dst_root, args.overwrite, args.check_lookat)

    if not tasks:
        print("Nothing to do.")
        return

    if args.num_workers > 1:
        cv2.setNumThreads(1)

    done = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [
            ex.submit(
                convert_one,
                task,
                args.max_size,
                args.depth_interp,
                args.png_compression,
                args.exr_compression,
                args.depth_half,
                args.check_lookat,
            )
            for task in tasks
        ]
        for fut in as_completed(futures):
            msg = fut.result()
            done += 1
            print(f"{done}/{len(tasks)} {msg}")

    print(f"Done. Converted {done} items into: {args.dst_root}")


if __name__ == "__main__":
    main()


"""
python scripts/convert/enrichaerial2wai.py \
    --src_root ../data/ENRICH-Aerial \
    --dst_root ../data/enrich/aerial \
    --max_size 1024 \
    --check_lookat

python scripts/convert/split_enrich_aerial.py \
    --src_root ../data/enrich/aerial \
    --dst_root ../data/enrich
"""