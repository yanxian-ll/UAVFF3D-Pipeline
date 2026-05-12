# -*- coding: utf-8 -*-
"""
将 BlendedMVS 数据集整理为如下结构：

输入每个场景原始结构：
scene_xxx/
├── blended_images/
│   ├── 00000000.jpg
│   ├── 00000000_masked.jpg
│   └── ...
├── rendered_depth_maps/
│   ├── 00000000.pfm
│   └── ...
└── cams/
    ├── 00000000_cam.txt
    └── ...

处理后：
scene_xxx/
├── images/
│   ├── 00000000.jpg
│   ├── 00000000_masked.jpg
│   └── ...
├── depth/
│   ├── 00000000.exr
│   └── ...
└── cams/
    ├── 00000000.txt
    └── ...

并且 cams/*.txt 会被重写为如下格式：
extrinsic opencv(x Right, y Down, z Forward) world2camera
...
...
...
...

intrinsic: fx fy cx cy (pixel)
...
...
...

h w fov
720 1024 28.841546255022

功能：
1. 将 blended_images 重命名为 images
2. 读取 rendered_depth_maps/*.pfm，保存为 depth/*.exr
3. 删除原 rendered_depth_maps
4. 将 cams 下 *_cam.txt 重命名为 *.txt
5. 读取相机参数 + 对应图像尺寸，计算 fov，重写 cams/*.txt

用法：
python convert_blendedmvs_layout_fast.py --root /path/to/blendedmvs
python convert_blendedmvs_layout_fast.py --root /path/to/blendedmvs --overwrite_depth
python convert_blendedmvs_layout_fast.py --root /path/to/blendedmvs --workers 8
"""

import os
import re
import math
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# 必须在 import cv2 之前
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def init_worker():
    """
    子进程初始化：
    避免 OpenCV / OpenMP 在每个进程里再开很多线程，导致线程过量争抢。
    """
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def load_pfm(file_path: str) -> np.ndarray:
    """
    读取 PFM 深度图，返回 float32 numpy 数组。
    支持单通道 Pf 和三通道 PF。
    """
    with open(file_path, "rb") as file:
        header = file.readline().decode("utf-8").strip()

        if header == "PF":
            is_color = True
        elif header == "Pf":
            is_color = False
        else:
            raise ValueError(f"Invalid PFM file: {file_path}")

        dim_line = file.readline().decode("utf-8")
        while dim_line.startswith("#"):
            dim_line = file.readline().decode("utf-8")

        match = re.match(r"^(\d+)\s+(\d+)\s*$", dim_line)
        if match is None:
            raise ValueError(f"Invalid PFM header format in {file_path}: {dim_line}")

        img_width, img_height = map(int, match.groups())

        endian_scale = float(file.readline().decode("utf-8").strip())
        dtype = "<f4" if endian_scale < 0 else ">f4"

        img_data = np.fromfile(file, dtype=dtype)

        expected_size = img_width * img_height * (3 if is_color else 1)
        if img_data.size != expected_size:
            raise ValueError(
                f"PFM size mismatch in {file_path}: got {img_data.size}, expected {expected_size}"
            )

        if is_color:
            img_data = img_data.reshape((img_height, img_width, 3))
        else:
            img_data = img_data.reshape((img_height, img_width))

        # PFM 通常是从底到顶存储
        img_data = np.flipud(img_data).astype(np.float32, copy=False)

    return img_data


def save_exr_depth(depth: np.ndarray, out_path: str):
    """
    使用 OpenCV 保存 EXR。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    depth = depth.astype(np.float32, copy=False)

    params = []
    if hasattr(cv2, "IMWRITE_EXR_TYPE") and hasattr(cv2, "IMWRITE_EXR_TYPE_FLOAT"):
        params.extend([cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

    ok = cv2.imwrite(str(out_path), depth, params)
    if not ok:
        raise RuntimeError(f"Failed to write EXR: {out_path}")


def convert_one_pfm_to_exr(task):
    """
    单个 PFM -> EXR 的子进程任务。
    """
    pfm_path, out_path, overwrite_depth = task
    pfm_path = str(pfm_path)
    out_path = str(out_path)

    try:
        if (not overwrite_depth) and Path(out_path).exists():
            return ("skip", pfm_path, out_path, "")

        depth = load_pfm(pfm_path)
        save_exr_depth(depth, out_path)
        return ("ok", pfm_path, out_path, "")
    except Exception as e:
        return ("fail", pfm_path, out_path, str(e))


def rename_blended_images(scene_dir: Path):
    blended_dir = scene_dir / "blended_images"
    images_dir = scene_dir / "images"

    if blended_dir.exists():
        if images_dir.exists():
            raise FileExistsError(
                f"'images' already exists in {scene_dir}, cannot rename 'blended_images'."
            )
        blended_dir.rename(images_dir)
        print(f"[Rename] {scene_dir.name}: blended_images -> images")
    else:
        if images_dir.exists():
            print(f"[Skip] {scene_dir.name}: images already exists")
        else:
            print(f"[Warn] {scene_dir.name}: neither blended_images nor images exists")


def convert_depth_maps(scene_dir: Path, overwrite_depth: bool = False, workers: int = 4):
    src_depth_dir = scene_dir / "rendered_depth_maps"
    dst_depth_dir = scene_dir / "depth"

    if not src_depth_dir.exists():
        if dst_depth_dir.exists():
            print(f"[Skip] {scene_dir.name}: depth exists and rendered_depth_maps not found")
            return
        print(f"[Warn] {scene_dir.name}: rendered_depth_maps not found")
        return

    dst_depth_dir.mkdir(parents=True, exist_ok=True)

    pfm_files = sorted(src_depth_dir.glob("*.pfm"))
    if len(pfm_files) == 0:
        print(f"[Warn] {scene_dir.name}: no .pfm files found")
        return

    tasks = [
        (pfm_path, dst_depth_dir / f"{pfm_path.stem}.exr", overwrite_depth)
        for pfm_path in pfm_files
    ]

    ok_count = 0
    skip_count = 0
    fail_list = []

    if workers <= 1:
        for task in tqdm(tasks, desc=f"{scene_dir.name} depth", leave=False):
            status, pfm_path, out_path, msg = convert_one_pfm_to_exr(task)
            if status == "ok":
                ok_count += 1
            elif status == "skip":
                skip_count += 1
            else:
                fail_list.append((pfm_path, msg))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
            futures = [executor.submit(convert_one_pfm_to_exr, task) for task in tasks]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{scene_dir.name} depth",
                leave=False,
            ):
                status, pfm_path, out_path, msg = future.result()
                if status == "ok":
                    ok_count += 1
                elif status == "skip":
                    skip_count += 1
                else:
                    fail_list.append((pfm_path, msg))

    if fail_list:
        print(f"[Fail] {scene_dir.name}: {len(fail_list)} files failed, keep rendered_depth_maps")
        for pfm_path, msg in fail_list[:10]:
            print(f"    - {pfm_path}: {msg}")
        if len(fail_list) > 10:
            print(f"    ... and {len(fail_list) - 10} more")
        return

    shutil.rmtree(src_depth_dir)
    print(
        f"[Depth] {scene_dir.name}: converted={ok_count}, skipped={skip_count}, deleted rendered_depth_maps"
    )


def extract_floats_from_line(line: str):
    """
    从一行文本中提取所有浮点数（支持科学计数法）。
    """
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", line)
    return [float(x) for x in nums]


def parse_cam_file(cam_path: Path):
    """
    解析 cam txt，提取 extrinsic(4x4) 和 intrinsic(3x3)。
    兼容原始格式与重写后的目标格式。
    """
    lines = cam_path.read_text(encoding="utf-8").splitlines()
    lines_strip = [line.strip() for line in lines]

    extrinsic_start = None
    intrinsic_start = None

    for i, line in enumerate(lines_strip):
        low = line.lower()
        if extrinsic_start is None and low.startswith("extrinsic"):
            extrinsic_start = i
        if intrinsic_start is None and low.startswith("intrinsic"):
            intrinsic_start = i

    if extrinsic_start is None:
        raise ValueError(f"'extrinsic' section not found in {cam_path}")
    if intrinsic_start is None:
        raise ValueError(f"'intrinsic' section not found in {cam_path}")

    extrinsic_rows = []
    for line in lines_strip[extrinsic_start + 1:]:
        vals = extract_floats_from_line(line)
        if len(vals) >= 4:
            extrinsic_rows.append(vals[:4])
            if len(extrinsic_rows) == 4:
                break

    if len(extrinsic_rows) != 4:
        raise ValueError(f"Failed to parse 4x4 extrinsic matrix in {cam_path}")

    intrinsic_rows = []
    for line in lines_strip[intrinsic_start + 1:]:
        vals = extract_floats_from_line(line)
        if len(vals) >= 3:
            intrinsic_rows.append(vals[:3])
            if len(intrinsic_rows) == 3:
                break

    if len(intrinsic_rows) != 3:
        raise ValueError(f"Failed to parse 3x3 intrinsic matrix in {cam_path}")

    extrinsic = np.asarray(extrinsic_rows, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_rows, dtype=np.float64)
    return extrinsic, intrinsic


def find_corresponding_image(scene_dir: Path, cam_stem: str) -> Path:
    """
    根据 cam 文件名寻找对应原图。
    优先找:
        images/{cam_stem}.jpg / .png / ...
    若找不到，则退化为 scene 中任意一张非 masked 原图。
    """
    images_dir = scene_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")

    # 先严格匹配同名原图
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
                ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP"]:
        p = images_dir / f"{cam_stem}{ext}"
        if p.exists():
            return p

    # 再宽松匹配同 stem 且不是 masked
    for p in sorted(images_dir.glob(f"{cam_stem}.*")):
        if p.is_file() and p.suffix in IMAGE_EXTS and "_masked" not in p.stem:
            return p

    # 再退化到任意非 masked 原图
    candidates = [
        p for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "_masked" not in p.stem
    ]
    if len(candidates) == 0:
        raise FileNotFoundError(f"No valid source image found in: {images_dir}")

    return candidates[0]


def read_image_hw(image_path: Path):
    """
    读取图像高宽。
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    h, w = img.shape[:2]
    return h, w


def compute_vertical_fov_deg(intrinsic: np.ndarray, image_h: int) -> float:
    """
    按示例使用 fy 和图像高 h 计算垂直 FOV:
        fov = 2 * atan(h / (2 * fy))
    返回角度制。
    """
    fy = float(intrinsic[1, 1])
    if fy <= 0:
        raise ValueError(f"Invalid fy={fy}, cannot compute FOV")
    fov_rad = 2.0 * math.atan(float(image_h) / (2.0 * fy))
    return math.degrees(fov_rad)


def format_matrix_rows(mat: np.ndarray, decimals: int = 12):
    """
    将矩阵格式化为多行字符串。
    """
    lines = []
    fmt = "{:." + str(decimals) + "f}"
    for row in mat:
        lines.append(" ".join(fmt.format(float(v)) for v in row))
    return "\n".join(lines)


def write_target_cam_file(cam_path: Path, extrinsic: np.ndarray, intrinsic: np.ndarray, h: int, w: int):
    """
    将相机参数写为目标格式。
    """
    fov_deg = compute_vertical_fov_deg(intrinsic, h)

    text = (
        "extrinsic opencv(x Right, y Down, z Forward) world2camera\n"
        f"{format_matrix_rows(extrinsic, decimals=12)}\n\n"
        "intrinsic: fx fy cx cy (pixel)\n"
        f"{format_matrix_rows(intrinsic, decimals=12)}\n\n"
        "h w fov\n"
        f"{int(h)} {int(w)} {fov_deg:.12f}\n"
    )

    cam_path.write_text(text, encoding="utf-8")


def rename_cam_files_if_needed(scene_dir: Path):
    """
    将 *_cam.txt 重命名为 *.txt
    """
    cams_dir = scene_dir / "cams"
    if not cams_dir.exists():
        print(f"[Warn] {scene_dir.name}: cams not found")
        return

    cam_files = sorted(cams_dir.glob("*_cam.txt"))
    if len(cam_files) == 0:
        print(f"[Skip] {scene_dir.name}: no *_cam.txt found")
        return

    rename_count = 0
    for cam_path in cam_files:
        new_name = cam_path.name.replace("_cam.txt", ".txt")
        new_path = cams_dir / new_name

        if new_path.exists():
            raise FileExistsError(f"Target cam file already exists: {new_path}")

        cam_path.rename(new_path)
        rename_count += 1

    print(f"[Cam-Rename] {scene_dir.name}: renamed {rename_count} files")


def is_scene_dir(path: Path) -> bool:
    """
    简单判断一个目录是否像 BlendedMVS 场景目录。
    """
    if not path.is_dir():
        return False

    has_blended = (path / "blended_images").exists()
    has_images = (path / "images").exists()
    has_depth = (path / "rendered_depth_maps").exists() or (path / "depth").exists()
    has_cams = (path / "cams").exists()

    return (has_blended or has_images) and has_depth and has_cams

    
def rewrite_one_cam_file(task):
    """
    单个 cam txt 的子进程任务：
    1. 解析 extrinsic / intrinsic
    2. 找对应图像
    3. 读取 h, w
    4. 计算 fov
    5. 重写 txt
    """
    cam_path, scene_dir = task
    cam_path = Path(cam_path)
    scene_dir = Path(scene_dir)

    try:
        extrinsic, intrinsic = parse_cam_file(cam_path)

        cam_stem = cam_path.stem
        image_path = find_corresponding_image(scene_dir, cam_stem)
        h, w = read_image_hw(image_path)

        write_target_cam_file(cam_path, extrinsic, intrinsic, h, w)
        return ("ok", str(cam_path), "")
    except Exception as e:
        return ("fail", str(cam_path), str(e))

def rewrite_cam_files(scene_dir: Path, workers: int = 4):
    """
    并行解析 cams/*.txt，读取对应图像尺寸，计算 fov，并重写为目标格式。
    """
    cams_dir = scene_dir / "cams"
    if not cams_dir.exists():
        print(f"[Warn] {scene_dir.name}: cams not found")
        return

    txt_files = sorted(cams_dir.glob("*.txt"))
    if len(txt_files) == 0:
        print(f"[Warn] {scene_dir.name}: no .txt cam files found")
        return

    tasks = [(cam_path, scene_dir) for cam_path in txt_files]

    ok_count = 0
    fail_list = []

    if workers <= 1:
        for task in tqdm(tasks, desc=f"{scene_dir.name} cams", leave=False):
            status, cam_path, msg = rewrite_one_cam_file(task)
            if status == "ok":
                ok_count += 1
            else:
                fail_list.append((cam_path, msg))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
            futures = [executor.submit(rewrite_one_cam_file, task) for task in tasks]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{scene_dir.name} cams",
                leave=False,
            ):
                status, cam_path, msg = future.result()
                if status == "ok":
                    ok_count += 1
                else:
                    fail_list.append((cam_path, msg))

    if fail_list:
        print(f"[Cam-Rewrite] {scene_dir.name}: rewritten={ok_count}, failed={len(fail_list)}")
        for cam_path, msg in fail_list[:10]:
            print(f"    - {cam_path}: {msg}")
        if len(fail_list) > 10:
            print(f"    ... and {len(fail_list) - 10} more")
    else:
        print(f"[Cam-Rewrite] {scene_dir.name}: rewritten {ok_count} files")

def move_pair_txt_to_scene_root(scene_dir: Path):
    """
    将 cams/pair.txt 移动到 scene_dir/pair.txt
    """
    cams_dir = scene_dir / "cams"
    if not cams_dir.exists():
        print(f"[Warn] {scene_dir.name}: cams not found")
        return

    src_pair = cams_dir / "pair.txt"
    if not src_pair.exists():
        print(f"[Skip] {scene_dir.name}: pair.txt not found in cams")
        return

    dst_pair = scene_dir / "pair.txt"

    if dst_pair.exists():
        raise FileExistsError(f"Target pair.txt already exists: {dst_pair}")

    src_pair.rename(dst_pair)
    print(f"[Pair] {scene_dir.name}: moved cams/pair.txt -> {dst_pair.name}")


def rename_jpg_to_png_in_images(scene_dir: Path):
    """
    将 images 目录下所有 .jpg / .jpeg 图像直接重命名为 .png
    注意：这里只改文件名，不做格式转码。
    """
    images_dir = scene_dir / "images"
    if not images_dir.exists():
        print(f"[Warn] {scene_dir.name}: images not found")
        return

    img_files = sorted([
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    ])

    if len(img_files) == 0:
        print(f"[Skip] {scene_dir.name}: no .jpg/.jpeg found in images")
        return

    rename_count = 0
    for img_path in img_files:
        new_path = img_path.with_suffix(".png")

        if new_path.exists():
            raise FileExistsError(f"Target png already exists: {new_path}")

        img_path.rename(new_path)
        rename_count += 1

    print(f"[Image-Rename] {scene_dir.name}: renamed {rename_count} jpg/jpeg -> png")

def process_scene(scene_dir: Path, overwrite_depth: bool = False, workers: int = 4):
    print(f"\n========== Processing scene: {scene_dir.name} ==========")

    rename_blended_images(scene_dir)

    # 直接改扩展名，加速，不转码
    rename_jpg_to_png_in_images(scene_dir)

    convert_depth_maps(scene_dir, overwrite_depth=overwrite_depth, workers=workers)

    rename_cam_files_if_needed(scene_dir)

    # 必须在 rewrite_cam_files 之前移动，
    # 否则 pair.txt 会被当成普通相机 txt 去解析
    move_pair_txt_to_scene_root(scene_dir)

    rewrite_cam_files(scene_dir, workers=workers)

    print(f"========== Done: {scene_dir.name} ==========")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/home/csuzhang/disk/a3dscenes/blendedmvs",
        help="BlendedMVS 数据集根目录，里面应包含多个场景子目录",
    )
    parser.add_argument(
        "--overwrite_depth",
        action="store_true",
        help="若 depth/*.exr 已存在，是否覆盖重写",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="PFM->EXR 并行进程数；机械硬盘建议 2~4，SSD 建议 4~16",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    scene_dirs = sorted([p for p in root.iterdir() if is_scene_dir(p)])
    if len(scene_dirs) == 0:
        raise RuntimeError(f"No valid BlendedMVS scene directories found under: {root}")

    print(f"Found {len(scene_dirs)} scene(s) under: {root}")
    print(f"Using workers = {args.workers}")

    for scene_dir in scene_dirs:
        process_scene(
            scene_dir,
            overwrite_depth=args.overwrite_depth,
            workers=args.workers,
        )

    print("\nAll scenes processed successfully.")


if __name__ == "__main__":
    main()