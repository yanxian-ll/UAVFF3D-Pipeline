# resize_dataset_inplace.py
import os
import re
import math
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 必须在 import cv2 前设置
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


# -----------------------------
# 可按需要修改的默认参数
# -----------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEPTH_EXTS = {".exr"}

# 插值策略
# 图像：缩小时 area，放大时 linear
# mask：最近邻
# depth：默认最近邻，避免深度边缘被插值污染；若你想平滑可改成 cv2.INTER_LINEAR
DEPTH_INTERP = cv2.INTER_NEAREST
MASK_INTERP = cv2.INTER_NEAREST


def find_existing_subdir(root: Path, name: str) -> Path:
    p = root / name
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"缺少目录: {p}")
    return p


def list_files(folder: Path, exts: set[str]):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def get_hw_from_array(arr: np.ndarray):
    if arr is None:
        return None
    return int(arr.shape[0]), int(arr.shape[1])  # h, w


def choose_image_interp(src_h: int, src_w: int, dst_h: int, dst_w: int):
    if dst_h < src_h or dst_w < src_w:
        return cv2.INTER_AREA
    return cv2.INTER_LINEAR


def resize_and_overwrite_image(path: Path, target_w: int, target_h: int, kind: str):
    """
    kind in {"image", "mask", "depth"}
    """
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return ("error", str(path), "读取失败")

    src_h, src_w = get_hw_from_array(arr)
    if src_h == target_h and src_w == target_w:
        return ("skip", str(path), f"{src_w}x{src_h}")

    if kind == "image":
        interp = choose_image_interp(src_h, src_w, target_h, target_w)
    elif kind == "mask":
        interp = MASK_INTERP
    elif kind == "depth":
        interp = DEPTH_INTERP
    else:
        return ("error", str(path), f"未知 kind: {kind}")

    resized = cv2.resize(arr, (target_w, target_h), interpolation=interp)

    ok = cv2.imwrite(str(path), resized)
    if not ok:
        return ("error", str(path), "写回失败")

    return ("resize", str(path), f"{src_w}x{src_h} -> {target_w}x{target_h}")


# -----------------------------
# cams 解析与写回
# -----------------------------
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_floats_from_line(line: str):
    vals = _FLOAT_RE.findall(line)
    return [float(v) for v in vals]


def find_header_index(lines: list[str], prefix: str):
    prefix = prefix.lower()
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(prefix):
            return i
    return -1


def next_n_numeric_lines(lines: list[str], start_idx: int, n: int):
    vals = []
    idxs = []
    i = start_idx
    while i < len(lines) and len(vals) < n:
        cur = lines[i].strip()
        nums = parse_floats_from_line(cur)
        if len(nums) > 0:
            vals.append(nums)
            idxs.append(i)
        i += 1
    if len(vals) != n:
        raise ValueError(f"从第 {start_idx} 行开始未找到足够的 {n} 行数字")
    return vals, idxs


def parse_cam_txt(path: Path):
    """
    解析如下格式：
    extrinsic opencv(x Right, y Down, z Forward) world2camera
    ...
    intrinsic: fx fy cx cy (pixel)
    ...
    h w hfov
    ...
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    extr_idx = find_header_index(lines, "extrinsic")
    intr_idx = find_header_index(lines, "intrinsic")
    hw_idx = find_header_index(lines, "h w hfov")

    if extr_idx < 0 or intr_idx < 0 or hw_idx < 0:
        raise ValueError(f"{path} 不是预期的 cams txt 格式")

    extr_lines, _ = next_n_numeric_lines(lines, extr_idx + 1, 4)
    intr_lines, _ = next_n_numeric_lines(lines, intr_idx + 1, 3)
    hw_lines, _ = next_n_numeric_lines(lines, hw_idx + 1, 1)

    extr = np.array(extr_lines, dtype=np.float64)
    K = np.array(intr_lines, dtype=np.float64)

    hw = hw_lines[0]
    if len(hw) < 2:
        raise ValueError(f"{path} 的 h w hfov 行解析失败")

    h = int(round(hw[0]))
    w = int(round(hw[1]))
    hfov = float(hw[2]) if len(hw) >= 3 else None

    return {
        "extr": extr,
        "K": K,
        "h": h,
        "w": w,
        "hfov": hfov,
    }


def compute_hfov_deg(width: int, fx: float) -> float:
    # hfov = 2 * atan(w / (2*fx))
    return math.degrees(2.0 * math.atan(width / (2.0 * fx)))


def format_cam_txt(extr: np.ndarray, K: np.ndarray, h: int, w: int, hfov: float):
    lines = []
    lines.append("extrinsic opencv(x Right, y Down, z Forward) world2camera")
    for r in range(4):
        lines.append(" ".join(f"{extr[r, c]:.12f}" for c in range(4)))
    lines.append("")
    lines.append("intrinsic: fx fy cx cy (pixel)")
    for r in range(3):
        lines.append(" ".join(f"{K[r, c]:.12f}" for c in range(3)))
    lines.append("")
    lines.append("h w hfov")
    lines.append(f"{h:d} {w:d} {hfov:.12f}")
    lines.append("")
    return "\n".join(lines)


def resize_and_overwrite_cam(path: Path, target_w: int, target_h: int):
    try:
        cam = parse_cam_txt(path)
    except Exception as e:
        return ("error", str(path), f"解析失败: {e}")

    src_h, src_w = cam["h"], cam["w"]

    if src_h == target_h and src_w == target_w:
        return ("skip", str(path), f"{src_w}x{src_h}")

    K = cam["K"].copy()

    # 使用更稳妥的像素中心缩放方式
    sx = target_w / float(src_w)
    sy = target_h / float(src_h)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    K[0, 0] = fx * sx
    K[1, 1] = fy * sy
    K[0, 2] = (cx + 0.5) * sx - 0.5
    K[1, 2] = (cy + 0.5) * sy - 0.5

    new_hfov = compute_hfov_deg(target_w, K[0, 0])

    new_text = format_cam_txt(
        extr=cam["extr"],
        K=K,
        h=target_h,
        w=target_w,
        hfov=new_hfov,
    )

    try:
        path.write_text(new_text, encoding="utf-8")
    except Exception as e:
        return ("error", str(path), f"写回失败: {e}")

    return ("resize", str(path), f"{src_w}x{src_h} -> {target_w}x{target_h}")


# -----------------------------
# 调度
# -----------------------------
def process_root(root: Path, target_w: int, target_h: int, workers: int = 1):
    cams_dir = find_existing_subdir(root, "cams")
    depth_dir = find_existing_subdir(root, "depth")
    mask_dir = find_existing_subdir(root, "mask")
    images_dir = find_existing_subdir(root, "images")

    cam_files = list_files(cams_dir, {".txt"})
    depth_files = list_files(depth_dir, DEPTH_EXTS)
    mask_files = list_files(mask_dir, MASK_EXTS)
    image_files = list_files(images_dir, IMAGE_EXTS)

    tasks = []

    for p in cam_files:
        tasks.append(("cam", p))

    for p in depth_files:
        tasks.append(("depth", p))

    for p in mask_files:
        tasks.append(("mask", p))

    for p in image_files:
        tasks.append(("image", p))

    total = len(tasks)
    if total == 0:
        print("没有找到任何可处理文件。")
        return

    print(f"开始处理: {root}")
    print(f"目标尺寸: {target_w}x{target_h}")
    print(f"文件总数: {total}")
    print(f"线程数: {workers}")

    # 避免 OpenCV 内部线程和外部线程池过度竞争
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    stats = {"resize": 0, "skip": 0, "error": 0}

    def _run_one(item):
        kind, path = item
        if kind == "cam":
            return resize_and_overwrite_cam(path, target_w, target_h)
        elif kind == "depth":
            return resize_and_overwrite_image(path, target_w, target_h, "depth")
        elif kind == "mask":
            return resize_and_overwrite_image(path, target_w, target_h, "mask")
        elif kind == "image":
            return resize_and_overwrite_image(path, target_w, target_h, "image")
        return ("error", str(path), f"未知任务类型: {kind}")

    if workers <= 1:
        for idx, item in enumerate(tasks, 1):
            status, path_str, msg = _run_one(item)
            stats[status] = stats.get(status, 0) + 1
            if status == "error":
                print(f"[{idx}/{total}] [ERROR] {path_str} | {msg}")
            elif status == "resize":
                print(f"[{idx}/{total}] [RESIZE] {path_str} | {msg}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_run_one, item): item for item in tasks}
            done_cnt = 0
            for fut in as_completed(futures):
                done_cnt += 1
                try:
                    status, path_str, msg = fut.result()
                except Exception as e:
                    status, path_str, msg = "error", str(futures[fut][1]), str(e)

                stats[status] = stats.get(status, 0) + 1
                if status == "error":
                    print(f"[{done_cnt}/{total}] [ERROR] {path_str} | {msg}")
                elif status == "resize":
                    print(f"[{done_cnt}/{total}] [RESIZE] {path_str} | {msg}")

    print("\n处理完成")
    print(f"  resize: {stats.get('resize', 0)}")
    print(f"  skip:   {stats.get('skip', 0)}")
    print(f"  error:  {stats.get('error', 0)}")


def main():
    parser = argparse.ArgumentParser(
        description="将根目录下 cams/depth/mask/images 中的文件统一缩放到目标宽高，并原地覆盖。"
    )
    parser.add_argument("--root", type=str, required=True, help="根目录，下面应有 cams/depth/mask/images")
    parser.add_argument("--width", type=int, required=True, help="目标宽度")
    parser.add_argument("--height", type=int, required=True, help="目标高度")
    parser.add_argument("--workers", type=int, default=16, help="线程数，默认 1。数据量大可设为 4/8")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"无效根目录: {root}")

    if args.width <= 0 or args.height <= 0:
        raise ValueError("width 和 height 必须为正整数")

    process_root(
        root=root,
        target_w=args.width,
        target_h=args.height,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

"""

python scripts/resize_dataset.py --root /opt/data/private/dataset/data/urbanscene3d/artsci --width 1500 --height 999


"""