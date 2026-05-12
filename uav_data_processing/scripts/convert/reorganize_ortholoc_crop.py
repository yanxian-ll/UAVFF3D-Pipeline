import os
import argparse
import shutil
import json
import random
import hashlib
from typing import Tuple, List

import numpy as np
import cv2
from plyfile import PlyData
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from reorganize_utils import save_cam_txt

# 让 OpenCV 支持 EXR（前提：你的 opencv 编译时启用了 OpenEXR）
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


# ----------------------------
# Basic IO
# ----------------------------
def read_ply(ply_file: str) -> np.ndarray:
    plydata = PlyData.read(ply_file)
    vertices = plydata["vertex"]
    x = vertices["x"]
    y = vertices["y"]
    z = vertices["z"]
    points = np.vstack([x, y, z]).T
    return points


def parse_camera_json(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(json_path, "r") as f:
        data = json.load(f)

    intrinsics = np.array(data["intrinsics"], dtype=np.float32)
    extr3x4 = np.array(data["extrinsics"], dtype=np.float32)  # world2camera

    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, :] = extr3x4
    return intrinsics, extrinsics


def point_to_depth_map(point_map: np.ndarray, extrinsics_3x4: np.ndarray) -> np.ndarray:
    """
    point_map: HxWx3 (world)
    extrinsics_3x4: world2cam
    return depth: HxW (z in cam)
    """
    h, w, _ = point_map.shape
    ones = np.ones((h, w, 1), dtype=point_map.dtype)
    homogeneous_points = np.concatenate((point_map, ones), axis=-1)  # H W 4
    homogeneous_points = homogeneous_points.reshape(-1, 4).T         # 4 HW
    transformed_points = extrinsics_3x4 @ homogeneous_points         # 3 HW
    depth_map = transformed_points[2, :].reshape(h, w)
    return depth_map


def save_depth_exr_cv2(exr_path: str, depth: np.ndarray):
    """
    使用 OpenCV 写 EXR：要求 depth 为 float32 单通道
    """
    depth32 = depth.astype(np.float32, copy=False)
    ok = cv2.imwrite(exr_path, depth32)
    if not ok:
        raise RuntimeError(
            f"cv2.imwrite failed for EXR: {exr_path}\n"
            "可能原因：你的 OpenCV 没有启用 OpenEXR（即使设置了 OPENCV_IO_ENABLE_OPENEXR 也不行）。\n"
            "解决：安装/编译带 OpenEXR 的 opencv，或改保存为 .npy/.pfm。"
        )


# ----------------------------
# Cropping helpers (2x2 with overlap)
# ----------------------------
def compute_2x2_overlapped_crops(H: int, W: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """
    返回 4 个裁剪框 (x0, y0, x1, y1)，对应 2x2 网格。
    严格保证：同一行的两块在 x 方向有重叠；同一列的两块在 y 方向有重叠。
    overlap: [0, 0.70]，表示相邻块在该方向上的重叠比例（相对于裁剪块尺寸）。
    """
    overlap = float(overlap)
    overlap = max(0.0, min(overlap, 0.70))

    # 先按你原来的推导确定 crop 尺寸（保证两块覆盖全图且同尺寸）
    w_c = int(round(W / (2.0 - overlap)))
    h_c = int(round(H / (2.0 - overlap)))
    w_c = max(2, min(w_c, W))  # 至少2像素，避免 step==crop
    h_c = max(2, min(h_c, H))

    # 由 overlap 定义步长 step = crop*(1-overlap)，并强制 step <= crop-1，确保必重叠
    step_x = int(round(w_c * (1.0 - overlap)))
    step_y = int(round(h_c * (1.0 - overlap)))
    step_x = max(1, min(step_x, w_c - 1))
    step_y = max(1, min(step_y, h_c - 1))

    # 2列2行左上角（先按步长放置）
    x0_0 = 0
    x0_1 = x0_0 + step_x
    y0_0 = 0
    y0_1 = y0_0 + step_y

    # 如果越界：整体向左/上平移，保持两个块之间的相对间距=step
    # 使得 x0_1 + w_c <= W
    if x0_1 + w_c > W:
        shift = (x0_1 + w_c) - W
        x0_0 = max(0, x0_0 - shift)
        x0_1 = x0_0 + step_x

    if y0_1 + h_c > H:
        shift = (y0_1 + h_c) - H
        y0_0 = max(0, y0_0 - shift)
        y0_1 = y0_0 + step_y

    # 再保险：确保两个块都在范围内
    x0_0 = int(max(0, min(x0_0, W - w_c)))
    x0_1 = int(max(0, min(x0_1, W - w_c)))
    y0_0 = int(max(0, min(y0_0, H - h_c)))
    y0_1 = int(max(0, min(y0_1, H - h_c)))

    # 最终 4 个 crop
    crops = [
        (x0_0, y0_0, x0_0 + w_c, y0_0 + h_c),  # top-left
        (x0_1, y0_0, x0_1 + w_c, y0_0 + h_c),  # top-right
        (x0_0, y0_1, x0_0 + w_c, y0_1 + h_c),  # bottom-left
        (x0_1, y0_1, x0_1 + w_c, y0_1 + h_c),  # bottom-right
    ]
    return crops


def crop_intrinsics(
    K: np.ndarray, x0: int, y0: int
) -> np.ndarray:
    """
    裁剪后相机内参更新：principal point 平移
    K' = [[fx, 0, cx-x0],
          [0, fy, cy-y0],
          [0,  0,   1 ]]
    """
    K2 = K.copy()
    K2[0, 2] = K2[0, 2] - float(x0)
    K2[1, 2] = K2[1, 2] - float(y0)
    return K2


# ----------------------------
# Helper to generate deterministic hash seed
# ----------------------------
def get_sample_seed(base_name: str, global_seed: int) -> int:
    """
    生成确定性种子：基于样本名和全局种子
    """
    # 使用hashlib创建确定性哈希值
    hash_obj = hashlib.md5(base_name.encode('utf-8'))
    name_hash = int(hash_obj.hexdigest()[:8], 16)  # 取前8位作为哈希值
    
    # 将全局种子与名称哈希组合
    combined = (global_seed ^ name_hash) & 0xFFFFFFFF
    
    # 进一步混合
    combined = (combined * 0x5DEECE66D + 0xB) & 0xFFFFFFFFFFFFFFFF
    return int(combined & 0xFFFFFFFF)


# ----------------------------
# Per-sample processing
# ----------------------------
def process_one_image_with_crops(
    base_name: str,
    img_path: str,
    ply_path: str,
    cam_path: str,
    out_dir: str,
    split_name: str,
    n_crops: int,
    overlap_min: float,
    overlap_max: float,
    seed: int = 0,
):
    """
    对单张图做 N 次随机 overlap 裁剪，每次生成 4 个 patch。
    输出结构保持和原来一致：out_dir/<new_sample_name>/{images,cams,depth,mask}
    
    如果输出目录已存在且包含所有预期的文件，则跳过处理。
    """
    
    # ---- 为当前样本生成确定性种子 ----
    sample_seed = get_sample_seed(base_name, seed)
    
    # ---- 使用确定性种子初始化所有随机数生成器 ----
    rng = random.Random(sample_seed)
    
    # ---- 检查是否所有输出已存在（提前检查以跳过处理） ----
    all_outputs_exist = True
    
    # 预计算所有输出文件路径
    expected_outputs = []
    
    # 生成所有裁剪参数（使用确定性RNG）
    for c_idx in range(n_crops):
        overlap = rng.uniform(overlap_min, overlap_max)
        overlap = max(0.0, min(overlap, 0.70))
        
        # 为每个patch生成输出路径
        for p_idx in range(4):  # 总是4个patch
            patch_name = f"{base_name}_c{c_idx:02d}_p{p_idx:02d}_ov{overlap:.3f}"
            out_path = os.path.join(out_dir, f"{split_name}_{base_name}_ov{int(overlap*100):03d}")
            
            # 检查所有输出文件
            img_file = os.path.join(out_path, "images", patch_name + ".png")
            depth_file = os.path.join(out_path, "depth", patch_name + ".exr")
            mask_file = os.path.join(out_path, "mask", patch_name + ".png")
            cam_file = os.path.join(out_path, "cams", patch_name + ".txt")
            
            expected_outputs.extend([img_file, depth_file, mask_file, cam_file])
    
    # 重置RNG状态（因为我们需要重新生成相同的序列）
    rng = random.Random(sample_seed)
    
    # 检查所有文件是否存在
    for file_path in expected_outputs:
        if not os.path.exists(file_path):
            all_outputs_exist = False
            break
    
    # 如果所有输出都已存在，跳过处理
    if all_outputs_exist:
        print(f"跳过 {base_name}，所有输出已存在")
        return
    
    # ---- read original image (for cropping) ----
    jpg_path = os.path.join(img_path, base_name + ".jpg")
    if not os.path.isfile(jpg_path):
        raise FileNotFoundError(jpg_path)

    # 用 PIL 读尺寸，cv2 读图像内容（更快）
    with Image.open(jpg_path) as im:
        W, H = im.size

    img_bgr = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"cv2.imread failed: {jpg_path}")
    if img_bgr.shape[0] != H or img_bgr.shape[1] != W:
        # 极少数情况 PIL/cv2 读出来尺寸不一致，直接按 cv2 为准
        H, W = img_bgr.shape[:2]

    # ---- camera ----
    K, extr_world2cam = parse_camera_json(os.path.join(cam_path, base_name + ".json"))
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    # ---- depth & mask (compute once from point map) ----
    # point map: (H*W,3) -> (H,W,3)
    pts = read_ply(os.path.join(ply_path, base_name + ".ply"))
    if pts.shape[0] != H * W:
        raise ValueError(
            f"{base_name}: point count mismatch. got {pts.shape[0]}, expected {H*W} "
            f"(H={H}, W={W})"
        )
    points = pts.reshape(H, W, 3)

    depth_full = point_to_depth_map(points, extr_world2cam[:3, :])  # H W
    mask_full = np.isfinite(depth_full) & (depth_full > 1e-6)
    depth_full = depth_full.astype(np.float32, copy=False)
    depth_full[~mask_full] = 0.0
    mask_full_u8 = (mask_full.astype(np.uint8) * 255)

    # ---- generate crops ----
    for c_idx in range(n_crops):
        overlap = rng.uniform(overlap_min, overlap_max)
        overlap = max(0.0, min(overlap, 0.70))

        crop_boxes = compute_2x2_overlapped_crops(H, W, overlap)

        for p_idx, (x0, y0, x1, y1) in enumerate(crop_boxes):
            patch_name = f"{base_name}_c{c_idx:02d}_p{p_idx:02d}_ov{overlap:.3f}"
            out_path = os.path.join(out_dir, f"{split_name}_{base_name}_ov{int(overlap*100):03d}")

            os.makedirs(os.path.join(out_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(out_path, "cams"), exist_ok=True)
            os.makedirs(os.path.join(out_path, "depth"), exist_ok=True)
            os.makedirs(os.path.join(out_path, "mask"), exist_ok=True)

            # 检查单个文件是否存在
            out_img = os.path.join(out_path, "images", patch_name + ".png")
            out_depth = os.path.join(out_path, "depth", patch_name + ".exr")
            out_mask = os.path.join(out_path, "mask", patch_name + ".png")
            out_cam = os.path.join(out_path, "cams", patch_name + ".txt")
            
            # 如果所有文件都存在，跳过这个patch
            if (os.path.exists(out_img) and os.path.exists(out_depth) and 
                os.path.exists(out_mask) and os.path.exists(out_cam)):
                continue

            # ---- crop image ----
            patch_img = img_bgr[y0:y1, x0:x1, :]
            if not os.path.exists(out_img):
                cv2.imwrite(out_img, patch_img)

            # ---- crop depth & mask ----
            patch_depth = depth_full[y0:y1, x0:x1]
            patch_mask = mask_full_u8[y0:y1, x0:x1]

            if not os.path.exists(out_depth):
                save_depth_exr_cv2(out_depth, patch_depth)
            if not os.path.exists(out_mask):
                cv2.imwrite(out_mask, patch_mask)

            # ---- update intrinsics & save cam ----
            K_patch = crop_intrinsics(K, x0, y0)
            Hc, Wc = patch_depth.shape[:2]

            if not os.path.exists(out_cam):
                save_cam_txt(
                    out_cam,
                    np.linalg.inv(extr_world2cam),  # cam2world unchanged
                    float(K_patch[0, 0]), float(K_patch[1, 1]),
                    float(K_patch[0, 2]), float(K_patch[1, 2]),
                    int(Hc), int(Wc),
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../raw_data/ortholoc/unpacked")
    parser.add_argument("--out_dir", type=str, default="../data/ortholoc_crop")
    parser.add_argument("--workers", type=int, default=8, help="线程数（磁盘IO多，别太大）")

    # new args
    parser.add_argument("--splits", type=str, default="train,val,test_inPlace,test_outPlace",
                        help="逗号分隔，例如 train,val,test_inPlace,test_outPlace")
    parser.add_argument("--n_crops", type=int, default=2, help="每张原图随机裁剪的次数（每次产出4张patch）")
    parser.add_argument("--overlap_min", type=float, default=0.10, help="重叠度下限（0~0.70）")
    parser.add_argument("--overlap_max", type=float, default=0.70, help="重叠度上限（0~0.70）")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子（影响 overlap 采样）")
    args = parser.parse_args()
    
    # ---- 设置全局随机种子以保证可重复性 ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"使用随机种子: {args.seed}")

    overlap_min = float(args.overlap_min)
    overlap_max = float(args.overlap_max)
    if overlap_min > overlap_max:
        overlap_min, overlap_max = overlap_max, overlap_min
    overlap_min = max(0.0, min(overlap_min, 0.70))
    overlap_max = max(0.0, min(overlap_max, 0.70))

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("No splits specified.")

    for split in splits:
        img_dir = os.path.join(args.data_dir, split, "queries")
        cam_dir = os.path.join(args.data_dir, split, "cameras")
        ply_dir = os.path.join(args.data_dir, split, "point_maps")

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(img_dir)

        list_filenames = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
        list_filenames = sorted(list_filenames)

        os.makedirs(args.out_dir, exist_ok=True)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for base_name in list_filenames:
                futures.append(
                    executor.submit(
                        process_one_image_with_crops,
                        base_name,
                        img_dir,
                        ply_dir,
                        cam_dir,
                        args.out_dir,
                        split,
                        args.n_crops,
                        overlap_min,
                        overlap_max,
                        args.seed,
                    )
                )

            for fu in tqdm(as_completed(futures), total=len(futures), desc=f"{split}"):
                fu.result()


if __name__ == "__main__":
    main()
    