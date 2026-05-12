import os
import argparse
import shutil
from plyfile import PlyData
from PIL import Image
import numpy as np
import cv2
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from reorganize_utils import save_cam_txt

# 让 OpenCV 支持 EXR（前提：你的 opencv 编译时启用了 OpenEXR）
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def read_ply(ply_file):
    plydata = PlyData.read(ply_file)
    vertices = plydata["vertex"]
    x = vertices["x"]
    y = vertices["y"]
    z = vertices["z"]
    points = np.vstack([x, y, z]).T
    return points


def parse_camera_json(json_path):
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


def process(filename, img_path, ply_path, cam_path, out_dir, split):
    out_path = os.path.join(out_dir, f"{split}_{filename}")
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "cams"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "depth"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "mask"), exist_ok=True)

    # ---- copy image as png ----
    out_img = os.path.join(out_path, "images", filename + ".png")
    if not os.path.exists(out_img):
        shutil.copy2(
            os.path.join(img_path, filename + ".jpg"),
            out_img,
        )

    # ---- read size ----
    with Image.open(os.path.join(img_path, filename + ".jpg")) as img:
        width, height = img.size

    # ---- camera ----
    intr, extr_world2cam = parse_camera_json(os.path.join(cam_path, filename + ".json"))

    # ---- depth & mask ----
    out_depth = os.path.join(out_path, "depth", filename + ".exr")
    out_mask = os.path.join(out_path, "mask", filename + ".png")

    if (not os.path.exists(out_depth)) or (not os.path.exists(out_mask)):
        # point map: (H*W,3) -> (H,W,3)
        pts = read_ply(os.path.join(ply_path, filename + ".ply"))
        if pts.shape[0] != height * width:
            raise ValueError(
                f"{filename}: point count mismatch. got {pts.shape[0]}, expected {height*width} "
                f"(height={height}, width={width})"
            )
        points = pts.reshape(height, width, 3)

        depth = point_to_depth_map(points, extr_world2cam[:3, :])  # H W
        # mask: depth finite & positive (你也可以只用 ~nan)
        mask = np.isfinite(depth) & (depth > 1e-6)
        depth = depth.astype(np.float32, copy=False)
        depth[~mask] = 0.0

        # 写 EXR（用 cv2）
        save_depth_exr_cv2(out_depth, depth)

        # 写 mask（uint8）
        cv2.imwrite(out_mask, (mask.astype(np.uint8) * 255))

    # ---- save cam txt ----
    save_cam_txt(
        os.path.join(out_path, "cams", filename + ".txt"),
        np.linalg.inv(extr_world2cam),  # cam2world
        float(intr[0, 0]), float(intr[1, 1]), float(intr[0, 2]), float(intr[1, 2]),
        int(height), int(width),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../raw_data/ortholoc/unpacked")
    parser.add_argument("--out_dir", type=str, default="../data/ortholoc")
    parser.add_argument("--workers", type=int, default=8, help="线程数（磁盘IO多，别太大）")
    args = parser.parse_args()

    splits = ["test_inPlace", "test_outPlace"]

    for split in splits:
        img_dir = os.path.join(args.data_dir, split, "queries")
        cam_dir = os.path.join(args.data_dir, split, "cameras")
        ply_dir = os.path.join(args.data_dir, split, "point_maps")

        list_filenames = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
        list_filenames = sorted(list_filenames)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for filename in list_filenames:
                futures.append(
                    executor.submit(process, filename, img_dir, ply_dir, cam_dir, args.out_dir, split)
                )

            for fu in tqdm(as_completed(futures), total=len(futures), desc=f"{split}"):
                fu.result()


if __name__ == "__main__":
    main()
