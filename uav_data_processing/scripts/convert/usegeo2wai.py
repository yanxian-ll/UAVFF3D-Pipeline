import os
import shutil
import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff

# 必须放在 import cv2 之前
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

from tqdm import tqdm

# =========================================================
# 坐标系约定
# ---------------------------------------------------------
# OpenCV相机系（由像素+depth/range反投影得到）:
#   x: 向右
#   y: 向下
#   z: 向前
#
# 摄影测量 photo 相机系:
#   x: 向右
#   y: 向上
#   z: 向后
#
# 因此:
#   p_photo = F_cv_to_photo @ p_cv
# =========================================================
F_CV_TO_PHOTO = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


# =========================================================
# 1. 读取 pose 文件
# =========================================================
def read_pose_file(path: Path):
    data_dict = {}

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header = None
    for line in lines:
        s = line.strip()
        if s.startswith("#") and "label" in s:
            header = s.replace("#", "").split()
            break

    if header is None:
        raise RuntimeError(f"Cannot find header in pose file: {path}")

    print("Pose header:", header)

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        parts = s.split()
        if len(parts) != len(header):
            continue

        item = {}
        for k, v in zip(header, parts):
            if k == "label":
                item[k] = v
            else:
                item[k] = float(v)

        data_dict[item["label"]] = item

    return data_dict


# =========================================================
# 2. 基础旋转矩阵
# =========================================================
def Rx(a):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(a), -np.sin(a)],
            [0, np.sin(a), np.cos(a)],
        ],
        dtype=np.float64,
    )


def Ry(a):
    return np.array(
        [
            [np.cos(a), 0, np.sin(a)],
            [0, 1, 0],
            [-np.sin(a), 0, np.cos(a)],
        ],
        dtype=np.float64,
    )


def Rz(a):
    return np.array(
        [
            [np.cos(a), -np.sin(a), 0],
            [np.sin(a), np.cos(a), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


# =========================================================
# 3. B 方案旋转
# ---------------------------------------------------------
# modelB:
#   R = Rx(omega) @ Ry(phi) @ Rz(kappa)
#   解释为 photo_camera -> world
# =========================================================
def rotation_model_b(omega_deg, phi_deg, kappa_deg):
    omega = np.deg2rad(omega_deg)
    phi = np.deg2rad(phi_deg)
    kappa = np.deg2rad(kappa_deg)
    return Rx(omega) @ Ry(phi) @ Rz(kappa)


# =========================================================
# 4. UseGeo 内参 -> 当前图像 OpenCV 内参
# =========================================================
def convert_usegeo_intrinsics(c, x0, y0, orig_w, orig_h, new_w, new_h):
    """
    输入:
        c, x0, y0 为原始尺寸下参数
    输出:
        缩放到当前图像尺寸下的 OpenCV 像素内参

    这里采用:
        cx = x0 * sx
        cy = (-y0) * sy
    """
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid original size: ({orig_w}, {orig_h})")

    sx = new_w / float(orig_w)
    sy = new_h / float(orig_h)

    fx = c * sx
    fy = c * sy
    cx = x0 * sx
    cy = (-y0) * sy

    return fx, fy, cx, cy, sx, sy


# =========================================================
# 5. range map -> z-depth
# ---------------------------------------------------------
# UseGeo range/ray depth 转 z-depth:
#   xn = (u-cx)/fx
#   yn = (v-cy)/fy
#   denom = sqrt(xn^2 + yn^2 + 1)
#   z = range / denom
# =========================================================
def range_to_zdepth(range_map, fx, fy, cx, cy):
    h, w = range_map.shape

    u, v = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    r = range_map.astype(np.float32)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    xn = (u - cx) / fx
    yn = (v - cy) / fy
    denom = np.sqrt(xn * xn + yn * yn + 1.0)

    z = r / denom
    z[~np.isfinite(z)] = 0.0
    z[z < 0] = 0.0

    return z.astype(np.float32)


# =========================================================
# 6. 全局平移基准
# ---------------------------------------------------------
# 为了避免保存 cam 时相机中心坐标过大，先把整个场景整体平移：
#   C_shifted = C - global_offset
# 这样不会改变相机之间的相对几何关系，只是把世界坐标原点挪走。
# =========================================================
def get_camera_center_from_pose(pose):
    return np.array([pose["X0"], pose["Y0"], pose["Z0"]], dtype=np.float64)


def compute_global_translation_offset(pose_dict, mode="mean", manual_offset=None):
    mode = str(mode).lower()

    if mode == "none":
        return np.zeros(3, dtype=np.float64)

    centers = np.stack([get_camera_center_from_pose(p) for p in pose_dict.values()], axis=0)

    if mode == "mean":
        return centers.mean(axis=0)
    if mode == "first":
        return centers[0].copy()
    if mode == "manual":
        if manual_offset is None:
            raise ValueError("manual_offset must be provided when shift_mode='manual'")
        manual_offset = np.asarray(manual_offset, dtype=np.float64).reshape(3)
        return manual_offset

    raise ValueError(f"Unsupported shift_mode: {mode}")


def save_global_shift(out_dir: Path, shift_mode: str, global_offset: np.ndarray):
    shift_path = out_dir / "cam_global_shift.txt"
    with open(shift_path, "w", encoding="utf-8") as f:
        f.write("# Global translation offset applied before exporting cams\n")
        f.write("# shifted_camera_center = original_camera_center - global_offset\n")
        f.write(f"shift_mode {shift_mode}\n")
        f.write(f"global_offset_x {global_offset[0]:.12f}\n")
        f.write(f"global_offset_y {global_offset[1]:.12f}\n")
        f.write(f"global_offset_z {global_offset[2]:.12f}\n")


# =========================================================
# 7. B 方案 world -> cv camera 外参
# ---------------------------------------------------------
# modelB:
#   p_world = R_photo * p_photo + C
#   p_photo = R_photo^T * (p_world - C)
#   p_cv    = F * p_photo
#
# 若加入整体平移：
#   C_shifted = C - global_offset
#
# 则:
#   p_cv = (F @ R_photo^T) p_world_shifted + (-F @ R_photo^T @ C_shifted)
# =========================================================
def build_cv_extrinsic_from_model_b(pose, global_offset=None):
    C = get_camera_center_from_pose(pose)
    if global_offset is None:
        global_offset = np.zeros(3, dtype=np.float64)
    global_offset = np.asarray(global_offset, dtype=np.float64).reshape(3)
    C_shifted = C - global_offset

    R_photo = rotation_model_b(
        pose["omega[deg]"],
        pose["phi[deg]"],
        pose["kappa[deg]"],
    )

    R_cv = F_CV_TO_PHOTO @ R_photo.T
    t_cv = -R_cv @ C_shifted.reshape(3, 1)

    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = R_cv
    extrinsic[:3, 3] = t_cv.flatten()
    return extrinsic, C_shifted


# =========================================================
# 8. 写 cam 文件
# =========================================================
def calculate_fov(fy, h):
    return 2 * np.arctan(h / (2 * fy)) * 180.0 / np.pi


def write_cam_file(cam_path, extrinsic, fx, fy, cx, cy, h, w):
    fov = calculate_fov(fy, h)

    with open(cam_path, "w", encoding="utf-8") as f:
        f.write("extrinsic opencv(x Right, y Down, z Forward) world2camera\n")
        for i in range(4):
            r = extrinsic[i]
            f.write(f"{r[0]} {r[1]} {r[2]} {r[3]}\n")

        f.write("\nintrinsic: fx fy cx cy (pixel)\n")
        f.write(f"{fx} 0 {cx}\n")
        f.write(f"0 {fy} {cy}\n")
        f.write("0 0 1\n")

        f.write("\nh w fov\n")
        f.write(f"{h} {w} {fov}\n")


# =========================================================
# 9. 单张处理
# =========================================================
def process_one(depth_path, image_dir, pose_dict, args, images_dir, depth_out_dir, cams_dir, global_offset):
    base_name = depth_path.stem.replace("_depth_res", "")
    pose_name = base_name + ".jpg"
    src_rgb_name = depth_path.stem.replace("_depth_res", "_res") + ".jpg"
    rgb_path = image_dir / src_rgb_name

    if not rgb_path.exists():
        print(f"❌ missing RGB: {rgb_path}")
        return

    if pose_name not in pose_dict:
        print(f"❌ missing pose: {pose_name}")
        return

    pose = pose_dict[pose_name]
    print("=" * 100)
    print(f"Processing: {pose_name}")
    print(pose)

    range_map = tiff.imread(depth_path)
    range_map = np.nan_to_num(range_map, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        print(f"❌ failed to read RGB: {rgb_path}")
        return

    cur_h, cur_w = range_map.shape[:2]
    if rgb_bgr.shape[0] != cur_h or rgb_bgr.shape[1] != cur_w:
        raise ValueError(
            f"Shape mismatch: range={range_map.shape}, rgb={rgb_bgr.shape}"
        )

    fx, fy, cx, cy, sx, sy = convert_usegeo_intrinsics(
        c=pose["c"],
        x0=pose["x0"],
        y0=pose["y0"],
        orig_w=args.width,
        orig_h=args.height,
        new_w=cur_w,
        new_h=cur_h,
    )

    # 1) range -> z-depth（全分辨率）
    z_depth = range_to_zdepth(
        range_map=range_map,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    # 2) 保存 image 到 images/*.jpg
    out_img_path = images_dir / f"{base_name}.jpg"
    shutil.copy2(rgb_path, out_img_path)

    # 3) 保存 z-depth 到 depth/*.exr
    out_depth_path = depth_out_dir / f"{base_name}.exr"
    ok = cv2.imwrite(str(out_depth_path), z_depth)
    if not ok:
        raise RuntimeError(
            f"Failed to write EXR: {out_depth_path}. "
            f"Please check OpenCV OpenEXR support."
        )

    # 4) 保存 cam 到 cams/*.txt（保存前先整体平移）
    extrinsic, C_shifted = build_cv_extrinsic_from_model_b(pose, global_offset=global_offset)
    out_cam_path = cams_dir / f"{base_name}.txt"
    write_cam_file(out_cam_path, extrinsic, fx, fy, cx, cy, cur_h, cur_w)

    if args.verbose_cam_shift:
        C_raw = get_camera_center_from_pose(pose)
        print(f"  raw camera center     : {C_raw}")
        print(f"  shifted camera center : {C_shifted}")
        print(f"  global offset         : {global_offset}")


# =========================================================
# 10. 主流程
# =========================================================
def main(args):
    depth_dir = Path(args.depth_dir)
    image_dir = Path(args.image_dir)
    pose_path = Path(args.pose)
    out_dir = Path(args.out)

    images_dir = out_dir / "images"
    depth_out_dir = out_dir / "depth"
    cams_dir = out_dir / "cams"

    images_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir.mkdir(parents=True, exist_ok=True)
    cams_dir.mkdir(parents=True, exist_ok=True)

    pose_dict = read_pose_file(pose_path)

    manual_offset = None
    if str(args.shift_mode).lower() == "manual":
        manual_offset = [args.shift_tx, args.shift_ty, args.shift_tz]

    global_offset = compute_global_translation_offset(
        pose_dict=pose_dict,
        mode=args.shift_mode,
        manual_offset=manual_offset,
    )
    print(f"Global shift mode   : {args.shift_mode}")
    print(f"Global shift offset : {global_offset}")
    save_global_shift(out_dir, args.shift_mode, global_offset)

    depth_files = sorted(depth_dir.glob("*.tiff"))

    if args.max_count > 0:
        depth_files = depth_files[:args.max_count]

    print(f"Total depth files: {len(depth_files)}")

    for depth_path in tqdm(depth_files):
        process_one(
            depth_path=depth_path,
            image_dir=image_dir,
            pose_dict=pose_dict,
            args=args,
            images_dir=images_dir,
            depth_out_dir=depth_out_dir,
            cams_dir=cams_dir,
            global_offset=global_offset,
        )

    print("✅ All done!")


# =========================================================
# 11. CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset1
    parser.add_argument("--depth_dir", default="/opt/data/private/dataset/raw_data/dataset1/Depth_resized/depth_maps")
    parser.add_argument("--image_dir", default="/opt/data/private/dataset/raw_data/dataset1/Depth_resized/undistorted_images")
    parser.add_argument("--pose", default="/opt/data/private/dataset/raw_data/dataset1/Image_orientations_dataset1.xyz")
    parser.add_argument("--width", default=7953, type=int, help="原始图像宽度")
    parser.add_argument("--height", default=5279, type=int, help="原始图像高度")
    parser.add_argument("--out", default="./output/dataset1_export")

    # dataset2
    # parser.add_argument("--depth_dir", default="/opt/data/private/dataset/raw_data/dataset2/Depth_resized/depth_maps")
    # parser.add_argument("--image_dir", default="/opt/data/private/dataset/raw_data/dataset2/Depth_resized/undistorted_images")
    # parser.add_argument("--pose", default="/opt/data/private/dataset/raw_data/dataset2/Image_orientations_dataset2.xyz")
    # parser.add_argument("--width", default=7954, type=int, help="原始图像宽度")
    # parser.add_argument("--height", default=5279, type=int, help="原始图像高度")
    # parser.add_argument("--out", default="./output/dataset2_export")

    # dataset3
    # parser.add_argument("--depth_dir", default="/opt/data/private/dataset/raw_data/dataset3/Depth_resized/depth_maps")
    # parser.add_argument("--image_dir", default="/opt/data/private/dataset/raw_data/dataset3/Depth_resized/undistorted_images")
    # parser.add_argument("--pose", default="/opt/data/private/dataset/raw_data/dataset3/Image_orientations_dataset3.xyz")
    # parser.add_argument("--width", default=7955, type=int, help="原始图像宽度")
    # parser.add_argument("--height", default=5279, type=int, help="原始图像高度")
    # parser.add_argument("--out", default="./output/dataset3_export")

    parser.add_argument("--max_count", default=0, type=int, help="只处理前 N 张，0 表示全部")

    parser.add_argument(
        "--shift_mode",
        default="mean",
        choices=["none", "mean", "first", "manual"],
        help="导出 cam 前对所有相机中心施加统一平移。mean=减去所有相机中心均值，first=减去第一张相机中心，manual=减去手动给定偏移，none=不平移",
    )
    parser.add_argument("--shift_tx", default=0.0, type=float, help="manual 模式下的全局平移 X")
    parser.add_argument("--shift_ty", default=0.0, type=float, help="manual 模式下的全局平移 Y")
    parser.add_argument("--shift_tz", default=0.0, type=float, help="manual 模式下的全局平移 Z")
    parser.add_argument("--verbose_cam_shift", action="store_true", help="打印每张图平移前后的相机中心")

    args = parser.parse_args()
    main(args)

