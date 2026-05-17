#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import re

# OpenCV 写 EXR：需要你的 opencv 构建支持 OpenEXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import open3d as o3d
import numpy as np
from tqdm import tqdm

# ---------------------------
# 基础 IO / resize
# ---------------------------
def resize_image_to_size(image_path: str, size_wh):
    """严格 resize 到固定 (W,H)，返回 img, new_w, new_h, (sx,sy), (orig_w,orig_h)"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None, None, None, None, None
    orig_h, orig_w = img.shape[:2]
    new_w, new_h = int(size_wh[0]), int(size_wh[1])
    if (orig_w, orig_h) == (new_w, new_h):
        sx = sy = 1.0
        return img, new_w, new_h, (sx, sy), (orig_w, orig_h)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    sx = float(new_w) / float(orig_w)
    sy = float(new_h) / float(orig_h)
    return resized, new_w, new_h, (sx, sy), (orig_w, orig_h)


def resize_depth_to_size(depth_path: Path, size_wh):
    """严格 resize depth 到固定 (W,H)"""
    depth = _read_depth_any(depth_path)
    if depth is None:
        return None
    new_w, new_h = int(size_wh[0]), int(size_wh[1])
    if depth.shape[1] == new_w and depth.shape[0] == new_h:
        return depth
    return cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def resize_mask_to_size(mask_path: Path, size_wh):
    """严格 resize mask 到固定 (W,H)"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    new_w, new_h = int(size_wh[0]), int(size_wh[1])
    if mask.shape[1] == new_w and mask.shape[0] == new_h:
        return mask
    return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def _read_depth_any(path: Path):
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        d = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if d is None:
        return None
    if d.ndim == 3:
        d = d[..., 0]
    return d

def save_exr(depth_2d, filepath: Path):
    if depth_2d is None:
        return False
    depth_2d = depth_2d.astype(np.float32, copy=False)
    return bool(cv2.imwrite(str(filepath), depth_2d))

# ---------------------------
# COLMAP 解析
# ---------------------------

def parse_cameras_text(path: Path):
    cameras = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[camera_id] = dict(model=model, width=width, height=height, params=params)
    return cameras


def parse_images_text(path: Path):
    """
    COLMAP images.txt: 每个image两行：第一行 pose+name，第二行 2D points（可为空）
    """
    images = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if (not line) or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue

        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)   # w x y z
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)   # world2cam translation
        camera_id = int(parts[8])
        image_name = " ".join(parts[9:])

        images.append(dict(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name))

        i += 2  # skip points2D line
    return images


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)


def rotmat2extrinsic(R, t):
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    return extrinsic

def intrinsic_from_params(params, model_name, sx, sy, new_w, new_h):
    """
    用 sx, sy 分别缩放，避免单一 scale + round 导致的 1~2px 偏移
    """
    if model_name == "PINHOLE":
        fx = params[0] * sx
        fy = params[1] * sy
        cx = params[2] * sx
        cy = params[3] * sy
    elif model_name == "SIMPLE_PINHOLE":
        f = params[0]
        # SIMPLE_PINHOLE 只有一个 f：对非等比例缩放，通常取平均或分别缩放都行
        # 为了和图像实际变换一致，取 sx/sy 平均缩放 f（也可取 min/max，看你需求）
        f_s = f * (0.5 * (sx + sy))
        fx = fy = f_s
        cx = params[1] * sx
        cy = params[2] * sy
    elif model_name in ("SIMPLE_RADIAL", "RADIAL"):
        f = params[0]
        f_s = f * (0.5 * (sx + sy))
        fx = fy = f_s
        cx = params[1] * sx
        cy = params[2] * sy
    else:
        fx = params[0] * sx if len(params) > 0 else new_w / 2.0
        fy = params[1] * sy if len(params) > 1 else fx
        cx = params[2] * sx if len(params) > 2 else new_w / 2.0
        cy = params[3] * sy if len(params) > 3 else new_h / 2.0
    return fx, fy, cx, cy


def calculate_fov(fy, h):
    return 2.0 * np.arctan(float(h) / (2.0 * float(fy))) * 180.0 / np.pi


def write_cam_file(cam_path: Path, extrinsic, fx, fy, cx, cy, h, w):
    fov = calculate_fov(fy, h)
    with open(cam_path, "w", encoding="utf-8") as f:
        f.write("extrinsic opencv(x Right, y Down, z Forward) world2camera\n")
        for i in range(4):
            r = extrinsic[i]
            f.write(f"{r[0]:.12f} {r[1]:.12f} {r[2]:.12f} {r[3]:.12f}\n")
        f.write("\n")

        f.write("intrinsic: fx fy cx cy (pixel)\n")
        f.write(f"{fx:.12f} 0.000000000000 {cx:.12f}\n")
        f.write(f"0.000000000000 {fy:.12f} {cy:.12f}\n")
        f.write("0.000000000000 0.000000000000 1.000000000000\n")
        f.write("\n")

        f.write("h w fov\n")
        f.write(f"{h} {w} {fov:.12f}\n")


# ---------------------------
# LiDAR -> depth/mask 渲染（Open3D legacy Visualizer）
# ---------------------------

def load_point_cloud(ply_path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {ply_path}")
    if not pcd.has_colors():
        n = np.asarray(pcd.points).shape[0]
        colors = np.full((n, 3), 0.75, dtype=np.float32)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


class VisCache:
    """缓存不同 (w,h) 的 Open3D legacy Visualizer，并记录是否已添加几何体"""
    def __init__(self, point_size=2.0, visible=False):
        self.cache = {}  # (w,h) -> {"vis": vis, "inited": bool}
        self.point_size = float(point_size)
        self.visible = bool(visible)

    def get(self, w: int, h: int):
        key = (int(w), int(h))
        if key in self.cache:
            return self.cache[key]["vis"], self.cache[key]["inited"], key

        vis = o3d.visualization.Visualizer()
        ok = vis.create_window(window_name=f"o3d_{w}x{h}", width=w, height=h, visible=self.visible)
        if not ok:
            raise RuntimeError(
                "Open3D Visualizer create_window failed.\n"
                "无显示环境请用 xvfb-run；Windows 请确保有图形界面可用。"
            )

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        opt.point_size = self.point_size

        self.cache[key] = {"vis": vis, "inited": False}
        return vis, False, key

    def mark_inited(self, key):
        self.cache[key]["inited"] = True

    def close_all(self):
        for v in self.cache.values():
            try:
                v["vis"].destroy_window()
            except Exception:
                pass
        self.cache.clear()


def set_camera_world2cam(vis: o3d.visualization.Visualizer, fx, fy, cx, cy, world2cam: np.ndarray, w: int, h: int):
    """
    Open3D legacy 需要 extrinsic = world2cam
    """
    params = o3d.camera.PinholeCameraParameters()

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=int(w),
        height=int(h),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
    )
    params.intrinsic = intrinsic
    params.extrinsic = world2cam.astype(np.float64, copy=False)

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)


def render_depth_only(vis: o3d.visualization.Visualizer):
    vis.poll_events()
    vis.update_renderer()
    depth_f = np.asarray(vis.capture_depth_float_buffer(do_render=True), dtype=np.float32)
    return depth_f.astype(np.float32)


def depth_to_mask(depth: np.ndarray, eps: float = 1e-6, max_depth: float = 0.0) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > float(eps))
    if max_depth is not None and float(max_depth) > 0.0:
        valid &= depth < float(max_depth)
    return (valid.astype(np.uint8) * 255)


class LidarDepthRenderer:
    def __init__(self, lidar_ply: str, point_size=2.0, visible=False):
        self.pcd = load_point_cloud(lidar_ply)
        self.cache = VisCache(point_size=point_size, visible=visible)

    def render(self, fx, fy, cx, cy, world2cam: np.ndarray, w: int, h: int) -> np.ndarray:
        vis, inited, key = self.cache.get(w, h)
        if not inited:
            vis.clear_geometries()
            vis.add_geometry(self.pcd, reset_bounding_box=True)
            self.cache.mark_inited(key)

        set_camera_world2cam(vis, fx, fy, cx, cy, world2cam, w, h)
        depth = render_depth_only(vis)

        # 防止某些平台 viewport 不等于请求分辨率：强制 resize 回 (h,w)
        if depth.shape[0] != h or depth.shape[1] != w:
            depth = cv2.resize(depth, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)

        return depth

    def close(self):
        self.cache.close_all()


# ---------------------------
# 文件名解析和分组
# ---------------------------

def extract_timestamp_and_camera(image_name: str, camera_extract_method="first_char", camera_separator="_", camera_position=-1):
    """
    从图像文件名中提取时间戳和相机标识
    
    参数:
    image_name: 图像文件名
    camera_extract_method: 相机标识提取方法
        - "first_char": 第一个字符
        - "last_char": 最后一个字符
        - "prefix": 前缀（直到第一个分隔符）
        - "suffix": 后缀（最后一个分隔符之后）
        - "split": 按分隔符分割后取指定位置
        - "custom_regex": 自定义正则表达式（需要在函数外处理）
    camera_separator: 分隔符，用于split方法
    camera_position: 位置索引，用于split方法（从0开始，负数表示从后往前）
    """
    stem = Path(image_name).stem  # 去掉扩展名
    # TODO：
    stem = "_".join(stem.split("_")[:-1])
    
    camera = ""
    timestamp = stem
    
    if camera_extract_method == "first_char":
        # 第一个字符作为相机标识
        if len(stem) > 0:
            camera = stem[0]
            timestamp = stem[1:] if len(stem) > 1 else ""
    
    elif camera_extract_method == "last_char":
        # 最后一个字符作为相机标识
        if len(stem) > 0:
            camera = stem[-1]
            timestamp = stem[:-1] if len(stem) > 1 else ""
    
    elif camera_extract_method == "prefix":
        # 前缀作为相机标识（直到第一个分隔符）
        if camera_separator in stem:
            parts = stem.split(camera_separator, 1)
            camera = parts[0]
            timestamp = parts[1] if len(parts) > 1 else ""
    
    elif camera_extract_method == "suffix":
        # 后缀作为相机标识（最后一个分隔符之后）
        if camera_separator in stem:
            parts = stem.rsplit(camera_separator, 1)
            camera = parts[1] if len(parts) > 1 else ""
            timestamp = parts[0]
    
    elif camera_extract_method == "split":
        # 按分隔符分割后取指定位置
        if camera_separator in stem:
            parts = stem.split(camera_separator)
            try:
                if camera_position >= 0:
                    if camera_position < len(parts):
                        camera = parts[camera_position]
                        # 移除相机标识部分，剩余作为时间戳
                        timestamp_parts = parts[:camera_position] + parts[camera_position+1:]
                        timestamp = camera_separator.join(timestamp_parts) if timestamp_parts else ""
                else:
                    # 负数索引，从后往前
                    pos = len(parts) + camera_position
                    if 0 <= pos < len(parts):
                        camera = parts[pos]
                        # 移除相机标识部分，剩余作为时间戳
                        timestamp_parts = parts[:pos] + parts[pos+1:]
                        timestamp = camera_separator.join(timestamp_parts) if timestamp_parts else ""
            except (IndexError, ValueError):
                pass
    
    elif camera_extract_method == "regex":
        # 使用正则表达式提取
        # 这里需要一个正则表达式模式，但我们在函数参数中没有传递
        # 所以这个模式需要在函数外部定义
        pass
    
    # 如果没有提取到相机标识，尝试使用最后一个下划线作为分隔符（兼容旧逻辑）
    if not camera and "_" in stem:
        parts = stem.rsplit("_", 1)
        if re.match(r'^[A-Za-z0-9]{1,3}$', parts[-1]):
            camera = parts[-1]
            timestamp = parts[0]
    
    return timestamp, camera


def group_images_by_timestamp(images, camera_types=None, camera_extract_method="first_char", 
                             camera_separator="_", camera_position=-1):
    """
    根据时间戳分组图像
    返回: dict{timestamp: list[image_info]}
    """
    groups = defaultdict(list)
    
    for img_info in images:
        image_name = img_info["name"]
        timestamp, camera = extract_timestamp_and_camera(
            image_name, 
            camera_extract_method=camera_extract_method,
            camera_separator=camera_separator,
            camera_position=camera_position
        )
        
        # 如果指定了相机类型，只处理指定相机的图像
        if camera_types and camera not in camera_types:
            continue
        
        # 添加相机标识到图像信息中
        img_info_with_camera = img_info.copy()
        img_info_with_camera["camera_identifier"] = camera
        img_info_with_camera["timestamp"] = timestamp
        
        groups[timestamp].append(img_info_with_camera)
    
    return groups


# ---------------------------
# 输出目录
# ---------------------------

def _prepare_output_dirs(output_dir: Path, scene_name: str):
    """准备输出目录，所有相机类型都输出到同一目录"""
    out_dir = output_dir / f"{scene_name}"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "cams").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    (out_dir / "mask").mkdir(parents=True, exist_ok=True)
    return out_dir

def compute_global_target_size(cameras: dict, max_size: int):
    """
    用 COLMAP camera 的宽高作为基准，按 max_size 缩放一次，得到全局固定 (W,H)
    - 如果存在多个 camera 分辨率，取出现最多的那组(更稳)
    """
    wh_list = []
    for cam_id, info in cameras.items():
        wh_list.append((int(info["width"]), int(info["height"])))

    # 取出现最多的分辨率
    from collections import Counter
    (base_w, base_h), _ = Counter(wh_list).most_common(1)[0]

    scale = min(1.0, float(max_size) / float(max(base_w, base_h)))
    target_w = max(1, int(round(base_w * scale)))
    target_h = max(1, int(round(base_h * scale)))
    return target_w, target_h, base_w, base_h


# ---------------------------
# 单帧处理（逻辑：优先 LiDAR 渲染，否则走 COLMAP depth 目录）
# ---------------------------
def _process_one(frame_id, timestamp, camera_images, cameras,
                 input_images_dir: Path,
                 input_depth_dir: Path,
                 out_dir: Path,
                 max_size: int,
                 lidar_renderer: LidarDepthRenderer | None,
                 fill_lidar_holes_with_mvs: bool,
                 camera_id_to_number: dict,
                 target_size_wh: tuple
                 ):  # 新增参数：相机编号映射

    processed_count = 0
    
    for img_info in camera_images:
        image_name = img_info["name"]
        camera_id = img_info["camera_identifier"]
        
        # 使用映射将相机标识转换为数字编号（如cam001）
        if camera_id and len(camera_id_to_number.keys()) > 1:
            cam_number = camera_id_to_number[camera_id]
            file_id = f"{frame_id:08d}_{cam_number}"
        else:
            file_id = f"{frame_id:08d}"

        input_image_path = input_images_dir / image_name
        if not input_image_path.exists():
            continue

        # COLMAP depth/mask 文件名（假设与原始图像名对应）
        stem = Path(image_name).stem
        stem = "_".join(stem.split("_")[:-1])

        depth_name = f"{stem}.tif"
        mask_name = f"{stem}_mask.pgm"
        input_depth_path = input_depth_dir / depth_name
        input_mask_path = input_depth_dir / mask_name

        if (not input_depth_path.exists()) or (not input_mask_path.exists()):
            continue
        
        cam_id = img_info["camera_id"]
        cam_info = cameras.get(cam_id, None)
        if cam_info is None:
            continue

        # RGB resize & save
        # ===== 统一到全局固定尺寸 =====
        resized_img, new_w, new_h, (sx, sy), (orig_w, orig_h) = resize_image_to_size(
            str(input_image_path), size_wh=target_size_wh
        )
        if resized_img is None:
            continue

        out_img_path = out_dir / "images" / f"{file_id}.png"
        cv2.imwrite(str(out_img_path), resized_img)

        # Camera params：用 sx, sy 分别缩放（而不是单一 scale）
        fx, fy, cx, cy = intrinsic_from_params(cam_info["params"], cam_info["model"], sx, sy, new_w, new_h)

        R = qvec2rotmat(img_info["qvec"])
        t = img_info["tvec"]
        extrinsic = rotmat2extrinsic(R, t)  # world2cam

        out_cam_path = out_dir / "cams" / f"{file_id}.txt"
        write_cam_file(out_cam_path, extrinsic, fx, fy, cx, cy, new_h, new_w)

        out_depth_path = out_dir / "depth" / f"{file_id}.exr"
        out_mask_path = out_dir / "mask" / f"{file_id}.png"

        # 1) LiDAR 渲染优先：w/h 也用固定 new_w/new_h
        if lidar_renderer is not None:
            depth = lidar_renderer.render(fx, fy, cx, cy, extrinsic, new_w, new_h)

            lidar_mask_u8 = depth_to_mask(depth)
            lidar_valid = lidar_mask_u8 > 0

            if fill_lidar_holes_with_mvs and input_depth_path.exists():
                mvs_depth = resize_depth_to_size(input_depth_path, size_wh=target_size_wh)
                if mvs_depth is not None:
                    if input_mask_path.exists():
                        mvs_mask_u8 = resize_mask_to_size(input_mask_path, size_wh=target_size_wh)
                        if mvs_mask_u8 is not None:
                            mvs_valid = mvs_mask_u8 > 0
                        else:
                            mvs_valid = np.isfinite(mvs_depth) & (mvs_depth > 1e-6)
                    else:
                        mvs_valid = np.isfinite(mvs_depth) & (mvs_depth > 1e-6)

                    fill_idx = (~lidar_valid) & (mvs_valid)
                    if np.any(fill_idx):
                        depth[fill_idx] = mvs_depth[fill_idx]

            mask_u8 = depth_to_mask(depth)
            valid = mask_u8 > 0
            depth = depth.astype(np.float32, copy=False)
            depth[~valid] = 0.0

            ok = save_exr(depth, out_depth_path)
            if not ok:
                raise RuntimeError(
                    "Failed to write EXR (OpenCV likely lacks OpenEXR).\n"
                    "Workaround: save depth as .npy, or install OpenCV with OpenEXR enabled."
                )
            cv2.imwrite(str(out_mask_path), mask_u8)
            processed_count += 1
            continue

        # 2) 否则用 COLMAP depth/mask（同样严格到固定尺寸）
        if input_depth_path.exists():
            resized_depth = resize_depth_to_size(input_depth_path, size_wh=target_size_wh)
            resized_mask = None
            if input_mask_path.exists():
                resized_mask = resize_mask_to_size(input_mask_path, size_wh=target_size_wh)

            if resized_depth is not None and resized_mask is not None:
                resized_depth = resized_depth.astype(np.float32, copy=False)
                resized_depth[resized_mask <= 0] = 0.0
                cv2.imwrite(str(out_mask_path), resized_mask)

            if resized_depth is not None:
                save_exr(resized_depth, out_depth_path)
                processed_count += 1

    
    return processed_count


# ---------------------------
# 主流程：COLMAP -> 新格式（可选 LiDAR 渲染 depth）
# ---------------------------

def process_dataset_to_new_format(scene_dir: str,
                                  scene_name: str,
                                  output_dir: str,
                                  camera_types,
                                  max_size=1024,
                                  workers=0,
                                  lidar_ply: str | None = None,
                                  point_size=2.0,
                                  visible=False,
                                  fill_lidar_holes_with_mvs: bool = True,
                                  camera_extract_method: str = "first_char",
                                  camera_separator: str = "_",
                                  camera_position: int = -1):
    print(f"fill_lidar_holes_with_mvs: {fill_lidar_holes_with_mvs}")
    print(f"camera_extract_method: {camera_extract_method}")
    
    input_dir = Path(scene_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"scene_dir not found: {input_dir}")

    input_images_dir = input_dir / "images"
    sparse_dir = input_dir / "sparse" / "0"
    cameras_file = sparse_dir / "cameras.txt"
    images_file = sparse_dir / "images.txt"

    # 你原先的 depth 位置：depth/Chunk 1
    input_depth_dir = input_dir / "depth" / "Chunk 1"

    if not cameras_file.exists() or not images_file.exists():
        raise FileNotFoundError(f"missing cameras/images txt in: {sparse_dir}")

    output_dir = Path(output_dir)
    out_dir = _prepare_output_dirs(output_dir, scene_name)
    
    # 处理相机类型集合
    if camera_types is not None:
        camera_set = set(camera_types)
    else:
        camera_set = None

    cameras = parse_cameras_text(cameras_file)
    images = parse_images_text(images_file)

    target_w, target_h, base_w, base_h = compute_global_target_size(cameras, max_size)
    print(f"[{scene_name}] Global target size: {target_w}x{target_h} (from base {base_w}x{base_h}, max_size={max_size})")

    
    # 按时间戳分组图像
    grouped_images = group_images_by_timestamp(
        images, 
        camera_set,
        camera_extract_method=camera_extract_method,
        camera_separator=camera_separator,
        camera_position=camera_position
    )
    
    # 获取所有相机标识并按字母顺序排序，为每个分配固定编号
    all_camera_ids = set()
    for timestamp, img_list in grouped_images.items():
        for img_info in img_list:
            camera_id = img_info.get("camera_identifier", "")
            if camera_id:
                all_camera_ids.add(camera_id)
    
    # 按字母顺序排序并分配编号（cam001, cam002...）
    sorted_camera_ids = sorted(list(all_camera_ids))
    camera_id_to_number = {}
    for idx, cam_id in enumerate(sorted_camera_ids, 1):
        camera_id_to_number[cam_id] = f"cam{idx:03d}"  # 3位数字编号
    
    print(f"[{scene_name}] cameras={len(cameras)} images={len(images)} groups={len(grouped_images)}")
    print(f"[{scene_name}] use_lidar_render={bool(lidar_ply and Path(lidar_ply).exists())}")
    print(f"[{scene_name}] Camera mappings: {camera_id_to_number}")
    if camera_set is not None:
        print(f"[{scene_name}] camera_types={camera_set}")

    lidar_renderer = None
    if lidar_ply and Path(lidar_ply).exists():
        lidar_renderer = LidarDepthRenderer(lidar_ply, point_size=point_size, visible=visible)

    try:
        # 按时间戳排序（确保帧顺序一致）
        sorted_timestamps = sorted(grouped_images.keys())
        
        if workers and workers > 0 and lidar_renderer is None:
            # 多线程处理（不支持 LiDAR 渲染）
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = []
                for frame_id, timestamp in enumerate(sorted_timestamps):
                    camera_images = grouped_images[timestamp]
                    futures.append(ex.submit(
                        _process_one, frame_id, timestamp, camera_images, cameras,
                        input_images_dir, input_depth_dir, out_dir,
                        max_size, None,  # 多线程路径：不做 open3d 渲染
                        fill_lidar_holes_with_mvs,
                        camera_id_to_number,  # 传递相机编号映射
                        target_size_wh=(target_w, target_h),
                    ))
                for fu in tqdm(as_completed(futures), total=len(futures), desc=f"{scene_name}"):
                    done += fu.result()
            print(f"[{scene_name}] processed={done} images in {len(grouped_images)} frames")
            return

        # 单线程（支持 LiDAR 渲染）
        done = 0
        for frame_id, timestamp in enumerate(tqdm(sorted_timestamps, desc=scene_name)):
            camera_images = grouped_images[timestamp]
            done += _process_one(
                frame_id, timestamp, camera_images, cameras,
                input_images_dir, input_depth_dir, out_dir,
                max_size, lidar_renderer,
                fill_lidar_holes_with_mvs,
                camera_id_to_number,  # 传递相机编号映射
                target_size_wh=(target_w, target_h),
            )
        print(f"[{scene_name}] processed={done} images in {len(grouped_images)} frames")

    finally:
        if lidar_renderer is not None:
            lidar_renderer.close()


####################################################
# 输入格式要求
# scene_dir/
# --| colmap/
# --| --| images/
# --| --| --| {filename}_{idx}.jpg
# --| --| --| ...
# --| --| depth/Chunk 1/
# --| --| --| {filename}.tif
# --| --| --| {filename}_mask.pgm
# --| --| sparse/0/
# --| --| --| cameras.txt
# --| --| --| images.txt
# --| --| --| points3D.txt

## how to use
# python colmap2wai.py --scene_dir scene_dir --scene_name scene_name --output_dir output_dir

## 输出格式
# output_dir/
# --| scene_name/
# --| --| cams / 00000000.txt ...
# --| --| depth / 00000000.txt ...
# --| --| images / 00000000.txt ...
# --| --| mask / 00000000.txt ...


def main():
    parser = argparse.ArgumentParser(description="COLMAP -> ours format (optional LiDAR render depth/mask)")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="COLMAP 场景根目录，需包含 colmap/images/ colmap/sparse/0/(cameras.txt,images.txt)；可选 depth/Chunk 1/")
    parser.add_argument("--scene_name", type=str, required=True, help="输出前缀名")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")

    parser.add_argument("--camera_types", default=None, type=str, help="逗号分隔，如 A,D,W,X,S。指定要处理的相机类型，如果不指定则处理所有")
    parser.add_argument("--max_size", type=int, default=1024, help="最大边长（默认: 1024）")

    # 非 LiDAR 路径可用多线程；LiDAR 路径会自动强制单线程
    parser.add_argument("--workers", type=int, default=8, help="线程数(0=单线程；建议 4~16)")

    # LiDAR 渲染相关
    parser.add_argument("--lidar_ply", type=str, default="",
                        help="若提供且存在：用 LiDAR 点云渲染 depth+mask；否则用 COLMAP depth/Chunk 1")
    parser.add_argument("--point_size", type=float, default=2.0, help="Open3D 点大小")
    parser.add_argument("--visible", action="store_true", help="显示 Open3D 窗口(调试用)")

    # === 新增：控制是否用 MVS depth 填补 LiDAR 空缺 ===
    parser.add_argument(
        "--fill_lidar_holes_with_mvs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="当使用 LiDAR 渲染时，是否用 COLMAP/MVS depth 去填补 LiDAR 空缺区域（默认: False）"
    )
    
    # === 新增：相机标识提取方法 ===
    parser.add_argument("--camera_extract_method", type=str, default="first_char",
                        choices=["first_char", "last_char", "prefix", "suffix", "split", "regex"],
                        help="相机标识提取方法：first_char=第一个字符, last_char=最后一个字符, "
                             "prefix=前缀（直到第一个分隔符）, suffix=后缀（最后一个分隔符之后）, "
                             "split=按分隔符分割, regex=正则表达式（需要配合--camera_regex使用）")
    parser.add_argument("--camera_separator", type=str, default="_",
                        help="分隔符，用于prefix、suffix和split方法（默认: '_'）")
    parser.add_argument("--camera_position", type=int, default=-1,
                        help="位置索引，用于split方法（从0开始，负数表示从后往前，默认: -1表示最后一个）")
    parser.add_argument("--camera_regex", type=str, default=None,
                        help="自定义正则表达式，用于regex方法，必须包含两个分组：第一个是时间戳，第二个是相机标识")
    args = parser.parse_args()

    if args.camera_types is not None:
        camera_types = [c.strip() for c in args.camera_types.split(",") if c.strip()]
        if len(camera_types) == 0:
            camera_types = None
    else:
        camera_types = None
    
    args.scene_dir = os.path.join(args.scene_dir, "colmap")

    process_dataset_to_new_format(
        scene_dir=args.scene_dir,
        scene_name=args.scene_name,
        output_dir=args.output_dir,
        camera_types=camera_types,
        max_size=args.max_size,
        workers=args.workers,
        lidar_ply=args.lidar_ply if args.lidar_ply else None,
        point_size=args.point_size,
        visible=args.visible,
        fill_lidar_holes_with_mvs=bool(args.fill_lidar_holes_with_mvs),
        camera_extract_method=args.camera_extract_method,
        camera_separator=args.camera_separator,
        camera_position=args.camera_position,
    )


if __name__ == "__main__":
    main()
