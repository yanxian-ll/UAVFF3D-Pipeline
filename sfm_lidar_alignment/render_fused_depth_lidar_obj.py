# -*- coding: utf-8 -*-
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # 必须放在 import cv2 之前

import gc
import cv2
import shutil
import argparse
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import open3d as o3d
from tqdm import tqdm


# =========================
# IO
# =========================
def load_point_cloud(ply_path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd is None or pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {ply_path}")
    return pcd


def find_all_plys(ply_root: Path):
    if not ply_root.exists():
        raise FileNotFoundError(f"输入路径不存在: {ply_root}")

    if ply_root.is_file():
        if ply_root.suffix.lower() != ".ply":
            raise ValueError(f"输入文件不是 ply: {ply_root}")
        return [ply_root]

    plys = sorted(ply_root.glob("*.ply"))
    if len(plys) == 0:
        raise FileNotFoundError(f"目录下没有找到 ply 文件: {ply_root}")
    return plys


def resolve_cams_dir(cam_root: Path):
    if cam_root.name == "cams" and cam_root.is_dir():
        return cam_root

    cams_dir = cam_root / "cams"
    if cams_dir.is_dir():
        return cams_dir

    raise FileNotFoundError(f"未找到 cams 目录: {cam_root} 或 {cams_dir}")


def find_cam_txts(cams_dir: Path):
    txts = sorted(cams_dir.glob("*.txt"))
    if len(txts) == 0:
        raise FileNotFoundError(f"cams 下没有找到 txt 文件: {cams_dir}")
    return txts


def read_view_list_txt(view_list_txt: Optional[str], pair_list_txt: Optional[str]):
    names = []

    if view_list_txt is not None:
        path = Path(view_list_txt)
        if not path.exists():
            raise FileNotFoundError(f"视角名单文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    names.append(name)

    if pair_list_txt is not None:
        path = Path(pair_list_txt)
        if not path.exists():
            raise FileNotFoundError(f"配对名单文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                pair = line.strip()
                if not pair:
                    continue
                ss = pair.split()
                if len(ss) >= 2:
                    names.append(ss[0].strip())
                    names.append(ss[1].strip())

    if len(names) == 0:
        return None

    return list(set(names))


def filter_cam_txts_by_view_list(cam_txts, selected_names):
    if selected_names is None:
        return cam_txts

    stem_map = {p.stem: p for p in cam_txts}
    name_map = {p.name: p for p in cam_txts}

    selected = []
    missing = []

    for x in selected_names:
        if x in stem_map:
            selected.append(stem_map[x])
        elif x in name_map:
            selected.append(name_map[x])
        elif x.endswith(".txt") and x[:-4] in stem_map:
            selected.append(stem_map[x[:-4]])
        elif (x + ".txt") in name_map:
            selected.append(name_map[x + ".txt"])
        else:
            missing.append(x)

    if len(missing) > 0:
        print("[WARN] 以下视角在 cams 中未找到：")
        for x in missing:
            print(f"       {x}")

    if len(selected) == 0:
        raise RuntimeError("根据 view list 过滤后，没有任何有效视角。")

    return selected


def safe_remove(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception as e:
        print(f"[WARN] 删除失败: {path} | {e}")


def safe_rmtree(path: Path):
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception as e:
        print(f"[WARN] 删除目录失败: {path} | {e}")


# =========================
# Done Flags / Resume
# =========================
def touch_done_flag(flag_path: Path):
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(flag_path, "w", encoding="utf-8") as f:
        f.write("done\n")


def has_done_flag(done_dir: Path, name: str) -> bool:
    return (done_dir / f"{name}.done").exists()


def sanitize_source_done_flags(cams, out_npy_dir: Path, done_dir: Path):
    """
    如果 done 标记存在，但 npy 丢了，说明状态不一致，移除 done 标记。
    """
    for cam in cams:
        name = cam["name"]
        flag_path = done_dir / f"{name}.done"
        npy_path = out_npy_dir / f"{name}.npy"
        if flag_path.exists() and (not npy_path.exists()):
            safe_remove(flag_path)


def is_fuse_output_complete(out_dirs: dict, name: str):
    required = [
        out_dirs["fused_exr"] / f"{name}.exr",
        out_dirs["fused_vis"] / f"{name}.png",
        out_dirs["depth_lidar_vis"] / f"{name}.png",
        out_dirs["depth_obj_vis"] / f"{name}.png",
        out_dirs["fusion_debug"] / f"{name}.png",
        out_dirs["replace_mask"] / f"{name}.png",
        out_dirs["fuse_done"] / f"{name}.done",
    ]
    return all(p.exists() for p in required)


def sanitize_fuse_done_flags(cams, out_dirs: dict):
    """
    如果 fuse done 存在但最终结果文件不完整，移除 fuse done，允许自动重跑。
    """
    for cam in cams:
        name = cam["name"]
        done_flag = out_dirs["fuse_done"] / f"{name}.done"
        if done_flag.exists() and (not is_fuse_output_complete(out_dirs, name)):
            safe_remove(done_flag)


def get_completed_fuse_names(cams, out_dirs: dict):
    completed = set()
    for cam in cams:
        name = cam["name"]
        if is_fuse_output_complete(out_dirs, name):
            completed.add(name)
    return completed


# =========================
# Image IO
# =========================
def resolve_input_jpg(image_root: Path, name: str) -> Path:
    candidates = [
        image_root / f"{name}.jpg",
        image_root / f"{name}.jpeg",
        image_root / f"{name}.png",
        image_root / f"{name}.JPG",
        image_root / f"{name}.JPEG",
        image_root / f"{name}.PNG",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到原始图像: {name}.jpg/.jpeg/.png in {image_root}")


def read_and_resize_image_to_cam(image_path: Path, out_w: int, out_h: int):
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"无法读取图像: {image_path}")

    in_h, in_w = img_bgr.shape[:2]
    if in_w == out_w and in_h == out_h:
        return img_bgr

    return cv2.resize(img_bgr, (int(out_w), int(out_h)), interpolation=cv2.INTER_AREA)


def save_png_image(out_path: Path, img_bgr):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), img_bgr)
    if not ok:
        raise RuntimeError(f"保存 PNG 失败: {out_path}")


def build_image_index(image_root: Optional[Path]):
    if image_root is None:
        return None

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    index = {}
    for p in image_root.iterdir():
        if not p.is_file():
            continue
        if p.suffix not in exts:
            continue
        if p.stem not in index:
            index[p.stem] = p
    return index


def resolve_input_image_from_index(image_index: Optional[dict], image_root: Path, name: str) -> Path:
    if image_index is not None and name in image_index:
        return image_index[name]
    return resolve_input_jpg(image_root, name)


# =========================
# Camera TXT Parser
# =========================
def parse_cam_txt(cam_txt_path: Path):
    with open(cam_txt_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f.readlines()]

    lines = [x for x in raw_lines if x != ""]

    def find_line_idx(prefix):
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                return i
        return -1

    ext_idx = find_line_idx("extrinsic")
    intr_idx = find_line_idx("intrinsic")
    hwfov_idx = find_line_idx("h w fov")

    if ext_idx < 0 or intr_idx < 0 or hwfov_idx < 0:
        raise ValueError(f"相机文件格式错误: {cam_txt_path}")

    ext_lines = lines[ext_idx + 1: ext_idx + 5]
    intr_lines = lines[intr_idx + 1: intr_idx + 4]
    hwfov_line = lines[hwfov_idx + 1]

    if len(ext_lines) != 4:
        raise ValueError(f"extrinsic 行数不正确: {cam_txt_path}")
    if len(intr_lines) != 3:
        raise ValueError(f"intrinsic 行数不正确: {cam_txt_path}")

    extrinsic = np.array([[float(v) for v in line.split()] for line in ext_lines], dtype=np.float64)
    intrinsic = np.array([[float(v) for v in line.split()] for line in intr_lines], dtype=np.float64)

    parts = hwfov_line.split()
    if len(parts) != 3:
        raise ValueError(f"h w fov 行格式错误: {cam_txt_path}")

    h = int(float(parts[0]))
    w = int(float(parts[1]))
    fov = float(parts[2])

    return {
        "name": cam_txt_path.stem,
        "path": str(cam_txt_path),
        "extrinsic": extrinsic,  # world -> camera
        "intrinsic": intrinsic,
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "h": h,
        "w": w,
        "fov": fov,
    }


def scale_camera(cam: dict, max_image_edge: int):
    if max_image_edge is None or max_image_edge <= 0:
        return cam

    h = int(cam["h"])
    w = int(cam["w"])
    max_edge = max(h, w)
    if max_edge <= max_image_edge:
        return cam

    scale = float(max_image_edge) / float(max_edge)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    new_intrinsic = cam["intrinsic"].copy().astype(np.float64)
    new_intrinsic[0, 0] *= scale
    new_intrinsic[1, 1] *= scale
    new_intrinsic[0, 2] *= scale
    new_intrinsic[1, 2] *= scale

    return {
        "name": cam["name"],
        "path": cam["path"],
        "extrinsic": cam["extrinsic"].copy(),
        "intrinsic": new_intrinsic,
        "fx": float(new_intrinsic[0, 0]),
        "fy": float(new_intrinsic[1, 1]),
        "cx": float(new_intrinsic[0, 2]),
        "cy": float(new_intrinsic[1, 2]),
        "h": new_h,
        "w": new_w,
        "fov": cam["fov"],
    }


def save_cam_txt(cam: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ext = cam["extrinsic"]
    intr = cam["intrinsic"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("extrinsic opencv(x Right, y Down, z Forward) world2camera\n")
        for i in range(4):
            f.write(f"{ext[i,0]:.12f} {ext[i,1]:.12f} {ext[i,2]:.12f} {ext[i,3]:.12f}\n")
        f.write("\n")

        f.write("intrinsic: fx fy cx cy (pixel)\n")
        for i in range(3):
            f.write(f"{intr[i,0]:.12f} {intr[i,1]:.12f} {intr[i,2]:.12f}\n")
        f.write("\n")

        f.write("h w fov\n")
        f.write(f"{cam['h']} {cam['w']} {cam['fov']:.12f}\n")


# =========================
# Open3D Visualizer Cache
# =========================
class VisCache:
    def __init__(self, point_size=2.0, visible=False):
        self.cache = {}
        self.point_size = float(point_size)
        self.visible = bool(visible)

    def get(self, w: int, h: int):
        key = (int(w), int(h))
        if key in self.cache:
            return self.cache[key]["vis"], key

        vis = o3d.visualization.Visualizer()
        ok = vis.create_window(
            window_name=f"o3d_{w}x{h}",
            width=int(w),
            height=int(h),
            visible=self.visible,
        )
        if not ok:
            raise RuntimeError(
                "Open3D Visualizer create_window failed.\n"
                "Linux 无显示环境请尝试 xvfb-run；Windows 请确保有图形界面。"
            )

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        opt.point_size = self.point_size

        self.cache[key] = {"vis": vis, "geom_tag": None}
        return vis, key

    def ensure_geometry(self, key, pcd: o3d.geometry.PointCloud, geom_tag):
        item = self.cache[key]
        if item["geom_tag"] == geom_tag:
            return

        vis = item["vis"]
        vis.clear_geometries()
        vis.add_geometry(pcd, reset_bounding_box=True)
        item["geom_tag"] = geom_tag

    def close_all(self):
        for v in self.cache.values():
            try:
                v["vis"].destroy_window()
            except Exception:
                pass
        self.cache.clear()


# =========================
# Camera / Prepare
# =========================
def _prepare_one_camera_worker(
    idx,
    cam_txt,
    max_image_edge,
    cam_out_dir,
    image_root,
    images_dir,
    image_index,
):
    cam = parse_cam_txt(cam_txt)
    cam = scale_camera(cam, max_image_edge)

    out_cam_path = cam_out_dir / f"{cam['name']}.txt"
    if not out_cam_path.exists():
        save_cam_txt(cam, out_cam_path)

    if image_root is not None and images_dir is not None:
        out_img_path = images_dir / f"{cam['name']}.png"

        if not out_img_path.exists():
            src_img_path = resolve_input_image_from_index(image_index, image_root, cam["name"])

            direct_copy = False
            if src_img_path.suffix.lower() == ".png":
                img = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"无法读取图像: {src_img_path}")
                in_h, in_w = img.shape[:2]
                if in_w == cam["w"] and in_h == cam["h"]:
                    shutil.copy2(str(src_img_path), str(out_img_path))
                    direct_copy = True
                del img

            if not direct_copy:
                img_bgr = read_and_resize_image_to_cam(src_img_path, cam["w"], cam["h"])
                save_png_image(out_img_path, img_bgr)
                del img_bgr

    return idx, cam


def prepare_cameras(
    cam_txts,
    max_image_edge: int,
    cam_out_dir: Path,
    image_root: Optional[Path] = None,
    images_dir: Optional[Path] = None,
    prepare_workers: int = 0,
):
    if len(cam_txts) == 0:
        return []

    cam_out_dir.mkdir(parents=True, exist_ok=True)
    if images_dir is not None:
        images_dir.mkdir(parents=True, exist_ok=True)

    image_index = build_image_index(image_root) if image_root is not None else None

    if prepare_workers is None or prepare_workers <= 0:
        prepare_workers = min(16, max(1, (os.cpu_count() or 8)))

    if len(cam_txts) <= 4 or prepare_workers == 1:
        cams = []
        for idx, cam_txt in enumerate(tqdm(cam_txts, desc="Prepare Cameras")):
            _, cam = _prepare_one_camera_worker(
                idx, cam_txt, max_image_edge, cam_out_dir, image_root, images_dir, image_index
            )
            cams.append(cam)
        return cams

    results = [None] * len(cam_txts)
    with ThreadPoolExecutor(max_workers=prepare_workers) as ex:
        futures = []
        for idx, cam_txt in enumerate(cam_txts):
            futures.append(
                ex.submit(
                    _prepare_one_camera_worker,
                    idx,
                    cam_txt,
                    max_image_edge,
                    cam_out_dir,
                    image_root,
                    images_dir,
                    image_index,
                )
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Prepare Cameras"):
            idx, cam = fut.result()
            results[idx] = cam

    return results


def set_camera_from_world2cam(
    vis: o3d.visualization.Visualizer,
    K: np.ndarray,
    world2cam: np.ndarray,
    w: int,
    h: int,
):
    params = o3d.camera.PinholeCameraParameters()

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=int(w),
        height=int(h),
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
    )
    params.intrinsic = intrinsic
    params.extrinsic = world2cam.astype(np.float64)

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)


def render_depth_only(vis: o3d.visualization.Visualizer):
    vis.poll_events()
    vis.update_renderer()
    depth_f = np.asarray(vis.capture_depth_float_buffer(do_render=True), dtype=np.float32)
    return depth_f.astype(np.float32)


# =========================
# Depth / Mask Utils
# =========================
def is_valid_depth(depth):
    return np.isfinite(depth) & (depth > 1e-8)


def fuse_depth_keep_nearest(prev_depth: np.ndarray, curr_depth: np.ndarray):
    if prev_depth is None:
        return curr_depth.astype(np.float32)

    if prev_depth.shape != curr_depth.shape:
        raise ValueError(f"深度图尺寸不一致: prev={prev_depth.shape}, curr={curr_depth.shape}")

    prev_valid = is_valid_depth(prev_depth)
    curr_valid = is_valid_depth(curr_depth)

    fused = np.zeros_like(curr_depth, dtype=np.float32)
    only_prev = prev_valid & (~curr_valid)
    only_curr = (~prev_valid) & curr_valid
    both_valid = prev_valid & curr_valid

    fused[only_prev] = prev_depth[only_prev]
    fused[only_curr] = curr_depth[only_curr]
    fused[both_valid] = np.minimum(prev_depth[both_valid], curr_depth[both_valid])

    del prev_valid, curr_valid, only_prev, only_curr, both_valid
    return fused.astype(np.float32)


def save_depth_npy(out_path: Path, depth):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), depth.astype(np.float32), allow_pickle=False)


def load_depth_npy_memmap(in_path: Path):
    if not in_path.exists():
        return None
    return np.load(str(in_path), mmap_mode="r", allow_pickle=False)


def save_depth_exr(out_path: Path, depth):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), depth.astype(np.float32))
    if not ok:
        raise RuntimeError(f"保存 EXR 失败: {out_path}")


def resize_vis_if_needed(img, vis_max_edge: Optional[int], is_mask: bool = False):
    if vis_max_edge is None or vis_max_edge <= 0:
        return img

    h, w = img.shape[:2]
    max_edge = max(h, w)
    if max_edge <= vis_max_edge:
        return img

    scale = float(vis_max_edge) / float(max_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def make_depth_vis_bgr(depth, invalid_color=(0, 0, 0), percentile_min=1.0, percentile_max=99.0):
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 1e-8)

    vis_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    if not np.any(valid):
        return vis_bgr

    d = np.asarray(depth[valid], dtype=np.float32)
    dmin = np.percentile(d, percentile_min)
    dmax = np.percentile(d, percentile_max)
    if dmax <= dmin:
        dmax = dmin + 1e-6

    depth_norm = np.zeros((h, w), dtype=np.float32)
    depth_norm[valid] = np.clip((depth[valid] - dmin) / (dmax - dmin), 0.0, 1.0)

    depth_inv = np.zeros((h, w), dtype=np.float32)
    depth_inv[valid] = 1.0 - depth_norm[valid]

    depth_u8 = np.zeros((h, w), dtype=np.uint8)
    depth_u8[valid] = np.clip(depth_inv[valid] * 255.0, 0, 255).astype(np.uint8)

    vis_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    vis_bgr[~valid] = np.array(invalid_color, dtype=np.uint8)

    del valid, d, depth_norm, depth_inv, depth_u8
    return vis_bgr


def save_vis_png(out_path: Path, vis_bgr, vis_max_edge=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis_bgr = resize_vis_if_needed(vis_bgr, vis_max_edge, is_mask=False)
    ok = cv2.imwrite(str(out_path), vis_bgr)
    if not ok:
        raise RuntimeError(f"保存 PNG 失败: {out_path}")


def save_mask_png(out_path: Path, mask_bool, vis_max_edge=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    mask_u8 = resize_vis_if_needed(mask_u8, vis_max_edge, is_mask=True)
    ok = cv2.imwrite(str(out_path), mask_u8)
    if not ok:
        raise RuntimeError(f"保存 mask 失败: {out_path}")


def resolve_mask_file(mask_root: Path, name: str):
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".npy", ".PNG", ".JPG", ".JPEG"]
    for ext in exts:
        p = mask_root / f"{name}{ext}"
        if p.exists():
            return p
    return None


def load_optional_binary_mask(mask_root: Optional[Path], name: str, h: int, w: int, dilate_px: int = 0):
    if mask_root is None:
        return None

    mask_path = resolve_mask_file(mask_root, name)
    if mask_path is None:
        return None

    if mask_path.suffix.lower() == ".npy":
        mask = np.load(str(mask_path), allow_pickle=False)
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"无法读取掩码: {mask_path}")

    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2)
    else:
        mask = mask > 0

    mask = mask.astype(np.uint8)

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    if dilate_px > 0:
        k = int(dilate_px) * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask.astype(bool)


def remove_small_connected_components(mask: Optional[np.ndarray], min_area: int, connectivity: int = 8):
    if mask is None:
        return None

    mask_u8 = mask.astype(np.uint8)
    if min_area is None or min_area <= 1:
        return mask_u8.astype(bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=connectivity)
    out = np.zeros_like(mask_u8, dtype=np.uint8)

    for label_id in range(1, num_labels):  # 0 是背景
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            out[labels == label_id] = 1

    return out.astype(bool)


def dilate_binary_mask(mask: Optional[np.ndarray], dilate_px: int):
    if mask is None:
        return None

    if dilate_px is None or dilate_px <= 0:
        return mask.astype(bool)

    mask_u8 = mask.astype(np.uint8)
    k = int(dilate_px) * 2 + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    return mask_u8.astype(bool)


def fuse_lidar_obj_depth(
    lidar_depth: np.ndarray,
    obj_depth: np.ndarray,
    veg_mask: np.ndarray = None,
    replace_abs_thr: float = 1.5,
    replace_rel_thr: float = 0.15,
    min_replace_area: int = 64,
    replace_dilate_px: int = 2,
):
    """
    返回:
        fused_depth
        debug_mask      = fill_from_obj | replace_mask
        replace_mask    = 仅“两者都有值但需要替换”的区域，并已去掉小连通域
        fill_from_obj   = 激光无值、OBJ 直接覆盖的区域（不做小区域过滤）
    """
    if lidar_depth is None and obj_depth is None:
        return None, None, None, None

    if lidar_depth is None:
        out = np.asarray(obj_depth, dtype=np.float32).copy()
        fill_from_obj = is_valid_depth(obj_depth)
        replace_mask = np.zeros_like(fill_from_obj, dtype=bool)
        debug_mask = fill_from_obj | replace_mask
        return out, debug_mask, replace_mask, fill_from_obj

    if obj_depth is None:
        out = np.asarray(lidar_depth, dtype=np.float32).copy()
        fill_from_obj = np.zeros_like(out, dtype=bool)
        replace_mask = np.zeros_like(out, dtype=bool)
        debug_mask = fill_from_obj | replace_mask
        return out, debug_mask, replace_mask, fill_from_obj

    if lidar_depth.shape != obj_depth.shape:
        raise ValueError(f"lidar_depth.shape != obj_depth.shape: {lidar_depth.shape} vs {obj_depth.shape}")

    lidar_valid = is_valid_depth(lidar_depth)
    obj_valid = is_valid_depth(obj_depth)
    both_valid = lidar_valid & obj_valid

    out = np.zeros_like(np.asarray(lidar_depth), dtype=np.float32)
    out[lidar_valid] = lidar_depth[lidar_valid]

    # 1) 直接覆盖区域：激光无值而 OBJ 有值
    fill_from_obj = (~lidar_valid) & obj_valid
    out[fill_from_obj] = obj_depth[fill_from_obj]

    # 2) 需要替换区域：两者都有值，OBJ 更近且差异足够大
    diff = np.abs(lidar_depth - obj_depth)
    rel = diff / np.maximum(lidar_depth, 1e-6)
    big_diff = (diff > float(replace_abs_thr)) & (rel > float(replace_rel_thr))
    obj_is_closer = obj_depth < lidar_depth

    replace_mask_raw = both_valid & big_diff & obj_is_closer

    if veg_mask is not None:
        replace_mask_raw = replace_mask_raw & (~veg_mask)


    # replace_mask = remove_small_connected_components(
    #     replace_mask_raw,
    #     min_area=min_replace_area,
    #     connectivity=8,
    # )

    # replace_mask = dilate_binary_mask(
    #     replace_mask,
    #     dilate_px=replace_dilate_px,
    # )
    
    replace_mask = replace_mask_raw

    if veg_mask is not None:
        replace_mask = replace_mask & (~veg_mask)

    out[replace_mask] = obj_depth[replace_mask]
    debug_mask = fill_from_obj | replace_mask

    del lidar_valid, obj_valid, both_valid
    del diff, rel, big_diff, obj_is_closer, replace_mask_raw
    return out.astype(np.float32), debug_mask, replace_mask, fill_from_obj


# =========================
# Output Dirs
# =========================
def ensure_dirs(output_root: Path, save_images: bool):
    output_root.mkdir(parents=True, exist_ok=True)

    dirs = {
        "fused_exr": output_root / "fused_exr",
        "fused_vis": output_root / "fused_vis",
        "depth_lidar_vis": output_root / "depth_lidar_vis",
        "depth_obj_vis": output_root / "depth_obj_vis",
        "fusion_debug": output_root / "fusion_debug",
        "replace_mask": output_root / "replace_mask",
        "cams": output_root / "cams",
        "images": (output_root / "images") if save_images else None,

        "temp_lidar_npy": output_root / "_temp_lidar_npy",
        "temp_obj_npy": output_root / "_temp_obj_npy",

        "temp_lidar_done": output_root / "_temp_lidar_done",
        "temp_obj_done": output_root / "_temp_obj_done",
        "fuse_done": output_root / "_fuse_done",
    }

    for key, path in dirs.items():
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)

    return dirs


# =========================
# Render Source Depth (Temp NPY)
# =========================
def render_source_depths_to_npy(
    source_name: str,
    ply_paths,
    cams,
    vis_cache: VisCache,
    out_npy_dir: Path,
    done_dir: Path,
    skip_names: set,
):
    print(f"\n[INFO] 开始渲染 {source_name} 点云，共 {len(ply_paths)} 个 ply")
    out_npy_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    pending_cams = []
    for cam in cams:
        name = cam["name"]
        npy_path = out_npy_dir / f"{name}.npy"

        # 最终结果已完整，整个 source render 跳过
        if name in skip_names:
            continue

        # source npy 已经完整存在，跳过
        if npy_path.exists() and has_done_flag(done_dir, name):
            continue

        # 半成品，删掉重做
        if npy_path.exists() and (not has_done_flag(done_dir, name)):
            safe_remove(npy_path)

        pending_cams.append(cam)

    if len(pending_cams) == 0:
        print(f"[INFO] {source_name}: 全部视角已有结果，跳过")
        return

    for ply_idx, ply_path in enumerate(ply_paths):
        print(f"[INFO] {source_name} ply ({ply_idx + 1}/{len(ply_paths)}): {ply_path}")
        pcd = load_point_cloud(str(ply_path))
        geom_tag = str(ply_path.resolve())

        for cam in tqdm(pending_cams, desc=f"Render {source_name}:{ply_path.name}"):
            name = cam["name"]
            h, w = cam["h"], cam["w"]
            npy_path = out_npy_dir / f"{name}.npy"

            vis, key = vis_cache.get(w, h)
            vis_cache.ensure_geometry(key, pcd, geom_tag)
            set_camera_from_world2cam(vis, cam["intrinsic"], cam["extrinsic"], w, h)

            curr_depth = render_depth_only(vis)

            if ply_idx == 0:
                fused_depth = curr_depth
            else:
                prev_depth = load_depth_npy_memmap(npy_path)
                fused_depth = fuse_depth_keep_nearest(prev_depth, curr_depth)
                del prev_depth

            save_depth_npy(npy_path, fused_depth)
            del curr_depth, fused_depth

        del pcd
        gc.collect()

    for cam in pending_cams:
        touch_done_flag(done_dir / f"{cam['name']}.done")


# =========================
# Fuse / Export
# =========================
def _export_one_view_worker(
    cam: dict,
    temp_lidar_dir: Path,
    temp_obj_dir: Path,
    out_dirs: dict,
    veg_mask_root: Optional[Path],
    replace_abs_thr: float,
    replace_rel_thr: float,
    veg_dilate_px: int,
    vis_max_edge: Optional[int],
    min_replace_area: int,
):
    name = cam["name"]
    h, w = cam["h"], cam["w"]

    if is_fuse_output_complete(out_dirs, name):
        return name, "skip"

    lidar_npy_path = temp_lidar_dir / f"{name}.npy"
    obj_npy_path = temp_obj_dir / f"{name}.npy"

    if not lidar_npy_path.exists():
        raise FileNotFoundError(f"缺少 lidar 临时深度: {lidar_npy_path}")
    if not obj_npy_path.exists():
        raise FileNotFoundError(f"缺少 obj 临时深度: {obj_npy_path}")

    lidar_depth = load_depth_npy_memmap(lidar_npy_path)
    obj_depth = load_depth_npy_memmap(obj_npy_path)

    veg_mask = load_optional_binary_mask(
        veg_mask_root,
        name,
        h=h,
        w=w,
        dilate_px=veg_dilate_px,
    )

    fused_depth, debug_mask, replace_mask, fill_from_obj = fuse_lidar_obj_depth(
        lidar_depth=lidar_depth,
        obj_depth=obj_depth,
        veg_mask=veg_mask,
        replace_abs_thr=replace_abs_thr,
        replace_rel_thr=replace_rel_thr,
        min_replace_area=min_replace_area,
    )

    fused_exr_path = out_dirs["fused_exr"] / f"{name}.exr"
    lidar_vis_path = out_dirs["depth_lidar_vis"] / f"{name}.png"
    obj_vis_path = out_dirs["depth_obj_vis"] / f"{name}.png"
    fused_vis_path = out_dirs["fused_vis"] / f"{name}.png"
    debug_mask_path = out_dirs["fusion_debug"] / f"{name}.png"
    replace_mask_path = out_dirs["replace_mask"] / f"{name}.png"

    if fused_depth is not None and (not fused_exr_path.exists()):
        save_depth_exr(fused_exr_path, fused_depth)

    if lidar_depth is not None and (not lidar_vis_path.exists()):
        lidar_vis = make_depth_vis_bgr(lidar_depth)
        save_vis_png(lidar_vis_path, lidar_vis, vis_max_edge)
        del lidar_vis

    if obj_depth is not None and (not obj_vis_path.exists()):
        obj_vis = make_depth_vis_bgr(obj_depth)
        save_vis_png(obj_vis_path, obj_vis, vis_max_edge)
        del obj_vis

    if fused_depth is not None and (not fused_vis_path.exists()):
        fused_vis = make_depth_vis_bgr(fused_depth)
        save_vis_png(fused_vis_path, fused_vis, vis_max_edge)
        del fused_vis

    if debug_mask is not None and (not debug_mask_path.exists()):
        save_mask_png(debug_mask_path, debug_mask, vis_max_edge)

    if replace_mask is not None and (not replace_mask_path.exists()):
        save_mask_png(replace_mask_path, replace_mask, vis_max_edge)

    # 只在最终文件完整时打 done 标记
    required_now = [
        fused_exr_path,
        lidar_vis_path,
        obj_vis_path,
        fused_vis_path,
        debug_mask_path,
        replace_mask_path,
    ]
    if all(p.exists() for p in required_now):
        touch_done_flag(out_dirs["fuse_done"] / f"{name}.done")

    del fused_depth, debug_mask, replace_mask, fill_from_obj, veg_mask
    del lidar_depth, obj_depth
    gc.collect()

    # 该视角最终结果完整后，删除 temp
    if is_fuse_output_complete(out_dirs, name):
        safe_remove(lidar_npy_path)
        safe_remove(obj_npy_path)

    return name, "done"


def fuse_and_export_per_view(
    cams,
    temp_lidar_dir: Path,
    temp_obj_dir: Path,
    out_dirs: dict,
    veg_mask_root: Optional[Path],
    replace_abs_thr: float,
    replace_rel_thr: float,
    veg_dilate_px: int,
    vis_max_edge: Optional[int],
    min_replace_area: int,
    export_workers: int,
):
    print("\n[INFO] 开始逐视角融合并导出最终结果")

    pending_cams = [cam for cam in cams if not is_fuse_output_complete(out_dirs, cam["name"])]
    if len(pending_cams) == 0:
        print("[INFO] Fuse + Export: 全部视角已有结果，跳过")
        return

    if export_workers is None or export_workers <= 0:
        export_workers = min(8, max(1, (os.cpu_count() or 8)))

    if len(pending_cams) <= 2 or export_workers == 1:
        for cam in tqdm(pending_cams, desc="Fuse + Export"):
            _export_one_view_worker(
                cam=cam,
                temp_lidar_dir=temp_lidar_dir,
                temp_obj_dir=temp_obj_dir,
                out_dirs=out_dirs,
                veg_mask_root=veg_mask_root,
                replace_abs_thr=replace_abs_thr,
                replace_rel_thr=replace_rel_thr,
                veg_dilate_px=veg_dilate_px,
                vis_max_edge=vis_max_edge,
                min_replace_area=min_replace_area,
            )
        return

    futures = []
    with ThreadPoolExecutor(max_workers=export_workers) as ex:
        for cam in pending_cams:
            futures.append(
                ex.submit(
                    _export_one_view_worker,
                    cam,
                    temp_lidar_dir,
                    temp_obj_dir,
                    out_dirs,
                    veg_mask_root,
                    replace_abs_thr,
                    replace_rel_thr,
                    veg_dilate_px,
                    vis_max_edge,
                    min_replace_area,
                )
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fuse + Export"):
            fut.result()


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "先分别渲染激光点云和 OBJ 点云到临时 NPY，再逐视角读取并融合。"
            "支持按视角断点续跑；已有结果自动跳过；replace 区域可去掉小连通域。"
        )
    )
    parser.add_argument("lidar_ply_root", type=str, help="激光点云 ply 文件或目录")
    parser.add_argument("obj_ply_root", type=str, help="OBJ 采样点云 ply 文件或目录")
    parser.add_argument("cam_root", type=str, help="相机目录，或其上一级目录（内部应包含 cams/）")
    parser.add_argument("output_root", type=str, help="输出目录")

    parser.add_argument("--image-root", type=str, default=None, help="原始图像目录；提供后会输出 images/")
    parser.add_argument("--veg-mask-root", type=str, default=None, help="植被掩码目录，非零视为植被")
    parser.add_argument("--view-list-txt", type=str, default=None, help="一行一个视角名")
    parser.add_argument("--pair-list-txt", type=str, default=None, help="一行两个视角名")

    parser.add_argument("--max-image-edge", type=int, default=1600, help="渲染与输出 cams/images 使用的最大图像边")
    parser.add_argument("--vis-max-edge", type=int, default=400, help="可视化 PNG 的最大边，<=0 表示不缩小")

    parser.add_argument("--prepare-workers", type=int, default=8, help="Prepare Cameras 线程数，<=0 表示自动")
    parser.add_argument("--export-workers", type=int, default=8, help="Fuse + Export 线程数，<=0 表示自动")

    parser.add_argument("--lidar-point-size", type=float, default=3.0, help="激光点大小")
    parser.add_argument("--obj-point-size", type=float, default=3.0, help="OBJ 点大小")
    parser.add_argument("--visible", action="store_true", help="显示 Open3D 窗口")

    parser.add_argument("--replace-abs-thr", type=float, default=1.5, help="差异绝对阈值，单位米")
    parser.add_argument("--replace-rel-thr", type=float, default=0.0, help="差异相对阈值，相对激光深度")
    parser.add_argument("--veg-dilate-px", type=int, default=2, help="植被掩码膨胀像素，保护边界")
    parser.add_argument(
        "--min-replace-area",
        type=int,
        default=64,
        help="仅对 replace 区域做连通域去小块，单位像素；direct fill 不处理",
    )

    parser.add_argument("--keep-temp", action="store_true", help="保留 _temp_* 临时结果，默认最终会删除")
    args = parser.parse_args()

    lidar_ply_root = Path(args.lidar_ply_root)
    obj_ply_root = Path(args.obj_ply_root)
    cam_root = Path(args.cam_root)
    output_root = Path(args.output_root)

    image_root = None
    if args.image_root is not None:
        image_root = Path(args.image_root)
        if not image_root.exists() or not image_root.is_dir():
            raise FileNotFoundError(f"原始图像目录不存在: {image_root}")

    veg_mask_root = None
    if args.veg_mask_root is not None:
        veg_mask_root = Path(args.veg_mask_root)
        if not veg_mask_root.exists() or not veg_mask_root.is_dir():
            raise FileNotFoundError(f"植被掩码目录不存在: {veg_mask_root}")

    dirs = ensure_dirs(
        output_root=output_root,
        save_images=(image_root is not None),
    )

    lidar_plys = find_all_plys(lidar_ply_root)
    obj_plys = find_all_plys(obj_ply_root)

    print(f"[INFO] 激光 ply 数量: {len(lidar_plys)}")
    print(f"[INFO] OBJ ply 数量: {len(obj_plys)}")

    cams_dir = resolve_cams_dir(cam_root)
    cam_txts = find_cam_txts(cams_dir)

    selected_names = read_view_list_txt(args.view_list_txt, args.pair_list_txt)
    cam_txts = filter_cam_txts_by_view_list(cam_txts, selected_names)
    print(f"[INFO] 实际处理视角数: {len(cam_txts)} | {cams_dir}")

    cams = prepare_cameras(
        cam_txts=cam_txts,
        max_image_edge=args.max_image_edge,
        cam_out_dir=dirs["cams"],
        image_root=image_root,
        images_dir=dirs["images"],
        prepare_workers=args.prepare_workers,
    )

    # 修正可能的脏状态
    sanitize_source_done_flags(cams, dirs["temp_lidar_npy"], dirs["temp_lidar_done"])
    sanitize_source_done_flags(cams, dirs["temp_obj_npy"], dirs["temp_obj_done"])
    sanitize_fuse_done_flags(cams, out_dirs=dirs)

    completed_names = get_completed_fuse_names(cams, out_dirs=dirs)
    if len(completed_names) > 0:
        print(f"[INFO] 已完成视角数（最终结果完整）: {len(completed_names)}")

    lidar_cache = VisCache(point_size=args.lidar_point_size, visible=args.visible)
    obj_cache = VisCache(point_size=args.obj_point_size, visible=args.visible)

    try:
        render_source_depths_to_npy(
            source_name="lidar",
            ply_paths=lidar_plys,
            cams=cams,
            vis_cache=lidar_cache,
            out_npy_dir=dirs["temp_lidar_npy"],
            done_dir=dirs["temp_lidar_done"],
            skip_names=completed_names,
        )

        render_source_depths_to_npy(
            source_name="obj",
            ply_paths=obj_plys,
            cams=cams,
            vis_cache=obj_cache,
            out_npy_dir=dirs["temp_obj_npy"],
            done_dir=dirs["temp_obj_done"],
            skip_names=completed_names,
        )

    finally:
        lidar_cache.close_all()
        obj_cache.close_all()

    fuse_and_export_per_view(
        cams=cams,
        temp_lidar_dir=dirs["temp_lidar_npy"],
        temp_obj_dir=dirs["temp_obj_npy"],
        out_dirs=dirs,
        veg_mask_root=veg_mask_root,
        replace_abs_thr=args.replace_abs_thr,
        replace_rel_thr=args.replace_rel_thr,
        veg_dilate_px=args.veg_dilate_px,
        vis_max_edge=args.vis_max_edge,
        min_replace_area=args.min_replace_area,
        export_workers=args.export_workers,
    )

    if not args.keep_temp:
        safe_rmtree(dirs["temp_lidar_npy"])
        safe_rmtree(dirs["temp_obj_npy"])

    print(f"\n[INFO] 完成，输出目录: {output_root}")
    print("[INFO] 最终保留：")
    print(f"       {dirs['fused_exr']}")
    print(f"       {dirs['fused_vis']}")
    print(f"       {dirs['depth_lidar_vis']}")
    print(f"       {dirs['depth_obj_vis']}")
    print(f"       {dirs['fusion_debug']}")
    print(f"       {dirs['replace_mask']}")
    print(f"       {dirs['cams']}")
    print(f"       {dirs['fuse_done']}")
    if dirs["images"] is not None:
        print(f"       {dirs['images']}")
    if args.keep_temp:
        print(f"       {dirs['temp_lidar_npy']}")
        print(f"       {dirs['temp_obj_npy']}")


if __name__ == "__main__":
    main()



"""
python render_fused_depth_lidar_obj.py \
    ./lidar/nanfang/transform/lidar_finall.ply \
    ./recon/nanfang/models/pc/0 \
    ./recon/nanfang/cams \
    ./scene_fused_render/nanfang \
    --image-root ./undistort/nanfang/ImageCorrection/undistort

python render_fused_depth_lidar_obj.py \
    ./lidar/yanghaitang/transform/lidar_finall.ply \
    ./recon/yanghaitang/models/pc/0 \
    ./recon/yanghaitang/cams \
    ./scene_fused_render/yanghaitang \
    --image-root ./undistort/yanghaitang/ImageCorrection/undistort

python render_fused_depth_lidar_obj.py \
    ./lidar/xiaoxiang/transform/lidar_finall.ply \
    ./recon/xiaoxiang/models/pc/0 \
    ./recon/xiaoxiang/cams \
    ./scene_fused_render/xiaoxiang \
    --image-root ./undistort/xiaoxiang/ImageCorrection/undistort


"""