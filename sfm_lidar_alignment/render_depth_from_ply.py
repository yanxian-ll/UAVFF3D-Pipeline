import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # 必须放在 import cv2 之前

import cv2
import argparse
from pathlib import Path

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


def read_view_list_txt(view_list_txt: str, pair_list_txt: str):
    
    names = []

    if view_list_txt is not None:
        path = Path(view_list_txt)
        if not path.exists():
            raise FileNotFoundError(f"视角名单文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                names.append(name)
    
    if pair_list_txt is not None:
        path = Path(pair_list_txt)
        if not path.exists():
            raise FileNotFoundError(f"视角名单文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                pair = line.strip()
                if not pair:
                    continue
                names.append(pair.split(" ")[0].strip())
                names.append(pair.split(" ")[1].strip())

    if len(names) == 0:
        return

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
        print("[WARN] 以下视角在 cams 目录中未找到：")
        for x in missing:
            print(f"       {x}")

    if len(selected) == 0:
        raise RuntimeError("根据 view list 过滤后，没有任何有效视角。")

    return selected


def resolve_input_jpg(image_root: Path, name: str) -> Path:
    candidates = [
        image_root / f"{name}.jpg",
        image_root / f"{name}.jpeg",
        image_root / f"{name}.JPG",
        image_root / f"{name}.JPEG",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到原始图像: {name}.jpg / .jpeg in {image_root}")


def read_and_resize_image_to_cam(image_path: Path, out_w: int, out_h: int):
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"无法读取图像: {image_path}")

    in_h, in_w = img_bgr.shape[:2]
    if in_w == out_w and in_h == out_h:
        return img_bgr

    resized = cv2.resize(img_bgr, (int(out_w), int(out_h)), interpolation=cv2.INTER_AREA)
    return resized


def save_png_image(out_path: Path, img_bgr):
    ok = cv2.imwrite(str(out_path), img_bgr)
    if not ok:
        raise RuntimeError(f"保存 PNG 失败: {out_path}")


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

    extrinsic = np.array(
        [[float(v) for v in line.split()] for line in ext_lines],
        dtype=np.float64,
    )
    intrinsic = np.array(
        [[float(v) for v in line.split()] for line in intr_lines],
        dtype=np.float64,
    )

    parts = hwfov_line.split()
    if len(parts) != 3:
        raise ValueError(f"h w fov 行格式错误: {cam_txt_path}")

    h = int(float(parts[0]))
    w = int(float(parts[1]))
    fov = float(parts[2])

    return {
        "name": cam_txt_path.stem,
        "path": str(cam_txt_path),
        "extrinsic": extrinsic,   # world -> camera
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

    cam_scaled = {
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
    return cam_scaled


def save_cam_txt(cam: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ext = cam["extrinsic"]
    intr = cam["intrinsic"]
    h = cam["h"]
    w = cam["w"]
    fov = cam["fov"]

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
        f.write(f"{h} {w} {fov:.12f}\n")


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

        self.cache[key] = {
            "vis": vis,
            "geom_tag": None,   # 当前窗口里装的是哪个 ply
        }
        return vis, key

    def ensure_geometry(self, key, pcd: o3d.geometry.PointCloud, geom_tag):
        """
        只有当当前窗口中的几何不是这个 ply 时，才 clear/add 一次。
        """
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


def prepare_cameras(
    cam_txts,
    max_image_edge: int,
    cam_out_dir: Path,
    image_root: Path = None,
    images_dir: Path = None,
):
    cams = []

    for cam_txt in tqdm(cam_txts, desc="Prepare Cameras"):
        cam = parse_cam_txt(cam_txt)
        cam = scale_camera(cam, max_image_edge)

        # 相机文件只保存一次
        out_cam_path = cam_out_dir / f"{cam['name']}.txt"
        if not out_cam_path.exists():
            save_cam_txt(cam, out_cam_path)

        # 原图只保存一次
        if image_root is not None and images_dir is not None:
            out_img_path = images_dir / f"{cam['name']}.png"
            if not out_img_path.exists():
                src_img_path = resolve_input_jpg(image_root, cam["name"])
                img_bgr = read_and_resize_image_to_cam(
                    src_img_path, cam["w"], cam["h"]
                )
                save_png_image(out_img_path, img_bgr)

        cams.append(cam)

    return cams

# =========================
# Camera Set / Render
# =========================
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
    depth_f = np.asarray(
        vis.capture_depth_float_buffer(do_render=True),
        dtype=np.float32,
    )
    return depth_f.astype(np.float32)


# =========================
# Save / Load Depth
# =========================
def ensure_dirs(output_root: Path, save_npy: bool, save_exr: bool, save_vis: bool, save_images: bool):
    cam_out_dir = output_root / "cams"
    cam_out_dir.mkdir(parents=True, exist_ok=True)

    image_dir = None
    if save_images:
        image_dir = output_root / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

    npy_dir = None
    exr_dir = None
    vis_dir = None

    if save_npy:
        npy_dir = output_root / "depth_npy"
        npy_dir.mkdir(parents=True, exist_ok=True)

    if save_exr:
        exr_dir = output_root / "depth_exr"
        exr_dir.mkdir(parents=True, exist_ok=True)

    if save_vis:
        vis_dir = output_root / "depth_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

    return image_dir, cam_out_dir, npy_dir, exr_dir, vis_dir


def save_depth_npy(out_path: Path, depth):
    np.save(str(out_path), depth.astype(np.float32))


def save_depth_exr(out_path: Path, depth):
    ok = cv2.imwrite(str(out_path), depth.astype(np.float32))
    if not ok:
        raise RuntimeError(f"保存 EXR 失败: {out_path}")


def load_depth_npy(in_path: Path):
    if not in_path.exists():
        return None
    return np.load(str(in_path)).astype(np.float32)


def load_depth_exr(in_path: Path):
    if not in_path.exists():
        return None
    depth = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"读取 EXR 失败: {in_path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32)


def is_valid_depth(depth):
    return np.isfinite(depth) & (depth > 1e-8)


def fuse_depth_keep_nearest(prev_depth: np.ndarray, curr_depth: np.ndarray):
    """
    融合策略：
    - 两者都无效 -> 无效
    - 只有一个有效 -> 取有效那个
    - 两者都有效 -> 取较小深度（更近）
    """
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

    return fused.astype(np.float32)


def make_depth_vis(depth, invalid_color=(0, 0, 0), percentile_min=1.0, percentile_max=99.0):
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 1e-8)

    vis = np.zeros((h, w, 3), dtype=np.uint8)
    if not np.any(valid):
        return vis

    d = depth[valid]
    dmin = np.percentile(d, percentile_min)
    dmax = np.percentile(d, percentile_max)

    if dmax <= dmin:
        dmax = dmin + 1e-6

    depth_norm = np.zeros_like(depth, dtype=np.float32)
    depth_norm[valid] = np.clip((depth[valid] - dmin) / (dmax - dmin), 0.0, 1.0)

    depth_inv = np.zeros_like(depth_norm, dtype=np.float32)
    depth_inv[valid] = 1.0 - depth_norm[valid]

    depth_u8 = np.zeros((h, w), dtype=np.uint8)
    depth_u8[valid] = np.clip(depth_inv[valid] * 255.0, 0, 255).astype(np.uint8)

    vis_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    vis[~valid] = np.array(invalid_color, dtype=np.uint8)
    return vis


def save_depth_vis_png(out_path: Path, vis_rgb):
    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(out_path), vis_bgr)
    if not ok:
        raise RuntimeError(f"保存深度可视化失败: {out_path}")


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="输入一个目录，依次读取其中所有 ply 渲染深度图；每处理一个新的 ply，都与之前已保存的深度图融合并覆盖保存。"
    )
    parser.add_argument("ply_root", type=str, help="输入 ply 文件夹，或单个 ply 文件")
    parser.add_argument("cam_root", type=str, help="相机目录，或其上一级目录（内部应包含 cams/）")
    parser.add_argument("output_root", type=str, help="输出根目录")

    parser.add_argument("--image-root", type=str, default=None, help="原始 JPG 图像目录，按相机名匹配 name.jpg，并缩放后保存到 output/images/*.png")
    parser.add_argument("--view-list-txt", type=str, default=None, help="视角名单 txt，一行一个名字；不传则用 cams 下全部视角")
    parser.add_argument("--pair-list-txt", type=str, default=None, help="视角名单 txt，一行两个名字；不传则用 cams 下全部视角")
    parser.add_argument("--max-image-edge", type=int, default=1600, help="图像最大边；>0 时对相机分辨率和内参做等比例缩放，并保存到输出目录/cams")

    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D 点大小")
    parser.add_argument("--visible", action="store_true", help="显示窗口（调试用）")
    parser.add_argument("--save-depth-npy", action="store_true", help="保存融合后的深度为 .npy")
    parser.add_argument("--save-depth-exr", action="store_true", help="保存融合后的深度为 .exr")
    parser.add_argument("--save-depth-vis", action="store_true", help="保存融合后的深度可视化 .png")

    args = parser.parse_args()

    if (not args.save_depth_npy) and (not args.save_depth_exr) and (not args.save_depth_vis):
        args.save_depth_exr = True

    ply_root = Path(args.ply_root)
    cam_root = Path(args.cam_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image_root = None
    if args.image_root is not None:
        image_root = Path(args.image_root)
        if not image_root.exists() or not image_root.is_dir():
            raise FileNotFoundError(f"原始图像目录不存在: {image_root}")

    images_dir, cam_out_dir, npy_dir, exr_dir, vis_dir = ensure_dirs(
        output_root,
        save_npy=args.save_depth_npy,
        save_exr=args.save_depth_exr,
        save_vis=args.save_depth_vis,
        save_images=(image_root is not None),
    )

    # 1. 找到所有 ply
    plys = find_all_plys(ply_root)
    print(f"[INFO] 找到 ply 数量: {len(plys)}")
    for i, p in enumerate(plys):
        print(f"  [{i:03d}] {p}")

    # 2. 读取并筛选相机
    cams_dir = resolve_cams_dir(cam_root)
    cam_txts = find_cam_txts(cams_dir)

    selected_names = read_view_list_txt(args.view_list_txt, args.pair_list_txt)

    cam_txts = filter_cam_txts_by_view_list(cam_txts, selected_names)
    print(f"[INFO] 实际处理视角数: {len(cam_txts)} | {cams_dir}")

    # 3. 相机只预处理一次
    cams = prepare_cameras(
        cam_txts=cam_txts,
        max_image_edge=args.max_image_edge,
        cam_out_dir=cam_out_dir,
        image_root=image_root,
        images_dir=images_dir,
    )

    vis_cache = VisCache(point_size=args.point_size, visible=args.visible)

    try:
        for ply_idx, ply_path in enumerate(plys):
            print(f"\n[INFO] 开始处理 ply ({ply_idx + 1}/{len(plys)}): {ply_path}")
            pcd = load_point_cloud(str(ply_path))
            n_points = np.asarray(pcd.points).shape[0]
            print(f"[INFO] 当前点云点数: {n_points}")

            # 用 ply 路径做当前几何标识
            geom_tag = str(ply_path.resolve())

            for cam in tqdm(cams, desc=f"Render {ply_path.name}"):
                name = cam["name"]
                h, w = cam["h"], cam["w"]

                vis, key = vis_cache.get(w, h)

                # 关键优化：
                # 同一个 ply + 同一个分辨率窗口，只在第一次真正 clear/add geometry
                vis_cache.ensure_geometry(key, pcd, geom_tag)

                set_camera_from_world2cam(
                    vis,
                    cam["intrinsic"],
                    cam["extrinsic"],
                    w,
                    h,
                )

                curr_depth = render_depth_only(vis)

                fused_depth = curr_depth
                prev_depth = None
                npy_path = None
                exr_path = None

                if npy_dir is not None:
                    npy_path = npy_dir / f"{name}.npy"
                    prev_depth = load_depth_npy(npy_path)
                elif exr_dir is not None:
                    exr_path = exr_dir / f"{name}.exr"
                    prev_depth = load_depth_exr(exr_path)

                if prev_depth is not None:
                    fused_depth = fuse_depth_keep_nearest(prev_depth, curr_depth)

                if npy_dir is not None:
                    if npy_path is None:
                        npy_path = npy_dir / f"{name}.npy"
                    save_depth_npy(npy_path, fused_depth)

                if exr_dir is not None:
                    if exr_path is None:
                        exr_path = exr_dir / f"{name}.exr"
                    save_depth_exr(exr_path, fused_depth)

                if vis_dir is not None:
                    vis_rgb = make_depth_vis(fused_depth)
                    vis_path = vis_dir / f"{name}.png"
                    save_depth_vis_png(vis_path, vis_rgb)

    finally:
        vis_cache.close_all()

    print(f"\n[INFO] 完成，输出目录: {output_root}")


if __name__ == "__main__":
    main()

"""
从采样的密集点云中渲染选取的一些视角，用于后续配准优化；

python render_depth_from_ply.py nanfang/models/pc/0 \
    ./nanfang ./nanfang_render \
    --save-depth-npy --save-depth-vis \
    --image-root ../undistort/nanfang/ImageCorrection/undistort \
    --view-list-txt ../lidar/nanfang/selected_views.txt \
    --point-size 3.0

python render_depth_from_ply.py yanghaitang/models/pc/0 \
    ./yanghaitang ./yanghaitang_render \
    --save-depth-npy --save-depth-vis \
    --image-root ../undistort/yanghaitang/ImageCorrection/undistort \
    --view-list-txt ../lidar/yanghaitang/selected_views.txt \
    --point-size 3.0

python render_depth_from_ply.py xiaoxiang/models/pc/0 \
    ./xiaoxiang ./xiaoxiang_render \
    --save-depth-npy --save-depth-vis \
    --image-root ../undistort/xiaoxiang/ImageCorrection/undistort \
    --view-list-txt ../lidar/xiaoxiang/selected_views.txt \
    --point-size 3.0


"""

"""
python render_depth_from_ply.py nanfang/models/pc/0 \
    ./nanfang ./nanfang_render \
    --save-depth-npy --save-depth-vis \
    --image-root ../undistort/nanfang/ImageCorrection/undistort \
    --pair-list-txt ./nanfang_pair_selected/selected_pairs.txt \
    --point-size 3.0

"""