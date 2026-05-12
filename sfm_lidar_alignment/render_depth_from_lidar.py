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
def load_point_cloud(ply_path: str, default_color=(180, 180, 180)) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd is None or pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {ply_path}")

    if not pcd.has_colors():
        n = np.asarray(pcd.points).shape[0]
        colors = np.full(
            (n, 3),
            np.array(default_color, dtype=np.float32) / 255.0,
            dtype=np.float32
        )
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


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
    """
    按用户给的名单筛选视角
    优先保持用户 txt 中的顺序
    支持:
        stem
        filename
    """
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


# =========================
# Camera TXT Parser
# =========================
def parse_cam_txt(cam_txt_path: Path):
    """
    解析格式：

    extrinsic opencv(x Right, y Down, z Forward) world2camera
    4x4

    intrinsic: fx fy cx cy (pixel)
    3x3

    h w fov
    h w fov
    """
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

    ext_lines = lines[ext_idx + 1 : ext_idx + 5]
    intr_lines = lines[intr_idx + 1 : intr_idx + 4]
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
    """
    若 max(h, w) > max_image_edge，则缩放相机分辨率与内参
    """
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
    new_intrinsic[0, 0] *= scale  # fx
    new_intrinsic[1, 1] *= scale  # fy
    new_intrinsic[0, 2] *= scale  # cx
    new_intrinsic[1, 2] *= scale  # cy

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
        "fov": cam["fov"],  # 纯分辨率缩放，不改视场角
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
            f.write(
                f"{ext[i,0]:.12f} {ext[i,1]:.12f} {ext[i,2]:.12f} {ext[i,3]:.12f}\n"
            )
        f.write("\n")

        f.write("intrinsic: fx fy cx cy (pixel)\n")
        for i in range(3):
            f.write(
                f"{intr[i,0]:.12f} {intr[i,1]:.12f} {intr[i,2]:.12f}\n"
            )
        f.write("\n")

        f.write("h w fov\n")
        f.write(f"{h} {w} {fov:.12f}\n")


# =========================
# Open3D Visualizer Cache
# =========================
class VisCache:
    """
    缓存不同 (w, h) 分辨率的 Open3D Visualizer
    """
    def __init__(self, point_size=2.0, visible=False):
        self.cache = {}   # (w,h) -> {"vis": vis, "inited": bool}
        self.point_size = float(point_size)
        self.visible = bool(visible)

    def get(self, w: int, h: int):
        key = (int(w), int(h))
        if key in self.cache:
            return self.cache[key]["vis"], self.cache[key]["inited"], key

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


def render_rgb_depth(
    vis: o3d.visualization.Visualizer,
    render_rgb=True,
    render_depth=True,
):
    vis.poll_events()
    vis.update_renderer()

    rgb = None
    depth = None

    if render_rgb:
        rgb_f = np.asarray(
            vis.capture_screen_float_buffer(do_render=True),
            dtype=np.float32,
        )
        rgb = np.clip(rgb_f * 255.0, 0, 255).astype(np.uint8)

    if render_depth:
        depth_f = np.asarray(
            vis.capture_depth_float_buffer(do_render=True),
            dtype=np.float32,
        )
        depth = depth_f.astype(np.float32)

    return rgb, depth


# =========================
# Save
# =========================
def ensure_dirs(output_root: Path, save_npy: bool, save_exr: bool, save_vis: bool):
    rgb_dir = output_root / "images"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    cam_out_dir = output_root / "cams"
    cam_out_dir.mkdir(parents=True, exist_ok=True)

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

    return rgb_dir, cam_out_dir, npy_dir, exr_dir, vis_dir


def save_rgb_jpg(out_path: Path, rgb, jpg_quality=90):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(
        str(out_path),
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)],
    )
    if not ok:
        raise RuntimeError(f"保存 JPG 失败: {out_path}")


def save_depth_npy(out_path: Path, depth):
    np.save(str(out_path), depth.astype(np.float32))


def save_depth_exr(out_path: Path, depth):
    ok = cv2.imwrite(str(out_path), depth.astype(np.float32))
    if not ok:
        raise RuntimeError(f"保存 EXR 失败: {out_path}")


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
        description="使用 Open3D 根据 cams/*.txt 从 LiDAR PLY 渲染 RGB/JPG 和深度图"
    )
    parser.add_argument("ply_path", type=str, help="输入 LiDAR PLY 点云")
    parser.add_argument("cam_root", type=str, help="相机目录，或其上一级目录（内部应包含 cams/）")
    parser.add_argument("output_root", type=str, help="输出根目录")

    parser.add_argument("--view-list-txt", type=str, default=None, help="视角名单 txt，一行一个名字；不传则用 cams 下全部视角")
    parser.add_argument("--pair-list-txt", type=str, default=None, help="视角名单 txt，一行一个名字；不传则用 cams 下全部视角")
    parser.add_argument("--max-image-edge", type=int, default=1600, help="图像最大边；>0 时对相机分辨率和内参做等比例缩放，并保存到输出目录/cams")

    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D 点大小")
    parser.add_argument("--visible", action="store_true", help="显示窗口（调试用）")
    parser.add_argument("--jpg-quality", type=int, default=50, help="JPG 质量")
    parser.add_argument("--save-depth-npy", action="store_true", help="保存深度为 .npy")
    parser.add_argument("--save-depth-exr", action="store_true", help="保存深度为 .exr")
    parser.add_argument("--save-depth-vis", action="store_true", help="保存深度可视化 .png")
    parser.add_argument("--default-color", type=int, nargs=3, default=[180, 180, 180], help="点云无颜色时默认 RGB")

    args = parser.parse_args()

    if (not args.save_depth_npy) and (not args.save_depth_exr) and (not args.save_depth_vis):
        args.save_depth_exr = True

    ply_path = Path(args.ply_path)
    cam_root = Path(args.cam_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rgb_dir, cam_out_dir, npy_dir, exr_dir, vis_dir = ensure_dirs(
        output_root,
        save_npy=args.save_depth_npy,
        save_exr=args.save_depth_exr,
        save_vis=args.save_depth_vis,
    )

    print("[INFO] 读取点云...")
    pcd = load_point_cloud(str(ply_path), default_color=tuple(args.default_color))
    n_points = np.asarray(pcd.points).shape[0]
    print(f"[INFO] 点数: {n_points}")

    cams_dir = resolve_cams_dir(cam_root)
    cam_txts = find_cam_txts(cams_dir)

    # 读取视角名单并筛选
    selected_names = read_view_list_txt(args.view_list_txt, args.pair_list_txt)
    cam_txts = filter_cam_txts_by_view_list(cam_txts, selected_names)

    print(f"[INFO] 实际渲染视角数: {len(cam_txts)} | {cams_dir}")

    vis_cache = VisCache(point_size=args.point_size, visible=args.visible)

    try:
        for cam_txt in tqdm(cam_txts, desc="Open3D Rendering"):
            cam = parse_cam_txt(cam_txt)

            # 缩放相机
            cam = scale_camera(cam, args.max_image_edge)

            # 把实际使用的 cam 保存到输出目录/cams
            save_cam_txt(cam, cam_out_dir / f"{cam['name']}.txt")

            name = cam["name"]
            h, w = cam["h"], cam["w"]

            vis, inited, key = vis_cache.get(w, h)

            if not inited:
                vis.clear_geometries()
                vis.add_geometry(pcd, reset_bounding_box=True)
                vis_cache.mark_inited(key)

            set_camera_from_world2cam(
                vis,
                cam["intrinsic"],
                cam["extrinsic"],
                w,
                h,
            )

            rgb, depth = render_rgb_depth(
                vis,
                render_rgb=True,
                render_depth=True,
            )

            # RGB -> jpg
            rgb_path = rgb_dir / f"{name}.jpg"
            save_rgb_jpg(rgb_path, rgb, jpg_quality=args.jpg_quality)

            # Depth -> npy
            if npy_dir is not None:
                npy_path = npy_dir / f"{name}.npy"
                save_depth_npy(npy_path, depth)

            # Depth -> exr
            if exr_dir is not None:
                exr_path = exr_dir / f"{name}.exr"
                save_depth_exr(exr_path, depth)

            # Depth visualization
            if vis_dir is not None:
                vis_rgb = make_depth_vis(depth)
                vis_path = vis_dir / f"{name}.png"
                save_depth_vis_png(vis_path, vis_rgb)

    finally:
        vis_cache.close_all()

    print(f"[INFO] 完成，输出目录: {output_root}")


if __name__ == "__main__":
    main()



"""
从粗对齐的lidar中渲染选取的一些视角，用于后续配准优化；

python render_depth_from_lidar.py nanfang/transform/lidar_full.ply \
    ../recon/nanfang ./nanfang \
    --save-depth-npy --save-depth-vis \
    --view-list-txt nanfang/selected_views.txt \
    --pair-list-txt ../recon/nanfang_pair_selected/selected_pairs.txt \
    --point-size 3.0

python render_depth_from_lidar.py yanghaitang/transform/lidar_full.ply \
    ../recon/yanghaitang ./yanghaitang \
    --save-depth-npy --save-depth-vis \
    --view-list-txt yanghaitang/selected_views.txt \
    --point-size 3.0

python render_depth_from_lidar.py xiaoxiang/transform/lidar_full.ply \
    ../recon/xiaoxiang ./xiaoxiang \
    --save-depth-npy --save-depth-vis \
    --view-list-txt xiaoxiang/selected_views.txt \
    --point-size 3.0

"""


"""
从精对齐的lidar中渲染选取的一些立体像对，用于后续评估；

python render_depth_from_lidar.py nanfang/transform/lidar_finall.ply \
    ../recon/nanfang ./nanfang_final \
    --save-depth-npy --save-depth-vis \
    --view-list-txt nanfang/selected_views.txt \
    --point-size 3.0

python render_depth_from_lidar.py yanghaitang/transform/lidar_finall.ply \
    ../recon/yanghaitang ./yanghaitang_final \
    --save-depth-npy --save-depth-vis \
    --view-list-txt yanghaitang/selected_views.txt \
    --point-size 3.0

python render_depth_from_lidar.py xiaoxiang/transform/lidar_finall.ply \
    ../recon/xiaoxiang ./xiaoxiang_final \
    --save-depth-npy --save-depth-vis \
    --view-list-txt xiaoxiang/selected_views.txt \
    --point-size 3.0

"""
