import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
import open3d as o3d
import open3d.visualization.rendering as rendering
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import math
import imageio.v2 as imageio

def copy_image_as_png(src_path: str, dst_png_path: str):
    if os.path.exists(dst_png_path):
        return
    try:
        import cv2
        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cv2.imread failed: {src_path}")
        cv2.imwrite(dst_png_path, img)
    except Exception:
        import imageio.v2 as imageio
        img = imageio.imread(src_path)
        imageio.imwrite(dst_png_path, img)

def sorted_image_list_by_id(cameras, image_names):
    """按 SortedImageID 排序，保证相机/图像/深度顺序一致。"""
    items = []
    for name in image_names:
        if name in cameras:
            items.append((cameras[name].sorted_image_id, name))
    items.sort(key=lambda x: x[0])
    return [n for _, n in items]

def compute_hfov_deg(width_px: int, fx: float) -> float:
    """水平FOV：2*atan(w/(2fx))，与你示例 720/360.606 -> 89.90° 一致。"""
    return float(2.0 * math.degrees(math.atan(width_px / (2.0 * fx + 1e-12))))


def save_cam_txt(
    out_txt_path: str,
    T_cam2world: np.ndarray,   # 输入是 cam2world
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int
):
    """
    保存你指定格式：
    - extrinsic: OpenCV(x Right, y Down, z Forward), world2camera
    - intrinsic 3x3
    - h w fov  (注意：你写的是 h w fov)
    """
    T_world2cam = np.linalg.inv(T_cam2world).astype(np.float64)

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    fov = compute_hfov_deg(W, fx)

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("extrinsic opencv(x Right, y Down, z Forward) world2camera\n")
        for r in range(4):
            f.write(f"{T_world2cam[r,0]:.12f} {T_world2cam[r,1]:.12f} {T_world2cam[r,2]:.12f} {T_world2cam[r,3]:.12f}\n")
        f.write("\n")
        f.write("intrinsic: fx fy cx cy (pixel)\n")
        for r in range(3):
            f.write(f"{K[r,0]:.12f} {K[r,1]:.12f} {K[r,2]:.12f}\n")
        f.write("\n")
        f.write("h w hfov\n")
        f.write(f"{H} {W} {fov:.12f}\n")


def export_png_and_cam_parallel(images_sorted, cameras, image_path, image_dir, cam_dir, max_workers=None):
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(cam_dir, exist_ok=True)

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 8) * 2)

    def _job(idx_img):
        idx, img_name = idx_img
        cam = cameras[img_name]
        src_img = os.path.join(image_path, img_name)

        fname = f"{idx:08d}"
        dst_png = os.path.join(image_dir, fname + ".png")
        dst_cam = os.path.join(cam_dir, fname + ".txt")

        copy_image_as_png(src_img, dst_png)
        save_cam_txt(
            out_txt_path=dst_cam,
            T_cam2world=cam.T4x4.astype(np.float64),
            fx=float(cam.fx), fy=float(cam.fy), cx=float(cam.cx), cy=float(cam.cy),
            H=int(cam.height), W=int(cam.width)
        )
        return fname

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_job, (i, n)) for i, n in enumerate(images_sorted)]
        for fu in as_completed(futures):
            _ = fu.result()


class CameraCalibration:
    """相机标定数据类"""
    def __init__(self, data: Dict[str, Any]):
        self.original_scene_name = data.get("OriginalSceneName", "")
        self.original_image_name = data.get("OriginalImageName", "")
        self.sorted_image_id = data.get("SortedImageID", -1)
        
        self.T4x4 = np.array(data.get("T4x4", []))  # 外参矩阵 (4x4)  cam2world
        self.P3x3 = np.array(data.get("P3x3", []))  # 内参矩阵 (3x3)
        
        # 畸变参数
        self.k1 = data.get("K1", 0.0)
        self.k2 = data.get("K2", 0.0)
        self.k3 = data.get("K3", 0.0)
        self.p1 = data.get("P1", 0.0)
        self.p2 = data.get("P2", 0.0)
        
        # 图像尺寸
        self.width = data.get("Width", 0)
        self.height = data.get("Height", 0)
        
        # 提取旋转矩阵和平移向量
        if self.T4x4.shape == (4, 4):
            self.rotation_matrix = self.T4x4[:3, :3]
            self.translation_vector = self.T4x4[:3, 3]
        else:
            self.rotation_matrix = np.eye(3)
            self.translation_vector = np.zeros(3)
        
        # 提取内参参数
        if self.P3x3.shape == (3, 3):
            self.fx = self.P3x3[0, 0]
            self.fy = self.P3x3[1, 1]
            self.cx = self.P3x3[0, 2]
            self.cy = self.P3x3[1, 2]
        else:
            self.fx = self.fy = self.cx = self.cy = 0.0
    
    def __repr__(self):
        return f"CameraCalibration(image={self.original_image_name}, id={self.sorted_image_id})"

    def get_camera_center(self) -> np.ndarray:
        """获取相机中心在世界坐标系中的位置"""
        return self.translation_vector
    
    def get_camera_axes(self, scale: float = 1.0) -> tuple:
        """获取相机坐标系轴在世界坐标系中的方向
        返回: (x_axis, y_axis, z_axis)
        """
        # 旋转矩阵的列向量表示相机坐标系轴在世界坐标系中的方向
        x_axis = self.rotation_matrix[:, 0] * scale
        y_axis = self.rotation_matrix[:, 1] * scale
        z_axis = self.rotation_matrix[:, 2] * scale
        return x_axis, y_axis, z_axis

def read_camera_calibration_json(file_path: str) -> Dict:
    """
    读取相机标定JSON文件
    
    参数:
        file_path: JSON文件路径
        
    返回:
        List[CameraCalibration]: 相机标定数据对象列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cameras = dict()
    for i, camera_data in enumerate(data):
        camera = CameraCalibration(camera_data)
        cameras[camera.original_image_name] = camera
    print(f"成功加载 {len(cameras)} 个相机的标定数据")
    return cameras

import open3d.visualization.rendering as rendering
import open3d as o3d
import numpy as np
import os

def _ensure_renderer(renderer, scene_has_mesh, W, H, mesh, mat):
    """
    由于 open3d.cuda OffscreenRenderer 可能没有 width/height 属性，
    我们通过外部记录分辨率来判断是否需要重建 renderer。
    """
    if renderer is None:
        renderer = rendering.OffscreenRenderer(W, H)
        renderer.scene.set_background([0, 0, 0, 1])
        renderer.scene.add_geometry("mesh", mesh, mat)
        return renderer

    # 尝试释放并重建（某些版本 release_resources 也可能不存在）
    try:
        renderer.release_resources()
    except Exception:
        pass

    # 直接重建一个新的 renderer
    try:
        del renderer
    except Exception:
        pass

    renderer = rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background([0, 0, 0, 1])
    renderer.scene.add_geometry("mesh", mesh, mat)
    return renderer


def render_depth_maps_from_mesh_open3d(mesh_path, cameras, image_names,
                                      out_depth_dir, 
                                      depth_max_m=200.0,
                                      use_z_in_view_space=True,
                                      convention="cv"):

    os.makedirs(out_depth_dir, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"

    renderer = None
    cur_wh = None  # ✅ 自己维护分辨率

    image_names = sorted_image_list_by_id(cameras, image_names)

    intrinsic_cache = {}
    world2cam_list = []
    intr_key_list = []
    wh_list = []
    for img_name in image_names:
        cam = cameras[img_name]
        W, H = int(cam.width), int(cam.height)
        fx, fy, cx, cy = float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)
        wh_list.append((W, H))
        intr_key = (W, H, fx, fy, cx, cy)
        intr_key_list.append(intr_key)
        
        T_cam2world = cam.T4x4.astype(np.float64)
        T_world2cam = np.linalg.inv(T_cam2world)

        if convention.lower() == "gl":
            cv_to_gl = np.array([[1, 0, 0, 0],
                                 [0,-1, 0, 0],
                                 [0, 0,-1, 0],
                                 [0, 0, 0, 1]], dtype=np.float64)
            T_world2cam = cv_to_gl @ T_world2cam

        world2cam_list.append(T_world2cam)

    for idx, img_name in enumerate(image_names):
        fname = f"{idx:08d}"
        exr_path = os.path.join(out_depth_dir, fname + ".exr")
        if os.path.exists(exr_path):
            continue  # ✅ 断点续跑

        W, H = wh_list[idx]
        if cur_wh != (W, H):
            renderer = _ensure_renderer(renderer, scene_has_mesh=True, W=W, H=H, mesh=mesh, mat=mat)
            cur_wh = (W, H)

        key = intr_key_list[idx]
        intrinsic = intrinsic_cache.get(key)
        if intrinsic is None:
            W, H, fx, fy, cx, cy = key
            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            intrinsic_cache[key] = intrinsic

        renderer.setup_camera(intrinsic, world2cam_list[idx])

        depth_o3d = renderer.render_to_depth_image(z_in_view_space=use_z_in_view_space)
        depth = np.asarray(depth_o3d).astype(np.float32)
        depth[depth > depth_max_m] = 0.0

        # EXR 写出
        imageio.imwrite(exr_path, depth)

    # 清理
    try:
        renderer.release_resources()
    except Exception:
        pass
    try:
        del renderer
    except Exception:
        pass


if __name__ == "__main__":
    data_dir = "./interval5_CAM_LIDAR"
    output_dir = "./uavscenes"
    os.makedirs(output_dir, exist_ok=True)

    mesh_dict = {
        "AMtown": "./terra_3dmap_pointcloud_mesh/AMtown/Mesh.ply",
        "AMvalley": "terra_3dmap_pointcloud_mesh/AMvalley/Mesh.ply",
        "HKairport": "terra_3dmap_pointcloud_mesh/HKairport/Mesh.ply",
        "HKairport_GNSS": "terra_3dmap_pointcloud_mesh/HKairport_GNSS/Mesh.ply",
        "HKairport_GNSS_Evening": "terra_3dmap_pointcloud_mesh/HKairport_GNSS_Evening/Mesh.ply",
        "HKisland": "terra_3dmap_pointcloud_mesh/HKisland/Mesh.ply",
        "HKisland_GNSS": "terra_3dmap_pointcloud_mesh/HKisland_GNSS/Mesh.ply",
        "HKisland_GNSS_Evening": "terra_3dmap_pointcloud_mesh/HKisland_GNSS_Evening/Mesh.ply",
    }
    def _find_mesh(sequence):
        for name in mesh_dict.keys():
            if name in sequence:
                return mesh_dict[name]

    list_sequences = os.listdir(data_dir)
    for sequence in tqdm(list_sequences):
        image_path = os.path.join(data_dir, sequence, "interval5_CAM")
        images = [f for f in os.listdir(image_path) if f.endswith(".jpg")]

        # read cameras
        camera_path = os.path.join(data_dir, sequence, "sampleinfos_interpolated.json")
        cameras = read_camera_calibration_json(camera_path)

        images_sorted = sorted_image_list_by_id(cameras, images)
        
        # 渲染深度图
        mesh_path = _find_mesh(sequence)
        seq_out_dir = os.path.join(output_dir, sequence)
        depth_dir = os.path.join(seq_out_dir, "depth")
        image_dir = os.path.join(seq_out_dir, "images")
        cam_dir = os.path.join(seq_out_dir, "cams")

        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(cam_dir, exist_ok=True)

        # ✅ 并行导出 png + cam（加速明显）
        export_png_and_cam_parallel(
            images_sorted=images_sorted,
            cameras=cameras,
            image_path=image_path,
            image_dir=image_dir,
            cam_dir=cam_dir,
            max_workers=None   # 自动
        )

        # 2) 渲染深度并保存为 exr
        render_depth_maps_from_mesh_open3d(
            mesh_path=mesh_path,
            cameras=cameras,
            image_names=images_sorted,
            out_depth_dir=depth_dir,
            use_z_in_view_space=True,
            convention="cv"
        )
