"""Interactive renderer for A3D-Syn-S style synthetic scenes.

This tool is intended for smaller or irregular synthetic scene units where a
human operator selects useful UAV viewpoints in Open3D, optionally records
camera trajectories, interpolates intermediate poses, and renders the selected
views to the common A3D camera/depth/image layout.
"""

import os
import json
import time
import math
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import imageio.v2 as imageio

# Enable OpenEXR support in OpenCV if the installed build provides it.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


# =========================
# Utils
# =========================

def make_dirs(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "depth").mkdir(exist_ok=True)
    (out_dir / "cams").mkdir(exist_ok=True)
    (out_dir / "viewpoints").mkdir(exist_ok=True)
    # Optional depth preview PNGs are useful for quick inspection.
    (out_dir / "depth_png").mkdir(exist_ok=True)


def safe_norm(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return None, 0.0
    return v / n, float(n)


def orthonormalize_rotation(R):
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    R_ = U @ Vt
    if np.linalg.det(R_) < 0:
        U[:, -1] *= -1
        R_ = U @ Vt
    return R_


def get_camera_center_from_extrinsic(T_w2c):
    return np.linalg.inv(T_w2c)[:3, 3]


def decompose_extrinsic_w2c(T_w2c):
    T_c2w = np.linalg.inv(T_w2c)
    cam_pos = T_c2w[:3, 3].copy()
    R_c2w = T_c2w[:3, :3].copy()
    return cam_pos, R_c2w


def compose_extrinsic_from_cam_pose(cam_pos, R_c2w):
    T_c2w = np.eye(4, dtype=np.float64)
    T_c2w[:3, :3] = R_c2w
    T_c2w[:3, 3] = np.asarray(cam_pos, dtype=np.float64)
    return np.linalg.inv(T_c2w)


def calculate_fov(fy, h):
    """Return the vertical field of view in degrees."""
    fy = float(fy)
    h = float(h)
    return float(np.degrees(2.0 * np.arctan(h / (2.0 * fy))))


def write_cam_file(cam_path: Path, extrinsic, fx, fy, cx, cy, h, w):
    """
    Camera text format:
    - 4x4 world-to-camera extrinsic matrix
    - 3x3 intrinsic matrix
    - image height, image width, vertical FOV
    """
    fov = calculate_fov(fy, h)
    extrinsic = np.asarray(extrinsic, dtype=np.float64)

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


# =========================
# Visualization helpers
# =========================
def create_frustum_lineset(K, width, height, T_w2c, frustum_len=0.5):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    T_c2w = np.linalg.inv(T_w2c)
    R_c2w = T_c2w[:3, :3]
    t_c2w = T_c2w[:3, 3]

    z = float(frustum_len)
    corners_px = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float64)

    corners_cam = []
    for u, v in corners_px:
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        corners_cam.append([x, y, z])
    corners_cam = np.asarray(corners_cam, dtype=np.float64)

    corners_w = (R_c2w @ corners_cam.T).T + t_c2w[None, :]
    cam_center = t_c2w

    forward_cam = np.array([0.0, 0.0, z * 1.5], dtype=np.float64)
    forward_w = R_c2w @ forward_cam + t_c2w

    points = np.vstack([cam_center[None, :], corners_w, forward_w[None, :]])

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
        [0, 5],
    ]
    colors = [
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
        [0, 0, 1],
    ]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return ls


def create_camera_axes(T_w2c, size=0.2):
    T_c2w = np.linalg.inv(T_w2c)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
    frame.transform(T_c2w)
    return frame


# =========================
# Main App
# =========================
class ViewSelectorAndRenderer:
    def __init__(self, mesh_path, out_dir, width, height, fx, fy,
                 cx=None, cy=None, frustum_ratio=0.05, axis_ratio=0.02, max_images=0,
                 max_invalid_depth_ratio=0.3):
        self.mesh_path = mesh_path
        self.out_dir = out_dir
        self.width = width
        self.height = height
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx) if cx is not None else width / 2.0
        self.cy = float(cy) if cy is not None else height / 2.0
        self.max_images = int(max_images)  # <= 0 means unlimited.

        self.max_invalid_depth_ratio = float(np.clip(max_invalid_depth_ratio, 0.0, 1.0))

        self.K_user = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.frustum_ratio = float(frustum_ratio)
        self.axis_ratio = float(axis_ratio)

        self.mesh = None
        self.vis = None

        # All viewpoints that will be rendered: manual, trajectory-generated, and manual-pose interpolation results.
        self.viewpoints = []
        # Viewpoints added manually with Space; used as control poses for manual-pose interpolation.
        self.manual_viewpoints = []
        # Trajectory keyframes recorded with O.
        self.trajectory_keyframes = []

        self.marker_geometries = []
        self.rendered = False

        # Scene scale estimation.
        self.obb = None
        self.obb_extent = None
        self.obb_R = None
        self.world_up = None

        # Step-size parameters.
        self.move_step = None
        self.traj_interp_pos_step = None
        self.manual_interp_pos_step = None
        self.interp_min_views_per_segment = 2

        # Curve trajectory parameters.
        self.curve_alpha = 0.5   # centripetal Catmull-Rom
        self.curve_samples_per_seg = 30  # Number of samples per segment for arc-length estimation.

        # Redundancy-pruning thresholds; tune as needed.
        self.prune_pos_thresh_ratio = 0.01
        self.prune_angle_thresh_deg = 3.0

        # Deferred marker buffer to avoid UI stalls while generating views.
        self._pending_marker_batch = []

        make_dirs(self.out_dir)

    # -------------------------
    # Basic helpers
    # -------------------------
    def remaining_slots(self):
        if self.max_images <= 0:
            return 10**9
        return max(0, self.max_images - len(self.viewpoints))

    def can_add_more(self):
        return self.remaining_slots() > 0

    def _warn_full(self, prefix=""):
        if self.max_images > 0 and len(self.viewpoints) >= self.max_images:
            print(f"{prefix}[LIMIT] Reached max_images={self.max_images}; no more viewpoints can be added.")
            return True
        return False

    def _scene_scale(self):
        if self.obb_extent is None:
            return 1.0
        return max(float(np.linalg.norm(self.obb_extent)), 1e-6)

    # -------------------------
    # Mesh / init
    # -------------------------
    def load_mesh(self):
        print(f"[INFO] Loading mesh: {self.mesh_path}")
        mesh = o3d.io.read_triangle_mesh(self.mesh_path, enable_post_processing=True)
        if mesh.is_empty():
            raise RuntimeError(f"Failed to load mesh: {self.mesh_path}")
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        self.mesh = mesh

        self.obb = mesh.get_oriented_bounding_box()
        self.obb.color = (1.0, 0.5, 0.0)
        self.obb_extent = np.asarray(self.obb.extent, dtype=np.float64)
        self.obb_R = np.asarray(self.obb.R, dtype=np.float64)

        up_axis_idx = int(np.argmin(self.obb_extent))
        self.world_up = self.obb_R[:, up_axis_idx]
        u, _ = safe_norm(self.world_up)
        self.world_up = np.array([0.0, 0.0, 1.0]) if u is None else u

        scene_scale = self._scene_scale()

        # Scale visualization helpers relative to scene size.
        self.frustum_len = max(scene_scale * self.frustum_ratio, 1e-4)
        self.axis_size = max(scene_scale * self.axis_ratio, 1e-4)

        # Camera movement step.
        self.move_step = max(scene_scale * 0.01, 0.02)

        # Interpolation step.
        self.traj_interp_pos_step = max(scene_scale * 0.01, 0.02)
        self.manual_interp_pos_step = max(scene_scale * 0.01, 0.02)

        print("[INFO] Mesh loaded.")
        print(f"       Vertices : {len(mesh.vertices)}")
        print(f"       Triangles: {len(mesh.triangles)}")
        print(f"       Textures : {len(mesh.textures)}")
        print(f"       OBB extent: {self.obb_extent.round(4).tolist()}")
        print(f"       world_up (for PageUp/PageDown): {self.world_up.round(4).tolist()}")
        print(f"       frustum_len: {self.frustum_len:.4f}")
        print(f"       axis_size: {self.axis_size:.4f}")
        print(f"       move_step: {self.move_step:.4f}")
        print(f"       traj_interp_pos_step: {self.traj_interp_pos_step:.4f}")
        print(f"       manual_interp_pos_step: {self.manual_interp_pos_step:.4f}")
        print(f"       max_images: {self.max_images if self.max_images > 0 else 'unlimited'}")

    # -------------------------
    # Camera I/O
    # -------------------------
    def get_current_camera(self):
        vc = self.vis.get_view_control()
        cam = vc.convert_to_pinhole_camera_parameters()
        K = np.asarray(cam.intrinsic.intrinsic_matrix, dtype=np.float64)
        T_w2c = np.asarray(cam.extrinsic, dtype=np.float64)
        return K, T_w2c

    def get_current_camera_center(self):
        _, T_w2c = self.get_current_camera()
        return get_camera_center_from_extrinsic(T_w2c)

    def set_camera_by_extrinsic(self, T_w2c, K=None):
        vc = self.vis.get_view_control()
        cam = vc.convert_to_pinhole_camera_parameters()

        if K is None:
            K = self.K_user

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            self.width, self.height,
            float(K[0, 0]), float(K[1, 1]),
            float(K[0, 2]), float(K[1, 2]),
        )
        cam.intrinsic = intrinsic
        cam.extrinsic = T_w2c.copy()

        try:
            vc.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
        except TypeError:
            vc.convert_from_pinhole_camera_parameters(cam)

        self.vis.poll_events()
        self.vis.update_renderer()

    def apply_user_intrinsics(self):
        vc = self.vis.get_view_control()
        cam = vc.convert_to_pinhole_camera_parameters()

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(self.width, self.height, self.fx, self.fy, self.cx, self.cy)
        cam.intrinsic = intrinsic

        try:
            vc.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
            print("[INFO] Applied custom camera intrinsics.")
        except TypeError:
            vc.convert_from_pinhole_camera_parameters(cam)
            print("[INFO] Applied custom camera intrinsics. (legacy mode)")
        except Exception as e:
            print(f"[WARN] Failed to apply intrinsics: {e}")

    # -------------------------
    # Marker rendering
    # -------------------------
    def add_marker_for_viewpoint(self, K, T_w2c):
        frustum = create_frustum_lineset(K, self.width, self.height, T_w2c, frustum_len=self.frustum_len)
        axes = create_camera_axes(T_w2c, size=self.axis_size)

        try:
            self.vis.add_geometry(frustum, reset_bounding_box=False)
            self.vis.add_geometry(axes, reset_bounding_box=False)
        except TypeError:
            self.vis.add_geometry(frustum)
            self.vis.add_geometry(axes)

        self.marker_geometries.append((frustum, axes))

    def add_markers_batch(self, vps):
        """Draw multiple viewpoint markers in one batch to reduce UI stalls."""
        if self.vis is None or len(vps) == 0:
            return
        for vp in vps:
            self.add_marker_for_viewpoint(vp["K"], vp["extrinsic"])
        self.vis.poll_events()
        self.vis.update_renderer()

    def rebuild_markers(self):
        for pair in self.marker_geometries:
            for g in pair:
                try:
                    self.vis.remove_geometry(g, reset_bounding_box=False)
                except TypeError:
                    self.vis.remove_geometry(g)
        self.marker_geometries = []

        for vp in self.viewpoints:
            self.add_marker_for_viewpoint(vp["K"], vp["extrinsic"])

        self.vis.poll_events()
        self.vis.update_renderer()

    # -------------------------
    # Viewpoint management
    # -------------------------
    def append_viewpoint(self, K, T_w2c, source="manual", draw_marker=True, verbose=True):
        if self._warn_full(prefix="[ADD] "):
            return False

        vp = {"K": K.copy(), "extrinsic": T_w2c.copy(), "source": source}
        self.viewpoints.append(vp)

        if source == "manual":
            self.manual_viewpoints.append({"K": K.copy(), "extrinsic": T_w2c.copy(), "source": source})

        if draw_marker:
            self.add_markers_batch([vp])

        if verbose:
            idx = len(self.viewpoints) - 1
            cam_center = get_camera_center_from_extrinsic(T_w2c)
            print(f"[ADD] Viewpoint #{idx} ({source}) | cam_center = {cam_center.round(4).tolist()}")
        return True

    def delete_nearest_viewpoint_to_current_camera(self):
        if len(self.viewpoints) == 0:
            print("[X] No viewpoints to delete.")
            return False

        cur_cam = self.get_current_camera_center()
        dists = []
        for i, vp in enumerate(self.viewpoints):
            c = get_camera_center_from_extrinsic(vp["extrinsic"])
            dists.append((float(np.linalg.norm(c - cur_cam)), i))
        dists.sort(key=lambda x: x[0])

        dmin, idx = dists[0]
        removed = self.viewpoints.pop(idx)

        # If a manual viewpoint was removed, also remove the closest matching manual control pose.
        if removed.get("source") == "manual" and len(self.manual_viewpoints) > 0:
            rc = get_camera_center_from_extrinsic(removed["extrinsic"])
            mdists = []
            for j, mv in enumerate(self.manual_viewpoints):
                mc = get_camera_center_from_extrinsic(mv["extrinsic"])
                mdists.append((float(np.linalg.norm(mc - rc)), j))
            mdists.sort(key=lambda x: x[0])
            if len(mdists) > 0:
                _, midx = mdists[0]
                self.manual_viewpoints.pop(midx)

        self.rebuild_markers()
        print(f"[X] Removed nearest viewpoint #{idx} (dist={dmin:.4f}). Remaining: {len(self.viewpoints)}")
        return True

    # -------------------------
    # Camera translation only (keyboard)
    # -------------------------
    def _move_camera_local(self, dx=0.0, dy=0.0, dz=0.0):
        """
        Translate the camera in its local coordinate system without changing rotation.
        dx: camera-right direction.
        dy: camera-down direction, following OpenCV camera coordinates.
        dz: camera-forward direction.
        """
        K, T_w2c = self.get_current_camera()
        cam_pos, R_c2w = decompose_extrinsic_w2c(T_w2c)
        R_c2w = orthonormalize_rotation(R_c2w)

        x_axis = R_c2w[:, 0]
        y_axis = R_c2w[:, 1]
        z_axis = R_c2w[:, 2]

        cam_pos_new = cam_pos + x_axis * float(dx) + y_axis * float(dy) + z_axis * float(dz)
        T_new = compose_extrinsic_from_cam_pose(cam_pos_new, R_c2w)
        self.set_camera_by_extrinsic(T_new, K=self.K_user)

    def _move_camera_world(self, delta_world):
        K, T_w2c = self.get_current_camera()
        cam_pos, R_c2w = decompose_extrinsic_w2c(T_w2c)
        R_c2w = orthonormalize_rotation(R_c2w)

        cam_pos_new = cam_pos + np.asarray(delta_world, dtype=np.float64)
        T_new = compose_extrinsic_from_cam_pose(cam_pos_new, R_c2w)
        self.set_camera_by_extrinsic(T_new, K=self.K_user)

    def _move_camera_world_up(self, step):
        self._move_camera_world(self.world_up * float(step))

    # -------------------------
    # Curved interpolation (Catmull-Rom over multi-points)
    # -------------------------
    def _tj(self, ti, pi, pj, alpha=0.5):
        d = float(np.linalg.norm(pj - pi))
        d = max(d, 1e-9)
        return ti + (d ** alpha)

    def _catmull_rom_point(self, p0, p1, p2, p3, t, alpha=0.5):
        """
        Centripetal Catmull-Rom spline point for t in [0,1], segment p1->p2
        """
        t0 = 0.0
        t1 = self._tj(t0, p0, p1, alpha)
        t2 = self._tj(t1, p1, p2, alpha)
        t3 = self._tj(t2, p2, p3, alpha)

        tt = t1 + t * (t2 - t1)

        def lerp(a, b, ta, tb, tcur):
            if abs(tb - ta) < 1e-12:
                return a.copy()
            return (tb - tcur) / (tb - ta) * a + (tcur - ta) / (tb - ta) * b

        A1 = lerp(p0, p1, t0, t1, tt)
        A2 = lerp(p1, p2, t1, t2, tt)
        A3 = lerp(p2, p3, t2, t3, tt)

        B1 = lerp(A1, A2, t0, t2, tt)
        B2 = lerp(A2, A3, t1, t3, tt)

        C = lerp(B1, B2, t1, t2, tt)
        return C

    def _estimate_curve_segment_length(self, p0, p1, p2, p3, alpha=0.5, n_samples=30):
        pts = []
        for k in range(n_samples + 1):
            t = k / float(n_samples)
            pts.append(self._catmull_rom_point(p0, p1, p2, p3, t, alpha))
        pts = np.asarray(pts, dtype=np.float64)
        return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))

    # -------------------------
    # Quaternion / rotation interpolation (SLERP)
    # -------------------------
    def _rotmat_to_quat_xyzw(self, R):
        R = np.asarray(R, dtype=np.float64)
        tr = np.trace(R)

        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

        q = np.array([x, y, z, w], dtype=np.float64)
        qn = np.linalg.norm(q)
        if qn < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q / qn

    def _quat_xyzw_to_rotmat(self, q):
        q = np.asarray(q, dtype=np.float64)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = q / n

        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        R = np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
            [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
            [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)],
        ], dtype=np.float64)
        return orthonormalize_rotation(R)

    def _quat_slerp_xyzw(self, q0, q1, t):
        q0 = np.asarray(q0, dtype=np.float64)
        q1 = np.asarray(q1, dtype=np.float64)

        q0 = q0 / max(np.linalg.norm(q0), 1e-12)
        q1 = q1 / max(np.linalg.norm(q1), 1e-12)

        dot = float(np.dot(q0, q1))

        # Take the shortest interpolation path.
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)

        # Use normalized linear interpolation when rotations are very close.
        if dot > 0.9995:
            q = (1.0 - t) * q0 + t * q1
            q = q / max(np.linalg.norm(q), 1e-12)
            return q

        theta_0 = np.arccos(dot)
        theta = theta_0 * t

        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        s0 = np.sin(theta_0 - theta) / max(sin_theta_0, 1e-12)
        s1 = sin_theta / max(sin_theta_0, 1e-12)

        q = s0 * q0 + s1 * q1
        q = q / max(np.linalg.norm(q), 1e-12)
        return q

    def _interp_rotation_c2w_slerp(self, R0, R1, t):
        q0 = self._rotmat_to_quat_xyzw(R0)
        q1 = self._rotmat_to_quat_xyzw(R1)
        q = self._quat_slerp_xyzw(q0, q1, float(t))
        return self._quat_xyzw_to_rotmat(q)

    def _interp_rotation_c2w_linear(self, R0, R1, t):
        return orthonormalize_rotation((1.0 - t) * R0 + t * R1)

    # -------------------------
    # Redundancy pruning
    # -------------------------
    def prune_redundant_views(self, planned_views,
                              pos_thresh_ratio=None,
                              angle_thresh_deg=None):
        """
        Remove redundant viewpoints using both:
            1) camera-center distance, and
            2) viewing-direction angle, measured from the camera z-axis.
        """
        if len(planned_views) <= 1:
            return planned_views

        if pos_thresh_ratio is None:
            pos_thresh_ratio = self.prune_pos_thresh_ratio
        if angle_thresh_deg is None:
            angle_thresh_deg = self.prune_angle_thresh_deg

        scene_scale = self._scene_scale()
        pos_thresh = scene_scale * float(pos_thresh_ratio)
        angle_thresh = np.deg2rad(float(angle_thresh_deg))

        kept = []
        last_keep = None

        for vp in planned_views:
            p, R = decompose_extrinsic_w2c(vp["extrinsic"])
            z_axis = R[:, 2]
            z_axis, _ = safe_norm(z_axis)
            if z_axis is None:
                z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            if last_keep is None:
                kept.append(vp)
                last_keep = (p, z_axis)
                continue

            p_last, z_last = last_keep

            dp = float(np.linalg.norm(p - p_last))
            cosang = float(np.clip(np.dot(z_axis, z_last), -1.0, 1.0))
            ang = float(np.arccos(cosang))

            if dp < pos_thresh and ang < angle_thresh:
                continue

            kept.append(vp)
            last_keep = (p, z_axis)

        print(f"[PRUNE] Before: {len(planned_views)}, After: {len(kept)} | "
              f"pos_thresh={pos_thresh:.4f}, angle_thresh={float(angle_thresh_deg):.2f}deg")
        return kept

    def _prune_against_existing(self, candidates, pos_eps_ratio=0.002, angle_eps_deg=1.0):
        """Remove candidates that are almost identical to existing viewpoints."""
        if len(candidates) == 0 or len(self.viewpoints) == 0:
            return candidates

        scene_scale = self._scene_scale()
        pos_eps = max(scene_scale * float(pos_eps_ratio), 1e-6)
        ang_eps = np.deg2rad(float(angle_eps_deg))

        existing_pose = []
        for vp in self.viewpoints:
            p, R = decompose_extrinsic_w2c(vp["extrinsic"])
            z = R[:, 2]
            z, _ = safe_norm(z)
            if z is None:
                z = np.array([0, 0, 1.0], dtype=np.float64)
            existing_pose.append((p, z))

        kept = []
        for vp in candidates:
            p, R = decompose_extrinsic_w2c(vp["extrinsic"])
            z = R[:, 2]
            z, _ = safe_norm(z)
            if z is None:
                z = np.array([0, 0, 1.0], dtype=np.float64)

            redundant = False
            for p0, z0 in existing_pose:
                if np.linalg.norm(p - p0) < pos_eps:
                    cosang = np.clip(np.dot(z, z0), -1.0, 1.0)
                    ang = np.arccos(cosang)
                    if ang < ang_eps:
                        redundant = True
                        break
            if not redundant:
                kept.append(vp)

        if len(kept) != len(candidates):
            print(f"[PRUNE-EXISTING] Before: {len(candidates)}, After: {len(kept)}")
        return kept

    def _view_feature_pos_dir(self, vp):
        """
        Extract the camera center and forward direction from a viewpoint.
        Returns:
        - p: camera center, shape (3,)
        - dir: camera z-axis / forward direction, shape (3,)
        """
        p, R = decompose_extrinsic_w2c(vp["extrinsic"])
        z = R[:, 2].astype(np.float64)  # OpenCV camera-forward direction.
        z, _ = safe_norm(z)
        if z is None:
            z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return p.astype(np.float64), z.astype(np.float64)
    
    def _select_diverse_views(self, candidates, k, existing_views=None,
                          pos_weight=1.0, dir_weight=0.35):
        """
        Select k diverse viewpoints from candidates using position and direction.
        This is similar to farthest point sampling (FPS). Existing views are treated
        as already occupied references so newly selected views do not duplicate them.

        Distance definition, larger is better:
            d = pos_weight * ||dp||_norm + dir_weight * angle_norm
        where:
            ||dp||_norm = ||p_i - p_j|| / scene_scale
            angle_norm  = arccos(dot(dir_i, dir_j)) / pi, in [0, 1]
        """
        n = len(candidates)
        if k <= 0 or n == 0:
            return []
        if k >= n:
            return candidates

        scene_scale = self._scene_scale()
        scene_scale = max(scene_scale, 1e-6)

        cand_feats = [self._view_feature_pos_dir(vp) for vp in candidates]

        ref_feats = []
        if existing_views is not None and len(existing_views) > 0:
            ref_feats = [self._view_feature_pos_dir(vp) for vp in existing_views]

        def pair_dist(feat_a, feat_b):
            pa, za = feat_a
            pb, zb = feat_b
            dp = np.linalg.norm(pa - pb) / scene_scale
            cosang = float(np.clip(np.dot(za, zb), -1.0, 1.0))
            ang = np.arccos(cosang) / np.pi  # [0, 1]
            return float(pos_weight * dp + dir_weight * ang)

        # For each candidate, keep its nearest distance to selected/existing views.
        min_dists = np.full((n,), np.inf, dtype=np.float64)

        # Initialize with existing views so candidates too close to old views are discouraged.
        if len(ref_feats) > 0:
            for i in range(n):
                dmin = np.inf
                for rf in ref_feats:
                    d = pair_dist(cand_feats[i], rf)
                    if d < dmin:
                        dmin = d
                min_dists[i] = dmin
        else:
            # Without existing views, choosing a representative first point would also work;
            # here we use the simplest policy: pick the first one, then expand by FPS.
            min_dists[:] = np.inf

        selected_indices = []

        # Select the first point.
        if len(ref_feats) > 0:
            # If existing views are present, start with the candidate farthest from them.
            first_idx = int(np.argmax(min_dists))
        else:
            first_idx = 0

        selected_indices.append(first_idx)

        # Update nearest distances using the first selected point.
        for i in range(n):
            d = pair_dist(cand_feats[i], cand_feats[first_idx])
            if d < min_dists[i]:
                min_dists[i] = d
        min_dists[first_idx] = -1.0  # Mark as selected.

        # Select the remaining k-1 views: each time choose the max-min-distance candidate.
        while len(selected_indices) < k:
            next_idx = int(np.argmax(min_dists))
            if min_dists[next_idx] < 0:
                break
            selected_indices.append(next_idx)

            for i in range(n):
                if min_dists[i] < 0:
                    continue
                d = pair_dist(cand_feats[i], cand_feats[next_idx])
                if d < min_dists[i]:
                    min_dists[i] = d
            min_dists[next_idx] = -1.0

        # Keep the output close to trajectory order for smoother rendering sequences.
        selected_indices = sorted(selected_indices)

        selected = [candidates[i] for i in selected_indices]
        print(f"[DIVERSE-SAMPLE] Select {len(selected)} / {len(candidates)} "
            f"(k={k}, pos_w={pos_weight}, dir_w={dir_weight})")
        return selected
        
    # -------------------------
    # Commit generated views (compute -> prune -> append -> draw once)
    # -------------------------
    def _commit_generated_views(self, planned, source_tag="interp"):
        if len(planned) == 0:
            print(f"[{source_tag}] No planned views.")
            return []

        # 1) Prune redundancy inside the newly generated candidates.
        planned = self.prune_redundant_views(planned)

        # 2) Remove near-duplicates against already-added viewpoints.
        planned = self._prune_against_existing(planned)

        if len(planned) == 0:
            print(f"[{source_tag}] All planned views removed by pruning.")
            return []

        # 3) Enforce max_images with diversity sampling instead of naive truncation.
        n_can_add = self.remaining_slots()
        if n_can_add <= 0:
            print(f"[{source_tag}] max_images reached, cannot append.")
            return []

        if len(planned) > n_can_add:
            print(f"[{source_tag}] Too many planned views: {len(planned)} > {n_can_add}. "
                f"Use diversity sampling instead of truncation.")

            planned = self._select_diverse_views(
                candidates=planned,
                k=n_can_add,
                existing_views=self.viewpoints,   # Use old views as references to avoid duplicates.
                pos_weight=1.0,
                dir_weight=0.35
            )

            # Optionally prune again to remove rare residual near-duplicates.
            planned = self.prune_redundant_views(planned)
            if len(planned) > n_can_add:
                # In rare cases, enforce the hard limit after pruning.
                planned = planned[:n_can_add]

        # 4) Append once and draw markers in one batch.
        self.viewpoints.extend(planned)
        self.add_markers_batch(planned)

        print(f"[{source_tag}] Appended {len(planned)} views. Total viewpoints: {len(self.viewpoints)}")
        return planned

    # -------------------------
    # Interpolation core
    # -------------------------
    def _generate_views_from_pose_sequence_linear_candidates(self, pose_seq, pos_step, source="interp"):
        """Generate candidate viewpoints only; do not append or draw markers."""
        if len(pose_seq) < 2:
            print("[INTERP] Need at least 2 poses.")
            return []

        if pos_step is None or pos_step <= 0:
            pos_step = 0.05

        planned = []
        K = self.K_user.copy()

        for i in range(len(pose_seq) - 1):
            T0 = pose_seq[i]["extrinsic"]
            T1 = pose_seq[i + 1]["extrinsic"]

            p0, R0 = decompose_extrinsic_w2c(T0)
            p1, R1 = decompose_extrinsic_w2c(T1)

            seg_len = float(np.linalg.norm(p1 - p0))
            if seg_len < 1e-9:
                n_views = self.interp_min_views_per_segment
            else:
                n_views = int(np.ceil(seg_len / float(pos_step))) + 1
                n_views = max(n_views, self.interp_min_views_per_segment)

            j_start = 0 if i == 0 else 1
            for j in range(j_start, n_views):
                t = 0.0 if n_views == 1 else (j / float(n_views - 1))
                p = (1.0 - t) * p0 + t * p1
                R = self._interp_rotation_c2w_slerp(R0, R1, t)
                T = compose_extrinsic_from_cam_pose(p, R)
                vp_new = {"K": K.copy(), "extrinsic": T, "source": source}
                planned.append(vp_new)

        return planned

    def _generate_views_from_pose_sequence_curved_candidates(self, pose_seq, pos_step, source="interp_curve"):
        """
        Generate candidate viewpoints with Catmull-Rom position interpolation.
        - Falls back to linear interpolation when fewer than 3 poses are available.
        - Uses SLERP to interpolate rotations between key poses.
        """
        if len(pose_seq) < 2:
            print("[INTERP] Need at least 2 poses.")
            return []

        if pos_step is None or pos_step <= 0:
            pos_step = 0.05

        if len(pose_seq) < 3:
            print("[INTERP] Fewer than 3 poses, fallback to linear interpolation.")
            return self._generate_views_from_pose_sequence_linear_candidates(
                pose_seq, pos_step=pos_step, source=source
            )

        planned = []
        K = self.K_user.copy()

        poses = []
        for item in pose_seq:
            p, R = decompose_extrinsic_w2c(item["extrinsic"])
            poses.append((p, orthonormalize_rotation(R)))

        n = len(poses)
        for i in range(n - 1):
            p1, R1 = poses[i]
            p2, R2 = poses[i + 1]

            if i - 1 >= 0:
                p0 = poses[i - 1][0]
            else:
                p0 = p1 + (p1 - p2)

            if i + 2 < n:
                p3 = poses[i + 2][0]
            else:
                p3 = p2 + (p2 - p1)

            seg_len = self._estimate_curve_segment_length(
                p0, p1, p2, p3, alpha=self.curve_alpha, n_samples=self.curve_samples_per_seg
            )
            if seg_len < 1e-9:
                n_views = self.interp_min_views_per_segment
            else:
                n_views = int(np.ceil(seg_len / float(pos_step))) + 1
                n_views = max(n_views, self.interp_min_views_per_segment)

            j_start = 0 if i == 0 else 1
            for j in range(j_start, n_views):
                t = 0.0 if n_views == 1 else (j / float(n_views - 1))

                # Position: multi-point fitted Catmull-Rom curve.
                p = self._catmull_rom_point(p0, p1, p2, p3, t, alpha=self.curve_alpha)

                # Rotation: SLERP between neighboring key poses.
                R = self._interp_rotation_c2w_slerp(R1, R2, t)

                T = compose_extrinsic_from_cam_pose(p, R)
                vp_new = {"K": K.copy(), "extrinsic": T, "source": source}
                planned.append(vp_new)

        return planned

    def generate_views_from_trajectory(self, append=True):
        if len(self.trajectory_keyframes) < 2:
            print("[G] Need at least 2 trajectory keyframes.")
            return []

        planned = self._generate_views_from_pose_sequence_curved_candidates(
            self.trajectory_keyframes,
            pos_step=self.traj_interp_pos_step,
            source="traj_curve_interp"
        )

        if append:
            planned = self._commit_generated_views(planned, source_tag="G")

        print(f"[G] Generated {len(planned)} curved views from trajectory ({len(self.trajectory_keyframes)} keyframes).")
        print(f"[G] Total viewpoints: {len(self.viewpoints)}")
        return planned

    def generate_views_from_manual_viewpoints(self, append=True):
        """Generate new viewpoints by interpolating manually added Space-key viewpoints."""
        if len(self.manual_viewpoints) < 2:
            print("[I] Need at least 2 manually added viewpoints (Space) to interpolate.")
            return []

        planned = self._generate_views_from_pose_sequence_curved_candidates(
            self.manual_viewpoints,
            pos_step=self.manual_interp_pos_step,
            source="manual_curve_interp"
        )

        if append:
            planned = self._commit_generated_views(planned, source_tag="I")

        print(f"[I] Generated {len(planned)} curved views from manual viewpoints ({len(self.manual_viewpoints)} manual poses).")
        print(f"[I] Total viewpoints: {len(self.viewpoints)}")
        return planned

    # -------------------------
    # Trajectory keyframes
    # -------------------------
    def add_trajectory_keyframe(self):
        _, T_w2c = self.get_current_camera()
        self.trajectory_keyframes.append({
            "K": self.K_user.copy(),
            "extrinsic": T_w2c.copy(),
            "source": "traj_kf"
        })
        idx = len(self.trajectory_keyframes) - 1
        c = get_camera_center_from_extrinsic(T_w2c)
        print(f"[O] Trajectory keyframe #{idx} recorded | cam_center = {c.round(4).tolist()}")

    def undo_trajectory_keyframe(self):
        if len(self.trajectory_keyframes) == 0:
            print("[N] No trajectory keyframe to undo.")
            return False
        self.trajectory_keyframes.pop(-1)
        print(f"[N] Removed last trajectory keyframe. Remaining: {len(self.trajectory_keyframes)}")
        return True

    def clear_trajectory_keyframes(self):
        self.trajectory_keyframes = []
        print("[M] Cleared all trajectory keyframes.")

    def print_trajectory_state(self):
        print("\n[TRAJECTORY STATE]")
        print(f"  keyframes: {len(self.trajectory_keyframes)}")
        print(f"  traj_interp_pos_step: {self.traj_interp_pos_step}")
        print(f"  manual_interp_pos_step: {self.manual_interp_pos_step}")
        print(f"  max_images: {self.max_images if self.max_images > 0 else 'unlimited'}")
        print(f"  viewpoints(total): {len(self.viewpoints)}")
        print(f"  viewpoints(manual): {len(self.manual_viewpoints)}")
        print(f"  prune_pos_thresh_ratio: {self.prune_pos_thresh_ratio}")
        print(f"  prune_angle_thresh_deg: {self.prune_angle_thresh_deg}")
        if len(self.trajectory_keyframes) > 0:
            start = max(0, len(self.trajectory_keyframes) - 5)
            for i in range(start, len(self.trajectory_keyframes)):
                c = get_camera_center_from_extrinsic(self.trajectory_keyframes[i]["extrinsic"])
                print(f"    traj_kf #{i}: {c.round(4).tolist()}")
        if len(self.manual_viewpoints) > 0:
            start = max(0, len(self.manual_viewpoints) - 5)
            for i in range(start, len(self.manual_viewpoints)):
                c = get_camera_center_from_extrinsic(self.manual_viewpoints[i]["extrinsic"])
                print(f"    manual #{i}: {c.round(4).tolist()}")
        print("")

    # -------------------------
    # UI / callbacks
    # -------------------------
    def setup_visualizer(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            "Manual viewpoints + trajectory interpolation",
            width=self.width, height=self.height, visible=True
        )
        self.vis.add_geometry(self.mesh)

        if self.obb is not None:
            try:
                self.vis.add_geometry(self.obb, reset_bounding_box=False)
            except TypeError:
                self.vis.add_geometry(self.obb)

        render_opt = self.vis.get_render_option()
        render_opt.mesh_show_back_face = True
        render_opt.light_on = True

        # Manual viewpoint management.
        self.vis.register_key_callback(32, self.on_add_viewpoint)          # Space
        self.vis.register_key_callback(ord("U"), self.on_undo_viewpoint)
        self.vis.register_key_callback(ord("C"), self.on_clear_viewpoints)
        self.vis.register_key_callback(ord("X"), self.on_delete_nearest_viewpoint)
        self.vis.register_key_callback(ord("R"), self.on_render_all)
        self.vis.register_key_callback(ord("H"), self.on_help)

        # Trajectory keyframes and trajectory-based generation.
        self.vis.register_key_callback(ord("O"), self.on_record_trajectory_keyframe)
        self.vis.register_key_callback(ord("N"), self.on_undo_trajectory_keyframe)
        self.vis.register_key_callback(ord("M"), self.on_clear_trajectory_keyframes)
        self.vis.register_key_callback(ord("G"), self.on_generate_views_from_trajectory)
        self.vis.register_key_callback(ord("B"), self.on_print_trajectory_state)

        # Generate views from manually selected control poses.
        self.vis.register_key_callback(ord("I"), self.on_generate_views_from_manual_viewpoints)

        # Arrow keys: translation only.
        try:
            self.vis.register_key_callback(262, self.on_key_right_arrow)   # Right
            self.vis.register_key_callback(263, self.on_key_left_arrow)    # Left
            self.vis.register_key_callback(264, self.on_key_down_arrow)    # Down
            self.vis.register_key_callback(265, self.on_key_up_arrow)      # Up
            self.vis.register_key_callback(266, self.on_key_page_up)       # PageUp
            self.vis.register_key_callback(267, self.on_key_page_down)     # PageDown
            self.arrow_key_registered = True
        except Exception:
            self.arrow_key_registered = False

        # Fallback letter keys for translation.
        self.vis.register_key_callback(ord("W"), self.on_move_forward)
        self.vis.register_key_callback(ord("S"), self.on_move_backward)
        self.vis.register_key_callback(ord("A"), self.on_move_left)        # Move left.
        self.vis.register_key_callback(ord("D"), self.on_move_right)
        self.vis.register_key_callback(ord("Q"), self.on_move_up_world)
        self.vis.register_key_callback(ord("E"), self.on_move_down_world)

        self.apply_user_intrinsics()

        print("\n[Controls]")
        print("  Left mouse drag: rotate the view")
        print("  Middle mouse / Shift+left mouse: pan")
        print("  Mouse wheel: zoom")
        print("")
        print("  Space: add the current camera as a manual viewpoint")
        print("  U: undo the last viewpoint")
        print("  C: clear all viewpoints")
        print("  X: delete the viewpoint nearest to the current camera")
        print("  R: render and save all viewpoints using a hidden render window")
        print("")
        print("  O: record a trajectory keyframe")
        print("  N: undo the last trajectory keyframe")
        print("  M: clear trajectory keyframes")
        print("  G: generate viewpoints from trajectory keyframes with a fitted curve")
        print("  I: generate viewpoints from manual viewpoints with a fitted curve")
        print("  B: print trajectory/viewpoint status")
        print("")
        print("  Arrow keys: Up forward / Down backward / Left left / Right right")
        print("  Letter keys: W/S forward-backward, A/D left-right, Q/E up-down")
        print("")
        print(f"  max_images: {self.max_images if self.max_images > 0 else 'unlimited'}")
        print(f"  max_invalid_depth_ratio: {self.max_invalid_depth_ratio}")
        print("  H: help")
        print("  Close the window to exit\n")

    # ---------- Manual viewpoint callbacks ----------
    def on_add_viewpoint(self, vis):
        _, T_w2c = self.get_current_camera()
        ok = self.append_viewpoint(self.K_user.copy(), T_w2c, source="manual", draw_marker=True, verbose=True)
        if ok:
            self.vis.poll_events()
            self.vis.update_renderer()
        return False

    def on_undo_viewpoint(self, vis):
        if not self.viewpoints:
            print("[U] No viewpoint to remove.")
            return False

        removed = self.viewpoints.pop(-1)

        if removed.get("source") == "manual" and len(self.manual_viewpoints) > 0:
            # Manual Space-key viewpoints are appended at the end, so remove the last one.
            self.manual_viewpoints.pop(-1)

        self.rebuild_markers()
        print(f"[U] Removed last viewpoint. Remaining: {len(self.viewpoints)}")
        return False

    def on_clear_viewpoints(self, vis):
        self.viewpoints = []
        self.manual_viewpoints = []
        self.rebuild_markers()
        print("[C] All viewpoints removed (manual/interpolated all cleared).")
        return False

    def on_delete_nearest_viewpoint(self, vis):
        self.delete_nearest_viewpoint_to_current_camera()
        return False

    # ---------- Keyboard move callbacks (translation only) ----------
    def on_key_up_arrow(self, vis):
        self._move_camera_local(dz=+self.move_step)
        return False

    def on_key_down_arrow(self, vis):
        self._move_camera_local(dz=-self.move_step)
        return False

    def on_key_left_arrow(self, vis):
        self._move_camera_local(dx=-self.move_step)
        return False

    def on_key_right_arrow(self, vis):
        self._move_camera_local(dx=+self.move_step)
        return False

    def on_key_page_up(self, vis):
        self._move_camera_world_up(+self.move_step)
        return False

    def on_key_page_down(self, vis):
        self._move_camera_world_up(-self.move_step)
        return False

    # Fallback letter-key movement callbacks.
    def on_move_forward(self, vis):
        self._move_camera_local(dz=+self.move_step)
        return False

    def on_move_backward(self, vis):
        self._move_camera_local(dz=-self.move_step)
        return False

    def on_move_left(self, vis):
        self._move_camera_local(dx=-self.move_step)
        return False

    def on_move_right(self, vis):
        self._move_camera_local(dx=+self.move_step)
        return False

    def on_move_up_world(self, vis):
        self._move_camera_world_up(+self.move_step)
        return False

    def on_move_down_world(self, vis):
        self._move_camera_world_up(-self.move_step)
        return False

    # ---------- Trajectory callbacks ----------
    def on_record_trajectory_keyframe(self, vis):
        self.add_trajectory_keyframe()
        return False

    def on_undo_trajectory_keyframe(self, vis):
        self.undo_trajectory_keyframe()
        return False

    def on_clear_trajectory_keyframes(self, vis):
        self.clear_trajectory_keyframes()
        return False

    def on_generate_views_from_trajectory(self, vis):
        self.generate_views_from_trajectory(append=True)
        return False

    def on_generate_views_from_manual_viewpoints(self, vis):
        self.generate_views_from_manual_viewpoints(append=True)
        return False

    def on_print_trajectory_state(self, vis):
        self.print_trajectory_state()
        return False

    def on_help(self, vis):
        print("\n[Help]")
        print("  Curve-based view generation uses Catmull-Rom for position and SLERP for rotation.")
        print("  Generated views are computed first and drawn in batches to avoid UI stalls.")
        print("  Space adds a manual view; O records a keyframe; G/I generates views; R renders.\n")
        return False

    # -------------------------
    # Save / render
    # -------------------------
    def _save_depth(self, depth_float, base_name):
        """Save raw depth as a single-channel float32 EXR file."""
        depth = np.asarray(depth_float, dtype=np.float32).copy()
        invalid = ~np.isfinite(depth)
        depth[invalid] = 0.0

        exr_path = Path(self.out_dir) / "depth" / f"{base_name}.exr"

        ok = cv2.imwrite(str(exr_path), depth)
        if not ok:
            raise RuntimeError(
                f"Failed to write EXR: {exr_path}. "
                f"Please ensure OpenCV is built with OpenEXR support."
            )

        # Optional normalized PNG preview for quick visual inspection.
        valid_vals = depth[depth > 0]
        if len(valid_vals) > 0:
            dmax = np.percentile(valid_vals, 99.5)
            dmax = max(dmax, 1e-6)
        else:
            dmax = 1.0
        depth_norm = np.clip(depth / dmax, 0, 1)
        depth_u16 = (depth_norm * 65535).astype(np.uint16)
        png_path = Path(self.out_dir) / "depth_png" / f"{base_name}.png"
        imageio.imwrite(png_path, depth_u16)

        return exr_path

    def save_viewpoint_list(self):
        serializable = []
        for i, vp in enumerate(self.viewpoints):
            serializable.append({
                "id": i,
                "source": vp.get("source", "unknown"),
                "width": self.width,
                "height": self.height,
                "intrinsic_matrix_3x3": vp["K"].tolist(),
                "extrinsic_world_to_camera_4x4": vp["extrinsic"].tolist()
            })

        path = Path(self.out_dir) / "viewpoints" / "selected_viewpoints.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved viewpoint list: {path}")

    def _compute_invalid_depth_ratio(self, depth_float):
        depth = np.asarray(depth_float, dtype=np.float32)
        invalid = (~np.isfinite(depth)) | (depth <= 0)
        return float(np.mean(invalid))

    def render_all_views(self):
        if len(self.viewpoints) == 0:
            print("[WARN] No viewpoints selected. Press Space / G / I first.")
            return

        n_render = len(self.viewpoints)
        if self.max_images > 0:
            n_render = min(n_render, self.max_images)

        self.save_viewpoint_list()
        print(f"[INFO] Start batch rendering {n_render} / {len(self.viewpoints)} viewpoints ...")

        # Use a hidden render window; the main UI should not redraw markers repeatedly.
        render_vis = o3d.visualization.Visualizer()
        render_vis.create_window(
            window_name="HiddenRenderer",
            width=self.width,
            height=self.height,
            visible=False
        )
        render_vis.add_geometry(self.mesh)

        render_opt = render_vis.get_render_option()
        render_opt.mesh_show_back_face = True
        render_opt.light_on = True

        vc = render_vis.get_view_control()

        out_dir = Path(self.out_dir)
        images_dir = out_dir / "images"
        cams_dir = out_dir / "cams"

        saved_count = 0
        skipped_count = 0

        for idx in range(n_render):
            vp = self.viewpoints[idx]
            K = vp["K"]
            T_w2c = vp["extrinsic"]

            cam = vc.convert_to_pinhole_camera_parameters()
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(
                self.width, self.height,
                float(K[0, 0]), float(K[1, 1]),
                float(K[0, 2]), float(K[1, 2])
            )
            cam.intrinsic = intrinsic
            cam.extrinsic = T_w2c

            try:
                vc.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
            except TypeError:
                vc.convert_from_pinhole_camera_parameters(cam)
            except Exception as e:
                print(f"[ERROR] Failed to set camera for view #{idx}: {e}")
                continue

            render_vis.poll_events()
            render_vis.update_renderer()
            time.sleep(0.02)

            # Capture depth first and skip invalid frames before writing files.
            depth_float = np.asarray(render_vis.capture_depth_float_buffer(do_render=True), dtype=np.float32)
            invalid_ratio = self._compute_invalid_depth_ratio(depth_float)

            if invalid_ratio > self.max_invalid_depth_ratio:
                print(f"[SKIP] #{idx} ({vp.get('source', 'unknown')}) "
                      f"invalid_depth_ratio={invalid_ratio:.3f} > {self.max_invalid_depth_ratio:.3f}")
                skipped_count += 1
                continue

            base_name = f"{saved_count:06d}"

            # RGB -> images/*.png after the depth-validity filter.
            rgb_float = np.asarray(render_vis.capture_screen_float_buffer(do_render=True))
            rgb_u8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
            rgb_path = images_dir / f"{base_name}.png"
            imageio.imwrite(rgb_path, rgb_u8)

            # Depth -> depth/*.exr
            depth_exr_path = self._save_depth(depth_float, base_name)

            # Cam -> cams/*.txt
            cam_txt_path = cams_dir / f"{base_name}.txt"
            write_cam_file(
                cam_path=cam_txt_path,
                extrinsic=T_w2c,
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                h=int(self.height),
                w=int(self.width),
            )

            print(f"[RENDERED] #{idx} ({vp.get('source', 'unknown')}) "
                  f"invalid_depth_ratio={invalid_ratio:.3f} "
                  f"-> image={rgb_path.name}, depth={depth_exr_path.name}, cam={cam_txt_path.name}")
            saved_count += 1

        render_vis.destroy_window()
        self.rendered = True
        print(f"[INFO] Batch rendering completed. saved={saved_count}, skipped={skipped_count}")

    def on_render_all(self, vis):
        # Keep the main window alive; rendering happens in a hidden window.
        self.render_all_views()
        # Refresh the main window once after rendering without rebuilding markers.
        self.vis.poll_events()
        self.vis.update_renderer()
        return False

    # -------------------------
    # Main loop
    # -------------------------
    def run(self):
        self.load_mesh()
        self.setup_visualizer()
        self.vis.run()
        self.vis.destroy_window()



def positive_int(value):
    """Argparse type helper: parse a strictly positive integer."""
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer, got {value!r}.") from exc
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {ivalue}.")
    return ivalue


def nonnegative_int(value):
    """Argparse type helper: parse a non-negative integer."""
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer, got {value!r}.") from exc
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative integer, got {ivalue}.")
    return ivalue


def positive_float(value):
    """Argparse type helper: parse a strictly positive float."""
    try:
        fvalue = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a number, got {value!r}.") from exc
    if not np.isfinite(fvalue) or fvalue <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive finite number, got {fvalue}.")
    return fvalue


def ratio_float(value):
    """Argparse type helper: parse a float ratio in [0, 1]."""
    try:
        fvalue = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a number, got {value!r}.") from exc
    if not np.isfinite(fvalue) or fvalue < 0.0 or fvalue > 1.0:
        raise argparse.ArgumentTypeError(f"Expected a ratio in [0, 1], got {fvalue}.")
    return fvalue


def build_arg_parser():
    """Create a standardized command-line interface for the renderer."""
    parser = argparse.ArgumentParser(
        description=(
            "Interactively select camera viewpoints for a mesh, optionally generate "
            "interpolated viewpoints, and batch-render RGB images, EXR depth maps, "
            "and camera-parameter files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required scene/output inputs.
    parser.add_argument(
        "--mesh",
        type=str,
        default="models/vatican-city-state-rome-italy/Sketchfab_2021_11_26_18_02_24.obj",
        help="Path to the input mesh file, usually .obj/.ply/.glb depending on Open3D support.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./captures/vatican-city",
        help="Output directory. Subdirectories images/depth/depth_png/cams/viewpoints are created inside it.",
    )

    # Image size.
    parser.add_argument("--width", type=positive_int, default=1024, help="Render width in pixels.")
    parser.add_argument("--height", type=positive_int, default=720, help="Render height in pixels.")

    # Camera intrinsics. Prefer explicit fx/fy. --focal is a convenience alias for square pixels.
    parser.add_argument("--fx", type=positive_float, default=1200.0, help="Focal length fx in pixels.")
    parser.add_argument("--fy", type=positive_float, default=None, help="Focal length fy in pixels. Defaults to fx.")
    parser.add_argument("--focal", type=positive_float, default=None, help="Set both fx and fy to the same value.")
    parser.add_argument("--cx", type=float, default=None, help="Principal point cx in pixels. Defaults to width / 2.")
    parser.add_argument("--cy", type=float, default=None, help="Principal point cy in pixels. Defaults to height / 2.")

    # Optional randomized focal length. This replaces the old ambiguous --mf/--Mf behavior.
    parser.add_argument(
        "--random_focal",
        action="store_true",
        help="Randomly sample an integer focal length and set fx=fy to that value.",
    )
    parser.add_argument("--focal_min", "--mf", type=positive_float, default=1200.0,
                        help="Minimum focal length for --random_focal. --mf is kept as a backward-compatible alias.")
    parser.add_argument("--focal_max", "--Mf", type=positive_float, default=2000.0,
                        help="Maximum focal length for --random_focal. --Mf is kept as a backward-compatible alias.")
    parser.add_argument("--random_seed", type=int, default=None, help="Optional random seed for reproducible focal sampling.")
    parser.add_argument(
        "--append_focal_to_out_dir",
        action="store_true",
        help="Append the selected focal length to the output directory name, matching the original script behavior.",
    )

    # Visualization and filtering parameters.
    parser.add_argument("--frustum_ratio", type=positive_float, default=0.02,
                        help="Frustum visualization length as a ratio of scene scale.")
    parser.add_argument("--axis_ratio", type=positive_float, default=0.01,
                        help="Camera-axis visualization size as a ratio of scene scale.")
    parser.add_argument("--max_images", type=nonnegative_int, default=500,
                        help="Maximum total viewpoints/images to keep and render. 0 means unlimited.")
    parser.add_argument("--max_invalid_depth_ratio", type=ratio_float, default=0.5,
                        help="Skip a rendered frame when its invalid-depth-pixel ratio is above this threshold.")

    return parser


def normalize_args(args, parser):
    """
    Normalize and validate command-line inputs.

    The original script used --mf/--Mf and always sampled a random focal length.
    This version makes the default deterministic and requires --random_focal when
    randomized intrinsics are desired.
    """
    mesh_path = Path(args.mesh).expanduser()
    if not mesh_path.exists():
        parser.error(f"Input mesh does not exist: {mesh_path}")
    if not mesh_path.is_file():
        parser.error(f"Input mesh is not a file: {mesh_path}")

    if args.random_focal:
        if args.focal_min > args.focal_max:
            parser.error("--focal_min must be <= --focal_max when --random_focal is used.")
        import random
        if args.random_seed is not None:
            random.seed(args.random_seed)
        focal = float(random.randint(int(round(args.focal_min)), int(round(args.focal_max))))
        fx = focal
        fy = focal
    elif args.focal is not None:
        fx = float(args.focal)
        fy = float(args.focal)
    else:
        fx = float(args.fx)
        fy = float(args.fy) if args.fy is not None else fx

    if args.cx is not None and (not np.isfinite(args.cx)):
        parser.error("--cx must be finite when provided.")
    if args.cy is not None and (not np.isfinite(args.cy)):
        parser.error("--cy must be finite when provided.")

    out_dir = Path(args.out_dir).expanduser()
    if args.append_focal_to_out_dir:
        focal_tag = int(round(fx)) if abs(fx - round(fx)) < 1e-9 else fx
        out_dir = Path(f"{out_dir}_{focal_tag}")

    args.mesh = str(mesh_path)
    args.out_dir = str(out_dir)
    args.fx = fx
    args.fy = fy
    return args


def main():
    parser = build_arg_parser()
    args = normalize_args(parser.parse_args(), parser)

    print("[CONFIG] Normalized inputs")
    print(f"  mesh: {args.mesh}")
    print(f"  out_dir: {args.out_dir}")
    print(f"  size: {args.width} x {args.height}")
    print(f"  intrinsics: fx={args.fx}, fy={args.fy}, cx={args.cx if args.cx is not None else 'width/2'}, cy={args.cy if args.cy is not None else 'height/2'}")
    print(f"  max_images: {args.max_images if args.max_images > 0 else 'unlimited'}")
    print(f"  max_invalid_depth_ratio: {args.max_invalid_depth_ratio}")

    app = ViewSelectorAndRenderer(
        mesh_path=args.mesh,
        out_dir=args.out_dir,
        width=args.width,
        height=args.height,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        frustum_ratio=args.frustum_ratio,
        axis_ratio=args.axis_ratio,
        max_images=args.max_images,
        max_invalid_depth_ratio=args.max_invalid_depth_ratio,
    )
    app.run()


if __name__ == "__main__":
    main()
