"""Controlled hFOV-height renderer for the A3D-FA diagnostic split.

The operator selects a reference viewpoint/trajectory once. The renderer then
derives multiple target horizontal-FOV settings and adjusts camera distance so
the observed footprint remains approximately comparable across settings.
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

# Enable OpenEXR writing in OpenCV. Your OpenCV build must include OpenEXR support.
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
    # Optional depth preview images. You may delete this folder if you do not need it.
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


def calculate_hfov(fx, w):
    """Return the horizontal field of view in degrees."""
    fx = float(fx)
    w = float(w)
    return float(np.degrees(2.0 * np.arctan(w / (2.0 * fx))))


def focal_from_hfov(hfov_deg, width):
    hfov_deg = float(hfov_deg)
    width = float(width)
    if not (0.0 < hfov_deg < 179.0):
        raise ValueError(f"hfov_deg must be in (0, 179), got {hfov_deg}")
    return float(width / (2.0 * np.tan(np.radians(hfov_deg) / 2.0)))


def parse_float_list(text_or_seq):
    if isinstance(text_or_seq, (list, tuple)):
        return [float(x) for x in text_or_seq]
    if text_or_seq is None:
        return []
    vals = []
    for part in str(text_or_seq).split(','):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def write_cam_file(cam_path: Path, extrinsic, fx, fy, cx, cy, h, w):
    """
    Write one camera file with:
    - OpenCV-style world-to-camera extrinsic matrix
    - 3x3 intrinsic matrix
    - image height, image width, and vertical field of view
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
                 max_invalid_depth_ratio=0.3,
                 reference_height=10.0,
                 reference_hfov=55.0,
                 target_hfovs=None,
                 render_multi_hfov=True):
        self.mesh_path = mesh_path
        self.out_dir = out_dir
        self.width = width
        self.height = height
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx) if cx is not None else width / 2.0
        self.cy = float(cy) if cy is not None else height / 2.0
        self.max_images = int(max_images)  # <= 0 means unlimited

        self.max_invalid_depth_ratio = float(np.clip(max_invalid_depth_ratio, 0.0, 1.0))

        self.reference_height = float(reference_height)
        if self.reference_height <= 0:
            raise ValueError(f"reference_height must be > 0, got {self.reference_height}")

        self.reference_hfov = float(reference_hfov)
        if not (0.0 < self.reference_hfov < 179.0):
            raise ValueError(f"reference_hfov must be in (0, 179), got {self.reference_hfov}")

        if target_hfovs is None:
            target_hfovs = [25, 35, 45, 55, 65, 75, 85, 95]
        self.target_hfovs = sorted(set(parse_float_list(target_hfovs)))
        self.render_multi_hfov = bool(render_multi_hfov)

        self.K_user = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.frustum_ratio = float(frustum_ratio)
        self.axis_ratio = float(axis_ratio)

        self.mesh = None
        self.vis = None

        # All viewpoints used for final rendering: manual, trajectory-generated, and manual-pose interpolation.
        self.viewpoints = []
        # Viewpoints explicitly added with Space. These are used for manual-pose interpolation.
        self.manual_viewpoints = []
        # Trajectory keyframes recorded with O.
        self.trajectory_keyframes = []

        self.marker_geometries = []
        self.rendered = False

        # Estimated scene scale.
        self.obb = None
        self.obb_extent = None
        self.obb_R = None
        self.world_up = None

        # Step-size parameters.
        self.move_step = None
        self.traj_interp_pos_step = None
        self.manual_interp_pos_step = None
        self.interp_min_views_per_segment = 2

        # Curved trajectory parameters.
        self.curve_alpha = 0.5   # centripetal Catmull-Rom
        self.curve_samples_per_seg = 30  # Number of samples per segment for approximate arc-length estimation.

        # Redundancy-pruning thresholds. Tune these if needed.
        self.prune_pos_thresh_ratio = 0.01
        self.prune_angle_thresh_deg = 3.0

        # Deferred marker cache to avoid UI stalls while generating views.
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
            print(f"{prefix}[LIMIT] Reached the maximum image limit max_images={self.max_images}")
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

        # Visualization sizes are scaled relative to the scene.
        self.frustum_len = max(scene_scale * self.frustum_ratio, 1e-4)
        self.axis_size = max(scene_scale * self.axis_ratio, 1e-4)

        # Camera translation step.
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
        print(f"       reference_height: {self.reference_height:.4f}")
        print(f"       reference_hfov: {self.reference_hfov:.4f}")
        print(f"       target_hfovs: {self.target_hfovs}")
        print(f"       render_multi_hfov: {self.render_multi_hfov}")

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
        """Draw multiple viewpoint markers at once to avoid UI stalls."""
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

        # If a manual viewpoint is deleted, remove the closest matching pose from manual_viewpoints too.
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
        dx: camera-right direction
        dy: camera-down direction, matching the OpenCV camera y-axis
        dz: camera-forward direction
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
    # Multi-hFOV scale compensation
    # -------------------------
    def _build_intrinsic_from_hfov(self, hfov_deg):
        fx = focal_from_hfov(hfov_deg, self.width)
        fy = fx
        K = np.array([
            [fx, 0.0, self.cx],
            [0.0, fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return K

    def _calc_distance_for_hfov(self, hfov_deg):
        ref_half = np.radians(self.reference_hfov) / 2.0
        tgt_half = np.radians(float(hfov_deg)) / 2.0
        return float(self.reference_height * np.tan(ref_half) / np.tan(tgt_half))

    def _retarget_pose_for_hfov(self, T_w2c_ref, hfov_deg):
        cam_pos_ref, R_c2w = decompose_extrinsic_w2c(T_w2c_ref)
        R_c2w = orthonormalize_rotation(R_c2w)
        forward_axis = R_c2w[:, 2]
        forward_axis, _ = safe_norm(forward_axis)
        if forward_axis is None:
            forward_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        ref_distance = float(self.reference_height)
        tgt_distance = self._calc_distance_for_hfov(hfov_deg)

        # Use a virtual anchor point reference_height units in front of the reference camera.
        # Retarget each hFOV along the camera-forward axis so the central target coverage stays approximately consistent.
        anchor_point = cam_pos_ref + forward_axis * ref_distance
        cam_pos_tgt = anchor_point - forward_axis * tgt_distance
        T_w2c_tgt = compose_extrinsic_from_cam_pose(cam_pos_tgt, R_c2w)
        return T_w2c_tgt, tgt_distance

    def build_scaled_viewpoints_for_hfov(self, hfov_deg):
        hfov_deg = float(hfov_deg)
        K = self._build_intrinsic_from_hfov(hfov_deg)
        scaled_viewpoints = []

        for idx, vp in enumerate(self.viewpoints):
            T_new, tgt_distance = self._retarget_pose_for_hfov(vp["extrinsic"], hfov_deg)
            scaled_viewpoints.append({
                "K": K.copy(),
                "extrinsic": T_new,
                "source": f"{vp.get('source', 'unknown')}_hfov_{int(round(hfov_deg))}",
                "ref_view_idx": idx,
                "hfov": hfov_deg,
                "camera_distance": tgt_distance,
            })
        return scaled_viewpoints

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

        # Use the shortest rotation path.
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)

        # Use a linear approximation when the two quaternions are very close.
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
        Remove redundant viewpoints using:
            1) camera-center distance
            2) viewing-direction angle, based on the camera z-axis
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
        """Remove candidates that are too close to existing self.viewpoints."""
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
        Extract compact viewpoint features:
        - camera center p, shape (3,)
        - viewing direction, the OpenCV camera-forward z-axis, shape (3,)
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
        Select k diverse viewpoints from candidates using position and orientation.
        - Similar to farthest point sampling (FPS).
        - existing_views acts as an occupied set so new views avoid existing ones.

        Larger distance means more diversity:
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

        # For each candidate, track its nearest distance to the selected/existing set.
        min_dists = np.full((n,), np.inf, dtype=np.float64)

        # Initialize with existing_views to avoid selecting poses that are too similar.
        if len(ref_feats) > 0:
            for i in range(n):
                dmin = np.inf
                for rf in ref_feats:
                    d = pair_dist(cand_feats[i], rf)
                    if d < dmin:
                        dmin = d
                min_dists[i] = dmin
        else:
            # With no existing viewpoints, a central/representative first point would also work.
            # This implementation uses the simplest strategy: pick the first candidate, then expand with FPS.
            min_dists[:] = np.inf

        selected_indices = []

        # Select the first point.
        if len(ref_feats) > 0:
            # If existing views are available, select the candidate farthest from them first.
            first_idx = int(np.argmax(min_dists))
        else:
            first_idx = 0

        selected_indices.append(first_idx)

        # Update min_dists using the first selected point.
        for i in range(n):
            d = pair_dist(cand_feats[i], cand_feats[first_idx])
            if d < min_dists[i]:
                min_dists[i] = d
        min_dists[first_idx] = -1.0  # Mark as selected.

        # Select the remaining k-1 points by maximizing distance to the selected set.
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

        # Preserve the original trajectory order for smoother render sequencing.
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

        # 1) Prune redundant newly generated views.
        planned = self.prune_redundant_views(planned)

        # 2) Remove near-duplicates against existing viewpoints.
        planned = self._prune_against_existing(planned)

        if len(planned) == 0:
            print(f"[{source_tag}] All planned views removed by pruning.")
            return []

        # 3) Respect max_images using diversity sampling instead of naive truncation.
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
                existing_views=self.viewpoints,   # Use existing views as references so selected views are not too similar.
                pos_weight=1.0,
                dir_weight=0.35
            )

            # Optional light pruning to remove occasional similar views after sampling.
            planned = self.prune_redundant_views(planned)
            if len(planned) > n_can_add:
                # In rare edge cases, clamp to the hard limit.
                planned = planned[:n_can_add]

        # 4) Append all views and draw markers in one batch.
        self.viewpoints.extend(planned)
        self.add_markers_batch(planned)

        print(f"[{source_tag}] Appended {len(planned)} views. Total viewpoints: {len(self.viewpoints)}")
        return planned

    # -------------------------
    # Interpolation core
    # -------------------------
    def _generate_views_from_pose_sequence_linear_candidates(self, pose_seq, pos_step, source="interp"):
        """Generate candidate viewpoints only; do not append them or draw markers."""
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
        Generate candidate poses with Catmull-Rom position interpolation.
        - Falls back to linear interpolation when fewer than 3 poses are available.
        - Uses SLERP between keyframe rotations.
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

                # Position: multi-point curve interpolation.
                p = self._catmull_rom_point(p0, p1, p2, p3, t, alpha=self.curve_alpha)

                # Rotation: SLERP between keyframe rotations.
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
        """Generate new views by interpolating manually added viewpoints."""
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

        # Trajectory keyframes and generation.
        self.vis.register_key_callback(ord("O"), self.on_record_trajectory_keyframe)
        self.vis.register_key_callback(ord("N"), self.on_undo_trajectory_keyframe)
        self.vis.register_key_callback(ord("M"), self.on_clear_trajectory_keyframes)
        self.vis.register_key_callback(ord("G"), self.on_generate_views_from_trajectory)
        self.vis.register_key_callback(ord("B"), self.on_print_trajectory_state)

        # Generate views from manual-viewpoint pose interpolation.
        self.vis.register_key_callback(ord("I"), self.on_generate_views_from_manual_viewpoints)

        # Arrow-key translation controls only.
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

        # Backup letter-key translation controls.
        self.vis.register_key_callback(ord("W"), self.on_move_forward)
        self.vis.register_key_callback(ord("S"), self.on_move_backward)
        self.vis.register_key_callback(ord("A"), self.on_move_left)        # Move left.
        self.vis.register_key_callback(ord("D"), self.on_move_right)
        self.vis.register_key_callback(ord("Q"), self.on_move_up_world)
        self.vis.register_key_callback(ord("E"), self.on_move_down_world)

        self.apply_user_intrinsics()

        print("\n[Controls]")
        print("  Left mouse drag: rotate the view")
        print("  Middle mouse / Shift + left mouse: pan")
        print("  Mouse wheel: zoom")
        print("")
        print("  Space: add the current camera as a manual viewpoint")
        print("  U: undo the last viewpoint")
        print("  C: clear all viewpoints")
        print("  X: delete the viewpoint nearest to the current camera")
        print("  R: render and save all viewpoints in a hidden background window")
        print("")
        print("  O: record a trajectory keyframe")
        print("  N: undo the last trajectory keyframe")
        print("  M: clear all trajectory keyframes")
        print("  G: generate views from trajectory keyframes using a multi-point curve")
        print("  I: generate views from manual viewpoints using a multi-point curve")
        print("  B: print trajectory/viewpoint state")
        print("")
        print("  Arrow keys: Up forward / Down backward / Left move left / Right move right")
        print("  Letter keys: W/S forward-backward, A/D left-right, Q/E up-down")
        print("")
        print(f"  max_images: {self.max_images if self.max_images > 0 else 'unlimited'}")
        print(f"  max_invalid_depth_ratio: {self.max_invalid_depth_ratio}")
        print(f"  reference_height: {self.reference_height}")
        print(f"  reference_hfov: {self.reference_hfov}")
        print(f"  target_hfovs: {self.target_hfovs}")
        print("  R: render multiple hFOV groups using reference_height/reference_hfov compensation")
        print("  H: help")
        print("  Close the window: exit\n")

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
            # Remove the last manual_viewpoints entry because Space-added views are appended.
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

    # Backup letter-key controls.
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
        print("  Multi-point curve generation: Catmull-Rom for position, SLERP for rotation.")
        print("  Views are computed first and then drawn in a batch to avoid UI stalls.")
        print("  Space adds a manual view; O records a keyframe; G/I generates views; R renders.\n")
        return False

    # -------------------------
    # Save / render
    # -------------------------
    def _save_depth(self, depth_float, base_name, out_dir=None):
        """Save the raw single-channel float32 depth map as EXR."""
        depth = np.asarray(depth_float, dtype=np.float32).copy()
        invalid = ~np.isfinite(depth)
        depth[invalid] = 0.0

        if out_dir is None:
            out_dir = self.out_dir
        out_dir = Path(out_dir)
        make_dirs(out_dir)

        exr_path = out_dir / "depth" / f"{base_name}.exr"

        ok = cv2.imwrite(str(exr_path), depth)
        if not ok:
            raise RuntimeError(
                f"Failed to write EXR: {exr_path}. "
                f"Please ensure OpenCV is built with OpenEXR support."
            )

        valid_vals = depth[depth > 0]
        if len(valid_vals) > 0:
            dmax = np.percentile(valid_vals, 99.5)
            dmax = max(dmax, 1e-6)
        else:
            dmax = 1.0

        depth_norm = np.clip(depth / dmax, 0, 1)
        depth_u16 = (depth_norm * 65535).astype(np.uint16)
        png_path = out_dir / "depth_png" / f"{base_name}.png"
        imageio.imwrite(png_path, depth_u16)

        return exr_path

    def save_viewpoint_list(self, viewpoints=None, out_dir=None, extra_meta=None):
        if viewpoints is None:
            viewpoints = self.viewpoints
        if out_dir is None:
            out_dir = self.out_dir

        serializable = []
        for i, vp in enumerate(viewpoints):
            item = {
                "id": i,
                "source": vp.get("source", "unknown"),
                "width": self.width,
                "height": self.height,
                "intrinsic_matrix_3x3": vp["K"].tolist(),
                "extrinsic_world_to_camera_4x4": vp["extrinsic"].tolist()
            }
            if "ref_view_idx" in vp:
                item["ref_view_idx"] = int(vp["ref_view_idx"])
            if "hfov" in vp:
                item["hfov"] = float(vp["hfov"])
            if "camera_distance" in vp:
                item["camera_distance"] = float(vp["camera_distance"])
            serializable.append(item)

        payload = {
            "reference_height": self.reference_height,
            "reference_hfov": self.reference_hfov,
            "target_hfovs": self.target_hfovs,
            "render_multi_hfov": self.render_multi_hfov,
            "viewpoints": serializable,
        }
        if extra_meta is not None:
            payload["extra_meta"] = extra_meta

        path = Path(out_dir) / "viewpoints" / "selected_viewpoints.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved viewpoint list: {path}")
        
    def _compute_invalid_depth_ratio(self, depth_float):
        depth = np.asarray(depth_float, dtype=np.float32)
        invalid = (~np.isfinite(depth)) | (depth <= 0)
        return float(np.mean(invalid))

    def _render_viewpoints_to_dir(self, viewpoints, out_dir, tag="default", extra_meta=None):
        if len(viewpoints) == 0:
            print(f"[WARN] No viewpoints to render for tag={tag}.")
            return

        out_dir = Path(out_dir)
        make_dirs(out_dir)
        self.save_viewpoint_list(viewpoints=viewpoints, out_dir=out_dir, extra_meta=extra_meta)
        print(f"[INFO] Start batch rendering {len(viewpoints)} viewpoints for {tag} -> {out_dir}")

        render_vis = o3d.visualization.Visualizer()
        render_vis.create_window(
            window_name=f"HiddenRenderer_{tag}",
            width=self.width,
            height=self.height,
            visible=False
        )
        render_vis.add_geometry(self.mesh)

        render_opt = render_vis.get_render_option()
        render_opt.mesh_show_back_face = True
        render_opt.light_on = True

        vc = render_vis.get_view_control()

        images_dir = out_dir / "images"
        cams_dir = out_dir / "cams"

        saved_count = 0
        skipped_count = 0

        for idx, vp in enumerate(viewpoints):
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
                print(f"[ERROR] Failed to set camera for view #{idx} ({tag}): {e}")
                continue

            render_vis.poll_events()
            render_vis.update_renderer()
            time.sleep(0.02)

            depth_float = np.asarray(render_vis.capture_depth_float_buffer(do_render=True), dtype=np.float32)
            invalid_ratio = self._compute_invalid_depth_ratio(depth_float)

            if invalid_ratio > self.max_invalid_depth_ratio:
                print(f"[SKIP] #{idx} ({vp.get('source', 'unknown')}, {tag}) "
                    f"invalid_depth_ratio={invalid_ratio:.3f} > {self.max_invalid_depth_ratio:.3f}")
                skipped_count += 1
                continue

            base_name = f"{saved_count:06d}"

            rgb_float = np.asarray(render_vis.capture_screen_float_buffer(do_render=True))
            rgb_u8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
            rgb_path = images_dir / f"{base_name}.png"
            imageio.imwrite(rgb_path, rgb_u8)

            depth_exr_path = self._save_depth(depth_float, base_name, out_dir=out_dir)

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

            print(f"[RENDERED] #{idx} ({vp.get('source', 'unknown')}, {tag}) "
                f"invalid_depth_ratio={invalid_ratio:.3f} "
                f"-> image={rgb_path.name}, depth={depth_exr_path.name}, cam={cam_txt_path.name}")
            saved_count += 1

        render_vis.destroy_window()
        self.rendered = True
        print(f"[INFO] Batch rendering completed for {tag}. saved={saved_count}, skipped={skipped_count}")

    def render_all_views(self):
        if len(self.viewpoints) == 0:
            print("[WARN] No viewpoints selected. Press Space / G / I first.")
            return

        if self.render_multi_hfov:
            print("[INFO] Multi-hFOV rendering enabled.")
            for hfov in self.target_hfovs:
                subdir = Path(self.out_dir) / f"hfov_{int(round(hfov)):02d}"
                scaled_viewpoints = self.build_scaled_viewpoints_for_hfov(hfov)
                tgt_distance = self._calc_distance_for_hfov(hfov)
                extra_meta = {
                    "hfov": float(hfov),
                    "target_camera_distance": float(tgt_distance),
                    "distance_formula": "target_distance = reference_height * tan(reference_hfov/2) / tan(target_hfov/2)",
                }
                self._render_viewpoints_to_dir(
                    viewpoints=scaled_viewpoints,
                    out_dir=subdir,
                    tag=f"hfov_{int(round(hfov)):02d}",
                    extra_meta=extra_meta,
                )
        else:
            self._render_viewpoints_to_dir(
                viewpoints=self.viewpoints,
                out_dir=self.out_dir,
                tag="reference",
                extra_meta={
                    "hfov": self.reference_hfov,
                    "target_camera_distance": self.reference_height,
                }
            )
    
    def on_render_all(self, vis):
        # Keep the main window alive; rendering runs in a hidden window.
        self.render_all_views()
        # Refresh the main window once after rendering; do not redraw markers.
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
    """argparse type: strictly positive integer."""
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {ivalue}")
    return ivalue


def nonnegative_int(value):
    """argparse type: non-negative integer."""
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {ivalue}")
    return ivalue


def positive_float(value):
    """argparse type: strictly positive float."""
    try:
        fvalue = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not np.isfinite(fvalue) or fvalue <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive finite number, got {value!r}")
    return fvalue


def ratio_float(value):
    """argparse type: float ratio in [0, 1]."""
    try:
        fvalue = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not np.isfinite(fvalue) or not (0.0 <= fvalue <= 1.0):
        raise argparse.ArgumentTypeError(f"expected a finite value in [0, 1], got {value!r}")
    return fvalue


def hfov_float(value):
    """argparse type: horizontal field of view in degrees."""
    fvalue = positive_float(value)
    if not (0.0 < fvalue < 179.0):
        raise argparse.ArgumentTypeError(f"hFOV must be in (0, 179) degrees, got {fvalue}")
    return fvalue


def parse_hfov_list_arg(value):
    """Parse and validate a comma-separated hFOV list."""
    values = parse_float_list(value)
    if not values:
        raise argparse.ArgumentTypeError("target hFOV list must not be empty")

    cleaned = []
    for item in values:
        if not np.isfinite(item) or not (0.0 < item < 179.0):
            raise argparse.ArgumentTypeError(
                f"each target hFOV must be in (0, 179) degrees, got {item}"
            )
        cleaned.append(float(item))

    return sorted(set(cleaned))


def build_arg_parser():
    """Build a standardized command-line interface for the renderer."""
    parser = argparse.ArgumentParser(
        description=(
            "Interactive Open3D viewpoint selector with trajectory interpolation "
            "and optional multi-hFOV batch rendering."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("Input and output")
    io_group.add_argument(
        "--mesh",
        type=str,
        required=True,
        help="Path to the input OBJ mesh.",
    )
    io_group.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=str,
        required=True,
        help="Base output directory.",
    )
    io_group.add_argument(
        "--no-output-suffix",
        "--no_output_suffix",
        dest="append_output_suffix",
        action="store_false",
        help="Do not append reference-distance/reference-hFOV metadata to the output directory name.",
    )
    io_group.set_defaults(append_output_suffix=True)

    image_group = parser.add_argument_group("Image size and camera intrinsics")
    image_group.add_argument("--width", type=positive_int, default=1024, help="Render width in pixels.")
    image_group.add_argument("--height", type=positive_int, default=720, help="Render height in pixels.")
    image_group.add_argument(
        "--reference-distance",
        "--reference-height",
        "--reference_height",
        dest="reference_height",
        type=positive_float,
        default=0.3,
        help=(
            "Reference camera-to-target distance. For top-down views this can be treated as height; "
            "for oblique views it is the baseline distance to the target-region center."
        ),
    )

    camera_group = image_group.add_mutually_exclusive_group()
    camera_group.add_argument(
        "--reference-hfov",
        "--reference_hfov",
        dest="reference_hfov",
        type=hfov_float,
        default=None,
        help="Reference horizontal field of view in degrees. Used for manual planning and hFOV compensation.",
    )
    camera_group.add_argument(
        "--focal",
        type=positive_float,
        default=None,
        help="Reference focal length in pixels. If set, reference hFOV is derived from width and focal.",
    )

    image_group.add_argument("--cx", type=float, default=None, help="Principal point x in pixels. Defaults to width / 2.")
    image_group.add_argument("--cy", type=float, default=None, help="Principal point y in pixels. Defaults to height / 2.")

    render_group = parser.add_argument_group("Rendering behavior")
    render_group.add_argument(
        "--target-hfovs",
        "--target_hfovs",
        dest="target_hfovs",
        type=parse_hfov_list_arg,
        default=parse_hfov_list_arg("25,35,45,55,65,75,85,95"),
        help="Comma-separated target horizontal FOV values for automatic multi-hFOV rendering.",
    )
    render_group.add_argument(
        "--single-hfov-only",
        "--single_hfov_only",
        dest="single_hfov_only",
        action="store_true",
        help="Render only the reference hFOV instead of expanding to multiple hFOVs.",
    )
    render_group.add_argument(
        "--max-images",
        "--max_images",
        dest="max_images",
        type=nonnegative_int,
        default=500,
        help="Maximum number of viewpoints/images to keep and render. Use 0 for unlimited.",
    )
    render_group.add_argument(
        "--max-invalid-depth-ratio",
        "--max_invalid_depth_ratio",
        dest="max_invalid_depth_ratio",
        type=ratio_float,
        default=0.5,
        help="Skip a rendered frame when the invalid-depth-pixel ratio exceeds this threshold.",
    )

    viz_group = parser.add_argument_group("Visualization")
    viz_group.add_argument(
        "--frustum-ratio",
        "--frustum_ratio",
        dest="frustum_ratio",
        type=positive_float,
        default=0.02,
        help="Camera-frustum visualization length relative to scene scale.",
    )
    viz_group.add_argument(
        "--axis-ratio",
        "--axis_ratio",
        dest="axis_ratio",
        type=positive_float,
        default=0.01,
        help="Camera-axis visualization size relative to scene scale.",
    )

    return parser


def resolve_reference_camera(args, parser):
    """Resolve reference focal length and hFOV from standardized CLI inputs."""
    if args.reference_hfov is None and args.focal is None:
        args.reference_hfov = 65.0

    if args.focal is not None:
        args.fx = float(args.focal)
        args.fy = float(args.focal)
        args.reference_hfov = calculate_hfov(args.fx, args.width)
    else:
        args.fx = focal_from_hfov(args.reference_hfov, args.width)
        args.fy = args.fx

    if args.cx is not None and not (0.0 <= float(args.cx) <= float(args.width)):
        parser.error(f"--cx must be in [0, width], got {args.cx}")
    if args.cy is not None and not (0.0 <= float(args.cy) <= float(args.height)):
        parser.error(f"--cy must be in [0, height], got {args.cy}")

    return args


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args = resolve_reference_camera(args, parser)

    if args.append_output_suffix:
        args.out_dir = f"{args.out_dir}_refD{args.reference_height:g}_refHFOV{args.reference_hfov:g}"

    print("[CONFIG] Standardized input")
    print(f"         mesh: {args.mesh}")
    print(f"         out_dir: {args.out_dir}")
    print(f"         size: {args.width} x {args.height}")
    print(f"         fx/fy: {args.fx:.6f} / {args.fy:.6f}")
    print(f"         reference_distance: {args.reference_height:g}")
    print(f"         reference_hfov: {args.reference_hfov:.6f}")
    print(f"         target_hfovs: {args.target_hfovs}")
    print(f"         render_multi_hfov: {not args.single_hfov_only}")

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
        reference_height=args.reference_height,
        reference_hfov=args.reference_hfov,
        target_hfovs=args.target_hfovs,
        render_multi_hfov=not args.single_hfov_only,
    )
    app.run()


if __name__ == "__main__":
    main()
