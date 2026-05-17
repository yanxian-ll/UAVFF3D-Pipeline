# -*- coding: utf-8 -*-
"""
UAVFF3D-Syn-L multi-OBJ tile grouping and aerial-style rendering pipeline.

Workflow:
1) Scan OBJ tiles and compute a 2D XY convex hull for each tile.
2) Visualize all tile hulls and centers in a normalized display space.
3) Manually group tiles by picking center points.
4) Merge each selected tile group with an external ``merge_tiles`` executable.
5) Estimate or manually define an ROI for each merged group.
6) Generate lawnmower waypoints and render RGB, depth EXR, and camera files with Open3D.

Notes:
- ``flight_height`` is interpreted as height above the manually selected global ground Z.
- OpenCV-style camera coordinates are used in saved camera files: x right, y down, z forward.
- The default camera rig renders the nadir camera only; use ``--camera-rig five`` to render all five cameras.
"""

import os
import math
import json
import time
import argparse
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import imageio.v2 as imageio
from tqdm import tqdm

# Enable EXR writing in OpenCV. Your OpenCV build must include OpenEXR support.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


# ==============================================================================
# Basic I/O dirs
# ==============================================================================
def make_root_dirs(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(exist_ok=True)
    return out_dir


def make_scene_dirs(scene_dir):
    scene_dir = Path(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "images").mkdir(exist_ok=True)
    (scene_dir / "depth").mkdir(exist_ok=True)
    (scene_dir / "cams").mkdir(exist_ok=True)
    (scene_dir / "meta").mkdir(exist_ok=True)
    return scene_dir


# ==============================================================================
# Math / Geometry Utils
# ==============================================================================
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


def calculate_hfov(fx, w):
    return float(math.degrees(2.0 * math.atan(float(w) / (2.0 * float(fx)))))

def calculate_vfov(fy, h):
    return float(math.degrees(2.0 * math.atan(float(h) / (2.0 * float(fy)))))


def write_cam_file(cam_path: Path, extrinsic, fx, fy, cx, cy, h, w):
    hfov = calculate_hfov(fx, w)
    vfov = calculate_vfov(fy, h)
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

        f.write("h w hfov\n")
        f.write(f"{h} {w} {hfov:.12f}\n")


# ==============================================================================
# OBJ Scanning / Tile Metadata
# ==============================================================================
def find_obj_files(obj_root, recursive=True):
    obj_root = Path(obj_root)
    if obj_root.is_file() and obj_root.suffix.lower() == ".obj":
        return [obj_root]

    if not obj_root.exists():
        raise FileNotFoundError(f"OBJ root not found: {obj_root}")

    pattern = "**/*.obj" if recursive else "*.obj"
    files = sorted(obj_root.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No OBJ files found under: {obj_root}")
    return files

def get_input_root_dir(obj_root):
    """
    Return the input root directory.

    If ``obj_root`` is a directory, return it directly.
    If ``obj_root`` is a single OBJ file, return its parent directory.
    """
    obj_root = Path(obj_root)
    if obj_root.is_file():
        return obj_root.parent
    return obj_root


def get_obj_tile_meta_save_path(obj_path: Path):
    """
    Return the per-OBJ tile metadata path saved next to the OBJ file.

    Example:
      xxx/mesh_001.obj -> xxx/mesh_001_tile_meta.json
    """
    obj_path = Path(obj_path)
    return obj_path.parent / f"{obj_path.stem}_tile_meta.json"


def load_or_build_obj_tile_meta(obj_path: Path, overwrite=False):
    """
    Load cached per-OBJ tile metadata when possible.

    If the cache is missing, or ``overwrite=True``, recompute the metadata and save it.
    """
    obj_path = Path(obj_path)
    meta_path = get_obj_tile_meta_save_path(obj_path)

    if meta_path.exists() and (not overwrite):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta

    meta = get_obj_tile_meta(obj_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def get_global_ground_meta_path(obj_root):
    """
    Return the global ground-Z metadata path.

    The file is saved in the input directory, not the output directory.
    - Directory input: obj_root/global_ground_z.json
    - Single OBJ input: parent_dir/global_ground_z.json
    """
    input_root = get_input_root_dir(obj_root)
    input_root.mkdir(parents=True, exist_ok=True)
    return input_root / "global_ground_z.json"

def cross_2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull_2d(points_xy):
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points_xy shape invalid: {pts.shape}")

    pts = np.unique(pts, axis=0)
    if len(pts) <= 2:
        return pts.copy()

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross_2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross_2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) == 0:
        hull = pts[:1].copy()
    return hull


def polygon_area_2d(poly_xy):
    poly = np.asarray(poly_xy, dtype=np.float64)
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_centroid_2d(poly_xy):
    poly = np.asarray(poly_xy, dtype=np.float64)
    if len(poly) == 0:
        return np.zeros((2,), dtype=np.float64)
    if len(poly) == 1:
        return poly[0].copy()
    if len(poly) == 2:
        return poly.mean(axis=0)

    area2 = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if abs(area2) < 1e-12:
        return poly.mean(axis=0)

    factor = (poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    cx = np.sum((poly[:, 0] + np.roll(poly[:, 0], -1)) * factor) / (3.0 * area2)
    cy = np.sum((poly[:, 1] + np.roll(poly[:, 1], -1)) * factor) / (3.0 * area2)
    return np.array([cx, cy], dtype=np.float64)


def polygon_bbox_xyxy(poly_xy):
    poly = np.asarray(poly_xy, dtype=np.float64)
    mn = poly.min(axis=0)
    mx = poly.max(axis=0)
    return [float(mn[0]), float(mx[0]), float(mn[1]), float(mx[1])]


def get_obj_tile_meta(obj_path: Path):
    m = o3d.io.read_triangle_mesh(str(obj_path), enable_post_processing=False)
    if m.is_empty():
        raise RuntimeError(f"Empty mesh: {obj_path}")

    verts = np.asarray(m.vertices, dtype=np.float64)
    if verts.shape[0] == 0:
        raise RuntimeError(f"No vertices in mesh: {obj_path}")

    mn = verts.min(axis=0)
    mx = verts.max(axis=0)

    dx = float(mx[0] - mn[0])
    dy = float(mx[1] - mn[1])
    dz = float(mx[2] - mn[2])

    verts_xy = verts[:, :2]
    hull_xy = convex_hull_2d(verts_xy)
    if len(hull_xy) == 0:
        raise RuntimeError(f"Convex hull failed: {obj_path}")

    hull_area = float(polygon_area_2d(hull_xy))
    center_xy = polygon_centroid_2d(hull_xy)
    bbox_xyxy = polygon_bbox_xyxy(hull_xy)

    return {
        "path": str(obj_path),
        "name": obj_path.name,
        "parent_dir": str(obj_path.parent),
        "mn_xyz": [float(v) for v in mn],
        "mx_xyz": [float(v) for v in mx],
        "hull_xy": hull_xy.astype(np.float64).tolist(),
        "hull_num_vertices": int(len(hull_xy)),
        "hull_area_xy": hull_area,
        "bbox_xyxy": bbox_xyxy,
        "center_xy": [float(center_xy[0]), float(center_xy[1])],
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "num_vertices": int(len(m.vertices)),
        "num_triangles": int(len(m.triangles)),
    }


def estimate_images_for_tile_bbox(tile_meta, lane_spacing, point_spacing, roles_count, coverage_efficiency=0.75):
    x0, x1, y0, y1 = tile_meta["bbox_xyxy"]
    area_bbox = max(float(x1 - x0), 0.0) * max(float(y1 - y0), 0.0)
    cell_area = max(float(lane_spacing) * float(point_spacing), 1e-6)
    est_waypoints = max(1, int(math.ceil(area_bbox / (cell_area * max(coverage_efficiency, 1e-3)))))
    est_images = int(est_waypoints * int(roles_count))
    return est_waypoints, est_images


# ==============================================================================
# Manual grouping visualization / interaction
# ==============================================================================
def make_color_bank():
    return np.array([
        [0.90, 0.20, 0.20],
        [0.20, 0.75, 0.25],
        [0.20, 0.55, 0.95],
        [0.95, 0.75, 0.20],
        [0.85, 0.30, 0.85],
        [0.20, 0.85, 0.85],
        [1.00, 0.50, 0.10],
        [0.60, 0.40, 1.00],
        [0.50, 0.80, 0.20],
        [0.90, 0.40, 0.60],
    ], dtype=np.float64)


def create_polygon_lineset_xy(poly_xy, z=0.0, color=(0.1, 0.8, 1.0), closed=True):
    poly = np.asarray(poly_xy, dtype=np.float64)
    if len(poly) < 2:
        return None

    pts_xyz = np.column_stack([poly[:, 0], poly[:, 1], np.full((len(poly),), float(z), dtype=np.float64)])

    lines = [[i, i + 1] for i in range(len(poly) - 1)]
    if closed and len(poly) >= 3:
        lines.append([len(poly) - 1, 0])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_xyz)
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1)))
    return ls


def sample_polyline_points_xy(poly_xy, step=0.25, closed=True):
    poly = np.asarray(poly_xy, dtype=np.float64)
    if len(poly) < 2:
        return np.zeros((0, 2), dtype=np.float64)

    pts_all = []
    n = len(poly)
    seg_count = n if closed and n >= 3 else (n - 1)

    for i in range(seg_count):
        p0 = poly[i]
        p1 = poly[(i + 1) % n]
        d = p1 - p0
        seg_len = float(np.linalg.norm(d))
        if seg_len < 1e-12:
            continue

        num = max(2, int(math.ceil(seg_len / max(step, 1e-6))) + 1)
        ts = np.linspace(0.0, 1.0, num=num, endpoint=True)
        seg_pts = p0[None, :] * (1.0 - ts[:, None]) + p1[None, :] * ts[:, None]
        pts_all.append(seg_pts)

    if len(pts_all) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    pts = np.concatenate(pts_all, axis=0)

    # Remove consecutive duplicate samples.
    out = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - out[-1]) > 1e-9:
            out.append(p)
    return np.asarray(out, dtype=np.float64)

def build_grouping_visualization_geometries(
    tile_metas,
    display_tf,
    grouped_ids=None,
    group_color_map=None,
    z=0.0,
    add_world_frame=True,
    ungrouped_hull_color=(0.1, 0.8, 1.0),
    ungrouped_center_color=(1.0, 0.2, 0.2),
):
    geoms = []
    grouped_ids = set() if grouped_ids is None else set(int(i) for i in grouped_ids)
    group_color_map = {} if group_color_map is None else group_color_map

    ungrouped_centers = []
    ungrouped_center_colors = []

    grouped_centers = []
    grouped_center_colors = []

    for tid, meta in enumerate(tile_metas):
        hull_xy_disp = get_display_hull_xy(meta, display_tf)
        center_xy_disp = get_display_center_xy(meta, display_tf)
        center_xyz = [center_xy_disp[0], center_xy_disp[1], float(z)]

        if tid in grouped_ids:
            color = np.asarray(group_color_map.get(tid, [0.6, 0.6, 0.6]), dtype=np.float64)
            ls = create_polygon_lineset_xy(hull_xy_disp, z=z, color=tuple(color.tolist()), closed=True)
            if ls is not None:
                geoms.append(ls)
            grouped_centers.append(center_xyz)
            grouped_center_colors.append(color)
        else:
            ls = create_polygon_lineset_xy(hull_xy_disp, z=z, color=ungrouped_hull_color, closed=True)
            if ls is not None:
                geoms.append(ls)
            ungrouped_centers.append(center_xyz)
            ungrouped_center_colors.append(np.asarray(ungrouped_center_color, dtype=np.float64))

    if len(grouped_centers) > 0:
        pcd_grouped = o3d.geometry.PointCloud()
        pcd_grouped.points = o3d.utility.Vector3dVector(np.asarray(grouped_centers, dtype=np.float64))
        pcd_grouped.colors = o3d.utility.Vector3dVector(np.asarray(grouped_center_colors, dtype=np.float64))
        geoms.append(pcd_grouped)

    pcd_ungrouped = o3d.geometry.PointCloud()
    if len(ungrouped_centers) > 0:
        pcd_ungrouped.points = o3d.utility.Vector3dVector(np.asarray(ungrouped_centers, dtype=np.float64))
        pcd_ungrouped.colors = o3d.utility.Vector3dVector(np.asarray(ungrouped_center_colors, dtype=np.float64))
    else:
        pcd_ungrouped.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        pcd_ungrouped.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
    geoms.append(pcd_ungrouped)

    if add_world_frame:
        frame_size = max(1.0, 0.08 * float(display_tf["target_span"]))
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size))

    return geoms, pcd_ungrouped


def build_grouping_pick_pointcloud(
    tile_metas,
    display_tf,
    grouped_ids=None,
    group_color_map=None,
    z=0.0,
    hull_sample_step=0.25,
    ungrouped_center_color=(1.0, 0.2, 0.2),
    ungrouped_hull_color=(0.1, 0.8, 1.0),
):
    """
    Build one point cloud for VisualizerWithEditing.

    The first N points are pickable ungrouped tile centers.
    The remaining points are visual context only: sampled polygon boundaries and grouped centers.
    """
    grouped_ids = set() if grouped_ids is None else set(int(i) for i in grouped_ids)
    group_color_map = {} if group_color_map is None else group_color_map

    pick_pts = []
    pick_cols = []

    context_pts = []
    context_cols = []

    ungrouped_global_ids = []

    for tid, meta in enumerate(tile_metas):
        hull_xy_disp = get_display_hull_xy(meta, display_tf)
        center_xy_disp = get_display_center_xy(meta, display_tf)

        # Sample polygon boundaries as visual context.
        hull_pts_xy = sample_polyline_points_xy(
            hull_xy_disp,
            step=float(hull_sample_step),
            closed=True
        )
        if len(hull_pts_xy) > 0:
            hull_pts_xyz = np.column_stack([
                hull_pts_xy[:, 0],
                hull_pts_xy[:, 1],
                np.full((len(hull_pts_xy),), float(z), dtype=np.float64)
            ])

            if tid in grouped_ids:
                col = np.asarray(group_color_map.get(tid, [0.6, 0.6, 0.6]), dtype=np.float64)
            else:
                col = np.asarray(ungrouped_hull_color, dtype=np.float64)

            context_pts.append(hull_pts_xyz)
            context_cols.append(np.tile(col[None, :], (len(hull_pts_xyz), 1)))

        center_xyz = np.array([[center_xy_disp[0], center_xy_disp[1], float(z)]], dtype=np.float64)

        if tid in grouped_ids:
            col = np.asarray(group_color_map.get(tid, [0.6, 0.6, 0.6]), dtype=np.float64)
            context_pts.append(center_xyz)
            context_cols.append(col[None, :])
        else:
            # Place ungrouped centers first because only these points should be pickable.
            pick_pts.append(center_xyz)
            pick_cols.append(np.asarray(ungrouped_center_color, dtype=np.float64)[None, :])
            ungrouped_global_ids.append(int(tid))

    if len(pick_pts) > 0:
        pick_pts = np.concatenate(pick_pts, axis=0)
        pick_cols = np.concatenate(pick_cols, axis=0)
    else:
        pick_pts = np.zeros((0, 3), dtype=np.float64)
        pick_cols = np.zeros((0, 3), dtype=np.float64)

    if len(context_pts) > 0:
        context_pts = np.concatenate(context_pts, axis=0)
        context_cols = np.concatenate(context_cols, axis=0)
    else:
        context_pts = np.zeros((0, 3), dtype=np.float64)
        context_cols = np.zeros((0, 3), dtype=np.float64)

    all_pts = np.concatenate([pick_pts, context_pts], axis=0)
    all_cols = np.concatenate([pick_cols, context_cols], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_cols)

    return pcd, ungrouped_global_ids, int(len(pick_pts))

def pick_centers_on_pointcloud(
    geom,
    pickable_count,
    geom_name="ungrouped centers",
    width=1600,
    height=900,
):
    """
    Pick tile centers from one combined point cloud.

    The first ``pickable_count`` points are the selectable ungrouped centers.
    Later points are context only, such as polygon boundary samples and already grouped centers.

    Returns:
      picked_local: local indices within the ungrouped-center subset.
    """
    print("\n[Manual grouping pick instructions]")
    print(f"  Current geometry: {geom_name}")
    print("  1) Shift + left click: pick one or more center points for this group")
    print("  2) Shift + right click: undo the most recent pick")
    print("  3) Press Q or close the window to finish this round")
    print("  4) Picking no points in a round means grouping is finished\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"Pick Centers on {geom_name}", width=int(width), height=int(height))
    vis.add_geometry(geom)

    render_opt = vis.get_render_option()
    render_opt.point_size = 6.0
    render_opt.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    vis.run()
    vis.destroy_window()

    picked_idx_all = [int(i) for i in vis.get_picked_points()]

    # Keep only the first pickable_count points, which are the real ungrouped centers.
    picked_local = sorted(set(i for i in picked_idx_all if 0 <= i < int(pickable_count)))

    ignored = [i for i in picked_idx_all if i >= int(pickable_count)]
    if len(ignored) > 0:
        print(f"[GROUP] ignored {len(ignored)} picked context point(s): {ignored[:20]}")

    return picked_local

def build_group_preview_geometries(tile_metas, display_tf, picked_indices, z=0.0, color=(1.0, 1.0, 0.0)):
    geoms = []
    picked_indices = list(sorted(set(int(i) for i in picked_indices)))

    if len(picked_indices) > 0:
        for i in picked_indices:
            hull_xy_disp = get_display_hull_xy(tile_metas[i], display_tf)
            ls = create_polygon_lineset_xy(hull_xy_disp, z=z, color=color, closed=True)
            if ls is not None:
                geoms.append(ls)

        centers = []
        for i in picked_indices:
            cxy = get_display_center_xy(tile_metas[i], display_tf)
            centers.append([cxy[0], cxy[1], float(z)])

        centers = np.asarray(centers, dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers)
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(np.asarray(color, dtype=np.float64)[None, :], (len(centers), 1))
        )
        geoms.append(pcd)

    return geoms


def confirm_group_by_text(group_idx, picked_global):
    print("\n[GROUP PREVIEW]")
    print(f"  group_id={group_idx:03d}")
    print(f"  selected_tiles={len(picked_global)}")
    print(f"  tile_indices={picked_global}")
    ans = input("Confirm this group? Type y to accept; anything else cancels and lets you reselect: ").strip().lower()
    return ans == "y"

def manual_group_tiles_by_picking_centers(
    tile_metas,
    root_out: Path,
    z=0.0,
    keep_ungrouped_as_one=True,
    target_span=10,
    margin_ratio=0.05
):
    total_ids = list(range(len(tile_metas)))
    remaining_ids = set(total_ids)
    groups = []
    group_colors = []
    group_color_map = {}
    round_idx = 0
    color_bank = make_color_bank()

    display_tf = build_display_transform_from_tile_metas(
        tile_metas,
        target_span=target_span,
        margin_ratio=margin_ratio
    )
    with open(root_out / "meta" / "group_display_transform.json", "w", encoding="utf-8") as f:
        json.dump(display_tf, f, ensure_ascii=False, indent=2)

    print("\n[Manual tile-center grouping instructions]")
    print("  Note: this grouping window uses normalized display coordinates only for easier overview and picking.")
    print("  The saved groups still reference original tile IDs; merge/render uses the original coordinates.")
    print("  Color legend:")
    print("    - Cyan boundary points: polygon boundaries of ungrouped tiles")
    print("    - Red points: ungrouped tile centers, directly pickable")
    print("    - Other colors: confirmed groups, shown only as context")
    print("  Workflow:")
    print("    1) Inspect hull boundaries and centers in the same window")
    print("    2) Shift + left click one or more red center points")
    print("    3) Close the window to preview this group")
    print("    4) Confirm in the terminal whether to keep the group")
    print("    5) Picking no points in a round finishes grouping\n")

    while len(remaining_ids) > 0:
        grouped_ids = set(total_ids) - remaining_ids

        pick_pcd, ungrouped_global_ids, pickable_count = build_grouping_pick_pointcloud(
            tile_metas=tile_metas,
            display_tf=display_tf,
            grouped_ids=grouped_ids,
            group_color_map=group_color_map,
            z=z,
            hull_sample_step=0.25,
            ungrouped_center_color=(1.0, 0.2, 0.2),
            ungrouped_hull_color=(0.1, 0.8, 1.0),
        )

        if pickable_count == 0:
            print("[GROUP] no ungrouped centers left.")
            break

        picked_local = pick_centers_on_pointcloud(
            pick_pcd,
            pickable_count=pickable_count,
            geom_name=f"ungrouped tile centers round {round_idx:03d}",
            width=1600,
            height=900,
        )

        if len(picked_local) == 0:
            print(f"[GROUP] round {round_idx:03d}: picked 0 centers, finish grouping.")
            break

        picked_global = [ungrouped_global_ids[i] for i in picked_local]
        picked_color = color_bank[round_idx % len(color_bank)]

        preview_geoms = []
        preview_geoms.extend(
            build_group_preview_geometries(
                tile_metas=tile_metas,
                display_tf=display_tf,
                picked_indices=picked_global,
                z=z,
                color=tuple(picked_color.tolist())
            )
        )

        frame_size = max(1.0, 0.08 * float(display_tf["target_span"]))
        preview_geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size))

        o3d.visualization.draw_geometries(
            preview_geoms,
            window_name=f"Grouped Tiles Preview #{round_idx:03d}",
            width=1400,
            height=850
        )

        ok = confirm_group_by_text(round_idx, picked_global)
        if not ok:
            print(f"[GROUP] round {round_idx:03d}: canceled, please reselect this group.")
            continue

        groups.append(picked_global)
        group_colors.append([float(x) for x in picked_color.tolist()])

        for tid in picked_global:
            group_color_map[int(tid)] = [float(x) for x in picked_color.tolist()]

        remaining_ids -= set(picked_global)

        print(f"[GROUP] round {round_idx:03d}: confirmed group with {len(picked_global)} tile(s)")
        print(f"        global tile indices = {picked_global}")
        round_idx += 1

    if keep_ungrouped_as_one and len(remaining_ids) > 0:
        picked_color = color_bank[round_idx % len(color_bank)]
        final_group = sorted(list(remaining_ids))
        print(f"[GROUP] remaining {len(final_group)} tile(s) -> append as final group")
        groups.append(final_group)
        group_colors.append([float(x) for x in picked_color.tolist()])
        for tid in final_group:
            group_color_map[int(tid)] = [float(x) for x in picked_color.tolist()]
        remaining_ids = set()

    if len(groups) == 0:
        raise RuntimeError("No manual groups were created.")

    with open(root_out / "meta" / "manual_group_pick_raw.json", "w", encoding="utf-8") as f:
        json.dump({
            "num_tiles_total": len(tile_metas),
            "num_groups": len(groups),
            "group_mode": "manual_pick_centers_single_pickable_pointcloud",
            "groups_tile_indices": [[int(x) for x in g] for g in groups],
            "group_colors": group_colors,
            "ungrouped_kept_as_one": bool(keep_ungrouped_as_one),
        }, f, ensure_ascii=False, indent=2)

    return groups, group_colors


def build_group_infos_from_manual_groups(groups, tile_metas, global_ground_z, group_colors=None):
    group_infos = []
    for gid, idxs in enumerate(groups):
        hull_areas = [float(tile_metas[i].get("hull_area_xy", 0.0)) for i in idxs]
        est_imgs_group = int(sum(tile_metas[i].get("est_images_tile", 0) for i in idxs))
        est_wpts_group = int(sum(tile_metas[i].get("est_waypoints_tile", 0) for i in idxs))

        bboxs = [tile_metas[i]["bbox_xyxy"] for i in idxs]
        x0 = min(b[0] for b in bboxs)
        x1 = max(b[1] for b in bboxs)
        y0 = min(b[2] for b in bboxs)
        y1 = max(b[3] for b in bboxs)

        rec = {
            "group_id": int(gid),
            "tile_indices": [int(i) for i in idxs],
            "tile_paths": [tile_metas[i]["path"] for i in idxs],
            "tile_names": [tile_metas[i]["name"] for i in idxs],
            "num_tiles": int(len(idxs)),
            "bbox_xyxy": [float(x0), float(x1), float(y0), float(y1)],
            "sum_hull_area_xy": float(sum(hull_areas)),
            "est_waypoints_group": int(est_wpts_group),
            "est_images_group": int(est_imgs_group),
            "global_ground_z": float(global_ground_z),
        }

        if group_colors is not None and gid < len(group_colors):
            rec["group_color"] = group_colors[gid]

        group_infos.append(rec)
    return group_infos


# ==============================================================================
# Temporary input preparation for merge_tiles
# ==============================================================================
_TEXTURE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".tga", ".exr", ".hdr"}


def _safe_copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_obj_with_sidecars_to_subdir(obj_path: Path, dst_subdir: Path):
    obj_path = Path(obj_path)
    src_dir = obj_path.parent
    dst_subdir = Path(dst_subdir)
    dst_subdir.mkdir(parents=True, exist_ok=True)

    _safe_copy_file(obj_path, dst_subdir / obj_path.name)

    for mtl in src_dir.glob("*.mtl"):
        _safe_copy_file(mtl, dst_subdir / mtl.name)

    for f in src_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() in _TEXTURE_EXTS:
            _safe_copy_file(f, dst_subdir / f.name)


def merge_tile_group_with_external_tool(group_obj_paths, merge_tiles_exe, merged_out_path, keep_tmp_inputs=False):
    merge_tiles_exe = str(merge_tiles_exe)
    merged_out_path = Path(merged_out_path)
    merged_out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = merged_out_path.parent / f"_tmp_merge_inputs_{merged_out_path.stem}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for k, p in enumerate(group_obj_paths):
        src_obj = Path(p)
        subdir_name = f"{k:03d}_{src_obj.stem}"
        dst_subdir = tmp_dir / subdir_name
        _copy_obj_with_sidecars_to_subdir(src_obj, dst_subdir)

    cmd = [merge_tiles_exe, "-i", str(tmp_dir), "-o", str(merged_out_path)]
    print(f"[MERGE] Running: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dt = time.time() - t0
    print(f"[MERGE] returncode={res.returncode} | time={dt:.2f}s")
    if res.stdout:
        print("[MERGE][stdout]\n" + res.stdout.strip())
    if res.stderr:
        print("[MERGE][stderr]\n" + res.stderr.strip())

    if res.returncode != 0:
        raise RuntimeError(f"merge_tiles failed for output: {merged_out_path}")
    if not merged_out_path.exists():
        raise RuntimeError(f"merge_tiles reported success but output missing: {merged_out_path}")

    if not keep_tmp_inputs:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return merged_out_path


def cleanup_merged_outputs(merged_obj_path: Path, delete_textures=True, verbose=True):
    merged_obj_path = Path(merged_obj_path)
    group_dir = merged_obj_path.parent

    exts_tex = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".tga", ".exr", ".hdr"}
    deleted = []
    failed = []

    if merged_obj_path.exists():
        try:
            merged_obj_path.unlink()
            deleted.append(str(merged_obj_path))
        except Exception as e:
            failed.append((str(merged_obj_path), str(e)))

    mtl_same_stem = merged_obj_path.with_suffix(".mtl")
    if mtl_same_stem.exists():
        try:
            mtl_same_stem.unlink()
            deleted.append(str(mtl_same_stem))
        except Exception as e:
            failed.append((str(mtl_same_stem), str(e)))

    if delete_textures and group_dir.exists():
        for f in group_dir.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() in exts_tex:
                try:
                    f.unlink()
                    deleted.append(str(f))
                except Exception as e:
                    failed.append((str(f), str(e)))

    if verbose:
        print(f"[CLEAN] deleted {len(deleted)} merged output files in {group_dir}")
        if len(failed) > 0:
            print(f"[CLEAN][WARN] failed to delete {len(failed)} files")
            for p, e in failed[:10]:
                print(f"  - {p} | {e}")
            if len(failed) > 10:
                print(f"  ... ({len(failed)-10} more)")

    return {
        "deleted_count": len(deleted),
        "failed_count": len(failed),
        "deleted_files": deleted,
        "failed_files": [{"path": p, "error": e} for p, e in failed],
    }


# ==============================================================================
# Load merged OBJ mesh
# ==============================================================================
def load_mesh_single(mesh_path):
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    if mesh_path.suffix.lower() != ".obj":
        print(f"[WARN] Input suffix is not .obj: {mesh_path.suffix}. Will still try Open3D load.")

    print(f"[INFO] Loading merged mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh or mesh is empty: {mesh_path}")

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    print(f"[INFO] Mesh loaded | V={len(mesh.vertices)} F={len(mesh.triangles)} | textures={len(mesh.textures)}")
    return mesh


# ==============================================================================
# Sparse point cloud sampling / picking
# ==============================================================================
def sample_sparse_pcd_from_mesh(mesh, n_points=200000, voxel_size=0.0):
    if n_points <= 0:
        n_points = 100000
    print(f"[PCD] sampling {n_points} points from mesh for ROI picking...")
    pcd = mesh.sample_points_uniformly(number_of_points=int(n_points))
    if voxel_size and voxel_size > 0:
        print(f"[PCD] voxel downsample: voxel_size={voxel_size}")
        pcd = pcd.voxel_down_sample(float(voxel_size))
    print(f"[PCD] sparse points={len(pcd.points)}")
    return pcd


def pick_polygon_points_on_geometry(geom, geom_name="geometry"):
    print("\n[Manual ROI picking instructions]")
    print(f"  Current geometry: {geom_name}")
    print("  1) Shift + left click: pick ROI polygon vertices in order")
    print("  2) Shift + right click: undo the most recent pick")
    print("  3) Press Q or close the window to finish picking")
    print("  4) Pick at least 3 points\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"Pick ROI Polygon Vertices on {geom_name}", width=1600, height=900)
    vis.add_geometry(geom)
    vis.run()
    vis.destroy_window()

    picked_idx = vis.get_picked_points()

    if isinstance(geom, o3d.geometry.PointCloud):
        verts = np.asarray(geom.points, dtype=np.float64)
    else:
        verts = np.asarray(geom.vertices, dtype=np.float64)

    if len(picked_idx) < 3:
        raise RuntimeError(f"Need at least 3 picked points, got {len(picked_idx)}")

    pts = verts[np.asarray(picked_idx, dtype=np.int64)]
    print(f"[PICK] picked {len(pts)} points")
    for i, p in enumerate(pts):
        print(f"  P{i+1}: {p.round(6).tolist()}")
    return pts


def pick_ground_points_on_geometry(geom, geom_name="geometry"):
    print("\n[Manual ground-point picking instructions]")
    print(f"  Current geometry: {geom_name}")
    print("  1) Shift + left click: pick one or more ground points")
    print("  2) Shift + right click: undo the most recent pick")
    print("  3) Press Q or close the window to finish picking")
    print("  4) Recommended: pick 3-10 reasonably well-distributed ground points\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"Pick Ground Points on {geom_name}", width=1600, height=900)
    vis.add_geometry(geom)
    vis.run()
    vis.destroy_window()

    picked_idx = vis.get_picked_points()

    if isinstance(geom, o3d.geometry.PointCloud):
        verts = np.asarray(geom.points, dtype=np.float64)
    else:
        verts = np.asarray(geom.vertices, dtype=np.float64)

    if len(picked_idx) < 1:
        raise RuntimeError("Need at least 1 picked ground point.")

    pts = verts[np.asarray(picked_idx, dtype=np.int64)]
    print(f"[GROUND PICK] picked {len(pts)} points")
    for i, p in enumerate(pts):
        print(f"  G{i+1}: {p.round(6).tolist()}")
    return pts


def auto_roi_polygon_from_bbox_xyxy(x0, x1, y0, y1, z_mean=0.0, margin_ratio=0.02):
    x0, x1, y0, y1 = map(float, [x0, x1, y0, y1])
    dx = max(x1 - x0, 1e-6)
    dy = max(y1 - y0, 1e-6)
    mxg = dx * float(margin_ratio)
    myg = dy * float(margin_ratio)

    roi_xy = np.array([
        [x0 + mxg, y0 + myg],
        [x1 - mxg, y0 + myg],
        [x1 - mxg, y1 - myg],
        [x0 + mxg, y1 - myg],
    ], dtype=np.float64)
    picked_pts_xyz = np.column_stack([roi_xy, np.full((4,), float(z_mean), dtype=np.float64)])
    return picked_pts_xyz, roi_xy


def auto_roi_polygon_from_mesh_bbox(mesh, margin_ratio=0.02):
    aabb = mesh.get_axis_aligned_bounding_box()
    mn = aabb.get_min_bound()
    mx = aabb.get_max_bound()
    return auto_roi_polygon_from_bbox_xyxy(
        mn[0], mx[0], mn[1], mx[1],
        z_mean=0.5 * (float(mn[2]) + float(mx[2])),
        margin_ratio=margin_ratio
    )


# ==============================================================================
# Global ground-Z picking
# ==============================================================================
def build_combined_ground_pick_pcd_from_random_obj_files(
    obj_files,
    num_tiles=3,
    random_seed=-1,
    points_per_tile=8000,
    voxel_size=0.0,
):
    obj_files = [Path(p) for p in obj_files]
    if len(obj_files) == 0:
        raise RuntimeError("obj_files is empty")

    n_pick = min(int(num_tiles), len(obj_files))
    if n_pick <= 0:
        raise ValueError(f"num_tiles must be > 0, got {num_tiles}")

    rng = np.random.default_rng(None if int(random_seed) < 0 else int(random_seed))
    chosen = sorted([int(i) for i in rng.choice(len(obj_files), size=n_pick, replace=False)])

    color_bank = np.array([
        [1.0, 0.2, 0.2],
        [0.2, 1.0, 0.2],
        [0.2, 0.6, 1.0],
        [1.0, 0.8, 0.2],
        [1.0, 0.2, 1.0],
        [0.2, 1.0, 1.0],
    ], dtype=np.float64)

    combined_pts = []
    combined_cols = []
    chosen_infos = []

    print(f"[GROUND] random choose {n_pick} OBJ file(s) for global ground picking: {chosen}")

    for k, idx in enumerate(chosen):
        obj_path = obj_files[idx]
        print(f"[GROUND] load obj[{idx}] => {obj_path}")

        mesh = o3d.io.read_triangle_mesh(str(obj_path), enable_post_processing=False)
        if mesh.is_empty():
            print(f"[GROUND][WARN] empty mesh skipped: {obj_path}")
            continue

        pcd = sample_sparse_pcd_from_mesh(
            mesh,
            n_points=int(points_per_tile),
            voxel_size=float(voxel_size)
        )
        pts = np.asarray(pcd.points, dtype=np.float64)
        if len(pts) == 0:
            print(f"[GROUND][WARN] sampled pcd empty: {obj_path}")
            continue

        col = color_bank[k % len(color_bank)]
        cols = np.tile(col[None, :], (len(pts), 1))

        combined_pts.append(pts)
        combined_cols.append(cols)

        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        chosen_infos.append({
            "obj_index": int(idx),
            "obj_path": str(obj_path),
            "num_sampled_points": int(len(pts)),
            "sampled_pcd_min_xyz": [float(v) for v in mn],
            "sampled_pcd_max_xyz": [float(v) for v in mx],
        })

    if len(combined_pts) == 0:
        raise RuntimeError("Failed to build combined PCD for ground picking.")

    all_pts = np.concatenate(combined_pts, axis=0)
    all_cols = np.concatenate(combined_cols, axis=0)

    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(all_pts)
    combined_pcd.colors = o3d.utility.Vector3dVector(all_cols)

    return combined_pcd, chosen_infos


def pick_global_ground_z_from_random_obj_files(obj_files, args, obj_root):
    combined_pcd, chosen_infos = build_combined_ground_pick_pcd_from_random_obj_files(
        obj_files=obj_files,
        num_tiles=int(args.ground_pick_num_tiles),
        random_seed=int(args.ground_pick_seed),
        points_per_tile=int(args.ground_pick_points_per_tile),
        voxel_size=float(args.ground_pick_voxel),
    )

    if bool(args.show_ground_pick_preview):
        geoms_preview = [combined_pcd]
        pts = np.asarray(combined_pcd.points, dtype=np.float64)
        span = np.max(pts, axis=0) - np.min(pts, axis=0)
        frame_size = max(1.0, 0.02 * float(np.linalg.norm(span)))
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        geoms_preview.append(world_frame)
        o3d.visualization.draw_geometries(
            geoms_preview,
            window_name="Ground Pick Preview (Combined Random OBJ Samples)",
            width=1600,
            height=900
        )

    picked_pts = pick_ground_points_on_geometry(combined_pcd, "combined random OBJ samples")
    z_vals = picked_pts[:, 2].astype(np.float64)
    ground_z = float(np.median(z_vals))

    print(f"[GROUND] picked z values = {np.round(z_vals, 6).tolist()}")
    print(f"[GROUND] global ground_z (median) = {ground_z:.6f}")

    ground_meta = {
        "method": "manual_pick_from_random_obj_files",
        "ground_z": ground_z,
        "picked_points_xyz": picked_pts.tolist(),
        "picked_z_values": [float(z) for z in z_vals],
        "picked_z_median": ground_z,
        "picked_z_mean": float(np.mean(z_vals)),
        "picked_z_min": float(np.min(z_vals)),
        "picked_z_max": float(np.max(z_vals)),
        "params": {
            "ground_pick_num_tiles": int(args.ground_pick_num_tiles),
            "ground_pick_seed": int(args.ground_pick_seed),
            "ground_pick_points_per_tile": int(args.ground_pick_points_per_tile),
            "ground_pick_voxel": float(args.ground_pick_voxel),
        },
        "chosen_objs": chosen_infos,
    }

    ground_meta_path = get_global_ground_meta_path(obj_root)
    with open(ground_meta_path, "w", encoding="utf-8") as f:
        json.dump(ground_meta, f, ensure_ascii=False, indent=2)

    print(f"[GROUND] saved global ground meta to: {ground_meta_path}")
    return ground_z, ground_meta
# ==============================================================================
# Planning + rendering on single merged group
# ==============================================================================
def polygon_bounds_xy(poly_xy):
    poly = np.asarray(poly_xy, dtype=np.float64)
    mn_xy = np.min(poly, axis=0)
    mx_xy = np.max(poly, axis=0)
    return mn_xy, mx_xy


def polygon_area_xy(poly_xy):
    poly = np.asarray(poly_xy, dtype=np.float64)
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    area2 = np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    return abs(0.5 * float(area2))


def shrink_polygon_extent_xy(poly_xy, inset_m):
    inset = float(inset_m)
    if inset < 0:
        raise ValueError(f"inset_m must be >= 0, got {inset}")

    poly = np.asarray(poly_xy, dtype=np.float64)
    if inset == 0:
        return poly.copy()

    mn_xy, mx_xy = polygon_bounds_xy(poly)
    dx = float(mx_xy[0] - mn_xy[0])
    dy = float(mx_xy[1] - mn_xy[1])

    if dx <= 0 or dy <= 0:
        raise ValueError(f"Invalid polygon bbox for shrink: dx={dx}, dy={dy}")

    if 2.0 * inset >= dx or 2.0 * inset >= dy:
        raise ValueError(
            f"roi_inset={inset} too large. Need 2*inset < min(dx,dy), got dx={dx:.3f}, dy={dy:.3f}"
        )

    cx = 0.5 * (mn_xy[0] + mx_xy[0])
    cy = 0.5 * (mn_xy[1] + mx_xy[1])
    sx = (dx - 2.0 * inset) / dx
    sy = (dy - 2.0 * inset) / dy

    out = poly.copy()
    out[:, 0] = cx + (out[:, 0] - cx) * sx
    out[:, 1] = cy + (out[:, 1] - cy) * sy
    return out


def calc_swath_width(height_m, hfov_deg):
    return 2.0 * height_m * math.tan(math.radians(hfov_deg) / 2.0)


def choose_spacing(height_m, hfov_deg, side_overlap, forward_overlap):
    swath = calc_swath_width(height_m, hfov_deg)
    lane_spacing = max(0.2, swath * (1.0 - side_overlap))
    point_spacing = max(0.2, swath * (1.0 - forward_overlap))
    return swath, lane_spacing, point_spacing


def _rotate_xy(points_xy, angle_deg, center_xy):
    pts = np.asarray(points_xy, dtype=np.float64)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    th = math.radians(float(angle_deg))
    c, s = math.cos(th), math.sin(th)

    p = pts.copy()
    p[:, 0] -= cx
    p[:, 1] -= cy

    x_new = c * p[:, 0] - s * p[:, 1]
    y_new = s * p[:, 0] + c * p[:, 1]
    return np.stack([x_new + cx, y_new + cy], axis=1)


def _dedup_sorted_vals(vals, eps=1e-9):
    if len(vals) <= 1:
        return vals
    out = [vals[0]]
    for v in vals[1:]:
        if abs(v - out[-1]) > eps:
            out.append(v)
    return out


def _poly_scan_intersections_x(poly_xy, y, eps=1e-12):
    xs = []
    M = poly_xy.shape[0]
    for i in range(M):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % M]

        if abs(y2 - y1) < eps:
            continue

        ymin = min(y1, y2)
        ymax = max(y1, y2)

        if (y >= ymin - eps) and (y < ymax - eps):
            t = (y - y1) / (y2 - y1)
            x = x1 + t * (x2 - x1)
            xs.append(float(x))

    xs.sort()
    xs = _dedup_sorted_vals(xs, eps=1e-9)
    return xs


def _prune_close_waypoints(wpts, min_dist=1e-3):
    if len(wpts) <= 1:
        return wpts
    out = [wpts[0]]
    md2 = float(min_dist) * float(min_dist)
    for p in wpts[1:]:
        dx = p[0] - out[-1][0]
        dy = p[1] - out[-1][1]
        dz = p[2] - out[-1][2]
        if dx * dx + dy * dy + dz * dz >= md2:
            out.append(p)
    return out


def gen_lawnmower_polygon_local(poly_local_xy, altitude_z, lane_spacing, point_spacing, short_seg_ratio=0.35):
    poly = np.asarray(poly_local_xy, dtype=np.float64)
    miny = float(np.min(poly[:, 1]))
    maxy = float(np.max(poly[:, 1]))

    lane_vals = np.arange(miny, maxy + 1e-9, lane_spacing, dtype=np.float64)
    wpts = []
    min_seg_len = max(1e-6, float(short_seg_ratio) * float(point_spacing))

    for i, y in enumerate(lane_vals):
        inter = _poly_scan_intersections_x(poly, float(y))
        if len(inter) < 2:
            continue
        if len(inter) % 2 != 0:
            inter = inter[:-1]
            if len(inter) < 2:
                continue

        segs = []
        for k in range(0, len(inter), 2):
            a, b = inter[k], inter[k + 1]
            xL, xR = (a, b) if a <= b else (b, a)
            if (xR - xL) >= min_seg_len:
                segs.append((xL, xR))

        if len(segs) == 0:
            continue

        if i % 2 == 1:
            segs = segs[::-1]
            segs = [(b, a) for (a, b) in segs]

        for a, b in segs:
            if a <= b:
                xs_line = np.arange(a, b + 1e-9, point_spacing, dtype=np.float64)
            else:
                xs_line = np.arange(a, b - 1e-9, -point_spacing, dtype=np.float64)

            if xs_line.size == 0:
                xs_line = np.array([(a + b) * 0.5], dtype=np.float64)

            for x in xs_line:
                wpts.append((float(x), float(y), float(altitude_z)))

    return wpts


def gen_lawnmower_polygon_world(poly_world_xy,
                                altitude_z,
                                lane_spacing,
                                point_spacing,
                                path_angle_deg=None,
                                axis="auto",
                                short_seg_ratio=0.35):
    poly = np.asarray(poly_world_xy, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError(f"poly_world_xy shape invalid: {poly.shape}, expected (N,2), N>=3")

    mn_xy, mx_xy = polygon_bounds_xy(poly)
    dx = float(mx_xy[0] - mn_xy[0])
    dy = float(mx_xy[1] - mn_xy[1])

    if path_angle_deg is None:
        if axis == "x":
            used_angle = 0.0
        elif axis == "y":
            used_angle = 90.0
        elif axis == "auto":
            used_angle = 0.0 if dx >= dy else 90.0
        else:
            raise ValueError(f"Invalid axis={axis}")
    else:
        used_angle = float(path_angle_deg)

    cx = 0.5 * (mn_xy[0] + mx_xy[0])
    cy = 0.5 * (mn_xy[1] + mx_xy[1])

    poly_local = _rotate_xy(poly, angle_deg=-used_angle, center_xy=(cx, cy))
    wpts_local = gen_lawnmower_polygon_local(
        poly_local_xy=poly_local,
        altitude_z=altitude_z,
        lane_spacing=lane_spacing,
        point_spacing=point_spacing,
        short_seg_ratio=short_seg_ratio
    )
    if len(wpts_local) == 0:
        return [], used_angle

    pts_local_xy = np.array([[p[0], p[1]] for p in wpts_local], dtype=np.float64)
    pts_world_xy = _rotate_xy(pts_local_xy, angle_deg=used_angle, center_xy=(cx, cy))

    wpts = [(float(xy[0]), float(xy[1]), float(altitude_z)) for xy in pts_world_xy]
    wpts = _prune_close_waypoints(wpts, min_dist=min(1e-3, 0.01 * point_spacing))
    return wpts, used_angle


def yaw_to_next(curr, nxt):
    dx = nxt[0] - curr[0]
    dy = nxt[1] - curr[1]
    if abs(dx) + abs(dy) < 1e-9:
        return 0.0
    return math.atan2(dy, dx)


# ==============================================================================
# Camera pose / rendering (Open3D)
# ==============================================================================
def compose_Tcw_from_cam_pose(cam_pos_w, R_c2w):
    T_c2w = np.eye(4, dtype=np.float64)
    T_c2w[:3, :3] = np.asarray(R_c2w, dtype=np.float64)
    T_c2w[:3, 3] = np.asarray(cam_pos_w, dtype=np.float64)
    T_w2c = np.linalg.inv(T_c2w)
    return T_w2c


def rotation_world_axis(angle_rad, axis_world):
    axis = np.asarray(axis_world, dtype=np.float64)
    axis_u, _ = safe_norm(axis)
    if axis_u is None:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis_u
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    R = np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ],
    ], dtype=np.float64)
    return R


def build_camera_orientation_from_forward_up(forward_world, world_up=(0, 0, 1), roll_rad=0.0):
    f, _ = safe_norm(forward_world)
    if f is None:
        raise ValueError("forward_world is near zero.")

    up = np.asarray(world_up, dtype=np.float64)
    up_u, _ = safe_norm(up)
    if up_u is None:
        up_u = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    x_cam_w = np.cross(f, up_u)
    x_cam_w_u, _ = safe_norm(x_cam_w)
    if x_cam_w_u is None:
        alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_cam_w = np.cross(f, alt_up)
        x_cam_w_u, _ = safe_norm(x_cam_w)
        if x_cam_w_u is None:
            raise RuntimeError("Cannot build camera axes (forward parallel to fallback ups).")

    y_cam_w = np.cross(f, x_cam_w_u)
    y_cam_w_u, _ = safe_norm(y_cam_w)

    R_c2w = np.stack([x_cam_w_u, y_cam_w_u, f], axis=1)

    if abs(roll_rad) > 1e-12:
        R_roll_world = rotation_world_axis(roll_rad, f)
        R_c2w = R_roll_world @ R_c2w

    return orthonormalize_rotation(R_c2w)


def intrinsics_matrix(width, height, fx, fy, cx=None, cy=None):
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0
    return np.array([
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def save_depth_exr(depth_float, out_path):
    depth = np.asarray(depth_float, dtype=np.float32).copy()
    invalid = ~np.isfinite(depth)
    depth[invalid] = 0.0
    ok = cv2.imwrite(str(out_path), depth)
    if not ok:
        raise RuntimeError(f"Failed to write EXR: {out_path}")
    return depth


def compute_invalid_depth_ratio(depth_float):
    depth = np.asarray(depth_float, dtype=np.float32)
    invalid = (~np.isfinite(depth)) | (depth <= 0)
    return float(np.mean(invalid))


# ==============================================================================
# Five-camera rig definition (N/F/R/B/L)
# ==============================================================================
@dataclass
class VirtualCamSpec:
    role: str
    yaw_offset_deg: float
    pitch_deg: float
    roll_deg: float
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def build_five_camera_rig(width, height, fx, fy, cx, cy,
                          nadir_fx=None, nadir_fy=None,
                          oblique_fx=None, oblique_fy=None,
                          nadir_pitch_deg=-90.0, oblique_pitch_deg=-45.0,
                          roll_deg=0.0):
    nfx = float(fx if nadir_fx is None else nadir_fx)
    nfy = float(fy if nadir_fy is None else nadir_fy)
    ofx = float(fx if oblique_fx is None else oblique_fx)
    ofy = float(fy if oblique_fy is None else oblique_fy)

    specs = [
        VirtualCamSpec("N", 0.0,   float(nadir_pitch_deg),   float(roll_deg), nfx, nfy, float(cx), float(cy), int(width), int(height)),
        VirtualCamSpec("F", 0.0,   float(oblique_pitch_deg), float(roll_deg), ofx, ofy, float(cx), float(cy), int(width), int(height)),
        VirtualCamSpec("R", 90.0,  float(oblique_pitch_deg), float(roll_deg), ofx, ofy, float(cx), float(cy), int(width), int(height)),
        VirtualCamSpec("B", 180.0, float(oblique_pitch_deg), float(roll_deg), ofx, ofy, float(cx), float(cy), int(width), int(height)),
        VirtualCamSpec("L", 270.0, float(oblique_pitch_deg), float(roll_deg), ofx, ofy, float(cx), float(cy), int(width), int(height)),
    ]
    return specs


def build_ndir_camera_rig(width, height, fx, fy, cx, cy,
                          nadir_fx=None, nadir_fy=None,
                          oblique_fx=None, oblique_fy=None,
                          nadir_pitch_deg=-90.0, oblique_pitch_deg=-45.0,
                          roll_deg=0.0):
    nfx = float(fx if nadir_fx is None else nadir_fx)
    nfy = float(fy if nadir_fy is None else nadir_fy)
    ofx = float(fx if oblique_fx is None else oblique_fx)
    ofy = float(fy if oblique_fy is None else oblique_fy)

    specs = [
        VirtualCamSpec("N", 0.0,   float(nadir_pitch_deg),   float(roll_deg), nfx, nfy, float(cx), float(cy), int(width), int(height)),
    ]
    return specs


def forward_vector_from_yaw_pitch(yaw_rad, pitch_deg):
    p = math.radians(float(pitch_deg))
    cp = math.cos(p)
    sp = math.sin(p)
    return np.array([cp * math.cos(yaw_rad), cp * math.sin(yaw_rad), sp], dtype=np.float64)


def setup_view_control_with_camera(vc, width, height, K, T_w2c):
    cam = vc.convert_to_pinhole_camera_parameters()
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        int(width), int(height),
        float(K[0, 0]), float(K[1, 1]),
        float(K[0, 2]), float(K[1, 2]),
    )
    cam.intrinsic = intrinsic
    cam.extrinsic = T_w2c
    try:
        vc.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
    except TypeError:
        vc.convert_from_pinhole_camera_parameters(cam)

def render_single_view_with_vis(render_vis, vc, width, height, K, T_w2c, settle=0.0,
                                out_w=None, out_h=None,
                                rgb_median_ksize=0,
                                rgb_use_bilateral=False,
                                depth_resize_mode="valid_mean"):
    setup_view_control_with_camera(vc, width, height, K, T_w2c)
    render_vis.poll_events()
    render_vis.update_renderer()
    if settle > 0:
        time.sleep(float(settle))

    depth_raw = np.asarray(render_vis.capture_depth_float_buffer(do_render=True), dtype=np.float32)
    rgb_float = np.asarray(render_vis.capture_screen_float_buffer(do_render=True), dtype=np.float32)
    rgb_u8 = (np.clip(rgb_float, 0.0, 1.0) * 255).astype(np.uint8)

    if out_w is None:
        out_w = width
    if out_h is None:
        out_h = height

    rgb_out = postprocess_rgb(
        rgb_u8,
        out_w=int(out_w),
        out_h=int(out_h),
        median_ksize=int(rgb_median_ksize),
        use_bilateral=bool(rgb_use_bilateral),
    )

    depth_out = postprocess_depth(
        depth_raw,
        out_w=int(out_w),
        out_h=int(out_h),
        resize_mode=str(depth_resize_mode),
    )

    return rgb_out, depth_out


def normalize_odd_ksize(ksize):
    if ksize is None:
        return 0
    ksize = int(ksize)
    if ksize <= 1:
        return 0
    if ksize % 2 == 0:
        ksize += 1
    return ksize

def postprocess_rgb(rgb_u8, out_w, out_h, median_ksize=0, use_bilateral=False):
    rgb_u8 = np.asarray(rgb_u8, dtype=np.uint8)
    if rgb_u8.shape[1] != int(out_w) or rgb_u8.shape[0] != int(out_h):
        rgb_u8 = cv2.resize(rgb_u8, (int(out_w), int(out_h)), interpolation=cv2.INTER_AREA)
    median_ksize = normalize_odd_ksize(median_ksize)
    if median_ksize > 0:
        rgb_u8 = cv2.medianBlur(rgb_u8, median_ksize)
    if use_bilateral:
        rgb_u8 = cv2.bilateralFilter(rgb_u8, d=5, sigmaColor=25, sigmaSpace=25)
    return rgb_u8

def resize_depth_valid_mean(depth_float, out_w, out_h, eps=1e-6):
    depth = np.asarray(depth_float, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)

    depth_safe = np.where(valid, depth, 0.0).astype(np.float32)
    valid_f = valid.astype(np.float32)

    if depth.shape[1] == int(out_w) and depth.shape[0] == int(out_h):
        out = depth_safe
    else:
        # INTER_AREA computes an area average.
        avg_depth = cv2.resize(depth_safe, (int(out_w), int(out_h)), interpolation=cv2.INTER_AREA)
        avg_valid = cv2.resize(valid_f, (int(out_w), int(out_h)), interpolation=cv2.INTER_AREA)

        out = np.zeros((int(out_h), int(out_w)), dtype=np.float32)
        m = avg_valid > float(eps)
        out[m] = avg_depth[m] / avg_valid[m]

    out[~np.isfinite(out)] = 0.0
    out[out < 0] = 0.0
    return out.astype(np.float32)

def postprocess_depth(depth_float, out_w, out_h, resize_mode="valid_mean"):
    depth = np.asarray(depth_float, dtype=np.float32)

    if depth.shape[1] == int(out_w) and depth.shape[0] == int(out_h):
        out = depth.copy()
        out[~np.isfinite(out)] = 0.0
        out[out <= 0] = 0.0
        return out.astype(np.float32)

    if resize_mode == "nearest":
        depth_safe = depth.copy()
        depth_safe[~np.isfinite(depth_safe)] = 0.0
        depth_safe[depth_safe <= 0] = 0.0
        out = cv2.resize(depth_safe, (int(out_w), int(out_h)), interpolation=cv2.INTER_NEAREST)

    elif resize_mode == "area":
        depth_safe = depth.copy()
        depth_safe[~np.isfinite(depth_safe)] = 0.0
        depth_safe[depth_safe <= 0] = 0.0
        out = cv2.resize(depth_safe, (int(out_w), int(out_h)), interpolation=cv2.INTER_AREA)

    elif resize_mode == "valid_mean":
        out = resize_depth_valid_mean(depth, out_w, out_h)

    else:
        raise ValueError(f"Unsupported depth resize_mode: {resize_mode}")

    out = np.asarray(out, dtype=np.float32)
    out[~np.isfinite(out)] = 0.0
    out[out <= 0] = 0.0
    return out

def render_waypoints_open3d_fivecams(mesh, wpts, out_dir,
                                     cam_specs,
                                     yaw_mode="path",
                                     max_invalid_depth_ratio=0.3,
                                     settle=0.02,
                                     save_plan_json=True,
                                     chunk_idx=None,
                                     lane_id=12,
                                     ssaa=1,
                                     rgb_median_ksize=0,
                                     rgb_use_bilateral=False,
                                     depth_resize_mode="valid_mean"):
    out_dir = Path(out_dir)
    ssaa = max(1, int(ssaa))

    first_render_w = int(cam_specs[0].width) * ssaa
    first_render_h = int(cam_specs[0].height) * ssaa

    if save_plan_json:
        plan = {
            "chunk_idx": None if chunk_idx is None else int(chunk_idx),
            "num_waypoints": len(wpts),
            "yaw_mode": str(yaw_mode),
            "max_invalid_depth_ratio": float(max_invalid_depth_ratio),
            "ssaa": int(ssaa),
            "rgb_median_ksize": int(rgb_median_ksize),
            "rgb_use_bilateral": bool(rgb_use_bilateral),
            "depth_resize_mode": str(depth_resize_mode),
            "camera_specs": [
                {
                    "role": c.role,
                    "yaw_offset_deg": c.yaw_offset_deg,
                    "pitch_deg": c.pitch_deg,
                    "roll_deg": c.roll_deg,
                    "width": c.width,
                    "height": c.height,
                    "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy
                } for c in cam_specs
            ],
            "waypoints_xyz": [[float(x), float(y), float(z)] for (x, y, z) in wpts],
        }
        with open(out_dir / "meta" / "planned_waypoints.json", "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

    render_vis = o3d.visualization.Visualizer()
    render_vis.create_window(
        window_name="HiddenRenderer",
        width=int(first_render_w),
        height=int(first_render_h),
        visible=False
    )
    render_vis.add_geometry(mesh)

    ropt = render_vis.get_render_option()
    ropt.mesh_show_back_face = True
    ropt.light_on = True
    vc = render_vis.get_view_control()

    curr_render_w = int(first_render_w)
    curr_render_h = int(first_render_h)

    saved_waypoints = 0
    fully_skipped_waypoints = 0
    saved_frames = 0
    skipped_frames = 0
    total_waypoints = len(wpts)

    role_saved_counts = {c.role: 0 for c in cam_specs}
    role_skipped_counts = {c.role: 0 for c in cam_specs}

    saved_waypoint_idx = 0

    for i in tqdm(range(len(wpts)), desc=f"Render{' chunk'+str(chunk_idx) if chunk_idx is not None else ''}"):
        curr = wpts[i]
        nxt = wpts[i + 1] if i + 1 < len(wpts) else curr

        x, y, z = curr
        path_yaw = yaw_to_next(curr, nxt) if yaw_mode == "path" else 0.0
        cam_pos_w = np.array([x, y, z], dtype=np.float64)

        role_results = []

        for spec in cam_specs:
            yaw_role = path_yaw + math.radians(float(spec.yaw_offset_deg))
            fwd_world = forward_vector_from_yaw_pitch(yaw_role, spec.pitch_deg)
            R_c2w = build_camera_orientation_from_forward_up(
                forward_world=fwd_world,
                world_up=(0.0, 0.0, 1.0),
                roll_rad=math.radians(float(spec.roll_deg))
            )
            T_w2c = compose_Tcw_from_cam_pose(cam_pos_w, R_c2w)

            # Intrinsics for the final output resolution, written to cam.txt.
            K_out = intrinsics_matrix(
                spec.width, spec.height,
                spec.fx, spec.fy, spec.cx, spec.cy
            )

            # Intrinsics for the supersampled render resolution only.
            render_w = int(spec.width) * ssaa
            render_h = int(spec.height) * ssaa
            render_fx = float(spec.fx) * ssaa
            render_fy = float(spec.fy) * ssaa
            render_cx = float(spec.cx) * ssaa
            render_cy = float(spec.cy) * ssaa

            K_render = intrinsics_matrix(
                render_w, render_h,
                render_fx, render_fy, render_cx, render_cy
            )

            # Recreate the hidden window when the current camera needs a different render size.
            if render_w != curr_render_w or render_h != curr_render_h:
                render_vis.destroy_window()
                render_vis = o3d.visualization.Visualizer()
                render_vis.create_window(
                    window_name="HiddenRenderer",
                    width=int(render_w),
                    height=int(render_h),
                    visible=False
                )
                render_vis.add_geometry(mesh)

                ropt = render_vis.get_render_option()
                ropt.mesh_show_back_face = True
                ropt.light_on = True
                vc = render_vis.get_view_control()

                curr_render_w = int(render_w)
                curr_render_h = int(render_h)

            rgb_u8, depth_float = render_single_view_with_vis(
                render_vis, vc,
                render_w, render_h,
                K_render, T_w2c,
                settle=settle,
                out_w=spec.width,
                out_h=spec.height,
                rgb_median_ksize=rgb_median_ksize,
                rgb_use_bilateral=rgb_use_bilateral,
                depth_resize_mode=depth_resize_mode,
            )

            invalid_ratio = compute_invalid_depth_ratio(depth_float)

            role_results.append({
                "spec": spec,
                "K_out": K_out,
                "T_w2c": T_w2c,
                "rgb_u8": rgb_u8,
                "depth_float": depth_float,
                "invalid_ratio": invalid_ratio,
                "keep": bool(invalid_ratio <= float(max_invalid_depth_ratio)),
            })

        keep_any = any(rr["keep"] for rr in role_results)
        if not keep_any:
            fully_skipped_waypoints += 1
            skipped_frames += len(role_results)
            for rr in role_results:
                role_skipped_counts[rr["spec"].role] += 1

            ratios_str = ", ".join([f"{r['spec'].role}:{r['invalid_ratio']:.3f}" for r in role_results])
            print(f"[SKIP] wp={i:06d} all roles invalid (> {max_invalid_depth_ratio:.3f}) | {ratios_str}")
            continue

        base_idx = saved_waypoint_idx
        saved_waypoint_idx += 1

        saved_this_wp = 0
        saved_roles = []
        skipped_roles = []

        for rr in role_results:
            spec = rr["spec"]
            role = spec.role

            if not rr["keep"]:
                skipped_frames += 1
                role_skipped_counts[role] += 1
                skipped_roles.append(f"{role}:{rr['invalid_ratio']:.3f}")
                continue

            stem = f"{role}_{int(lane_id):03d}_{base_idx:08d}"

            img_path = out_dir / "images" / f"{stem}.png"
            dep_path = out_dir / "depth" / f"{stem}.exr"
            cam_path = out_dir / "cams" / f"{stem}.txt"

            imageio.imwrite(img_path, rr["rgb_u8"])
            save_depth_exr(rr["depth_float"], dep_path)

            # Write output-resolution intrinsics, not supersampled intrinsics.
            write_cam_file(
                cam_path=cam_path,
                extrinsic=rr["T_w2c"],
                fx=float(rr["K_out"][0, 0]),
                fy=float(rr["K_out"][1, 1]),
                cx=float(rr["K_out"][0, 2]),
                cy=float(rr["K_out"][1, 2]),
                h=int(spec.height),
                w=int(spec.width)
            )

            saved_frames += 1
            saved_this_wp += 1
            role_saved_counts[role] += 1
            saved_roles.append(f"{role}:{rr['invalid_ratio']:.3f}")

        saved_waypoints += 1
        print(f"[SAVE] wp={i:06d} -> saved_idx={base_idx:08d} | saved={saved_this_wp}/5 | "
              f"keep[{', '.join(saved_roles)}]"
              + (f" | skip[{', '.join(skipped_roles)}]" if skipped_roles else ""))

    render_vis.destroy_window()

    print(f"[DONE] saved_waypoints={saved_waypoints}, fully_skipped_waypoints={fully_skipped_waypoints}, "
          f"saved_frames={saved_frames}, skipped_frames={skipped_frames}, total_waypoints={total_waypoints}")
    print(f"[DONE] role_saved_counts={role_saved_counts}")
    print(f"[DONE] role_skipped_counts={role_skipped_counts}")

    return {
        "saved_waypoints": int(saved_waypoints),
        "skipped_waypoints": int(fully_skipped_waypoints),
        "saved_frames": int(saved_frames),
        "skipped_frames": int(skipped_frames),
        "total_waypoints": int(total_waypoints),
        "role_saved_counts": {k: int(v) for k, v in role_saved_counts.items()},
        "role_skipped_counts": {k: int(v) for k, v in role_skipped_counts.items()},
    }

# ==============================================================================
# Optional visualization
# ==============================================================================
def create_polyline_3d(points_xyz, color=(1.0, 1.0, 0.0), closed=True):
    pts = np.asarray(points_xyz, dtype=np.float64)
    if len(pts) < 2:
        return None
    lines = [[i, i + 1] for i in range(len(pts) - 1)]
    if closed and len(pts) >= 3:
        lines.append([len(pts) - 1, 0])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1)))
    return ls


def create_waypoint_traj_lineset(wpts_xyz, color=(1.0, 0.0, 1.0)):
    pts = np.asarray(wpts_xyz, dtype=np.float64)
    if len(pts) < 2:
        return None
    lines = [[i, i + 1] for i in range(len(pts) - 1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1)))
    return ls


# ==============================================================================
# Planning + rendering on single merged group
# ==============================================================================
def run_planning_and_render_on_single_mesh(
    mesh, scene_root: Path, args, group_info=None, global_ground_z=None):
    if global_ground_z is None:
        raise ValueError("global_ground_z must be provided.")

    scene_root = make_scene_dirs(scene_root)

    aabb = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(aabb.get_extent(), dtype=np.float64)
    scene_scale = float(np.linalg.norm(extent))
    print(f"[INFO] scene_scale≈{scene_scale:.3f}")

    picked_geom = mesh
    picked_geom_name = "mesh"
    sparse_pcd = None

    if bool(args.auto_roi_from_group_bbox):
        used_group_bbox_union = False
        if (group_info is not None) and ("bbox_xyxy" in group_info):
            x0, x1, y0, y1 = group_info["bbox_xyxy"]

            picked_pts_xyz, roi_poly_xy = auto_roi_polygon_from_bbox_xyxy(
                x0, x1, y0, y1,
                z_mean=float(global_ground_z),
                margin_ratio=float(args.auto_roi_margin_ratio)
            )
            picked_geom_name = "auto_group_tiles_bbox_union"
            used_group_bbox_union = True
            print(f"[ROI] Auto ROI from GROUP tile bbox union (margin_ratio={float(args.auto_roi_margin_ratio):.3f})")

        if not used_group_bbox_union:
            aabb = mesh.get_axis_aligned_bounding_box()
            mn = aabb.get_min_bound()
            mx = aabb.get_max_bound()
            picked_pts_xyz, roi_poly_xy = auto_roi_polygon_from_bbox_xyxy(
                mn[0], mx[0], mn[1], mx[1],
                z_mean=float(global_ground_z),
                margin_ratio=float(args.auto_roi_margin_ratio)
            )
            picked_geom_name = "auto_mesh_bbox_roi"
            print(f"[ROI] Auto ROI from merged mesh bbox (fallback) (margin_ratio={float(args.auto_roi_margin_ratio):.3f})")
    else:
        if bool(args.pick_on_sparse_pcd):
            sparse_pcd = sample_sparse_pcd_from_mesh(
                mesh,
                n_points=int(args.roi_pick_points),
                voxel_size=float(args.roi_pick_voxel)
            )
            picked_geom = sparse_pcd
            picked_geom_name = "sparse point cloud"

            if bool(args.show_sparse_pcd_preview):
                geoms_preview = [sparse_pcd]
                wf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(1.0, 0.02 * scene_scale))
                geoms_preview.append(wf)
                o3d.visualization.draw_geometries(
                    geoms_preview,
                    window_name="Sparse PCD Preview for ROI Picking",
                    width=1600,
                    height=900
                )

        picked_pts_xyz = pick_polygon_points_on_geometry(picked_geom, picked_geom_name)
        roi_poly_xy = picked_pts_xyz[:, :2].copy()

    if float(args.roi_inset) > 0:
        roi_poly_xy = shrink_polygon_extent_xy(roi_poly_xy, float(args.roi_inset))
        print(f"[ROI] Applied inset={float(args.roi_inset):.3f} (bbox-extent shrink)")

    roi_ref_z = float(global_ground_z)
    cam_z = float(global_ground_z) + float(args.flight_height)
    plan_height_agl = float(args.flight_height)

    print(f"[ALT] global_ground_z={global_ground_z:.3f}, rel_flight_height={float(args.flight_height):.3f}, cam_z={cam_z:.3f}")

    width = int(args.width)
    height = int(args.height)
    cx = float(args.cx) if args.cx is not None else width / 2.0
    cy = float(args.cy) if args.cy is not None else height / 2.0

    if getattr(args, "camera_rig", "nadir") == "five":
        cam_specs = build_five_camera_rig(
            width=width, height=height,
            fx=float(args.nadir_fx), fy=float(args.nadir_fy),
            cx=cx, cy=cy,
            nadir_fx=float(args.nadir_fx), nadir_fy=float(args.nadir_fy),
            oblique_fx=float(args.oblique_fx), oblique_fy=float(args.oblique_fy),
            nadir_pitch_deg=float(args.nadir_pitch_deg),
            oblique_pitch_deg=float(args.oblique_pitch_deg),
            roll_deg=float(args.roll_deg),
        )
    else:
        cam_specs = build_ndir_camera_rig(
            width=width, height=height,
            fx=float(args.nadir_fx), fy=float(args.nadir_fy),
            cx=cx, cy=cy,
            nadir_fx=float(args.nadir_fx), nadir_fy=float(args.nadir_fy),
            oblique_fx=float(args.oblique_fx), oblique_fy=float(args.oblique_fy),
            nadir_pitch_deg=float(args.nadir_pitch_deg),
            oblique_pitch_deg=float(args.oblique_pitch_deg),
            roll_deg=float(args.roll_deg),
        )

    role_order = [c.role for c in cam_specs]
    roles_count = len(role_order)

    hfov_deg_plan = float(np.degrees(2.0 * np.arctan(width / (2.0 * float(args.nadir_fx)))))
    swath, lane_spacing, point_spacing = choose_spacing(
        height_m=plan_height_agl,
        hfov_deg=hfov_deg_plan,
        side_overlap=float(args.side_overlap),
        forward_overlap=float(args.forward_overlap),
    )

    wpts_full, used_angle = gen_lawnmower_polygon_world(
        poly_world_xy=roi_poly_xy,
        altitude_z=cam_z,
        lane_spacing=lane_spacing,
        point_spacing=point_spacing,
        path_angle_deg=(None if args.path_angle is None else float(args.path_angle)),
        axis=args.axis,
        short_seg_ratio=float(args.short_seg_ratio),
    )
    if len(wpts_full) == 0:
        raise RuntimeError("No waypoints generated. Check ROI polygon / overlaps / altitude.")

    area_m2 = polygon_area_xy(roi_poly_xy)
    mn_xy, mx_xy = polygon_bounds_xy(roi_poly_xy)
    total_est_images_full = int(len(wpts_full) * roles_count)

    print("\n[PLAN]")
    print(f"  ROI source={picked_geom_name}")
    print(f"  ROI vertices={len(roi_poly_xy)}")
    print(f"  ROI bbox size=({mx_xy[0]-mn_xy[0]:.3f}, {mx_xy[1]-mn_xy[1]:.3f})")
    print(f"  ROI area={area_m2:.3f}")
    print(f"  global_ground_z={global_ground_z:.3f}")
    print(f"  rel_flight_height={float(args.flight_height):.3f}")
    print(f"  cam_z(abs)={cam_z:.3f}")
    print(f"  HFOV(plan, nadir)={hfov_deg_plan:.3f} deg")
    print(f"  swath={swath:.3f}, lane_spacing={lane_spacing:.3f}, point_spacing={point_spacing:.3f}")
    print(f"  path_angle_used={used_angle:.3f} deg")
    print(f"  waypoints(full)={len(wpts_full)}")
    print(f"  est_total_images(full, five cams)={total_est_images_full}")
    print(f"  roles={role_order}")
    print(f"  yaw_mode={args.yaw_mode}, nadir_pitch={args.nadir_pitch_deg}, oblique_pitch={args.oblique_pitch_deg}")

    planning_info = {
        "picked_on": picked_geom_name,
        "picked_points_xyz": picked_pts_xyz.tolist(),
        "roi_polygon_xy": roi_poly_xy.tolist(),
        "roi_area_m2": area_m2,
        "global_ground_z": float(global_ground_z),
        "roi_reference_z": roi_ref_z,
        "camera_rig_z": cam_z,
        "flight_height_rel": float(args.flight_height),
        "plan_height_agl": float(plan_height_agl),
        "hfov_deg_plan_nadir": hfov_deg_plan,
        "swath": swath,
        "lane_spacing": lane_spacing,
        "point_spacing": point_spacing,
        "path_angle_used_deg": used_angle,
        "num_waypoints_full": len(wpts_full),
        "roles": role_order,
        "roles_count": roles_count,
        "est_total_images_full": total_est_images_full,
        "camera_specs": [
            {
                "role": c.role,
                "yaw_offset_deg": c.yaw_offset_deg,
                "pitch_deg": c.pitch_deg,
                "roll_deg": c.roll_deg,
                "width": c.width,
                "height": c.height,
                "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy
            } for c in cam_specs
        ]
    }
    if group_info is not None:
        planning_info["group_info"] = group_info

    with open(scene_root / "meta" / "roi_and_plan_meta.json", "w", encoding="utf-8") as f:
        json.dump(planning_info, f, ensure_ascii=False, indent=2)

    if args.show_plan:
        roi_poly_xyz = np.column_stack([roi_poly_xy, np.full((len(roi_poly_xy),), roi_ref_z, dtype=np.float64)])
        roi_ls = create_polyline_3d(roi_poly_xyz, color=(1.0, 1.0, 0.0), closed=True)
        traj_ls = create_waypoint_traj_lineset(np.asarray(wpts_full), color=(1.0, 0.0, 1.0))

        geoms = [mesh]
        if roi_ls is not None:
            geoms.append(roi_ls)
        if traj_ls is not None:
            geoms.append(traj_ls)

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(1.0, 0.02 * scene_scale))
        geoms.append(world_frame)

        o3d.visualization.draw_geometries(
            geoms,
            window_name="Planned ROI + Trajectory Preview",
            width=1600,
            height=900,
            mesh_show_back_face=True
        )

    out_scene_dir = make_scene_dirs(scene_root)
    stats = render_waypoints_open3d_fivecams(
        mesh=mesh,
        wpts=wpts_full,
        out_dir=out_scene_dir,
        cam_specs=cam_specs,
        yaw_mode=str(args.yaw_mode),
        max_invalid_depth_ratio=float(args.max_invalid_depth_ratio),
        settle=float(args.settle),
        save_plan_json=True,
        chunk_idx=None,
        lane_id=int(args.lane_id),
        ssaa=int(args.ssaa),
        rgb_median_ksize=int(args.rgb_median_ksize),
        rgb_use_bilateral=bool(args.rgb_use_bilateral),
        depth_resize_mode=str(args.depth_resize_mode),
    )

    return stats


def build_display_transform_from_tile_metas(tile_metas, target_span=300.0, margin_ratio=0.05):
    centers = np.asarray([m["center_xy"] for m in tile_metas], dtype=np.float64)
    if len(centers) == 0:
        raise RuntimeError("tile_metas is empty")

    mn = centers.min(axis=0)
    mx = centers.max(axis=0)
    center = 0.5 * (mn + mx)
    span = mx - mn
    max_span = max(float(span[0]), float(span[1]), 1e-6)

    usable_span = float(target_span) * max(1e-6, (1.0 - 2.0 * float(margin_ratio)))
    scale = usable_span / max_span

    return {
        "world_center_xy": [float(center[0]), float(center[1])],
        "scale": float(scale),
        "target_span": float(target_span),
        "margin_ratio": float(margin_ratio),
        "world_min_xy": [float(mn[0]), float(mn[1])],
        "world_max_xy": [float(mx[0]), float(mx[1])],
    }


def world_xy_to_display_xy(points_xy, tf):
    pts = np.asarray(points_xy, dtype=np.float64)
    c = np.asarray(tf["world_center_xy"], dtype=np.float64)
    s = float(tf["scale"])
    return (pts - c[None, :]) * s


def get_display_hull_xy(meta, display_tf):
    hull_xy = np.asarray(meta["hull_xy"], dtype=np.float64)
    return world_xy_to_display_xy(hull_xy, display_tf)


def get_display_center_xy(meta, display_tf):
    center_xy = np.asarray(meta["center_xy"], dtype=np.float64)[None, :]
    return world_xy_to_display_xy(center_xy, display_tf)[0]

def format_height_tag(height):
    h = float(height)
    s = f"{h:.3f}".rstrip("0").rstrip(".")
    s = s.replace("-", "neg").replace(".", "p")
    return s


def get_group_cache_dir(root_out: Path, gid: int):
    """
    Return the stable cache directory for merged OBJ files and pre-merge metadata.
    """
    return Path(root_out) / f"group_{gid:03d}"


def get_group_render_dir(root_out: Path, gid: int, flight_height: float):
    """
    Return the render output directory for the current flight height.

    Examples: group_h180_000, group_h120p5_003.
    """
    htag = format_height_tag(flight_height)
    return Path(root_out) / f"group_h{htag}_{gid:03d}"


def scene_outputs_exist(scene_dir: Path):
    """
    Return True when a scene directory already contains rendered images, depth maps, and camera files.
    """
    scene_dir = Path(scene_dir)
    images_dir = scene_dir / "images"
    depth_dir = scene_dir / "depth"
    cams_dir = scene_dir / "cams"

    if not images_dir.exists() or not depth_dir.exists() or not cams_dir.exists():
        return False

    n_img = len(list(images_dir.glob("*.png")))
    n_dep = len(list(depth_dir.glob("*.exr")))
    n_cam = len(list(cams_dir.glob("*.txt")))

    return (n_img > 0) and (n_dep > 0) and (n_cam > 0)


def find_reusable_merged_obj(root_out: Path, gid: int, current_render_dir: Path):
    """
    Find a reusable merged_group.obj.

    Priority:
      1) Stable cache directory: group_{gid}/merged_group.obj
      2) Current render directory: merged_group.obj
    """
    cache_dir = get_group_cache_dir(root_out, gid)
    candidates = [
        cache_dir / "merged_group.obj",
        Path(current_render_dir) / "merged_group.obj",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

# ==============================================================================
# Standardized CLI helpers
# ==============================================================================
def _parse_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}.")
    return ivalue


def _parse_nonnegative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative integer, got {value!r}.")
    return ivalue


def _parse_positive_float(value):
    fvalue = float(value)
    if fvalue <= 0.0:
        raise argparse.ArgumentTypeError(f"Expected a positive float, got {value!r}.")
    return fvalue


def _parse_nonnegative_float(value):
    fvalue = float(value)
    if fvalue < 0.0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative float, got {value!r}.")
    return fvalue


def _parse_ratio_0_1(value):
    fvalue = float(value)
    if not (0.0 <= fvalue < 1.0):
        raise argparse.ArgumentTypeError(f"Expected a ratio in [0, 1), got {value!r}.")
    return fvalue


def _parse_ratio_0_05(value):
    fvalue = float(value)
    if not (0.0 <= fvalue < 0.5):
        raise argparse.ArgumentTypeError(f"Expected a ratio in [0, 0.5), got {value!r}.")
    return fvalue


def _parse_depth_invalid_ratio(value):
    fvalue = float(value)
    if not (0.0 <= fvalue <= 1.0):
        raise argparse.ArgumentTypeError(f"Expected a ratio in [0, 1], got {value!r}.")
    return fvalue


def _parse_optional_float(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "auto", "null"}:
        return None
    return float(value)


def _add_path_arg(parser, *flags, dest, default, help_text):
    parser.add_argument(*flags, dest=dest, type=str, default=default, help=help_text)


def _add_bool_arg(parser, positive_flags, negative_flags, dest, default, help_text):
    """Add a predictable boolean flag pair while keeping underscore aliases."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(*positive_flags, dest=dest, action="store_true", help=help_text)
    group.add_argument(*negative_flags, dest=dest, action="store_false", help=f"Disable: {help_text}")
    parser.set_defaults(**{dest: bool(default)})


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Scan multi-OBJ tiles, manually group tiles by 2D centers, merge groups, "
            "plan a lawnmower ROI trajectory, and render camera images/depth with Open3D."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / external merge tool.
    _add_path_arg(parser, "--obj-root", "--obj_root", dest="obj_root",
                  default="models/terra_obj_002/terra_obj",
                  help_text="Directory containing OBJ tiles, or a single OBJ file.")
    _add_bool_arg(parser,
                  ["--recursive", "--obj-recursive", "--obj_recursive"],
                  ["--no-recursive", "--no-obj-recursive", "--no_obj_recursive"],
                  dest="obj_recursive", default=True,
                  help_text="Recursively search for OBJ files under obj_root.")
    _add_path_arg(parser, "--merge-tiles-exe", "--merge_tiles_exe", dest="merge_tiles_exe",
                  default="./merge_tiles", help_text="Path to the external merge_tiles executable.")
    _add_bool_arg(parser,
                  ["--reuse-merged-groups", "--reuse_merged_groups"],
                  ["--no-reuse-merged-groups", "--no_reuse_merged_groups"],
                  dest="reuse_merged_groups", default=True,
                  help_text="Reuse an existing merged_group.obj instead of running merge_tiles again.")
    _add_bool_arg(parser,
                  ["--keep-merge-tmp-inputs", "--keep_merge_tmp_inputs"],
                  ["--no-keep-merge-tmp-inputs", "--no_keep_merge_tmp_inputs"],
                  dest="keep_merge_tmp_inputs", default=False,
                  help_text="Keep temporary copied OBJ inputs used by merge_tiles for debugging.")
    _add_bool_arg(parser,
                  ["--delete-merged-obj-after-render", "--delete_merged_obj_after_render"],
                  ["--no-delete-merged-obj-after-render", "--no_delete_merged_obj_after_render"],
                  dest="delete_merged_obj_after_render", default=False,
                  help_text="Delete merged_group.obj and sidecar textures after successful rendering.")

    # Output naming.
    parser.add_argument("--lane-id", "--lane_id", dest="lane_id", type=_parse_nonnegative_int, default=12,
                        help="Lane/route id used in output filenames, e.g. N_012_00000001.")
    _add_path_arg(parser, "--out-dir", "--out_dir", dest="out_dir",
                  default="captures_obj_plan_fivecam/scene_002",
                  help_text="Output root directory for all groups.")

    # Manual grouping.
    _add_bool_arg(parser,
                  ["--reuse-manual-groups", "--reuse_manual_groups"],
                  ["--no-reuse-manual-groups", "--no_reuse_manual_groups"],
                  dest="reuse_manual_groups", default=True,
                  help_text="Reuse saved manual grouping metadata when available.")
    parser.add_argument("--group-vis-z", "--group_vis_z", dest="group_vis_z", type=float, default=0.0,
                        help="Z value used to visualize tile hulls and centers during manual grouping.")
    _add_bool_arg(parser,
                  ["--keep-ungrouped-as-one", "--keep_ungrouped_as_one"],
                  ["--no-keep-ungrouped-as-one", "--no_keep_ungrouped_as_one"],
                  dest="keep_ungrouped_as_one", default=False,
                  help_text="Append all remaining unpicked tiles as one final group.")
    parser.add_argument("--group-display-target-span", "--group_display_target_span", dest="group_display_target_span",
                        type=_parse_positive_float, default=100.0,
                        help="Normalized display span for the grouping visualization.")
    parser.add_argument("--group-display-margin-ratio", "--group_display_margin_ratio", dest="group_display_margin_ratio",
                        type=_parse_ratio_0_05, default=0.05,
                        help="Margin ratio added around the normalized grouping visualization.")

    # Ground-Z picking.
    parser.add_argument("--ground-pick-num-tiles", "--ground_pick_num_tiles", dest="ground_pick_num_tiles",
                        type=_parse_positive_int, default=1,
                        help="Number of random OBJ tiles used for global ground-Z picking.")
    parser.add_argument("--ground-pick-seed", "--ground_pick_seed", dest="ground_pick_seed", type=int, default=-1,
                        help="Random seed for selecting OBJ tiles for ground picking; -1 means random.")
    parser.add_argument("--ground-pick-points-per-tile", "--ground_pick_points_per_tile", dest="ground_pick_points_per_tile",
                        type=_parse_positive_int, default=8000,
                        help="Number of sampled points per chosen tile for global ground-Z picking.")
    parser.add_argument("--ground-pick-voxel", "--ground_pick_voxel", dest="ground_pick_voxel",
                        type=_parse_nonnegative_float, default=0.0,
                        help="Voxel size for ground-picking point-cloud downsampling; 0 disables downsampling.")
    _add_bool_arg(parser,
                  ["--show-ground-pick-preview", "--show_ground_pick_preview"],
                  ["--no-show-ground-pick-preview", "--no_show_ground_pick_preview"],
                  dest="show_ground_pick_preview", default=False,
                  help_text="Preview the combined sampled point cloud before picking global ground points.")

    # Skip / cache policy.
    _add_bool_arg(parser,
                  ["--skip-existing-height-groups", "--skip_existing_height_groups"],
                  ["--no-skip-existing-height-groups", "--no_skip_existing_height_groups"],
                  dest="skip_existing_height_groups", default=True,
                  help_text="Skip a flight-height group when images/depth/cams already exist.")

    # ROI mode.
    _add_bool_arg(parser,
                  ["--pick-on-sparse-pcd", "--pick_on_sparse_pcd"],
                  ["--no-pick-on-sparse-pcd", "--no_pick_on_sparse_pcd"],
                  dest="pick_on_sparse_pcd", default=True,
                  help_text="Use a sparse sampled point cloud for manual ROI picking instead of the full mesh.")
    parser.add_argument("--roi-pick-points", "--roi_pick_points", dest="roi_pick_points",
                        type=_parse_positive_int, default=20000,
                        help="Number of mesh points sampled for ROI-picking assistance.")
    parser.add_argument("--roi-pick-voxel", "--roi_pick_voxel", dest="roi_pick_voxel",
                        type=_parse_nonnegative_float, default=0.0,
                        help="Voxel size for sampled ROI point-cloud downsampling; 0 disables downsampling.")
    _add_bool_arg(parser,
                  ["--auto-roi-from-group-bbox", "--auto_roi_from_group_bbox"],
                  ["--no-auto-roi-from-group-bbox", "--no_auto_roi_from_group_bbox", "--disable-auto-roi-from-group-bbox", "--disable_auto_roi_from_group_bbox"],
                  dest="auto_roi_from_group_bbox", default=True,
                  help_text="Automatically derive ROI from the group tile bbox union.")
    parser.add_argument("--auto-roi-margin-ratio", "--auto_roi_margin_ratio", dest="auto_roi_margin_ratio",
                        type=_parse_ratio_0_05, default=0.02,
                        help="Margin ratio used to shrink the auto ROI bbox.")
    parser.add_argument("--roi-inset", "--roi_inset", dest="roi_inset", type=_parse_nonnegative_float, default=0.0,
                        help="Shrink the ROI polygon by a bbox-extent inset before planning.")

    # Image size and camera intrinsics.
    parser.add_argument("--width", type=_parse_positive_int, default=1024, help="Output image width in pixels.")
    parser.add_argument("--height", type=_parse_positive_int, default=720, help="Output image height in pixels.")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x in pixels; defaults to width / 2.")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y in pixels; defaults to height / 2.")
    parser.add_argument("--nadir-focal", "--nadir_focal", dest="nadir_focal", type=_parse_positive_float, default=599.0,
                        help="Default focal length in pixels for the nadir camera; used for fx/fy unless overridden.")
    parser.add_argument("--nadir-fx", "--nadir_fx", dest="nadir_fx", type=_parse_positive_float, default=None,
                        help="Nadir camera fx in pixels. Overrides --nadir-focal.")
    parser.add_argument("--nadir-fy", "--nadir_fy", dest="nadir_fy", type=_parse_positive_float, default=None,
                        help="Nadir camera fy in pixels. Overrides --nadir-focal.")
    parser.add_argument("--oblique-focal", "--oblique_focal", dest="oblique_focal", type=_parse_positive_float, default=599.0,
                        help="Default focal length in pixels for oblique cameras; used for fx/fy unless overridden.")
    parser.add_argument("--oblique-fx", "--oblique_fx", dest="oblique_fx", type=_parse_positive_float, default=None,
                        help="Oblique camera fx in pixels. Overrides --oblique-focal.")
    parser.add_argument("--oblique-fy", "--oblique_fy", dest="oblique_fy", type=_parse_positive_float, default=None,
                        help="Oblique camera fy in pixels. Overrides --oblique-focal.")

    # Flight / planning.
    parser.add_argument("--flight-height", "--flight_height", dest="flight_height",
                        type=_parse_positive_float, default=83.0,
                        help="Relative flight height above the manually picked global ground Z.")
    parser.add_argument("--side-overlap", "--side_overlap", dest="side_overlap", type=_parse_ratio_0_1, default=0.70,
                        help="Side overlap ratio used to derive lane spacing from swath width.")
    parser.add_argument("--forward-overlap", "--forward_overlap", dest="forward_overlap", type=_parse_ratio_0_1, default=0.75,
                        help="Forward overlap ratio used to derive waypoint spacing from swath width.")
    parser.add_argument("--axis", type=str, default="auto", choices=["x", "y", "auto"],
                        help="Axis preference for automatic lawnmower path orientation.")
    parser.add_argument("--path-angle", "--path_angle", dest="path_angle", type=_parse_optional_float, default=90.0,
                        help="Path direction angle in degrees. Use 'none' or 'auto' to let --axis choose the orientation.")
    parser.add_argument("--short-seg-ratio", "--short_seg_ratio", dest="short_seg_ratio", type=_parse_nonnegative_float, default=0.35,
                        help="Minimum scanline segment length as a fraction of point spacing.")

    # Camera rig and attitude.
    parser.add_argument("--camera-rig", "--camera_rig", dest="camera_rig", type=str, default="nadir",
                        choices=["nadir", "five"],
                        help="Render only the nadir camera or the full five-camera rig.")
    parser.add_argument("--yaw-mode", "--yaw_mode", dest="yaw_mode", type=str, default="path", choices=["path", "fixed"],
                        help="Use path-following yaw or fixed yaw.")
    parser.add_argument("--nadir-pitch-deg", "--nadir_pitch_deg", dest="nadir_pitch_deg", type=float, default=-90.0,
                        help="Nadir camera pitch in degrees.")
    parser.add_argument("--oblique-pitch-deg", "--oblique_pitch_deg", dest="oblique_pitch_deg", type=float, default=-35.0,
                        help="Oblique camera pitch in degrees.")
    parser.add_argument("--roll-deg", "--roll_deg", dest="roll_deg", type=float, default=0.0,
                        help="Camera roll in degrees.")

    # Rendering and postprocessing.
    parser.add_argument("--max-invalid-depth-ratio", "--max_invalid_depth_ratio", dest="max_invalid_depth_ratio",
                        type=_parse_depth_invalid_ratio, default=0.30,
                        help="Skip frames whose invalid depth ratio exceeds this threshold.")
    parser.add_argument("--settle", type=_parse_nonnegative_float, default=0.02,
                        help="Seconds to wait after setting the camera before capturing a frame.")
    parser.add_argument("--ssaa", type=_parse_positive_int, default=2,
                        help="Supersampling factor for RGB/depth rendering. 1 disables SSAA.")
    parser.add_argument("--rgb-median-ksize", "--rgb_median_ksize", dest="rgb_median_ksize",
                        type=_parse_nonnegative_int, default=0,
                        help="Median blur kernel size for RGB postprocessing. 0 disables it; even values are rounded up.")
    _add_bool_arg(parser,
                  ["--rgb-use-bilateral", "--rgb_use_bilateral"],
                  ["--no-rgb-use-bilateral", "--no_rgb_use_bilateral"],
                  dest="rgb_use_bilateral", default=False,
                  help_text="Enable bilateral filtering for RGB postprocessing.")
    parser.add_argument("--depth-resize-mode", "--depth_resize_mode", dest="depth_resize_mode", type=str,
                        default="nearest", choices=["nearest", "area", "valid_mean"],
                        help="Depth downsampling method when ssaa > 1; valid_mean preserves valid-depth averages.")

    # Preview windows.
    _add_bool_arg(parser,
                  ["--show-plan", "--show_plan"],
                  ["--no-show-plan", "--no_show_plan"],
                  dest="show_plan", default=False,
                  help_text="Show mesh, ROI, and planned trajectory before rendering.")
    _add_bool_arg(parser,
                  ["--show-sparse-pcd-preview", "--show_sparse_pcd_preview"],
                  ["--no-show-sparse-pcd-preview", "--no_show_sparse_pcd_preview"],
                  dest="show_sparse_pcd_preview", default=False,
                  help_text="Preview the sampled sparse point cloud before manual ROI picking.")

    return parser


def normalize_args(args, parser):
    """Fill derived values and run cross-field validation."""
    args.nadir_fx = float(args.nadir_focal if args.nadir_fx is None else args.nadir_fx)
    args.nadir_fy = float(args.nadir_focal if args.nadir_fy is None else args.nadir_fy)
    args.oblique_fx = float(args.oblique_focal if args.oblique_fx is None else args.oblique_fx)
    args.oblique_fy = float(args.oblique_focal if args.oblique_fy is None else args.oblique_fy)

    if args.cx is not None and not np.isfinite(float(args.cx)):
        parser.error("--cx must be finite when provided.")
    if args.cy is not None and not np.isfinite(float(args.cy)):
        parser.error("--cy must be finite when provided.")

    if float(args.side_overlap) + float(args.forward_overlap) >= 1.95:
        print("[WARN] Very high overlap values may generate a large number of waypoints.")

    args.rgb_median_ksize = normalize_odd_ksize(args.rgb_median_ksize)
    return args


def main():
    parser = build_arg_parser()
    args = normalize_args(parser.parse_args(), parser)

    root_out = make_root_dirs(args.out_dir)
    input_root = get_input_root_dir(args.obj_root)

    # --------------------------------------------------------------------------
    # A) Pick or load the global ground Z. The cache is saved beside the input OBJ root.
    # --------------------------------------------------------------------------
    obj_files = find_obj_files(args.obj_root, recursive=bool(args.obj_recursive))
    print(f"[INFO] Found {len(obj_files)} OBJ files")
    print(f"[INFO] Input root = {input_root}")

    global_ground_path = get_global_ground_meta_path(args.obj_root)
    if global_ground_path.exists():
        print(f"[INFO] Found existing global ground metadata: {global_ground_path}; loading it.")
        with open(global_ground_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        global_ground_z = data["ground_z"]
    else:
        global_ground_z, _ = pick_global_ground_z_from_random_obj_files(obj_files, args, args.obj_root)

    print(f"[GROUND] use global_ground_z = {global_ground_z:.6f} for all later groups")

    # --------------------------------------------------------------------------
    # B) Scan all OBJ tiles. Per-tile metadata is cached next to each OBJ, and a
    #    summary copy is written under out_dir/meta for convenient inspection.
    # --------------------------------------------------------------------------
    summary_path = root_out / "meta" / "tile_scan_summary.json"

    print("[INFO] Scanning OBJ tiles and caching metadata beside each OBJ...")
    tile_metas = []
    for p in tqdm(obj_files, desc="Scan tile convex hull"):
        try:
            meta = load_or_build_obj_tile_meta(Path(p), overwrite=False)
            tile_metas.append(meta)
        except Exception as e:
            print(f"[WARN] Failed to scan tile {p}: {e}")

    if len(tile_metas) == 0:
        raise RuntimeError("No valid OBJ tiles were scanned.")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_tiles": len(tile_metas),
                "input_root": str(input_root),
                "tiles": tile_metas,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------------------------
    # C) Estimate grouping cost from relative flight height. This is used only for
    #    metadata and terminal reporting; it does not force a grouping strategy.
    # --------------------------------------------------------------------------
    width = int(args.width)
    roles_count = 5 if args.camera_rig == "five" else 1
    hfov_deg_plan = float(np.degrees(2.0 * np.arctan(width / (2.0 * float(args.nadir_fx)))))
    swath_est, lane_spacing_est, point_spacing_est = choose_spacing(
        height_m=float(args.flight_height),
        hfov_deg=hfov_deg_plan,
        side_overlap=float(args.side_overlap),
        forward_overlap=float(args.forward_overlap),
    )
    print(
        f"[EST] coarse plan spacing for grouping | swath={swath_est:.3f}, "
        f"lane={lane_spacing_est:.3f}, point={point_spacing_est:.3f}, roles={roles_count}"
    )

    for meta in tile_metas:
        est_w, est_i = estimate_images_for_tile_bbox(
            meta,
            lane_spacing=lane_spacing_est,
            point_spacing=point_spacing_est,
            roles_count=roles_count,
            coverage_efficiency=0.75,
        )
        meta["est_waypoints_tile"] = int(est_w)
        meta["est_images_tile"] = int(est_i)

    with open(root_out / "meta" / "tile_scan_summary_with_estimation.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_tiles": len(tile_metas),
                "grouping_estimation": {
                    "hfov_deg_plan_nadir": hfov_deg_plan,
                    "lane_spacing_est": lane_spacing_est,
                    "point_spacing_est": point_spacing_est,
                    "roles_count": roles_count,
                    "camera_rig": args.camera_rig,
                    "flight_height_rel": float(args.flight_height),
                    "global_ground_z": float(global_ground_z),
                },
                "tiles": tile_metas,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------------------------
    # D) Manual grouping by picking tile centers.
    # --------------------------------------------------------------------------
    manual_group_summary_path = root_out / "meta" / "tile_grouping_summary.json"

    if len(tile_metas) == 1:
        print("[INFO] Only one OBJ tile found; skipping manual grouping.")
        single_group_color = [[0.90, 0.20, 0.20]]

        group_infos = build_group_infos_from_manual_groups(
            groups=[[0]],
            tile_metas=tile_metas,
            global_ground_z=float(global_ground_z),
            group_colors=single_group_color,
        )

        with open(manual_group_summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_tiles": 1,
                    "num_groups": 1,
                    "params": {
                        "group_mode": "single_obj_skip_grouping",
                        "group_vis_z": float(args.group_vis_z),
                        "keep_ungrouped_as_one": True,
                        "flight_height_rel": float(args.flight_height),
                        "global_ground_z": float(global_ground_z),
                    },
                    "groups": group_infos,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    else:
        if bool(args.reuse_manual_groups) and manual_group_summary_path.exists():
            print(f"[INFO] Reusing existing manual grouping summary: {manual_group_summary_path}")
            with open(manual_group_summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            group_infos = data["groups"]
        else:
            groups, group_colors = manual_group_tiles_by_picking_centers(
                tile_metas=tile_metas,
                root_out=root_out,
                z=float(args.group_vis_z),
                keep_ungrouped_as_one=bool(args.keep_ungrouped_as_one),
                target_span=args.group_display_target_span,
                margin_ratio=args.group_display_margin_ratio,
            )

            group_infos = build_group_infos_from_manual_groups(
                groups=groups,
                tile_metas=tile_metas,
                global_ground_z=float(global_ground_z),
                group_colors=group_colors,
            )

            print(f"[GROUP] manual grouping finished | total groups={len(group_infos)}")
            for gi in group_infos:
                print(
                    f"  - group {gi['group_id']:03d}: "
                    f"tiles={gi['num_tiles']} "
                    f"est_imgs={gi['est_images_group']} "
                    f"bbox={np.round(gi['bbox_xyxy'], 3).tolist()}"
                )

            with open(manual_group_summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "num_tiles": len(tile_metas),
                        "num_groups": len(group_infos),
                        "params": {
                            "group_mode": "manual_pick_centers_confirm_then_recolor",
                            "group_vis_z": float(args.group_vis_z),
                            "keep_ungrouped_as_one": bool(args.keep_ungrouped_as_one),
                            "flight_height_rel": float(args.flight_height),
                            "global_ground_z": float(global_ground_z),
                        },
                        "groups": group_infos,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    # --------------------------------------------------------------------------
    # E) For each group: merge tiles, plan ROI, render, optionally clean cache.
    # --------------------------------------------------------------------------
    all_groups_stats = []
    total_stats_all_groups = {
        "saved_waypoints": 0,
        "skipped_waypoints": 0,
        "saved_frames": 0,
        "skipped_frames": 0,
        "total_waypoints": 0,
        "groups_processed": 0,
        "groups_failed": 0,
    }

    for gi in group_infos:
        gid = int(gi["group_id"])

        # Stable cache directory, for example group_000.
        group_cache_dir = get_group_cache_dir(root_out, gid)
        make_scene_dirs(group_cache_dir)

        # Flight-height-specific render directory, for example group_h83_000.
        group_render_dir = get_group_render_dir(root_out, gid, args.flight_height)
        make_scene_dirs(group_render_dir)

        # Save group metadata both in the cache and render directories for easier inspection.
        with open(group_cache_dir / "meta" / "group_meta_premerge.json", "w", encoding="utf-8") as f:
            json.dump(gi, f, ensure_ascii=False, indent=2)

        with open(group_render_dir / "meta" / "group_meta_premerge.json", "w", encoding="utf-8") as f:
            json.dump(gi, f, ensure_ascii=False, indent=2)

        if bool(args.skip_existing_height_groups) and scene_outputs_exist(group_render_dir):
            print(f"[GROUP {gid:03d}] skipping existing render outputs at: {group_render_dir}")

            group_record = {
                "group_id": gid,
                "status": "skipped_existing_height",
                "render_dir": str(group_render_dir),
                "cache_dir": str(group_cache_dir),
                "flight_height": float(args.flight_height),
            }
            all_groups_stats.append(group_record)
            continue

        merged_obj_path = None
        mesh_input_path = None
        merge_mode = None

        try:
            if int(gi["num_tiles"]) == 1:
                mesh_input_path = Path(gi["tile_paths"][0])
                merge_mode = "direct_single_obj"
                print(f"[GROUP {gid:03d}] single OBJ group -> skip merge and use source OBJ directly: {mesh_input_path}")
            else:
                merged_obj_path = find_reusable_merged_obj(root_out, gid, group_render_dir)

                if bool(args.reuse_merged_groups) and (merged_obj_path is not None):
                    print(f"[GROUP {gid:03d}] Reusing merged OBJ: {merged_obj_path}")
                else:
                    merged_obj_path = group_cache_dir / "merged_group.obj"
                    merge_tile_group_with_external_tool(
                        group_obj_paths=gi["tile_paths"],
                        merge_tiles_exe=args.merge_tiles_exe,
                        merged_out_path=merged_obj_path,
                        keep_tmp_inputs=bool(args.keep_merge_tmp_inputs),
                    )

                mesh_input_path = merged_obj_path
                merge_mode = "merged_multi_obj"

            mesh = load_mesh_single(mesh_input_path)

            stats = run_planning_and_render_on_single_mesh(
                mesh=mesh,
                scene_root=group_render_dir,
                args=args,
                group_info=gi,
                global_ground_z=float(global_ground_z),
            )

            cleanup_info = None
            if bool(args.delete_merged_obj_after_render) and (merge_mode == "merged_multi_obj"):
                try:
                    del mesh
                except Exception:
                    pass

                if Path(merged_obj_path).resolve() == (group_cache_dir / "merged_group.obj").resolve():
                    last_del_err = None
                    for _ in range(3):
                        try:
                            cleanup_info = cleanup_merged_outputs(
                                merged_obj_path=merged_obj_path,
                                delete_textures=True,
                                verbose=True,
                            )
                            break
                        except Exception as e_del:
                            last_del_err = e_del
                            time.sleep(0.2)

                    if cleanup_info is None:
                        print(f"[WARN] Failed to clean up merged outputs for {merged_obj_path} | err={last_del_err}")

            group_record = {
                "group_id": gid,
                "status": "ok",
                "flight_height": float(args.flight_height),
                "render_dir": str(group_render_dir),
                "cache_dir": str(group_cache_dir),
                "merge_mode": merge_mode,
                "mesh_input": str(mesh_input_path),
                "merged_obj": None if merged_obj_path is None else str(merged_obj_path),
                "stats": stats,
            }

            if cleanup_info is not None:
                group_record["cleanup_info"] = cleanup_info

            all_groups_stats.append(group_record)

            total_stats_all_groups["groups_processed"] += 1
            for k in ["saved_waypoints", "skipped_waypoints", "saved_frames", "skipped_frames", "total_waypoints"]:
                total_stats_all_groups[k] += int(stats[k])

        except Exception as e:
            print(f"[ERROR] group {gid:03d} failed: {e}")
            group_record = {
                "group_id": gid,
                "status": "failed",
                "flight_height": float(args.flight_height),
                "render_dir": str(group_render_dir),
                "cache_dir": str(group_cache_dir),
                "merge_mode": merge_mode,
                "mesh_input": None if mesh_input_path is None else str(mesh_input_path),
                "merged_obj": None if merged_obj_path is None else str(merged_obj_path),
                "error": str(e),
            }
            all_groups_stats.append(group_record)
            total_stats_all_groups["groups_failed"] += 1
            continue

    with open(root_out / "meta" / "all_groups_stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": total_stats_all_groups,
                "global_ground_z": float(global_ground_z),
                "args": vars(args),
                "groups": all_groups_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n[ALL DONE - GROUPS]")
    print(json.dumps(total_stats_all_groups, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
