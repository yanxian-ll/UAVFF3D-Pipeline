# -*- coding: utf-8 -*-
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from PIL import Image
except Exception:
    Image = None

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"]
DEPTH_EXTS = [".npy", ".exr", ".pfm", ".png", ".tiff", ".tif"]


# =========================================================
# basic utils
# =========================================================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_file_for_stem(folder: Path, stem: str, exts: List[str]) -> Optional[Path]:
    for ext in exts:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    ext_set = {e.lower() for e in exts}
    for p in folder.glob(f"{stem}.*"):
        if p.suffix.lower() in ext_set:
            return p
    return None


def try_parse_float_line(line: str):
    vals = line.strip().split()
    if not vals:
        return None
    try:
        return [float(v) for v in vals]
    except Exception:
        return None


class LRUCache(dict):
    def __init__(self, max_items: int):
        super().__init__()
        self.max_items = max(1, int(max_items))
        self._order = []

    def get(self, key, default=None):
        if key in self:
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            return super().get(key)
        return default

    def put(self, key, value):
        if key in self:
            if key in self._order:
                self._order.remove(key)
        elif len(self._order) >= self.max_items:
            old = self._order.pop(0)
            if old in self:
                del self[old]
        super().__setitem__(key, value)
        self._order.append(key)


# =========================================================
# IO
# =========================================================
def read_cam_blendedmvs_txt(path_txt: str):
    with open(path_txt, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]

    ex_idx = None
    in_idx = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if low.startswith("extrinsic"):
            ex_idx = i
        elif low.startswith("intrinsic"):
            in_idx = i

    if ex_idx is None or in_idx is None:
        raise ValueError(f"Invalid cam txt format: {path_txt}")

    RT = []
    for j in range(ex_idx + 1, min(ex_idx + 8, len(lines))):
        vals = try_parse_float_line(lines[j])
        if vals is not None and len(vals) >= 4:
            RT.append(vals[:4])
        if len(RT) == 4:
            break
    if len(RT) != 4:
        raise ValueError(f"Failed to parse extrinsic from: {path_txt}")
    RT = np.asarray(RT, dtype=np.float64)

    K = []
    for j in range(in_idx + 1, min(in_idx + 8, len(lines))):
        vals = try_parse_float_line(lines[j])
        if vals is not None and len(vals) >= 3:
            K.append(vals[:3])
        if len(K) == 3:
            break
    if len(K) != 3:
        raise ValueError(f"Failed to parse intrinsic from: {path_txt}")
    K = np.asarray(K, dtype=np.float64)

    cam_h, cam_w, misc = 0, 0, 0.0
    for j in range(in_idx + 4, len(lines)):
        vals = try_parse_float_line(lines[j])
        if vals is not None and len(vals) >= 3:
            cam_h = int(round(vals[0]))
            cam_w = int(round(vals[1]))
            misc = float(vals[2])
            break

    cam2world = np.linalg.inv(RT)
    return K, cam2world, cam_h, cam_w, misc


def read_depth_any(path: Path) -> Optional[np.ndarray]:
    suf = path.suffix.lower()
    if suf == ".npy":
        d = np.load(path)
        if d is None:
            return None
        if d.ndim == 3:
            d = d[..., 0]
        return d.astype(np.float32)

    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    if d.ndim == 3:
        d = d[..., 0]
    return d.astype(np.float32)


def read_depth_shape(path: Path) -> Optional[Tuple[int, int]]:
    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(path, mmap_mode="r")
            if arr is None:
                return None
            return int(arr.shape[1]), int(arr.shape[0])
    except Exception:
        pass
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    h, w = d.shape[:2]
    return int(w), int(h)


def read_image_bgr(path: Path) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def get_image_size(path: Path) -> Optional[Tuple[int, int]]:
    if Image is not None:
        try:
            with Image.open(path) as im:
                return im.size
        except Exception:
            pass
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]
    return (w, h)


def scale_intrinsics(K, src_w, src_h, dst_w, dst_h):
    K2 = K.copy().astype(np.float64)
    if src_w <= 0 or src_h <= 0:
        return K2
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def resize_image_and_intrinsics_to_max_long_edge(img, K, max_long_edge):
    K_resized = K.copy().astype(np.float64)
    if img is None or max_long_edge is None or max_long_edge <= 0:
        return img, K_resized, 1.0
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return img, K_resized, 1.0
    scale = float(max_long_edge) / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    K_resized[0, 0] *= scale
    K_resized[1, 1] *= scale
    K_resized[0, 2] *= scale
    K_resized[1, 2] *= scale
    return img_resized, K_resized, scale


# =========================================================
# geometry
# =========================================================
def backproject_pixel_to_cam(u, v, z, K):
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = (u - cx) * z / max(fx, 1e-12)
    y = (v - cy) * z / max(fy, 1e-12)
    return np.array([x, y, z], dtype=np.float64)


def cam_to_world(Xc, cam2world):
    Xh = np.array([Xc[0], Xc[1], Xc[2], 1.0], dtype=np.float64)
    return (cam2world @ Xh)[:3]


def transform_world_to_cam(Xw, world2cam):
    Xh = np.array([Xw[0], Xw[1], Xw[2], 1.0], dtype=np.float64)
    return (world2cam @ Xh)[:3]


def project_world_points_to_image(world_pts, K, world2cam):
    if world_pts is None or len(world_pts) == 0:
        return None, None, None
    pts_h = np.concatenate([world_pts, np.ones((len(world_pts), 1), dtype=np.float64)], axis=1)
    cam_pts = (world2cam @ pts_h.T).T[:, :3]
    z = cam_pts[:, 2]
    valid = z > 1e-8
    if not np.any(valid):
        return None, None, None
    cam_pts_v = cam_pts[valid]
    uvw = (K @ cam_pts_v.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-12, None)
    idx = np.where(valid)[0]
    return uv.astype(np.float32), z[valid].astype(np.float32), idx


def sample_world_points_from_depth(depth, K_depth, cam2world, stride=4, min_depth=1e-3, max_points=30000, seed=0):
    h, w = depth.shape[:2]
    ys = np.arange(0, h, max(1, stride), dtype=np.int32)
    xs = np.arange(0, w, max(1, stride), dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    z = depth[yy, xx].astype(np.float64)
    valid = np.isfinite(z) & (z > min_depth)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    xx = xx[valid].astype(np.float64)
    yy = yy[valid].astype(np.float64)
    z = z[valid]

    if len(z) > max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(z), size=max_points, replace=False)
        xx, yy, z = xx[sel], yy[sel], z[sel]

    fx, fy = float(K_depth[0, 0]), float(K_depth[1, 1])
    cx, cy = float(K_depth[0, 2]), float(K_depth[1, 2])
    X = (xx - cx) * z / max(fx, 1e-12)
    Y = (yy - cy) * z / max(fy, 1e-12)
    pts_cam = np.stack([X, Y, z, np.ones_like(z)], axis=1)
    pts_world = (cam2world @ pts_cam.T).T[:, :3]
    return pts_world.astype(np.float64), z.astype(np.float64)


def frustum_corners_world_aabb(Hd: int, Wd: int, K_depth: np.ndarray, cam2world: np.ndarray, near: float, far: float):
    near = float(max(near, 1e-6))
    far = float(max(far, near + 1e-6))
    corners_uv = np.array([[0.0, 0.0], [Wd - 1.0, 0.0], [Wd - 1.0, Hd - 1.0], [0.0, Hd - 1.0]], dtype=np.float32)
    invK = np.linalg.inv(K_depth).astype(np.float32)
    pix = np.concatenate([corners_uv, np.ones((4, 1), dtype=np.float32)], axis=1)
    rays = (invK @ pix.T).T
    Xc_near = rays * near
    Xc_far = rays * far
    Xc = np.concatenate([Xc_near, Xc_far], axis=0)
    R = cam2world[:3, :3].astype(np.float32)
    t = cam2world[:3, 3].astype(np.float32)
    Xw = (R @ Xc.T).T + t[None, :]
    return Xw.min(axis=0), Xw.max(axis=0)


def aabb_overlap(a_min, a_max, b_min, b_max) -> bool:
    return bool(
        (a_min[0] <= b_max[0] and a_max[0] >= b_min[0]) and
        (a_min[1] <= b_max[1] and a_max[1] >= b_min[1]) and
        (a_min[2] <= b_max[2] and a_max[2] >= b_min[2])
    )


def angle_between_unit(v1, v2):
    c = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(c))


# =========================================================
# matching
# =========================================================
def create_feature_detector(feature_type: str):
    feature_type = feature_type.lower()
    if feature_type == "auto":
        if hasattr(cv2, "SIFT_create"):
            return "sift", cv2.SIFT_create(nfeatures=8000), cv2.NORM_L2
        return "orb", cv2.ORB_create(nfeatures=8000), cv2.NORM_HAMMING
    if feature_type == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("OpenCV does not provide SIFT_create on this build.")
        return "sift", cv2.SIFT_create(nfeatures=8000), cv2.NORM_L2
    if feature_type == "orb":
        return "orb", cv2.ORB_create(nfeatures=8000), cv2.NORM_HAMMING
    raise ValueError(f"Unsupported feature_type: {feature_type}")


def build_matcher(detector_name: str, norm_type: int):
    if detector_name == "sift":
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=64)
        return cv2.FlannBasedMatcher(index_params, search_params)
    return cv2.BFMatcher(norm_type)


def detect_and_compute(gray, detector):
    kps, des = detector.detectAndCompute(gray, None)
    if des is None or len(kps) == 0:
        return [], None
    if des.dtype != np.float32 and hasattr(cv2, "SIFT_create"):
        des = des.astype(np.float32)
    return kps, des


def knn_ratio_match_with_distance(des1, des2, matcher, ratio=0.75):
    if des1 is None or des2 is None:
        return []
    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def ransac_filter_2d_with_distance(pts1, pts2, dists, thr_px=3.0, min_inliers=16, min_inlier_ratio=0.25):
    if pts1 is None or pts2 is None or dists is None or len(pts1) < max(8, min_inliers):
        return None
    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        cv2.FM_RANSAC,
        ransacReprojThreshold=float(thr_px),
        confidence=0.999,
    )
    if F is None or mask is None:
        return None
    mask = mask.ravel().astype(bool)
    pts1_in = pts1[mask]
    pts2_in = pts2[mask]
    dists_in = dists[mask]
    num_in = int(mask.sum())
    ratio_in = float(num_in) / max(float(len(mask)), 1.0)
    if num_in < int(min_inliers) or ratio_in < float(min_inlier_ratio):
        return None
    return {
        "pts1": pts1_in.astype(np.float32),
        "pts2": pts2_in.astype(np.float32),
        "dists": dists_in.astype(np.float32),
        "num_inliers": num_in,
        "inlier_ratio": ratio_in,
    }


# =========================================================
# visualization
# =========================================================
def visualize_matches_points(img1, img2, pts1, pts2, out_path, max_lines=200, seed=0, title_text=None):
    if pts1 is None or pts2 is None or len(pts1) == 0:
        return
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)
    W = w1 + w2
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1+w2] = img2
    rng = np.random.default_rng(seed)
    n = len(pts1)
    if max_lines > 0 and n > max_lines:
        idx = rng.choice(n, size=max_lines, replace=False)
        pts1_draw = pts1[idx]
        pts2_draw = pts2[idx]
    else:
        pts1_draw = pts1
        pts2_draw = pts2
    for (x1, y1), (x2, y2) in zip(pts1_draw, pts2_draw):
        p1 = (int(round(float(x1))), int(round(float(y1))))
        p2 = (int(round(float(x2))) + w1, int(round(float(y2))))
        color = tuple(int(v) for v in rng.integers(0, 255, size=3))
        cv2.circle(canvas, p1, 3, color, -1)
        cv2.circle(canvas, p2, 3, color, -1)
        cv2.line(canvas, p1, p2, color, 1)
    if title_text:
        cv2.putText(canvas, title_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), canvas)


# =========================================================
# cached items
# =========================================================
@dataclass
class ViewMeta:
    stem: str
    img_path: Path
    cam_path: Path
    depth_path: Path
    img_wh: Tuple[int, int]
    depth_wh: Tuple[int, int]
    K_img: np.ndarray
    K_depth: np.ndarray
    cam2world: np.ndarray
    world2cam: np.ndarray
    center: np.ndarray
    forward: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray


@dataclass
class DepthCacheItem:
    depth: np.ndarray
    valid_mask: np.ndarray
    sampled_world_pts: np.ndarray


@dataclass
class FeatureCacheItem:
    img_bgr: np.ndarray
    gray: np.ndarray
    K_scaled: np.ndarray
    scale: float
    keypoints: list
    descriptors: np.ndarray


# =========================================================
# build meta / caches
# =========================================================
def build_view_metas(img_dir: Path, cam_dir: Path, depth_dir: Path, min_depth: float) -> List[ViewMeta]:
    metas: List[ViewMeta] = []
    for cam_path in tqdm(sorted(cam_dir.glob("*.txt")), desc="Build view metas", dynamic_ncols=True):
        stem = cam_path.stem
        img_path = find_file_for_stem(img_dir, stem, IMG_EXTS)
        depth_path = find_file_for_stem(depth_dir, stem, DEPTH_EXTS)
        if img_path is None or depth_path is None:
            continue
        img_wh = get_image_size(img_path)
        depth_wh = read_depth_shape(depth_path)
        if img_wh is None or depth_wh is None:
            continue
        w_img, h_img = img_wh
        w_d, h_d = depth_wh
        K, cam2world, cam_h, cam_w, _ = read_cam_blendedmvs_txt(str(cam_path))
        K_img = scale_intrinsics(K, cam_w, cam_h, w_img, h_img)
        K_depth = scale_intrinsics(K, cam_w, cam_h, w_d, h_d)
        world2cam = np.linalg.inv(cam2world)
        center = cam2world[:3, 3].copy()
        forward = cam2world[:3, 2].copy()
        n = np.linalg.norm(forward)
        if n > 1e-12:
            forward = forward / n

        depth = read_depth_any(depth_path)
        if depth is None:
            continue
        valid = np.isfinite(depth) & (depth > min_depth)
        if valid.any():
            near = float(depth[valid].min())
            far = float(depth[valid].max())
        else:
            near, far = 0.1, 1.0
        aabb_min, aabb_max = frustum_corners_world_aabb(h_d, w_d, K_depth, cam2world, near, far)

        metas.append(ViewMeta(
            stem=stem,
            img_path=img_path,
            cam_path=cam_path,
            depth_path=depth_path,
            img_wh=(w_img, h_img),
            depth_wh=(w_d, h_d),
            K_img=K_img,
            K_depth=K_depth,
            cam2world=cam2world,
            world2cam=world2cam,
            center=center,
            forward=forward,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
        ))
    return metas


def load_depth_item(meta: ViewMeta, depth_cache: LRUCache, sample_stride: int, sample_max_points: int, min_depth: float, seed: int):
    item = depth_cache.get(meta.stem)
    if item is not None:
        return item
    depth = read_depth_any(meta.depth_path)
    if depth is None:
        return None
    valid_mask = np.isfinite(depth) & (depth > min_depth)
    sampled_world_pts, _ = sample_world_points_from_depth(
        depth=depth,
        K_depth=meta.K_depth,
        cam2world=meta.cam2world,
        stride=sample_stride,
        min_depth=min_depth,
        max_points=sample_max_points,
        seed=seed,
    )
    item = DepthCacheItem(depth=depth, valid_mask=valid_mask, sampled_world_pts=sampled_world_pts)
    depth_cache.put(meta.stem, item)
    return item


def load_feature_item(meta: ViewMeta, feature_cache: LRUCache, detector, max_long_edge: int):
    item = feature_cache.get(meta.stem)
    if item is not None:
        return item
    img = read_image_bgr(meta.img_path)
    if img is None:
        return None
    img_scaled, K_scaled, scale = resize_image_and_intrinsics_to_max_long_edge(img, meta.K_img, max_long_edge)
    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    kps, des = detect_and_compute(gray, detector)
    item = FeatureCacheItem(
        img_bgr=img_scaled,
        gray=gray,
        K_scaled=K_scaled,
        scale=scale,
        keypoints=kps,
        descriptors=des,
    )
    feature_cache.put(meta.stem, item)
    return item


# =========================================================
# pair screening
# =========================================================
def estimate_overlap_ratio(meta_a: ViewMeta, meta_b: ViewMeta, depth_a: DepthCacheItem, depth_b: Optional[DepthCacheItem], min_depth: float, use_depth_consistency: bool, depth_abs_thr: float, depth_rel_thr: float):
    if depth_a is None or depth_a.sampled_world_pts.shape[0] == 0:
        return 0.0
    uv, z_proj, _ = project_world_points_to_image(depth_a.sampled_world_pts, meta_b.K_depth, meta_b.world2cam)
    if uv is None:
        return 0.0
    w_b, h_b = meta_b.depth_wh
    inside = (
        (uv[:, 0] >= 0) & (uv[:, 0] < w_b) &
        (uv[:, 1] >= 0) & (uv[:, 1] < h_b) &
        (z_proj > min_depth)
    )
    if not np.any(inside):
        return 0.0

    if not use_depth_consistency or depth_b is None:
        return float(np.mean(inside))

    uv_in = uv[inside]
    zp = z_proj[inside]
    x = np.clip(np.round(uv_in[:, 0]).astype(np.int32), 0, w_b - 1)
    y = np.clip(np.round(uv_in[:, 1]).astype(np.int32), 0, h_b - 1)
    d = depth_b.depth[y, x]
    valid = np.isfinite(d) & (d > min_depth)
    if not np.any(valid):
        return 0.0
    abs_err = np.abs(d[valid] - zp[valid])
    rel_err = abs_err / np.maximum(np.abs(d[valid]), 1e-8)
    ok = (abs_err <= depth_abs_thr) | (rel_err <= depth_rel_thr)
    return float(ok.sum()) / float(len(depth_a.sampled_world_pts) + 1e-12)


def generate_candidate_pairs_streaming(metas: List[ViewMeta], neighbors_per_view: int, max_view_angle_deg: float):
    centers = np.stack([m.center for m in metas], axis=0)
    N = len(metas)
    seen = set()
    for i in tqdm(range(N), desc="Generate candidate pairs", dynamic_ncols=True):
        d2 = np.sum((centers - centers[i:i+1]) ** 2, axis=1)
        k = min(N, neighbors_per_view + 1)
        nn = np.argpartition(d2, k - 1)[:k]
        nn = nn[nn != i]
        nn = nn[np.argsort(d2[nn])]
        for j in nn:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            if not aabb_overlap(metas[a].aabb_min, metas[a].aabb_max, metas[b].aabb_min, metas[b].aabb_max):
                continue
            if max_view_angle_deg < 180.0:
                ang = angle_between_unit(metas[a].forward, metas[b].forward)
                if ang > max_view_angle_deg:
                    continue
            yield a, b, float(d2[int(j)])


# =========================================================
# depth consistency on scaled image coords
# =========================================================
def depth_consistency_filter_scaled(
    pts1, pts2, desc_dists,
    depth1, depth2,
    K1_scaled, K2_scaled,
    cam2world1, cam2world2,
    depth_wh1, depth_wh2,
    img_wh_scaled1, img_wh_scaled2,
    reproj_thr_px,
    depth_abs_thr,
    depth_rel_thr,
    min_depth,
):
    if pts1 is None or pts2 is None or len(pts1) == 0:
        return None

    d_w1, d_h1 = depth_wh1
    d_w2, d_h2 = depth_wh2
    i_w1, i_h1 = img_wh_scaled1
    i_w2, i_h2 = img_wh_scaled2

    sx1 = d_w1 / max(i_w1, 1)
    sy1 = d_h1 / max(i_h1, 1)
    sx2 = d_w2 / max(i_w2, 1)
    sy2 = d_h2 / max(i_h2, 1)

    out_pts1 = []
    out_pts2 = []
    out_err = []
    out_desc = []

    world2cam2 = np.linalg.inv(cam2world2)

    for p1, p2, dist in zip(pts1, pts2, desc_dists):
        u1, v1 = float(p1[0]), float(p1[1])
        u2, v2 = float(p2[0]), float(p2[1])

        x1 = int(round(u1 * sx1))
        y1 = int(round(v1 * sy1))
        x2 = int(round(u2 * sx2))
        y2 = int(round(v2 * sy2))

        if not (0 <= x1 < d_w1 and 0 <= y1 < d_h1 and 0 <= x2 < d_w2 and 0 <= y2 < d_h2):
            continue

        z1 = float(depth1[y1, x1])
        z2 = float(depth2[y2, x2])
        if (not np.isfinite(z1)) or (not np.isfinite(z2)) or z1 <= min_depth or z2 <= min_depth:
            continue

        Xc1 = backproject_pixel_to_cam(u1, v1, z1, K1_scaled)
        Xw = cam_to_world(Xc1, cam2world1)
        Xc2 = transform_world_to_cam(Xw, world2cam2)
        if Xc2[2] <= min_depth:
            continue
        uvw2 = K2_scaled @ Xc2
        proj2 = uvw2[:2] / max(uvw2[2], 1e-12)
        reproj_err = float(np.linalg.norm(proj2 - np.array([u2, v2], dtype=np.float64)))
        if reproj_err > reproj_thr_px:
            continue

        pred_z2 = float(Xc2[2])
        abs_err = abs(pred_z2 - z2)
        rel_err = abs_err / max(abs(z2), 1e-8)
        if abs_err > depth_abs_thr and rel_err > depth_rel_thr:
            continue

        out_pts1.append([u1, v1])
        out_pts2.append([u2, v2])
        out_err.append(reproj_err)
        out_desc.append(float(dist))

    if len(out_pts1) == 0:
        return None

    return {
        "pts1": np.asarray(out_pts1, dtype=np.float32),
        "pts2": np.asarray(out_pts2, dtype=np.float32),
        "reproj_err": np.asarray(out_err, dtype=np.float32),
        "desc_dist": np.asarray(out_desc, dtype=np.float32),
    }


def keep_best_topk_matches(match_info, topk: int):
    if match_info is None:
        return None
    n = len(match_info["pts1"])
    if n == 0:
        return None
    score = match_info["desc_dist"].astype(np.float64) + 0.5 * match_info["reproj_err"].astype(np.float64)
    order = np.argsort(score)
    if topk > 0:
        order = order[:topk]
    return {
        "pts1": match_info["pts1"][order],
        "pts2": match_info["pts2"][order],
        "reproj_err": match_info["reproj_err"][order],
        "desc_dist": match_info["desc_dist"][order],
    }


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Strict stereo pair selection with top-k feature matches for evaluation")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--cam_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--num_pairs", type=int, default=50)
    parser.add_argument("--min_overlap", type=float, default=0.5)

    parser.add_argument("--feature_type", type=str, default="sift", choices=["auto", "sift", "orb"])
    parser.add_argument("--match_ratio", type=float, default=0.75)
    parser.add_argument("--fm_ransac_thr", type=float, default=2.0)
    parser.add_argument("--max_image_long_edge", type=int, default=1600)

    parser.add_argument("--overlap_sample_stride", type=int, default=4)
    parser.add_argument("--overlap_sample_max_points", type=int, default=25000)
    parser.add_argument("--use_depth_consistency_in_overlap", action="store_true")
    parser.add_argument("--depth_abs_thr", type=float, default=0.2)
    parser.add_argument("--depth_rel_thr", type=float, default=0.05)
    parser.add_argument("--reproj_thr_px", type=float, default=3.0)
    parser.add_argument("--min_depth", type=float, default=1e-3)

    parser.add_argument("--neighbors_per_view", type=int, default=30)
    parser.add_argument("--max_view_angle_deg", type=float, default=180.0)
    parser.add_argument("--max_pair_trials", type=int, default=500)

    parser.add_argument("--depth_cache_size", type=int, default=16)
    parser.add_argument("--feature_cache_size", type=int, default=64)
    parser.add_argument("--max_vis_lines", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--min_raw_good_matches", type=int, default=40, help="ratio test 后至少这么多匹配")
    parser.add_argument("--keep_top_for_ransac", type=int, default=200, help="RANSAC 前最多保留多少个 descriptor 最优匹配")
    parser.add_argument("--min_ransac_inliers", type=int, default=20, help="基础矩阵 RANSAC 至少内点数")
    parser.add_argument("--min_ransac_inlier_ratio", type=float, default=0.25, help="基础矩阵 RANSAC 至少内点率")
    parser.add_argument("--topk_per_pair", type=int, default=5, help="每个图像对最终最多保留多少个最优匹配")
    parser.add_argument("--min_final_matches", type=int, default=5, help="top-k 后最少保留多少个匹配，否则丢弃该图像对")
    parser.add_argument("--max_mean_desc_dist", type=float, default=180.0, help="最终匹配平均 descriptor distance 上限，<=0 不限制")
    parser.add_argument("--max_mean_reproj_err", type=float, default=2.5, help="最终匹配平均 reprojection error 上限，<=0 不限制")
    parser.add_argument("--max_p95_reproj_err", type=float, default=4.0, help="最终匹配 p95 reprojection error 上限，<=0 不限制")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    cam_dir = Path(args.cam_dir)
    depth_dir = Path(args.depth_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    vis_dir = ensure_dir(out_dir / "visualizations")
    match_dir = ensure_dir(out_dir / "match_data")

    detector_name, detector, norm_type = create_feature_detector(args.feature_type)
    matcher = build_matcher(detector_name, norm_type)

    print("[INFO] scanning views...")
    metas = build_view_metas(image_dir, cam_dir, depth_dir, min_depth=args.min_depth)
    if len(metas) < 2:
        raise RuntimeError("Not enough valid views found.")
    print(f"[INFO] usable views: {len(metas)}")

    depth_cache = LRUCache(args.depth_cache_size)
    feature_cache = LRUCache(args.feature_cache_size)
    selected_pairs = []
    tried = 0

    candidate_iter = generate_candidate_pairs_streaming(
        metas=metas,
        neighbors_per_view=args.neighbors_per_view,
        max_view_angle_deg=args.max_view_angle_deg,
    )

    pbar = tqdm(total=args.num_pairs, desc="Select stereo pairs", dynamic_ncols=True)
    for i, j, dist2 in candidate_iter:
        if len(selected_pairs) >= args.num_pairs:
            break
        tried += 1
        if args.max_pair_trials > 0 and tried > args.max_pair_trials:
            break

        meta_i = metas[i]
        meta_j = metas[j]

        depth_i = load_depth_item(meta_i, depth_cache, args.overlap_sample_stride, args.overlap_sample_max_points, args.min_depth, args.seed + i)
        depth_j = load_depth_item(meta_j, depth_cache, args.overlap_sample_stride, args.overlap_sample_max_points, args.min_depth, args.seed + j)
        if depth_i is None or depth_j is None:
            continue

        overlap_ij = estimate_overlap_ratio(
            meta_i, meta_j, depth_i, depth_j,
            min_depth=args.min_depth,
            use_depth_consistency=args.use_depth_consistency_in_overlap,
            depth_abs_thr=args.depth_abs_thr,
            depth_rel_thr=args.depth_rel_thr,
        )
        overlap_ji = estimate_overlap_ratio(
            meta_j, meta_i, depth_j, depth_i,
            min_depth=args.min_depth,
            use_depth_consistency=args.use_depth_consistency_in_overlap,
            depth_abs_thr=args.depth_abs_thr,
            depth_rel_thr=args.depth_rel_thr,
        )
        overlap = 0.5 * (overlap_ij + overlap_ji)
        if overlap < args.min_overlap:
            continue

        feat_i = load_feature_item(meta_i, feature_cache, detector, args.max_image_long_edge)
        feat_j = load_feature_item(meta_j, feature_cache, detector, args.max_image_long_edge)
        if feat_i is None or feat_j is None:
            continue
        if feat_i.descriptors is None or feat_j.descriptors is None:
            continue

        good = knn_ratio_match_with_distance(feat_i.descriptors, feat_j.descriptors, matcher, ratio=args.match_ratio)
        if len(good) < int(args.min_raw_good_matches):
            continue

        good = sorted(good, key=lambda m: float(m.distance))
        if args.keep_top_for_ransac > 0:
            good = good[:args.keep_top_for_ransac]

        pts1 = np.float32([feat_i.keypoints[m.queryIdx].pt for m in good])
        pts2 = np.float32([feat_j.keypoints[m.trainIdx].pt for m in good])
        desc_dists = np.float32([float(m.distance) for m in good])

        ransac_info = ransac_filter_2d_with_distance(
            pts1, pts2, desc_dists,
            thr_px=args.fm_ransac_thr,
            min_inliers=args.min_ransac_inliers,
            min_inlier_ratio=args.min_ransac_inlier_ratio,
        )
        if ransac_info is None:
            continue

        match_info = depth_consistency_filter_scaled(
            pts1=ransac_info["pts1"],
            pts2=ransac_info["pts2"],
            desc_dists=ransac_info["dists"],
            depth1=depth_i.depth,
            depth2=depth_j.depth,
            K1_scaled=feat_i.K_scaled,
            K2_scaled=feat_j.K_scaled,
            cam2world1=meta_i.cam2world,
            cam2world2=meta_j.cam2world,
            depth_wh1=meta_i.depth_wh,
            depth_wh2=meta_j.depth_wh,
            img_wh_scaled1=(feat_i.img_bgr.shape[1], feat_i.img_bgr.shape[0]),
            img_wh_scaled2=(feat_j.img_bgr.shape[1], feat_j.img_bgr.shape[0]),
            reproj_thr_px=args.reproj_thr_px,
            depth_abs_thr=args.depth_abs_thr,
            depth_rel_thr=args.depth_rel_thr,
            min_depth=args.min_depth,
        )
        if match_info is None:
            continue

        match_info = keep_best_topk_matches(match_info, args.topk_per_pair)
        if match_info is None or len(match_info["pts1"]) < int(args.min_final_matches):
            continue

        mean_desc = float(match_info["desc_dist"].mean()) if len(match_info["desc_dist"]) > 0 else float("inf")
        mean_reproj = float(match_info["reproj_err"].mean()) if len(match_info["reproj_err"]) > 0 else float("inf")
        p95_reproj = float(np.percentile(match_info["reproj_err"], 95)) if len(match_info["reproj_err"]) > 0 else float("inf")

        if args.max_mean_desc_dist > 0 and mean_desc > float(args.max_mean_desc_dist):
            continue
        if args.max_mean_reproj_err > 0 and mean_reproj > float(args.max_mean_reproj_err):
            continue
        if args.max_p95_reproj_err > 0 and p95_reproj > float(args.max_p95_reproj_err):
            continue

        rec = {
            "pair_id": len(selected_pairs),
            "idx1": i,
            "idx2": j,
            "stem1": meta_i.stem,
            "stem2": meta_j.stem,
            "candidate_center_dist": float(math.sqrt(max(dist2, 0.0))),
            "overlap_ij": float(overlap_ij),
            "overlap_ji": float(overlap_ji),
            "overlap": float(overlap),
            "num_raw_good_matches": int(len(good)),
            "num_ransac_inliers": int(ransac_info["num_inliers"]),
            "ransac_inlier_ratio": float(ransac_info["inlier_ratio"]),
            "num_matches": int(len(match_info["pts1"])),
            "mean_reproj_err": mean_reproj,
            "median_reproj_err": float(np.median(match_info["reproj_err"])),
            "p95_reproj_err": p95_reproj,
            "mean_desc_dist": mean_desc,
        }
        selected_pairs.append(rec)

        np.savez_compressed(
            match_dir / f"{rec['pair_id']:04d}_{meta_i.stem}__{meta_j.stem}.npz",
            pts1=match_info["pts1"],
            pts2=match_info["pts2"],
            reproj_err=match_info["reproj_err"],
            desc_dist=match_info["desc_dist"],
            overlap=np.array([overlap], dtype=np.float32),
            overlap_ij=np.array([overlap_ij], dtype=np.float32),
            overlap_ji=np.array([overlap_ji], dtype=np.float32),
            stem1=np.array([meta_i.stem]),
            stem2=np.array([meta_j.stem]),
        )

        title = (
            f"{meta_i.stem} <-> {meta_j.stem} | ov={overlap:.3f} | "
            f"m={len(match_info['pts1'])} | desc={mean_desc:.2f} | rep={mean_reproj:.2f}"
        )
        visualize_matches_points(
            feat_i.img_bgr,
            feat_j.img_bgr,
            match_info["pts1"],
            match_info["pts2"],
            vis_dir / f"{rec['pair_id']:04d}_{meta_i.stem}__{meta_j.stem}.png",
            max_lines=args.max_vis_lines,
            seed=args.seed,
            title_text=title,
        )

        pbar.update(1)
        pbar.set_postfix({
            "sel": len(selected_pairs),
            "raw": len(good),
            "in": rec["num_ransac_inliers"],
            "keep": rec["num_matches"],
            "rep": f"{mean_reproj:.2f}",
        })

    pbar.close()

    with open(out_dir / "selected_pairs.json", "w", encoding="utf-8") as f:
        json.dump(selected_pairs, f, ensure_ascii=False, indent=2)

    with open(out_dir / "selected_pairs.txt", "w", encoding="utf-8") as f:
        for rec in selected_pairs:
            f.write(
                f"{rec['stem1']} {rec['stem2']} "
                f"overlap={rec['overlap']:.6f} "
                f"num_matches={rec['num_matches']} "
                f"mean_reproj_err={rec['mean_reproj_err']:.6f} "
                f"mean_desc_dist={rec['mean_desc_dist']:.6f}\n"
            )

    summary = {
        "num_views": len(metas),
        "num_selected_pairs": len(selected_pairs),
        "num_pairs_target": args.num_pairs,
        "pair_trials": tried,
        "stopped_early": len(selected_pairs) >= args.num_pairs,
        "saved_items": [
            "visualizations/",
            "match_data/*.npz",
            "selected_pairs.json",
            "selected_pairs.txt",
            "summary.json",
        ],
        "args": vars(args),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] selected pairs: {len(selected_pairs)}")
    print(f"[INFO] tried candidates: {tried}")
    print(f"[INFO] outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()



"""
# 选择立体像对，sift特征提取，用于后续的快速配准验证。

python select_stereo_pairs_with_depth_consistency.py \
    --image_dir ../undistort/nanfang/ImageCorrection/undistort \
    --cam_dir ./nanfang/cams \
    --depth_dir ./nanfang_render_lowres/depth_npy \
    --out_dir ./nanfang_pair_selected

python select_stereo_pairs_with_depth_consistency.py \
    --image_dir ../undistort/yanghaitang/ImageCorrection/undistort \
    --cam_dir ./yanghaitang/cams \
    --depth_dir ./yanghaitang_render_lowres/depth_npy \
    --out_dir ./yanghaitang_pair_selected

python select_stereo_pairs_with_depth_consistency.py \
    --image_dir ../undistort/xiaoxiang/ImageCorrection/undistort \
    --cam_dir ./xiaoxiang/cams \
    --depth_dir ./xiaoxiang_render_lowres/depth_npy \
    --out_dir ./xiaoxiang_pair_selected
"""

