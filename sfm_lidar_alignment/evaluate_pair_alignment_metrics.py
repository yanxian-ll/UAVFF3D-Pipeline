#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据已选好的像对目录（selected_pairs.json + match_data/*.npz），结合 cams 和 depth_npy
计算基于重投影的一致性对齐指标。

输入：
- pair_dir: 由 select_stereo_pairs_streaming_ref.py 等脚本输出的像对目录
- cam_dir:  相机目录（BlendedMVS / MVSNet 风格 txt）
- depth_dir: 深度目录（通常是 depth_npy）

说明：
1) 本脚本默认 pair_dir/match_data/*.npz 中的 pts1 / pts2 处于“缩放后的图像坐标系”。
2) 缩放参数优先从 pair_dir/summary.json 中读取 max_image_long_edge；也可命令行覆盖。
3) 如果原始图像分辨率与 cam txt 中记录的 nominal h,w 不一致，而当前又不提供 image_dir，
   则本脚本会假设 nominal h,w 就是匹配时使用的原始图像尺度。大多数标准数据导出流程下这是成立的。

输出：
- metrics_per_pair.csv
- metrics_per_pair.json
- summary.txt
- summary.json
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"]
DEPTH_EXTS = [".npy", ".exr", ".pfm", ".png", ".tiff", ".tif"]


@dataclass
class CameraMeta:
    stem: str
    K_nom: np.ndarray
    cam2world: np.ndarray
    world2cam: np.ndarray
    nominal_wh: Tuple[int, int]   # (w, h)
    depth_wh: Tuple[int, int]     # (w, h)
    K_depth: np.ndarray
    K_scaled: np.ndarray
    img_scale: float


@dataclass
class PairEvalResult:
    pair_id: int
    stem1: str
    stem2: str
    file_npz: str
    num_input_matches: int
    num_valid_forward: int
    num_valid_backward: int
    num_valid_symmetric: int
    mean_fwd: Optional[float]
    median_fwd: Optional[float]
    rmse_fwd: Optional[float]
    p95_fwd: Optional[float]
    mean_bwd: Optional[float]
    median_bwd: Optional[float]
    rmse_bwd: Optional[float]
    p95_bwd: Optional[float]
    mean_sym: Optional[float]
    median_sym: Optional[float]
    rmse_sym: Optional[float]
    p95_sym: Optional[float]
    inlier_1px: Optional[float]
    inlier_2px: Optional[float]
    inlier_3px: Optional[float]
    inlier_5px: Optional[float]
    depth_consistent_ratio_fwd: Optional[float]
    depth_consistent_ratio_bwd: Optional[float]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# =========================================================
# IO helpers
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
        vals = _try_parse_float_line(lines[j])
        if vals is not None and len(vals) >= 4:
            RT.append(vals[:4])
        if len(RT) == 4:
            break
    if len(RT) != 4:
        raise ValueError(f"Failed to parse extrinsic from: {path_txt}")
    RT = np.asarray(RT, dtype=np.float64)

    K = []
    for j in range(in_idx + 1, min(in_idx + 8, len(lines))):
        vals = _try_parse_float_line(lines[j])
        if vals is not None and len(vals) >= 3:
            K.append(vals[:3])
        if len(K) == 3:
            break
    if len(K) != 3:
        raise ValueError(f"Failed to parse intrinsic from: {path_txt}")
    K = np.asarray(K, dtype=np.float64)

    cam_h, cam_w, misc = 0, 0, 0.0
    for j in range(in_idx + 4, len(lines)):
        vals = _try_parse_float_line(lines[j])
        if vals is not None and len(vals) >= 3:
            cam_h = int(round(vals[0]))
            cam_w = int(round(vals[1]))
            misc = float(vals[2])
            break

    cam2world = np.linalg.inv(RT)
    return K, cam2world, cam_h, cam_w, misc


def _try_parse_float_line(line: str):
    vals = line.strip().split()
    if len(vals) == 0:
        return None
    try:
        return [float(v) for v in vals]
    except Exception:
        return None


def read_depth_any(path: Path):
    path = Path(path)
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


def find_depth_for_stem(depth_dir: Path, stem: str):
    for ext in DEPTH_EXTS:
        p = depth_dir / f"{stem}{ext}"
        if p.exists():
            return p
    for p in depth_dir.glob(f"{stem}.*"):
        if p.suffix.lower() in {e.lower() for e in DEPTH_EXTS}:
            return p
    return None


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


def compute_scaled_wh_and_K(K_nom: np.ndarray, nominal_wh: Tuple[int, int], max_long_edge: int):
    w_nom, h_nom = nominal_wh
    if max_long_edge is None or max_long_edge <= 0:
        return (w_nom, h_nom), K_nom.copy().astype(np.float64), 1.0
    long_edge = max(w_nom, h_nom)
    if long_edge <= max_long_edge:
        return (w_nom, h_nom), K_nom.copy().astype(np.float64), 1.0
    scale = float(max_long_edge) / float(long_edge)
    w_scaled = max(1, int(round(w_nom * scale)))
    h_scaled = max(1, int(round(h_nom * scale)))
    K_scaled = scale_intrinsics(K_nom, w_nom, h_nom, w_scaled, h_scaled)
    return (w_scaled, h_scaled), K_scaled, scale


# =========================================================
# Geometry helpers
# =========================================================
def bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    Ia = img[y0c, x0c].astype(np.float32)
    Ib = img[y0c, x1c].astype(np.float32)
    Ic = img[y1c, x0c].astype(np.float32)
    Id = img[y1c, x1c].astype(np.float32)

    wa = (x1.astype(np.float32) - xs) * (y1.astype(np.float32) - ys)
    wb = (xs - x0.astype(np.float32)) * (y1.astype(np.float32) - ys)
    wc = (x1.astype(np.float32) - xs) * (ys - y0.astype(np.float32))
    wd = (xs - x0.astype(np.float32)) * (ys - y0.astype(np.float32))

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def scaled_img_pts_to_depth_coords(pts_scaled: np.ndarray, img_scale: float, nominal_wh: Tuple[int, int], depth_wh: Tuple[int, int]):
    w_nom, h_nom = nominal_wh
    w_d, h_d = depth_wh
    u_nom = pts_scaled[:, 0] / max(img_scale, 1e-12)
    v_nom = pts_scaled[:, 1] / max(img_scale, 1e-12)
    u_dep = u_nom * (float(w_d) / float(w_nom))
    v_dep = v_nom * (float(h_d) / float(h_nom))
    return u_dep.astype(np.float32), v_dep.astype(np.float32)


def backproject_pixels_to_world(us, vs, depth, K_depth, cam2world):
    invK = np.linalg.inv(K_depth).astype(np.float64)
    ones = np.ones_like(us, dtype=np.float64)
    pix = np.stack([us.astype(np.float64), vs.astype(np.float64), ones], axis=0)
    rays = (invK @ pix).T
    Xc = rays * depth[:, None].astype(np.float64)
    R = cam2world[:3, :3].astype(np.float64)
    t = cam2world[:3, 3].astype(np.float64)
    Xw = (R @ Xc.T).T + t[None, :]
    return Xw.astype(np.float64)


def project_world_to_scaled_image(Xw: np.ndarray, K_scaled: np.ndarray, world2cam: np.ndarray):
    R = world2cam[:3, :3].astype(np.float64)
    t = world2cam[:3, 3].astype(np.float64)
    Xc = (R @ Xw.T).T + t[None, :]
    z = Xc[:, 2]
    x = Xc[:, 0] / (z + 1e-12)
    y = Xc[:, 1] / (z + 1e-12)
    u = K_scaled[0, 0] * x + K_scaled[0, 2]
    v = K_scaled[1, 1] * y + K_scaled[1, 2]
    return u.astype(np.float32), v.astype(np.float32), z.astype(np.float32)


def compute_error_stats(err: np.ndarray) -> Dict[str, Optional[float]]:
    if err is None or len(err) == 0:
        return {
            "mean": None, "median": None, "rmse": None, "p95": None,
            "inlier_1px": None, "inlier_2px": None, "inlier_3px": None, "inlier_5px": None,
        }
    err = np.asarray(err, dtype=np.float64)
    return {
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "p95": float(np.percentile(err, 95)),
        "inlier_1px": float(np.mean(err < 1.0)),
        "inlier_2px": float(np.mean(err < 2.0)),
        "inlier_3px": float(np.mean(err < 3.0)),
        "inlier_5px": float(np.mean(err < 5.0)),
    }


def evaluate_direction(
    pts_src_scaled: np.ndarray,
    pts_tgt_scaled: np.ndarray,
    cam_src: CameraMeta,
    cam_tgt: CameraMeta,
    depth_src: np.ndarray,
    depth_tgt: Optional[np.ndarray],
    min_depth: float,
    use_target_depth_consistency: bool,
    depth_abs_thr: float,
    depth_rel_thr: float,
):
    if pts_src_scaled is None or pts_tgt_scaled is None or len(pts_src_scaled) == 0:
        return None

    u_dep, v_dep = scaled_img_pts_to_depth_coords(
        pts_src_scaled, cam_src.img_scale, cam_src.nominal_wh, cam_src.depth_wh
    )

    # sample source depth
    z_src = bilinear_sample(depth_src, u_dep, v_dep)
    valid = np.isfinite(z_src) & (z_src > min_depth)
    if valid.sum() == 0:
        return None

    pts_src_scaled = pts_src_scaled[valid]
    pts_tgt_scaled = pts_tgt_scaled[valid]
    u_dep = u_dep[valid]
    v_dep = v_dep[valid]
    z_src = z_src[valid]

    Xw = backproject_pixels_to_world(u_dep, v_dep, z_src, cam_src.K_depth, cam_src.cam2world)
    u_tgt_pred, v_tgt_pred, z_tgt_pred = project_world_to_scaled_image(Xw, cam_tgt.K_scaled, cam_tgt.world2cam)

    w_tgt, h_tgt = cam_tgt.nominal_wh
    w_tgt_scaled, h_tgt_scaled = compute_scaled_wh_and_K(cam_tgt.K_nom, cam_tgt.nominal_wh, int(round(max(cam_tgt.nominal_wh) * cam_tgt.img_scale)) if cam_tgt.img_scale != 1.0 else 0)[0]
    in_front = z_tgt_pred > min_depth
    in_img = (u_tgt_pred >= 0) & (u_tgt_pred < w_tgt_scaled) & (v_tgt_pred >= 0) & (v_tgt_pred < h_tgt_scaled)

    keep = in_front & in_img
    if keep.sum() == 0:
        return None

    pts_src_scaled = pts_src_scaled[keep]
    pts_tgt_scaled = pts_tgt_scaled[keep]
    u_tgt_pred = u_tgt_pred[keep]
    v_tgt_pred = v_tgt_pred[keep]
    z_tgt_pred = z_tgt_pred[keep]

    depth_consistent_ratio = None
    if use_target_depth_consistency and depth_tgt is not None:
        u_dep_tgt, v_dep_tgt = scaled_img_pts_to_depth_coords(
            np.stack([u_tgt_pred, v_tgt_pred], axis=1),
            cam_tgt.img_scale, cam_tgt.nominal_wh, cam_tgt.depth_wh
        )
        z_tgt = bilinear_sample(depth_tgt, u_dep_tgt, v_dep_tgt)
        valid_tgt = np.isfinite(z_tgt) & (z_tgt > min_depth)
        if valid_tgt.sum() == 0:
            return None
        abs_ok = np.abs(z_tgt - z_tgt_pred) <= depth_abs_thr
        rel_ok = np.abs(z_tgt - z_tgt_pred) <= (depth_rel_thr * np.maximum(z_tgt, z_tgt_pred))
        ok = valid_tgt & (abs_ok | rel_ok)
        depth_consistent_ratio = float(np.mean(ok)) if len(ok) > 0 else None
        if ok.sum() == 0:
            return None
        pts_src_scaled = pts_src_scaled[ok]
        pts_tgt_scaled = pts_tgt_scaled[ok]
        u_tgt_pred = u_tgt_pred[ok]
        v_tgt_pred = v_tgt_pred[ok]

    err = np.sqrt((u_tgt_pred - pts_tgt_scaled[:, 0]) ** 2 + (v_tgt_pred - pts_tgt_scaled[:, 1]) ** 2)
    return {
        "err": err.astype(np.float32),
        "num_valid": int(len(err)),
        "depth_consistent_ratio": depth_consistent_ratio,
    }


# =========================================================
# Pair loading
# =========================================================
def parse_pair_id_from_name(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.split("_")[0])
    except Exception:
        return -1


def load_pair_metadata(pair_dir: Path) -> Dict[Tuple[str, str], Dict]:
    meta_path = pair_dir / "selected_pairs.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    out = {}
    for rec in items:
        key = (str(rec["stem1"]), str(rec["stem2"]))
        out[key] = rec
    return out


def auto_read_max_long_edge(pair_dir: Path) -> int:
    summary_path = pair_dir / "summary.json"
    if not summary_path.exists():
        return 0
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        args = summary.get("args", {})
        return int(args.get("max_image_long_edge", 0) or 0)
    except Exception:
        return 0


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate alignment metrics from selected stereo pairs")
    parser.add_argument("--pair_dir", type=str, required=True, help="像对结果目录，包含 selected_pairs.json 和 match_data/")
    parser.add_argument("--cam_dir", type=str, required=True, help="cams 目录")
    parser.add_argument("--depth_dir", type=str, required=True, help="depth_npy / depth 目录")
    parser.add_argument("--out_dir", type=str, default="", help="输出目录，默认写到 pair_dir/eval_metrics")

    parser.add_argument("--max_image_long_edge", type=int, default=-1, help="覆盖 pair_dir/summary.json 中记录的最大图像边长；<0 表示自动读取")
    parser.add_argument("--min_depth", type=float, default=1e-3)
    parser.add_argument("--use_target_depth_consistency", action="store_true", help="额外要求投影到目标图后的深度与目标 depth 一致")
    parser.add_argument("--depth_abs_thr", type=float, default=0.2)
    parser.add_argument("--depth_rel_thr", type=float, default=0.05)
    args = parser.parse_args()

    pair_dir = Path(args.pair_dir)
    cam_dir = Path(args.cam_dir)
    depth_dir = Path(args.depth_dir)
    if args.out_dir:
        out_dir = ensure_dir(Path(args.out_dir))
    else:
        out_dir = ensure_dir(pair_dir / "eval_metrics")

    match_dir = pair_dir / "match_data"
    if not match_dir.is_dir():
        raise RuntimeError(f"match_data not found: {match_dir}")

    max_long_edge = args.max_image_long_edge
    if max_long_edge < 0:
        max_long_edge = auto_read_max_long_edge(pair_dir)
    print(f"[INFO] max_image_long_edge used for evaluation: {max_long_edge}")

    pair_meta = load_pair_metadata(pair_dir)
    npz_files = sorted(match_dir.glob("*.npz"), key=parse_pair_id_from_name)
    if len(npz_files) == 0:
        raise RuntimeError(f"No pair npz found in: {match_dir}")

    # collect stems from pair files
    stems_needed = set()
    pair_file_records = []
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        stem1 = str(data["stem1"][0]) if "stem1" in data else None
        stem2 = str(data["stem2"][0]) if "stem2" in data else None
        if stem1 is None or stem2 is None:
            # try parse from filename: 0000_a__b.npz
            tail = npz_path.stem.split("_", 1)[1] if "_" in npz_path.stem else npz_path.stem
            if "__" in tail:
                stem1, stem2 = tail.split("__", 1)
            else:
                raise RuntimeError(f"Cannot infer stems from {npz_path.name}")
        pair_file_records.append((npz_path, stem1, stem2))
        stems_needed.add(stem1)
        stems_needed.add(stem2)

    # preload camera metadata and depth paths for needed stems only
    cam_meta: Dict[str, CameraMeta] = {}
    depth_cache: Dict[str, np.ndarray] = {}

    for stem in sorted(stems_needed):
        cam_path = cam_dir / f"{stem}.txt"
        if not cam_path.exists():
            raise FileNotFoundError(f"cam not found for stem '{stem}': {cam_path}")
        depth_path = find_depth_for_stem(depth_dir, stem)
        if depth_path is None:
            raise FileNotFoundError(f"depth not found for stem '{stem}' in {depth_dir}")

        K_nom, cam2world, cam_h, cam_w, _ = read_cam_blendedmvs_txt(str(cam_path))
        depth = read_depth_any(depth_path)
        if depth is None:
            raise RuntimeError(f"failed to read depth: {depth_path}")
        h_d, w_d = depth.shape[:2]
        nominal_wh = (cam_w, cam_h)
        depth_wh = (w_d, h_d)
        K_depth = scale_intrinsics(K_nom, cam_w, cam_h, w_d, h_d)
        scaled_wh, K_scaled, img_scale = compute_scaled_wh_and_K(K_nom, nominal_wh, max_long_edge)
        cam_meta[stem] = CameraMeta(
            stem=stem,
            K_nom=K_nom,
            cam2world=cam2world,
            world2cam=np.linalg.inv(cam2world),
            nominal_wh=nominal_wh,
            depth_wh=depth_wh,
            K_depth=K_depth,
            K_scaled=K_scaled,
            img_scale=img_scale,
        )
        depth_cache[stem] = depth

    results: List[PairEvalResult] = []
    all_sym_err = []

    for npz_path, stem1, stem2 in tqdm(pair_file_records, desc="Evaluate selected pairs", dynamic_ncols=True):
        data = np.load(npz_path, allow_pickle=True)
        pts1 = np.asarray(data["pts1"], dtype=np.float32)
        pts2 = np.asarray(data["pts2"], dtype=np.float32)

        cam1 = cam_meta[stem1]
        cam2 = cam_meta[stem2]
        depth1 = depth_cache[stem1]
        depth2 = depth_cache[stem2]

        fwd = evaluate_direction(
            pts_src_scaled=pts1,
            pts_tgt_scaled=pts2,
            cam_src=cam1,
            cam_tgt=cam2,
            depth_src=depth1,
            depth_tgt=depth2,
            min_depth=args.min_depth,
            use_target_depth_consistency=args.use_target_depth_consistency,
            depth_abs_thr=args.depth_abs_thr,
            depth_rel_thr=args.depth_rel_thr,
        )
        bwd = evaluate_direction(
            pts_src_scaled=pts2,
            pts_tgt_scaled=pts1,
            cam_src=cam2,
            cam_tgt=cam1,
            depth_src=depth2,
            depth_tgt=depth1,
            min_depth=args.min_depth,
            use_target_depth_consistency=args.use_target_depth_consistency,
            depth_abs_thr=args.depth_abs_thr,
            depth_rel_thr=args.depth_rel_thr,
        )

        err_fwd = fwd["err"] if fwd is not None else np.array([], dtype=np.float32)
        err_bwd = bwd["err"] if bwd is not None else np.array([], dtype=np.float32)
        err_sym = np.concatenate([err_fwd, err_bwd], axis=0) if (len(err_fwd) + len(err_bwd)) > 0 else np.array([], dtype=np.float32)
        if len(err_sym) > 0:
            all_sym_err.append(err_sym)

        st_fwd = compute_error_stats(err_fwd)
        st_bwd = compute_error_stats(err_bwd)
        st_sym = compute_error_stats(err_sym)

        pair_id = parse_pair_id_from_name(npz_path)
        results.append(PairEvalResult(
            pair_id=pair_id,
            stem1=stem1,
            stem2=stem2,
            file_npz=npz_path.name,
            num_input_matches=int(len(pts1)),
            num_valid_forward=int(len(err_fwd)),
            num_valid_backward=int(len(err_bwd)),
            num_valid_symmetric=int(len(err_sym)),
            mean_fwd=st_fwd["mean"],
            median_fwd=st_fwd["median"],
            rmse_fwd=st_fwd["rmse"],
            p95_fwd=st_fwd["p95"],
            mean_bwd=st_bwd["mean"],
            median_bwd=st_bwd["median"],
            rmse_bwd=st_bwd["rmse"],
            p95_bwd=st_bwd["p95"],
            mean_sym=st_sym["mean"],
            median_sym=st_sym["median"],
            rmse_sym=st_sym["rmse"],
            p95_sym=st_sym["p95"],
            inlier_1px=st_sym["inlier_1px"],
            inlier_2px=st_sym["inlier_2px"],
            inlier_3px=st_sym["inlier_3px"],
            inlier_5px=st_sym["inlier_5px"],
            depth_consistent_ratio_fwd=(fwd["depth_consistent_ratio"] if fwd is not None else None),
            depth_consistent_ratio_bwd=(bwd["depth_consistent_ratio"] if bwd is not None else None),
        ))

    # save per-pair CSV
    csv_path = out_dir / "metrics_per_pair.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].__dict__.keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)

    # save per-pair JSON
    json_path = out_dir / "metrics_per_pair.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, ensure_ascii=False, indent=2)

    # overall summary
    all_sym_err_concat = np.concatenate(all_sym_err, axis=0) if len(all_sym_err) > 0 else np.array([], dtype=np.float32)
    overall = compute_error_stats(all_sym_err_concat)
    pair_mean_sym = np.array([r.mean_sym for r in results if r.mean_sym is not None], dtype=np.float64)
    pair_med_sym = np.array([r.median_sym for r in results if r.median_sym is not None], dtype=np.float64)
    pair_rmse_sym = np.array([r.rmse_sym for r in results if r.rmse_sym is not None], dtype=np.float64)

    summary = {
        "num_pairs": len(results),
        "num_pairs_with_valid_sym": int(np.sum([r.num_valid_symmetric > 0 for r in results])),
        "num_points_symmetric_total": int(len(all_sym_err_concat)),
        "max_image_long_edge": int(max_long_edge),
        "use_target_depth_consistency": bool(args.use_target_depth_consistency),
        "depth_abs_thr": float(args.depth_abs_thr),
        "depth_rel_thr": float(args.depth_rel_thr),
        "overall_pointwise_symmetric": overall,
        "pairwise_mean_of_mean_sym": float(pair_mean_sym.mean()) if len(pair_mean_sym) > 0 else None,
        "pairwise_median_of_median_sym": float(np.median(pair_med_sym)) if len(pair_med_sym) > 0 else None,
        "pairwise_mean_of_rmse_sym": float(pair_rmse_sym.mean()) if len(pair_rmse_sym) > 0 else None,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("========== Alignment Evaluation Summary ==========" + "\n")
        f.write(f"num_pairs: {summary['num_pairs']}\n")
        f.write(f"num_pairs_with_valid_sym: {summary['num_pairs_with_valid_sym']}\n")
        f.write(f"num_points_symmetric_total: {summary['num_points_symmetric_total']}\n")
        f.write(f"max_image_long_edge: {summary['max_image_long_edge']}\n")
        f.write(f"use_target_depth_consistency: {summary['use_target_depth_consistency']}\n")
        f.write(f"depth_abs_thr: {summary['depth_abs_thr']}\n")
        f.write(f"depth_rel_thr: {summary['depth_rel_thr']}\n")
        f.write("\n[overall_pointwise_symmetric]\n")
        for k, v in overall.items():
            f.write(f"{k}: {v}\n")
        f.write("\n[pairwise]\n")
        f.write(f"pairwise_mean_of_mean_sym: {summary['pairwise_mean_of_mean_sym']}\n")
        f.write(f"pairwise_median_of_median_sym: {summary['pairwise_median_of_median_sym']}\n")
        f.write(f"pairwise_mean_of_rmse_sym: {summary['pairwise_mean_of_rmse_sym']}\n")

    print(f"[INFO] pair metrics csv saved to: {csv_path}")
    print(f"[INFO] pair metrics json saved to: {json_path}")
    print(f"[INFO] summary saved to: {out_dir / 'summary.json'}")
    print("[INFO] overall symmetric metrics:")
    for k, v in overall.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()


"""
# 使用 OBJ 采样点渲染的深度图，评估模型作为对比（理论上误差为0）

python evaluate_pair_alignment_metrics.py \
    --pair_dir ../recon/nanfang_pair_selected \
    --cam_dir ../recon/nanfang/cams \
    --depth_dir ../recon/nanfang_render/depth_npy \
    --out_dir ./eval/nanfang_eval_base

python evaluate_pair_alignment_metrics.py \
    --pair_dir ../recon/yanghaitang_pair_selected \
    --cam_dir ../recon/yanghaitang/cams \
    --depth_dir ../recon/yanghaitang_render/depth_npy \
    --out_dir ./eval/yanghaitang_eval_base

python evaluate_pair_alignment_metrics.py \
    --pair_dir ../recon/xiaoxiang_pair_selected \
    --cam_dir ../recon/xiaoxiang/cams \
    --depth_dir ../recon/xiaoxiang_render/depth_npy \
    --out_dir ./eval/xiaoxiang_eval_base

"""


"""
# 使用对齐后的激光点云渲染的深度图，评估配准精度
python evaluate_pair_alignment_metrics.py \
    --pair_dir ../recon/nanfang_pair_selected \
    --cam_dir ../recon/nanfang/cams \
    --depth_dir ./nanfang_final/depth_npy \
    --out_dir ./eval/nanfang_eval_lidar

python evaluate_pair_alignment_metrics.py \
    --pair_dir ../recon/yanghaitang_pair_selected \
    --cam_dir ../recon/yanghaitang/cams \
    --depth_dir ./yanghaitang_final/depth_npy \
    --out_dir ./eval/yanghaitang_eval_lidar

python evaluate_pair_alignment_metrics.py \
    --pair_dir ../recon/xiaoxiang_pair_selected \
    --cam_dir ../recon/xiaoxiang/cams \
    --depth_dir ./xiaoxiang_final/depth_npy \
    --out_dir ./eval/xiaoxiang_eval_lidar

"""

