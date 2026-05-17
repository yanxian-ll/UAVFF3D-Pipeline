"""Build sparse view-covisibility graphs for UAVFF3D/WAI-format scenes.

For each source view, the script projects depth-derived 3D points into candidate
target views, checks depth consistency, and stores the strongest covisible
neighbors as a memory-mappable CSR graph. The resulting graph is referenced from
``scene_meta.json`` and can be consumed by downstream samplers.
"""

import os
import math
import json
import shutil
from pathlib import Path
from collections import defaultdict
import yaml
import numpy as np
import torch
from tqdm import tqdm
import argparse

from dataset.wai.core import load_data, store_data
from dataset.wai.scene_frame import get_scene_names

from dataset.utils.covis_utils import (
    load_scene_data,
    compute_frustum_intersection,
    project_points_to_views,            
    sample_depths_at_reprojections,    
)

def cfg_get(cfg, k, default=None):
    if hasattr(cfg, "get"):
        return cfg.get(k, default)
    return getattr(cfg, k, default)

def save_csr_npz(path: Path, indptr, indices, data, shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        data=np.asarray(data, dtype=np.float32),
        shape=np.asarray(shape, dtype=np.int64),
    )

def save_csr_mmap_npy(dirpath: Path, indptr, indices, data, shape,
                      indices_dtype=np.int32, data_dtype=np.float32):
    """
    Save CSR graph into multiple .npy files for mmap loading.
    Layout:
      dirpath/
        indptr.npy
        indices.npy
        data.npy
        shape.npy
        meta.json (optional)
    """
    dirpath.mkdir(parents=True, exist_ok=True)

    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=indices_dtype)
    data = np.asarray(data, dtype=data_dtype)
    shape = np.asarray(shape, dtype=np.int64)

    np.save(str(dirpath / "indptr.npy"), indptr, allow_pickle=False)
    np.save(str(dirpath / "indices.npy"), indices, allow_pickle=False)
    np.save(str(dirpath / "data.npy"), data, allow_pickle=False)
    np.save(str(dirpath / "shape.npy"), shape, allow_pickle=False)

    meta = {
        "format": "csr_mmap_npy",
        "indptr_dtype": str(indptr.dtype),
        "indices_dtype": str(indices.dtype),
        "data_dtype": str(data.dtype),
        "shape": shape.tolist(),
        "nnz": int(indices.shape[0]),
    }
    with open(dirpath / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


@torch.no_grad()
def compute_covisibility(cfg, scene_name: str, overwrite=False):
    scene_root = Path(cfg.root) / scene_name
    scene_meta = load_data(scene_root / "scene_meta.json", "scene_meta")

    out_dir = scene_root / cfg.out_path
    view_graph_dir = out_dir / "view_covis_graph_csr_mmap"
    view_graph_dir.mkdir(parents=True, exist_ok=True)

    if (view_graph_dir / "indptr.npy").exists() and not overwrite:
        scene_meta["scene_modalities"]["covis_graph_view_csr"] = {
            "scene_key": f"{cfg.out_path}/view_covis_graph_csr_mmap",  # 指向目录
            "format": "csr_mmap_npy",
        }
        store_data(scene_root / "scene_meta.json", scene_meta, "scene_meta")
        print(f"[{scene_name}] covisibility graph already exists, skipping.")
        return
    # mapanything format
    elif (out_dir / "v0").exists() and not overwrite:
        pairwise_npy = os.listdir(out_dir / "v0")[0]
        scene_meta["scene_modalities"]["pairwise_covisibility"] = {
            "scene_key": f"{cfg.out_path}/v0/{pairwise_npy}",
            "format": "mmap",
        }
        store_data(scene_root / "scene_meta.json", scene_meta, "scene_meta")
        print(f"[{scene_name}] covisibility graph already exists, skipping.")
        return
    
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data (depths/intrinsics/cam2worlds/valid masks/points)
    scene_data = load_scene_data(cfg, scene_name, device)

    depths = scene_data["depths"]
    depth_h, depth_w = scene_data["depth_dims"]
    valid_depth_masks = scene_data["valid_depth_masks"]
    intrinsics = scene_data["intrinsics"]
    cam2worlds = scene_data["cam2worlds"]
    world_pts3d = scene_data["world_pts3d"]   # list[tensor] or [N,...] depending on你的实现

    num_frames = depths.shape[0]
    print(f"[{scene_name}] num_views={num_frames}, depth_res=({depth_h},{depth_w})")

    # Candidate filtering: frustum intersection (optional)
    frustum_intersection = compute_frustum_intersection(
        cfg, depths, valid_depth_masks, intrinsics, cam2worlds, device
    )

    min_covis = float(cfg_get(cfg, "min_covis", 0.1))
    topk = int(cfg_get(cfg, "topk", 50))
    exclude_same_frame = bool(cfg_get(cfg, "exclude_same_frame", True))
    ov_chunk_size = int(cfg_get(cfg, "ov_chunk_size", 512))

    # 输出 CSR：每行 i 只保留 topk 且 >= min_covis 的邻居
    indptr = [0]
    indices = []
    data = []

    print(f"[{scene_name}] build VIEW-graph CSR (min_covis={min_covis}, topk={topk})")

    for idx in tqdm(range(num_frames), desc=f"VIEW graph ({scene_name})"):
        if cfg_get(cfg, "perform_frustum_check", True) and frustum_intersection is not None:
            ov_inds = frustum_intersection[idx].nonzero(as_tuple=False)[:, 0].to(device)
        else:
            ov_inds = torch.arange(num_frames, device=device)

        if ov_inds.numel() == 0:
            indptr.append(indptr[-1])
            continue

        # always drop self
        ov_inds = ov_inds[ov_inds != idx]
        if ov_inds.numel() == 0:
            indptr.append(indptr[-1])
            continue

        # compute overlap in chunks
        overlap_score = torch.zeros((num_frames,), device="cpu")
        for s in range(0, ov_inds.numel(), ov_chunk_size):
            e = min(s + ov_chunk_size, ov_inds.numel())
            ov_chunk = ov_inds[s:e]
            if ov_chunk.numel() == 0:
                continue

            reprojected_pts, valid_mask, _ = project_points_to_views(
                idx, ov_chunk, depth_h, depth_w,
                valid_depth_masks, cam2worlds, world_pts3d, intrinsics, device
            )

            if valid_mask.any():
                depth_lu, expected_depth = sample_depths_at_reprojections(
                    reprojected_pts, depths, ov_chunk, depth_h, depth_w, device
                )
                reprojection_error = torch.abs(expected_depth - depth_lu)

                # 深度关联阈值
                depth_assoc_error_thres = float(cfg_get(cfg, "depth_assoc_error_thres", 0.02))
                depth_assoc_rel_error_thres = float(cfg_get(cfg, "depth_assoc_rel_error_thres", 0.01))
                depth_assoc_error_temp = float(cfg_get(cfg, "depth_assoc_error_temp", 0.0))

                depth_assoc_thres = (
                    depth_assoc_error_thres
                    + depth_assoc_rel_error_thres * expected_depth
                    - math.log(0.5) * depth_assoc_error_temp
                )
                valid_depth_projection = (reprojection_error < depth_assoc_thres) & valid_mask

                denom_mode = cfg_get(cfg, "denominator_mode", "valid_target_depth")
                if denom_mode == "valid_target_depth":
                    score = valid_depth_projection.sum([1, 2]) / valid_depth_masks[ov_chunk].sum([1, 2]).clamp(1)
                    score = score.clamp(0, 1)
                elif denom_mode == "full":
                    score = valid_depth_projection.sum([1, 2]) / float(depth_h * depth_w)
                else:
                    raise NotImplementedError(f"denominator_mode={denom_mode}")

                overlap_score[ov_chunk.cpu()] = score.cpu()

        row = overlap_score.numpy()
        # keep >= min_covis
        cand = np.where(row >= min_covis)[0]
        if cand.size == 0:
            indptr.append(indptr[-1])
            continue

        # topk by score
        if cand.size > topk:
            part = np.argpartition(row[cand], -topk)[-topk:]
            cand = cand[part]
        # sort by score desc (optional)
        cand = cand[np.argsort(-row[cand])]

        indices.extend(cand.tolist())
        data.extend(row[cand].astype(np.float32).tolist())
        indptr.append(len(indices))
        torch.cuda.empty_cache()

    # save VIEW CSR
    save_csr_mmap_npy(
        view_graph_dir,
        indptr, indices, data,
        shape=(num_frames, num_frames),
        indices_dtype=np.int32,
        data_dtype=np.float16,
    )

    scene_meta["scene_modalities"]["covis_graph_view_csr"] = {
        "scene_key": f"{cfg.out_path}/view_covis_graph_csr_mmap",  # 指向目录
        "format": "csr_mmap_npy",
    }

    store_data(scene_root / "scene_meta.json", scene_meta, "scene_meta")
    print(f"[{scene_name}] done. graphs saved to {out_dir}")


class AttrDict(dict):
    """dict + attribute access + get()"""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e
    def __setattr__(self, k, v):
        self[k] = v


def _load_yaml_cfg(path: str) -> dict:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"YAML config must be a dict, got {type(cfg)}")
        return cfg
    return {}

def parse_args():
    default_config_path = "configs/covisibility_config.yaml"

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=default_config_path)
    pre_args, _ = pre.parse_known_args()
    cfg_yaml = _load_yaml_cfg(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Compute view-covisibility CSR graphs for UAVFF3D/WAI scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=pre_args.config)
    parser.add_argument("--root", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--out_path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--random_scene_processing_order", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--min_covis", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--topk", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--exclude_same_frame", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--build_frame_graph", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--frame_topk", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--frame_min_covis", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--denominator_mode", type=str, choices=["valid_target_depth", "full"], default=argparse.SUPPRESS)
    parser.add_argument("--depth_assoc_error_thres", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--depth_assoc_rel_error_thres", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--depth_assoc_error_temp", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--perform_frustum_check", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--ov_chunk_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_workers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--downscale_factor", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--target_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true", default=argparse.SUPPRESS)

    merged = dict(cfg_yaml)
    merged.update(vars(parser.parse_args()))
    return AttrDict(merged)


def main():
    cfg = parse_args()
    scene_names = get_scene_names(
        cfg, shuffle=cfg.get("random_scene_processing_order", False)
    )

    for scene_name in tqdm(sorted(scene_names), desc="Processing scenes"):
        compute_covisibility(cfg, scene_name, overwrite=cfg.get("overwrite", False))


if __name__ == "__main__":
    main()
