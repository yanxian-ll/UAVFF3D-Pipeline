"""Build sparse view-covisibility graphs for A3D/WAI-format scenes.

For each source view, the script projects depth-derived 3D points into candidate
target views, checks depth consistency, and stores the strongest covisible
neighbors as a memory-mappable CSR graph. The resulting graph is referenced from
``scene_meta.json`` and can be consumed by downstream samplers.
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataset.utils.covis_utils import (
    compute_frustum_intersection,
    load_scene_data,
    project_points_to_views,
    sample_depths_at_reprojections,
)
from dataset.wai.core import load_data, store_data
from dataset.wai.scene_frame import get_scene_names


def cfg_get(cfg, key, default=None):
    """Read either a dict key or an object attribute."""
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def save_csr_npz(path: Path, indptr, indices, data, shape):
    """Save a compact CSR graph as one compressed ``.npz`` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        data=np.asarray(data, dtype=np.float32),
        shape=np.asarray(shape, dtype=np.int64),
    )


def save_csr_mmap_npy(
    dirpath: Path,
    indptr,
    indices,
    data,
    shape,
    indices_dtype=np.int32,
    data_dtype=np.float32,
):
    """Save CSR arrays separately so large graphs can be memory-mapped."""
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


def _attach_existing_graph(scene_root: Path, scene_meta: dict, cfg, out_dir: Path) -> bool:
    """Update metadata when a compatible graph already exists."""
    view_graph_dir = out_dir / "view_covis_graph_csr_mmap"
    if (view_graph_dir / "indptr.npy").exists():
        scene_meta["scene_modalities"]["covis_graph_view_csr"] = {
            "scene_key": f"{cfg.out_path}/view_covis_graph_csr_mmap",
            "format": "csr_mmap_npy",
        }
        store_data(scene_root / "scene_meta.json", scene_meta, "scene_meta")
        return True

    # Legacy MapAnything-style pairwise graph.
    legacy_dir = out_dir / "v0"
    if legacy_dir.exists():
        pairwise_npy = os.listdir(legacy_dir)[0]
        scene_meta["scene_modalities"]["pairwise_covisibility"] = {
            "scene_key": f"{cfg.out_path}/v0/{pairwise_npy}",
            "format": "mmap",
        }
        store_data(scene_root / "scene_meta.json", scene_meta, "scene_meta")
        return True

    return False


@torch.no_grad()
def compute_covisibility(cfg, scene_name: str, overwrite=False):
    """Compute and store one scene's view-covisibility CSR graph."""
    scene_root = Path(cfg.root) / scene_name
    scene_meta = load_data(scene_root / "scene_meta.json", "scene_meta")

    out_dir = scene_root / cfg.out_path
    view_graph_dir = out_dir / "view_covis_graph_csr_mmap"

    if not overwrite and _attach_existing_graph(scene_root, scene_meta, cfg, out_dir):
        print(f"[{scene_name}] covisibility graph already exists, skipping.")
        return

    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    view_graph_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scene_data = load_scene_data(cfg, scene_name, device)

    depths = scene_data["depths"]
    depth_h, depth_w = scene_data["depth_dims"]
    valid_depth_masks = scene_data["valid_depth_masks"]
    intrinsics = scene_data["intrinsics"]
    cam2worlds = scene_data["cam2worlds"]
    world_pts3d = scene_data["world_pts3d"]

    num_frames = depths.shape[0]
    print(f"[{scene_name}] num_views={num_frames}, depth_res=({depth_h},{depth_w})")

    frustum_intersection = compute_frustum_intersection(
        cfg, depths, valid_depth_masks, intrinsics, cam2worlds, device
    )

    min_covis = float(cfg_get(cfg, "min_covis", 0.1))
    topk = int(cfg_get(cfg, "topk", 50))
    ov_chunk_size = int(cfg_get(cfg, "ov_chunk_size", 512))

    indptr = [0]
    indices = []
    data = []

    print(f"[{scene_name}] build VIEW-graph CSR (min_covis={min_covis}, topk={topk})")

    for idx in tqdm(range(num_frames), desc=f"VIEW graph ({scene_name})"):
        if cfg_get(cfg, "perform_frustum_check", True) and frustum_intersection is not None:
            ov_inds = frustum_intersection[idx].nonzero(as_tuple=False)[:, 0].to(device)
        else:
            ov_inds = torch.arange(num_frames, device=device)

        # Same-frame overlap is not useful for pair sampling.
        ov_inds = ov_inds[ov_inds != idx]
        if ov_inds.numel() == 0:
            indptr.append(indptr[-1])
            continue

        overlap_score = torch.zeros((num_frames,), device="cpu")
        for start in range(0, ov_inds.numel(), ov_chunk_size):
            end = min(start + ov_chunk_size, ov_inds.numel())
            ov_chunk = ov_inds[start:end]
            if ov_chunk.numel() == 0:
                continue

            reprojected_pts, valid_mask, _ = project_points_to_views(
                idx,
                ov_chunk,
                depth_h,
                depth_w,
                valid_depth_masks,
                cam2worlds,
                world_pts3d,
                intrinsics,
                device,
            )

            if not valid_mask.any():
                continue

            depth_lu, expected_depth = sample_depths_at_reprojections(
                reprojected_pts, depths, ov_chunk, depth_h, depth_w, device
            )
            reprojection_error = torch.abs(expected_depth - depth_lu)

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
        candidates = np.where(row >= min_covis)[0]
        if candidates.size == 0:
            indptr.append(indptr[-1])
            continue

        if candidates.size > topk:
            topk_part = np.argpartition(row[candidates], -topk)[-topk:]
            candidates = candidates[topk_part]
        candidates = candidates[np.argsort(-row[candidates])]

        indices.extend(candidates.tolist())
        data.extend(row[candidates].astype(np.float32).tolist())
        indptr.append(len(indices))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_csr_mmap_npy(
        view_graph_dir,
        indptr,
        indices,
        data,
        shape=(num_frames, num_frames),
        indices_dtype=np.int32,
        data_dtype=np.float16,
    )

    scene_meta["scene_modalities"]["covis_graph_view_csr"] = {
        "scene_key": f"{cfg.out_path}/view_covis_graph_csr_mmap",
        "format": "csr_mmap_npy",
    }
    store_data(scene_root / "scene_meta.json", scene_meta, "scene_meta")
    print(f"[{scene_name}] done. graphs saved to {out_dir}")


class AttrDict(dict):
    """Dictionary with attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


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
        description="Compute view-covisibility CSR graphs for A3D/WAI scenes.",
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
