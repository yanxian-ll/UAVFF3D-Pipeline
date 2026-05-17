"""Generate UAVFF3D/WAI-style ``scene_meta.json`` files.

The converter expects every scene to use the common UAVFF3D layout:

    scene/
      images/<stem>.png|jpg|jpeg
      cams/<stem>.txt
      depth/<stem>.exr

Optional same-stem modalities such as ``mask/`` are attached only when present.
"""

import argparse
import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import PurePosixPath
from typing import Dict, Iterable, Optional

import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable


DEFAULT_DATASETS = [
    "UAVFF3D-Real",
    "UAVFF3D-Syn-L",
    "UAVFF3D-Syn-S",
    "UAVFF3D-FA",
    "uavscenes",
    "usegeo",
    "blendedmvs",
    "whu_whuomvs",
    "urbanscene3d",
    "enrich",
]


def _posix_join(*parts: str) -> str:
    """Join metadata paths with POSIX separators, independent of host OS."""
    return str(PurePosixPath(*parts))


def _try_parse_float_line(line: str) -> Optional[list[float]]:
    parts = line.strip().split()
    if not parts:
        return None
    try:
        return [float(v) for v in parts]
    except ValueError:
        return None


def read_cam_blendedmvs_txt(path_txt: str):
    """Read the common camera text format and return K, cam2world, H, W, FOV."""
    with open(path_txt, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f]

    extrinsic_header = None
    intrinsic_header = None
    for idx, line in enumerate(lines):
        lower = line.lower()
        if lower.startswith("extrinsic"):
            extrinsic_header = idx
        elif lower.startswith("intrinsic"):
            intrinsic_header = idx

    if extrinsic_header is None or intrinsic_header is None:
        raise ValueError(f"Invalid camera file, missing headers: {path_txt}")

    extrinsic = []
    for line in lines[extrinsic_header + 1:]:
        vals = _try_parse_float_line(line)
        if vals is not None and len(vals) >= 4:
            extrinsic.append(vals[:4])
        if len(extrinsic) == 4:
            break
    if len(extrinsic) != 4:
        raise ValueError(f"Failed to parse 4x4 extrinsic from: {path_txt}")

    intrinsic = []
    for line in lines[intrinsic_header + 1:]:
        vals = _try_parse_float_line(line)
        if vals is not None and len(vals) >= 3:
            intrinsic.append(vals[:3])
        if len(intrinsic) == 3:
            break
    if len(intrinsic) != 3:
        raise ValueError(f"Failed to parse 3x3 intrinsic from: {path_txt}")

    height = width = 0
    fov = 0.0
    for line in lines[intrinsic_header + 4:]:
        vals = _try_parse_float_line(line)
        if vals is not None and len(vals) >= 3:
            height = int(round(vals[0]))
            width = int(round(vals[1]))
            fov = float(vals[2])
            break
    if height <= 0 or width <= 0:
        raise ValueError(f"Failed to parse image size from: {path_txt}")

    world_to_cam = np.asarray(extrinsic, dtype=np.float32)
    cam_to_world = np.linalg.inv(world_to_cam)
    K = np.asarray(intrinsic, dtype=np.float32)
    return K, cam_to_world, height, width, fov


def collect_stem_to_file(dir_path: str, exts: Iterable[str]) -> dict[str, str]:
    """Map file stems to filenames for one modality directory."""
    if not os.path.isdir(dir_path):
        return {}

    allowed_exts = {ext.lower() for ext in exts}
    mapping = {}
    for fname in os.listdir(dir_path):
        full_path = os.path.join(dir_path, fname)
        if not os.path.isfile(full_path):
            continue
        stem, ext = os.path.splitext(fname)
        if ext.lower() in allowed_exts:
            mapping[stem] = fname
    return mapping


def generate_scene_meta(data_dir: str, scene_name: str, dataset_name: str, version: str = "1.0"):
    """Create ``scene_meta.json`` for one processed scene directory."""
    cam_dir = os.path.join(data_dir, "cams")
    img_dir = os.path.join(data_dir, "images")
    depth_dir = os.path.join(data_dir, "depth")

    cam_files = collect_stem_to_file(cam_dir, [".txt"])
    img_files = collect_stem_to_file(img_dir, [".png", ".jpg", ".jpeg"])
    depth_files = collect_stem_to_file(depth_dir, [".exr"])

    common_stems = sorted(set(cam_files) & set(img_files) & set(depth_files))
    if not common_stems:
        raise RuntimeError(
            f"No common stems among images/cams/depth in {data_dir}; "
            f"images={len(img_files)}, cams={len(cam_files)}, depth={len(depth_files)}"
        )

    optional_dirs = {
        "mask": ("mask", [".png"], "binary"),
    }
    optional_files = {
        key: collect_stem_to_file(os.path.join(data_dir, dirname), exts)
        for key, (dirname, exts, _format) in optional_dirs.items()
    }
    used_optional = {key: False for key in optional_dirs}

    frames = []
    last_height = last_width = 0
    last_fov = 0.0

    for stem in common_stems:
        image_rel = _posix_join("images", img_files[stem])
        depth_rel = _posix_join("depth", depth_files[stem])

        intrinsics, cam_to_world_opencv, height, width, fov = read_cam_blendedmvs_txt(
            os.path.join(cam_dir, cam_files[stem])
        )
        last_height, last_width, last_fov = height, width, fov

        frame = {
            "frame_name": stem,
            "image": image_rel,
            "file_path": image_rel,
            "depth": depth_rel,
            "transform_matrix": cam_to_world_opencv.tolist(),
            "h": height,
            "w": width,
            "fl_x": float(intrinsics[0, 0]),
            "fl_y": float(intrinsics[1, 1]),
            "cx": float(intrinsics[0, 2]),
            "cy": float(intrinsics[1, 2]),
        }

        for key, files in optional_files.items():
            if stem not in files:
                continue
            dirname, _exts, _format = optional_dirs[key]
            frame[key] = _posix_join(dirname, files[stem])
            used_optional[key] = True

        frames.append(frame)

    frame_modalities = {
        "image": {"frame_key": "image", "format": "image"},
        "depth": {"frame_key": "depth", "format": "depth"},
    }
    for key, used in used_optional.items():
        if not used:
            continue
        _dirname, _exts, format_name = optional_dirs[key]
        frame_modalities[key] = {"frame_key": key, "format": format_name}

    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": dataset_name,
        "version": version,
        "shared_intrinsics": False,
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "scale_type": "colmap",
        "scene_modalities": {},
        "frames": frames,
        "frame_modalities": frame_modalities,
    }

    out_json = os.path.join(data_dir, "scene_meta.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(scene_meta, f, indent=2, ensure_ascii=False)

    return {
        "dataset": dataset_name,
        "scene": scene_name,
        "stats": {
            "num_images": len(common_stems),
            "height": last_height,
            "width": last_width,
            "fov": float(last_fov),
            "num_images_all": len(img_files),
            "num_cams_all": len(cam_files),
            "num_depth_all": len(depth_files),
        },
        "skipped": False,
    }


# Backward-compatible alias for the historical misspelling.
generate_scene_mata = generate_scene_meta


def process_single_scene(args_tuple):
    """Worker wrapper for one scene directory."""
    scene_dir, scene, dataset, version, overwrite = args_tuple
    try:
        out_json = os.path.join(scene_dir, "scene_meta.json")
        if os.path.isfile(out_json) and not overwrite:
            print(f"[{dataset}/{scene}] skip: scene_meta.json already exists")
            return {
                "dataset": dataset,
                "scene": scene,
                "stats": None,
                "skipped": True,
            }

        result = generate_scene_meta(
            data_dir=scene_dir,
            scene_name=scene,
            dataset_name=dataset,
            version=version,
        )

        stats = result["stats"]
        print(
            f"[{dataset}/{scene}] "
            f"kept={stats['num_images']} "
            f"(images={stats['num_images_all']}, cams={stats['num_cams_all']}, depth={stats['num_depth_all']}) "
            f"resolution=({stats['height']}, {stats['width']}) "
            f"fov={stats['fov']:.2f}"
        )
        return result
    except Exception as exc:
        print(f"Error processing {dataset}/{scene}: {exc}")
        return None


def find_scene_tasks(root_dir: str, datasets: Iterable[str], version: str, overwrite: bool):
    """Find scene folders that already contain images/cams/depth."""
    target_datasets = set(datasets)
    tasks = []
    for dataset in os.listdir(root_dir):
        if dataset not in target_datasets:
            continue

        data_dir = os.path.join(root_dir, dataset)
        if not os.path.isdir(data_dir):
            continue

        for scene in os.listdir(data_dir):
            scene_dir = os.path.join(data_dir, scene)
            if not os.path.isdir(scene_dir):
                continue

            required_dirs = [
                os.path.join(scene_dir, "cams"),
                os.path.join(scene_dir, "images"),
                os.path.join(scene_dir, "depth"),
            ]
            if all(os.path.isdir(path) for path in required_dirs):
                tasks.append((scene_dir, scene, dataset, version, overwrite))
    return tasks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate scene_meta.json for processed UAVFF3D/WAI-format scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--version", type=str, default="1.0")
    parser.add_argument("--dataset", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to min(cpu_count, number of scenes).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing scene_meta.json files instead of skipping them.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    tasks = find_scene_tasks(args.root_dir, args.dataset, args.version, args.overwrite)

    print(f"Found {len(tasks)} scenes to process")
    if not tasks:
        print("No valid scenes found.")
        return

    num_workers = args.num_workers if args.num_workers else min(cpu_count(), len(tasks))
    dataset_stats: Dict[str, dict] = {}
    skipped_count = 0

    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_single_scene, tasks),
                total=len(tasks),
                desc="Processing scenes",
            )
        )

    for result in results:
        if result is None:
            continue
        if result.get("skipped", False):
            skipped_count += 1
            continue

        dataset = result["dataset"]
        scene = result["scene"]
        dataset_stats.setdefault(dataset, {})[scene] = result["stats"]

    generated_count = sum(len(scenes) for scenes in dataset_stats.values())
    print(f"Done. Generated: {generated_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
