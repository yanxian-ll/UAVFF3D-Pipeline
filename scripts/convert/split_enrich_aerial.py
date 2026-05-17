#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将已经整理好的数据集目录（包含 images / depth / cams）按照拍摄方向拆分为 3 个子集：
  1) _ndir     : nadir
  2) _oblique  : forward / backward / left / right
  3) _ndiir2   : nadir-2 / ndir-2

每个子集下仍然包含：
  images/
  depth/
  cams/

并且使用 copy 复制文件，不移动原文件。

示例：
python split_enrich_views.py \
    --src_root /path/to/enrich_aerial_out \
    --dst_root /path/to/enrich_aerial_split

输入目录结构要求：
src_root/
  images/*.png
  depth/*.exr
  cams/*.txt

输出目录结构示例：
dst_root/
  _ndir/
    images/
    depth/
    cams/
  _oblique/
    images/
    depth/
    cams/
  _ndiir2/
    images/
    depth/
    cams/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMG_EXTS = {".png", ".jpg", ".jpeg"}
DEPTH_EXTS = {".exr"}
CAM_EXTS = {".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split converted ENRICH dataset by camera view.")
    parser.add_argument(
        "--src_root",
        type=Path,
        required=True,
        help="输入根目录，内部应包含 images / depth / cams",
    )
    parser.add_argument(
        "--dst_root",
        type=Path,
        required=True,
        help="输出根目录，将在其中创建 _ndir / _oblique / _ndiir2",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标文件已存在，是否覆盖",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：若 images/depth/cams 中任一文件缺失，则报错退出",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_group_from_stem(stem: str) -> str:
    """
    根据文件 stem 判断属于哪个子集。
    例如：
      01_nadir      -> _ndir
      01_forward    -> _oblique
      01_nadir-2    -> _ndiir2
      01_ndir-2     -> _ndiir2
    """
    s = stem.lower()

    if s.endswith("_nadir-2") or s.endswith("_ndir-2"):
        return "aerial_ndiir2"

    if s.endswith("_forward") or s.endswith("_backward") or s.endswith("_left") or s.endswith("_right"):
        return "aerial_oblique"

    if s.endswith("_nadir"):
        return "aerial_ndir"

    raise ValueError(f"无法识别文件属于哪个视角组: {stem}")


def collect_by_stem(folder: Path, valid_exts: set[str]) -> Dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"目录不存在: {folder}")

    mapping: Dict[str, Path] = {}
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in valid_exts:
            continue
        mapping[p.stem] = p
    return mapping


def build_output_dirs(dst_root: Path) -> Dict[str, Dict[str, Path]]:
    groups = ["aerial_ndir", "aerial_oblique", "aerial_ndiir2"]
    subfolders = ["images", "depth", "cams"]

    out: Dict[str, Dict[str, Path]] = {}
    for g in groups:
        out[g] = {}
        for sub in subfolders:
            d = dst_root / g / sub
            ensure_dir(d)
            out[g][sub] = d
    return out


def copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()

    src_images = args.src_root / "images"
    src_depth = args.src_root / "depth"
    src_cams = args.src_root / "cams"

    img_map = collect_by_stem(src_images, IMG_EXTS)
    depth_map = collect_by_stem(src_depth, DEPTH_EXTS)
    cam_map = collect_by_stem(src_cams, CAM_EXTS)

    all_stems = sorted(set(img_map.keys()) | set(depth_map.keys()) | set(cam_map.keys()))
    if not all_stems:
        raise RuntimeError("未找到任何可处理文件，请检查输入目录。")

    out_dirs = build_output_dirs(args.dst_root)

    copied_count = 0
    skipped_count = 0
    group_stats = {"aerial_ndir": 0, "aerial_oblique": 0, "aerial_ndiir2": 0}
    missing_records: List[Tuple[str, bool, bool, bool]] = []

    for stem in all_stems:
        has_img = stem in img_map
        has_depth = stem in depth_map
        has_cam = stem in cam_map

        if not (has_img and has_depth and has_cam):
            missing_records.append((stem, has_img, has_depth, has_cam))
            if args.strict:
                raise FileNotFoundError(
                    f"文件不完整: stem={stem}, "
                    f"image={has_img}, depth={has_depth}, cam={has_cam}"
                )
            skipped_count += 1
            continue

        group = infer_group_from_stem(stem)

        img_src = img_map[stem]
        depth_src = depth_map[stem]
        cam_src = cam_map[stem]

        img_dst = out_dirs[group]["images"] / img_src.name
        depth_dst = out_dirs[group]["depth"] / depth_src.name
        cam_dst = out_dirs[group]["cams"] / cam_src.name

        copy_file(img_src, img_dst, args.overwrite)
        copy_file(depth_src, depth_dst, args.overwrite)
        copy_file(cam_src, cam_dst, args.overwrite)

        copied_count += 1
        group_stats[group] += 1

    print("========== Split Finished ==========")
    print(f"Source root : {args.src_root}")
    print(f"Output root : {args.dst_root}")
    print(f"Copied sets : {copied_count}")
    print(f"Skipped sets: {skipped_count}")
    print(f"aerial_ndir       : {group_stats['aerial_ndir']}")
    print(f"aerial_oblique    : {group_stats['aerial_oblique']}")
    print(f"aerial_ndiir2     : {group_stats['aerial_ndiir2']}")

    if missing_records:
        print("\n以下 stem 因 images/depth/cams 不完整而被跳过：")
        for stem, has_img, has_depth, has_cam in missing_records:
            print(
                f"  {stem}: "
                f"images={has_img}, depth={has_depth}, cams={has_cam}"
            )


if __name__ == "__main__":
    main()
