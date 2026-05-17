#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import traceback
from pathlib import Path
from typing import Tuple, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
from tqdm import tqdm

from reorganize_utils import save_cam_txt


_SCENE_RE = re.compile(r"^scene_(\d{4})$")


def get_next_scene_counter(out_dir: Path) -> int:
    max_idx = -1
    if out_dir.is_dir():
        for p in out_dir.iterdir():
            if p.is_dir():
                m = _SCENE_RE.match(p.name)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def parse_wilduav_camera(json_path: Path) -> Tuple[int, int, float, float, float, float, np.ndarray]:
    """
    WildUAV JSON:
      - intrinsicMatrix, rotation: stored in column-order (each inner list is a column) -> transpose to row-major
      - translation: treated as camera center C in world coords
    Return:
      W0, H0, fx, fy, cx, cy, T_world2cam (4x4)
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))

    H0 = int(data["height"])
    W0 = int(data["width"])

    K_col = np.array(data["intrinsicMatrix"], dtype=np.float64)
    R_col = np.array(data["rotation"], dtype=np.float64)
    C = np.array(data["translation"], dtype=np.float64).reshape(3)

    K = K_col.T
    R_row = R_col.T

    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])

    # row-vector: pLocal = (pWorld - C) * R_row
    # -> column-vector world2cam: R_cw = R_row^T, t = -R_cw*C
    R_cw = R_row.T
    t_cw = -(R_cw @ C)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cw
    T[:3, 3] = t_cw
    return W0, H0, fx, fy, cx, cy, T


def process_one_frame(task: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        src_img = Path(task["src_img"])
        src_meta = Path(task["src_meta"])
        src_depth = Path(task["src_depth"])

        dst_img = Path(task["dst_img"])
        dst_cam = Path(task["dst_cam"])
        dst_depth = Path(task["dst_depth"])
        dst_mask = Path(task["dst_mask"])

        max_size = int(task["max_size"])
        depth_max = float(task["depth_max"])
        convention = str(task["convention"]).lower()

        # 续跑：全部都在就跳过
        if dst_img.exists() and dst_cam.exists() and dst_depth.exists() and dst_mask.exists():
            return True, "skip"

        # 读相机（拿原始尺寸 + w2c）
        W0, H0, fx, fy, cx, cy, T_w2c = parse_wilduav_camera(src_meta)

        # 缩放：最长边 -> max_size（保持宽高比）
        if max_size > 0:
            s = max_size / float(max(H0, W0))
        else:
            s = 1.0
        new_h = max(1, int(H0 * s + 0.5))
        new_w = max(1, int(W0 * s + 0.5))

        # convention: cv -> gl（y,z 翻转）
        if convention == "gl":
            cv_to_gl = np.array([[1, 0, 0, 0],
                                 [0, -1, 0, 0],
                                 [0, 0, -1, 0],
                                 [0, 0, 0, 1]], dtype=np.float64)
            T_w2c = cv_to_gl @ T_w2c

        # 写图像
        if not dst_img.exists():
            img = cv2.imread(str(src_img), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"cv2.imread failed: {src_img}")
            img2 = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(dst_img), img2):
                raise RuntimeError(f"cv2.imwrite failed: {dst_img}")

        # 写相机（你这里用 save_cam_txt：传 cam2world，所以对 w2c 取逆）
        if not dst_cam.exists():
            fx2, fy2, cx2, cy2 = fx * s, fy * s, cx * s, cy * s
            dst_cam.parent.mkdir(parents=True, exist_ok=True)
            save_cam_txt(
                dst_cam,
                np.linalg.inv(T_w2c).astype(np.float64),  # cam2world
                fx2, fy2, cx2, cy2,
                new_h, new_w
            )

        # 写深度 + mask
        if (not dst_depth.exists()) or (not dst_mask.exists()):
            depth = np.load(str(src_depth)).astype(np.float32)

            # 清洗
            depth[~np.isfinite(depth)] = 0.0
            depth[depth <= 0.0] = 0.0
            if depth_max > 0:
                depth[depth > depth_max] = 0.0

            # resize（depth 用 nearest）
            depth2 = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)

            if not dst_depth.exists():
                dst_depth.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(dst_depth), depth2):
                    raise RuntimeError(f"cv2.imwrite EXR failed: {dst_depth} (check OpenEXR enabled)")

            if not dst_mask.exists():
                valid = ((depth2 > 0.0) & np.isfinite(depth2)).astype(np.uint8) * 255
                dst_mask.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(dst_mask), valid):
                    raise RuntimeError(f"cv2.imwrite mask failed: {dst_mask}")

        return True, "ok"
    except Exception as e:
        return False, f"{e}\n{traceback.format_exc()}"


def export_seq_to_one_scene(
    img_meta_dir: Path,
    depth_dir: Path,
    out_dir: Path,
    scene_id: int,
    max_size: int,
    depth_max: float,
    convention: str,
    workers: int,
):
    img_dir = img_meta_dir / "img"
    meta_dir = img_meta_dir / "metadata"
    dep_dir = depth_dir / "depth"

    if not (img_dir.is_dir() and meta_dir.is_dir() and dep_dir.is_dir()):
        print(f"[SKIP] missing subdir(s): {img_meta_dir.name}")
        return

    imgs = {p.stem: p for p in img_dir.glob("*.png")}
    metas = {p.stem: p for p in meta_dir.glob("*.json")}
    deps = {p.stem: p for p in dep_dir.glob("*.npy")}

    keys = sorted(set(imgs) & set(metas) & set(deps), key=lambda s: int(s))
    if not keys:
        print(f"[SKIP] no matched frames: {img_meta_dir.name}")
        return

    scene_dir = out_dir / f"scene_{scene_id:04d}"
    (scene_dir / "images").mkdir(parents=True, exist_ok=True)
    (scene_dir / "cams").mkdir(parents=True, exist_ok=True)
    (scene_dir / "depth").mkdir(parents=True, exist_ok=True)
    (scene_dir / "mask").mkdir(parents=True, exist_ok=True)

    print(f"\n[EXPORT] {scene_dir.name}  src={img_meta_dir.name}  frames={len(keys)}  max_size={max_size}  depth_max={depth_max}  convention={convention}  workers={workers}")

    tasks: List[Dict[str, Any]] = []
    for k in keys:
        out_id = f"{int(k):08d}"
        tasks.append({
            "src_img": str(imgs[k]),
            "src_meta": str(metas[k]),
            "src_depth": str(deps[k]),
            "dst_img": str(scene_dir / "images" / f"{out_id}.png"),
            "dst_cam": str(scene_dir / "cams" / f"{out_id}.txt"),
            "dst_depth": str(scene_dir / "depth" / f"{out_id}.exr"),
            "dst_mask": str(scene_dir / "mask" / f"{out_id}.png"),
            "max_size": int(max_size),
            "depth_max": float(depth_max),
            "convention": str(convention),
        })

    fail = 0
    bar = tqdm(total=len(tasks), desc=scene_dir.name, ncols=110)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(process_one_frame, t) for t in tasks]
        for fut in as_completed(futs):
            ok, msg = fut.result()
            if not ok:
                fail += 1
                print("[FAIL]", msg)
            bar.update(1)

    bar.close()
    print(f"[DONE] {scene_dir.name}: {'OK' if fail == 0 else f'FAIL {fail}/{len(tasks)}'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="../raw_data/WildUAV/mapping_set")
    ap.add_argument("--out_dir", type=str, default="../data/wilduav")
    ap.add_argument("--max_size", type=int, default=1024, help="Resize so that max(H,W)=max_size, keep aspect. 0 disables.")
    ap.add_argument("--depth_max", type=float, default=300.0)
    ap.add_argument("--convention", type=str, default="cv", choices=["cv", "gl"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    # 避免 OpenCV 自己再开线程（线程池里会更稳）
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgmeta_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.endswith("_img+metadata")], key=lambda p: p.name)
    if not imgmeta_dirs:
        raise RuntimeError(f"No *img+metadata folders under: {data_dir}")

    scene_id = get_next_scene_counter(out_dir)
    if scene_id > 0:
        print(f"[INFO] Continue from scene_{scene_id:04d}")

    for i, img_meta_dir in enumerate(imgmeta_dirs, 1):
        seq_name = img_meta_dir.name.replace("_img+metadata", "")
        depth_dir = data_dir / f"{seq_name}_depth"
        print(f"\n========== ({i}/{len(imgmeta_dirs)}) Processing: {seq_name} ==========")
        export_seq_to_one_scene(
            img_meta_dir=img_meta_dir,
            depth_dir=depth_dir,
            out_dir=out_dir,
            scene_id=scene_id,
            max_size=int(args.max_size),
            depth_max=float(args.depth_max),
            convention=str(args.convention),
            workers=int(args.workers),
        )
        scene_id += 1

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
