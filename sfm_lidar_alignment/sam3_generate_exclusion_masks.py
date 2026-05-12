#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 SAM3 为选定视角生成简单排除掩码。

只做两类：
- vegetation
- water

输出：
- binary_masks/<stem>.png   255=剔除，0=保留
- overlays/<stem>_overlay.png

说明：
- 支持用 --input_long_edge 缩放输入图像，提高预测效率。
- 保存的 binary mask 和 overlay 也是缩放后的分辨率。
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"]

# 只保留最简单的两类
TARGET_CATEGORIES = {
    "vegetation": {
        "prompt": "vegetation",
        "color": (0, 200, 0),
        "id": 1,
    },
    "water": {
        "prompt": "water",
        "color": (0, 0, 200),
        "id": 2,
    },
    "car": {
        "prompt": "car",
        "color": (200, 0, 0),
        "id": 3,
    },
}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_device(dev: str) -> torch.device:
    if dev.startswith("cuda"):
        gpu_id = int(dev.split(":")[1]) if ":" in dev else 0
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device(dev)


def assert_all_on_device(model: torch.nn.Module, device: torch.device):
    bad = []
    for n, p in model.named_parameters(recurse=True):
        if p.device != device:
            bad.append(("param", n, str(p.device)))
    for n, b in model.named_buffers(recurse=True):
        if b.device != device:
            bad.append(("buffer", n, str(b.device)))
    if bad:
        for t in bad[:20]:
            print("[WARN] tensor not on target device:", t)
        raise RuntimeError(f"Model has tensors on wrong device, count={len(bad)}")


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return np.array([])
        if torch.is_tensor(x[0]):
            return torch.stack(list(x), dim=0).detach().cpu().numpy()
        return np.array(x)
    return np.array(x)


def find_image_for_name(images_dir: Path, name: str) -> Optional[Path]:
    p = Path(name)
    if p.suffix:
        cand = images_dir / p.name
        if cand.exists():
            return cand
    stem = p.stem if p.suffix else name.strip()
    for ext in IMG_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    valid_exts = {e.lower() for e in IMG_EXTS}
    for cand in images_dir.glob(f"{stem}.*"):
        if cand.suffix.lower() in valid_exts:
            return cand
    return None


def read_selected_names(txt_path: Path) -> List[str]:
    names = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) == 1:
                names.append(parts[0])
            else:
                names.extend(parts[:2])
    seen = set()
    out = []
    for n in names:
        key = Path(n).stem if Path(n).suffix else n
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def resize_rgb_to_max_long_edge(image_rgb: np.ndarray, max_long_edge: int):
    if max_long_edge is None or max_long_edge <= 0:
        return image_rgb, 1.0
    h, w = image_rgb.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return image_rgb, 1.0
    scale = float(max_long_edge) / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=interp)
    return resized, scale


@torch.inference_mode()
def segment_single_category(processor, pil_img, text_prompt: str, score_thr: float = 0.15, max_instances: int = 512):
    state = processor.set_image(pil_img)
    output = processor.set_text_prompt(state=state, prompt=text_prompt)

    masks = to_numpy(output.get("masks", None))
    scores = to_numpy(output.get("scores", None))

    if masks is None or scores is None:
        return None

    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if len(scores) == 0:
        return None

    if masks.ndim == 4:
        if masks.shape[1] == 1:
            masks = masks[:, 0, :, :]
        else:
            masks = masks.max(axis=1)
    elif masks.ndim == 3:
        pass
    elif masks.ndim == 2:
        masks = masks[None, ...]
    else:
        return None

    keep = np.where(scores >= score_thr)[0]
    if keep.size == 0:
        return None
    keep = keep[np.argsort(scores[keep])[::-1]]
    keep = keep[: min(max_instances, keep.size)]

    H, W = pil_img.size[1], pil_img.size[0]
    binary_mask = np.zeros((H, W), dtype=np.uint8)
    for idx in keep:
        m = masks[idx]
        m_bool = m > 0.5 if m.dtype != np.bool_ else m
        if np.any(m_bool):
            binary_mask[m_bool] = 1

    if np.sum(binary_mask) == 0:
        return None
    return binary_mask


@torch.inference_mode()
def segment_binary_mask(processor, pil_img, score_thr: float, max_instances_per_cat: int) -> np.ndarray:
    H, W = pil_img.size[1], pil_img.size[0]
    semantic_mask = np.zeros((H, W), dtype=np.uint8)

    for cat_key, cat_info in TARGET_CATEGORIES.items():
        binary = segment_single_category(
            processor=processor,
            pil_img=pil_img,
            text_prompt=cat_info["prompt"],
            score_thr=score_thr,
            max_instances=max_instances_per_cat,
        )
        if binary is not None:
            semantic_mask[binary > 0] = cat_info["id"]

    exclude_mask = (semantic_mask > 0).astype(np.uint8) * 255
    return exclude_mask


def make_overlay(image_rgb: np.ndarray, exclude_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image_rgb.copy().astype(np.float32)
    red = np.zeros_like(image_rgb, dtype=np.uint8)
    red[..., 0] = 255
    m = exclude_mask > 0
    if np.any(m):
        overlay[m] = cv2.addWeighted(overlay[m], 1.0, red[m].astype(np.float32), alpha, 0)
    return np.clip(overlay, 0, 255).astype(np.uint8)


# 优先兼容用户当前仓库结构

def build_processor(device: torch.device, checkpoint_path: str):
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    build_kwargs = {
        "bpe_path": "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        "checkpoint_path": checkpoint_path,
        "load_from_HF": False,
        "device": str(device),
    }
    model = build_sam3_image_model(**build_kwargs)
    model.eval()
    model.to(device)
    assert_all_on_device(model, device)
    processor = Sam3Processor(model)
    assert_all_on_device(model, device)
    return processor


def process_one_image(
    img_path: Path,
    out_dir: Path,
    processor,
    score_thr: float,
    max_instances_per_cat: int,
    input_long_edge: int,
    overwrite: bool = False,
):
    stem = img_path.stem
    binary_dir = ensure_dir(out_dir / "binary_masks")
    overlay_dir = ensure_dir(out_dir / "overlays")

    binary_path = binary_dir / f"{stem}.png"
    overlay_path = overlay_dir / f"{stem}_overlay.png"

    if (not overwrite) and binary_path.exists() and overlay_path.exists():
        return False

    image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"[WARN] failed to read image: {img_path}")
        return False
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb, _ = resize_rgb_to_max_long_edge(image_rgb, input_long_edge)

    pil_img = Image.fromarray(image_rgb)
    binary_mask = segment_binary_mask(
        processor=processor,
        pil_img=pil_img,
        score_thr=score_thr,
        max_instances_per_cat=max_instances_per_cat,
    )
    overlay = make_overlay(image_rgb, binary_mask)

    cv2.imwrite(str(binary_path), binary_mask)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return True


def main():
    parser = argparse.ArgumentParser(description="Use SAM3 to generate simple binary masks for selected views")
    parser.add_argument("--images_dir", type=str, required=True, help="images 路径")
    parser.add_argument("--selected_views_txt", type=str, required=True, help="一行一个名字；也兼容 selected_pairs.txt 前两列")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")

    parser.add_argument("--score_thr", type=float, default=0.15)
    parser.add_argument("--max_instances_per_cat", type=int, default=512)
    parser.add_argument("--input_long_edge", type=int, default=0, help="预测前缩放输入图像的最大边；<=0 表示不缩放")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_path", type=str, default="sam3/checkpoints/sam3.pt")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    selected_views_txt = Path(args.selected_views_txt)
    out_dir = ensure_dir(Path(args.out_dir))

    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not selected_views_txt.exists():
        raise FileNotFoundError(f"selected_views_txt not found: {selected_views_txt}")

    device = setup_device(args.device)
    processor = build_processor(device=device, checkpoint_path=args.checkpoint_path.strip())

    names = read_selected_names(selected_views_txt)
    image_paths = []
    missing = []
    for name in names:
        p = find_image_for_name(images_dir, name)
        if p is None:
            missing.append(name)
        else:
            image_paths.append(p)

    if len(image_paths) == 0:
        raise RuntimeError("No valid selected images found.")

    processed = 0
    for img_path in tqdm(image_paths, desc="Generating binary masks"):
        ok = process_one_image(
            img_path=img_path,
            out_dir=out_dir,
            processor=processor,
            score_thr=args.score_thr,
            max_instances_per_cat=args.max_instances_per_cat,
            input_long_edge=args.input_long_edge,
            overwrite=args.overwrite,
        )
        if ok:
            processed += 1

    print(f"[INFO] requested: {len(names)}")
    print(f"[INFO] found images: {len(image_paths)}")
    print(f"[INFO] processed: {processed}")
    print(f"[INFO] missing: {len(missing)}")
    if len(missing) > 0:
        print("[INFO] missing names example:", missing[:10])
    print(f"[INFO] outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()



"""
python sam3_generate_exclusion_masks.py \
    --images_dir recon/nanfang_render/images \
    --selected_views_txt lidar/nanfang/selected_views.txt \
    --out_dir ./eval_mask/nanfang_eval_masks \
    --device cuda:0  --input_long_edge 512 \
    --checkpoint_path sam3/checkpoints/sam3.pt 

python sam3_generate_exclusion_masks.py \
    --images_dir recon/yanghaitang_render/images \
    --selected_views_txt lidar/yanghaitang/selected_views.txt \
    --out_dir ./eval_mask/yanghaitang_eval_masks \
    --device cuda:0  --input_long_edge 512 \
    --checkpoint_path sam3/checkpoints/sam3.pt 

python sam3_generate_exclusion_masks.py \
    --images_dir recon/xiaoxiang_render/images \
    --selected_views_txt lidar/xiaoxiang/selected_views.txt \
    --out_dir ./eval_mask/xiaoxiang_eval_masks \
    --device cuda:0  --input_long_edge 512 \
    --checkpoint_path sam3/checkpoints/sam3.pt 
"""
