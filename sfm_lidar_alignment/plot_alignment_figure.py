# -*- coding: utf-8 -*-
import os
os.environ["MPLBACKEND"] = "Agg"   # 强制非交互后端，避免 TkAgg

import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")        
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.linewidth": 0.8,
})


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_depth_npy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find depth npy: {path}")
    depth = np.load(path)
    if depth.ndim != 2:
        raise ValueError(f"Depth must be HxW, got shape={depth.shape} from {path}")
    return depth.astype(np.float32)


def copy_original_inputs_to_output(args, out_dir: Path):
    """
    Copy the raw input RGB/depth files to the output directory for record keeping.
    """
    rgb_dir = out_dir / "inputs" / "rgbs"
    depth_dir = out_dir / "inputs" / "depths"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    copy_pairs = [
        (args.before_rgb,   rgb_dir / f"before_{Path(args.before_rgb).name}"),
        (args.after_rgb,    rgb_dir / f"after_{Path(args.after_rgb).name}"),
        (args.real_rgb,     rgb_dir / f"real_{Path(args.real_rgb).name}"),
        (args.before_depth, depth_dir / f"before_{Path(args.before_depth).name}"),
        (args.after_depth,  depth_dir / f"after_{Path(args.after_depth).name}"),
        (args.real_depth,   depth_dir / f"real_{Path(args.real_depth).name}"),
    ]

    for src, dst in copy_pairs:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Input file not found: {src_path}")
        shutil.copy2(str(src_path), str(dst))


def resize_to_match(img: np.ndarray, h: int, w: int, is_depth=False) -> np.ndarray:
    if img.shape[:2] == (h, w):
        return img
    interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def make_overlay(real_rgb: np.ndarray, lidar_rgb: np.ndarray,
                 alpha_real=0.55, alpha_lidar=0.45) -> np.ndarray:
    real_f = real_rgb.astype(np.float32)
    lidar_f = lidar_rgb.astype(np.float32)
    out = alpha_real * real_f + alpha_lidar * lidar_f
    return np.clip(out, 0, 255).astype(np.uint8)


def make_false_color_overlay(real_rgb: np.ndarray, lidar_rgb: np.ndarray) -> np.ndarray:
    h, w = real_rgb.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    real_gray = cv2.cvtColor(real_rgb, cv2.COLOR_RGB2GRAY)
    lidar_gray = cv2.cvtColor(lidar_rgb, cv2.COLOR_RGB2GRAY)

    canvas[..., 1] = real_gray
    canvas[..., 0] = lidar_gray
    canvas[..., 2] = lidar_gray

    return canvas


def get_line_points(h, w, mode, line_y=None, x0=None, y0=None, x1=None, y1=None):
    if mode == "horizontal":
        if line_y is None:
            line_y = h // 2
        line_y = int(np.clip(line_y, 0, h - 1))
        pts = [(x, line_y) for x in range(w)]
        return pts, (0, line_y, w - 1, line_y)

    elif mode == "custom":
        if None in [x0, y0, x1, y1]:
            raise ValueError("For custom line mode, x0,y0,x1,y1 must all be provided.")

        x0 = int(np.clip(x0, 0, w - 1))
        x1 = int(np.clip(x1, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        y1 = int(np.clip(y1, 0, h - 1))

        num = int(np.hypot(x1 - x0, y1 - y0)) + 1
        xs = np.linspace(x0, x1, num).astype(np.int32)
        ys = np.linspace(y0, y1, num).astype(np.int32)
        pts = list(zip(xs, ys))
        return pts, (x0, y0, x1, y1)

    else:
        raise ValueError(f"Unsupported line mode: {mode}")


def sample_depth_along_line(depth: np.ndarray, pts):
    vals = []
    for x, y in pts:
        v = depth[y, x]
        if np.isfinite(v) and v > 1e-8:
            vals.append(v)
        else:
            vals.append(np.nan)
    return np.array(vals, dtype=np.float32)


def draw_profile_line_on_axis(ax, line_xyxy, color="yellow", linewidth=2.2):
    x0, y0, x1, y1 = line_xyxy
    ax.plot(
        [x0, x1],
        [y0, y1],
        linestyle=(0, (6, 4)),
        color=color,
        linewidth=linewidth,
        solid_capstyle="round",
        dash_capstyle="round",
    )


def canny_edge(img_rgb: np.ndarray):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 80, 160)
    return edge


def edge_iou(a: np.ndarray, b: np.ndarray, dilate_px=2):
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)

    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = np.ones((k, k), np.uint8)
        a = cv2.dilate(a, kernel, iterations=1)
        b = cv2.dilate(b, kernel, iterations=1)

    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    if union == 0:
        return 0.0
    return inter / union


def add_panel_label(ax, text):
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="black", alpha=0.55, linewidth=0),
    )


def save_single_image_panel(img, line_xyxy, out_path, dpi=220):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.imshow(img)
    draw_profile_line_on_axis(ax, line_xyxy, color="yellow", linewidth=2.4)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_single_profile_panel(x_axis, prof_before, prof_after, prof_real, out_path, H, W, dpi=220):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.set_box_aspect(H / W)

    ax.plot(x_axis, prof_before, label="Before", linewidth=2.0)
    ax.plot(x_axis, prof_after, label="After", linewidth=2.0)
    ax.plot(x_axis, prof_real, label="Reference", linewidth=2.0)

    ax.set_xlabel("Profile position")
    ax.set_ylabel("Depth")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")

    finite_all = np.concatenate([
        prof_before[np.isfinite(prof_before)],
        prof_after[np.isfinite(prof_after)],
        prof_real[np.isfinite(prof_real)],
    ])
    if finite_all.size > 0:
        ymin = np.percentile(finite_all, 1)
        ymax = np.percentile(finite_all, 99)
        if ymax > ymin:
            pad = 0.05 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout(pad=0.3)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-rgb", type=str, required=True)
    parser.add_argument("--after-rgb", type=str, required=True)
    parser.add_argument("--real-rgb", type=str, required=True)

    parser.add_argument("--before-depth", type=str, required=True)
    parser.add_argument("--after-depth", type=str, required=True)
    parser.add_argument("--real-depth", type=str, required=True)

    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--line-mode", type=str, default="horizontal", choices=["horizontal", "custom"])
    parser.add_argument("--line-y", type=int, default=None)

    parser.add_argument("--x0", type=int, default=None)
    parser.add_argument("--y0", type=int, default=None)
    parser.add_argument("--x1", type=int, default=None)
    parser.add_argument("--y1", type=int, default=None)

    parser.add_argument("--use-false-color", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    before_rgb = read_rgb(args.before_rgb)
    after_rgb = read_rgb(args.after_rgb)
    real_rgb = read_rgb(args.real_rgb)

    before_depth = read_depth_npy(args.before_depth)
    after_depth = read_depth_npy(args.after_depth)
    real_depth = read_depth_npy(args.real_depth)

    H, W = real_rgb.shape[:2]
    before_rgb = resize_to_match(before_rgb, H, W, is_depth=False)
    after_rgb = resize_to_match(after_rgb, H, W, is_depth=False)

    before_depth = resize_to_match(before_depth, H, W, is_depth=True)
    after_depth = resize_to_match(after_depth, H, W, is_depth=True)
    real_depth = resize_to_match(real_depth, H, W, is_depth=True)

    pts, line_xyxy = get_line_points(
        H, W,
        mode=args.line_mode,
        line_y=args.line_y,
        x0=args.x0, y0=args.y0, x1=args.x1, y1=args.y1
    )

    if args.use_false_color:
        before_overlay = make_false_color_overlay(real_rgb, before_rgb)
        after_overlay = make_false_color_overlay(real_rgb, after_rgb)
    else:
        before_overlay = make_overlay(real_rgb, before_rgb)
        after_overlay = make_overlay(real_rgb, after_rgb)

    edge_real = canny_edge(real_rgb)
    edge_before = canny_edge(before_rgb)
    edge_after = canny_edge(after_rgb)
    iou_before = edge_iou(edge_real, edge_before)
    iou_after = edge_iou(edge_real, edge_after)

    prof_before = sample_depth_along_line(before_depth, pts)
    prof_after = sample_depth_along_line(after_depth, pts)
    prof_real = sample_depth_along_line(real_depth, pts)
    x_axis = np.arange(len(pts))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    copy_original_inputs_to_output(args, out_path.parent)

    stem = out_path.stem
    suffix = out_path.suffix if out_path.suffix else ".png"

    before_single = out_path.parent / f"{stem}_panel_before{suffix}"
    after_single = out_path.parent / f"{stem}_panel_after{suffix}"
    profile_single = out_path.parent / f"{stem}_panel_profile{suffix}"

    save_single_image_panel(before_overlay, line_xyxy, before_single, dpi=args.dpi)
    save_single_image_panel(after_overlay, line_xyxy, after_single, dpi=args.dpi)
    save_single_profile_panel(
        x_axis, prof_before, prof_after, prof_real,
        profile_single, H, W, dpi=args.dpi
    )

    fig = plt.figure(figsize=(16.5, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax1.imshow(before_overlay)
    draw_profile_line_on_axis(ax1, line_xyxy, color="yellow", linewidth=2.4)
    add_panel_label(ax1, "(a) Before")
    ax1.text(
        0.02, 0.05, f"Edge IoU: {iou_before:.3f}",
        transform=ax1.transAxes, ha="left", va="bottom",
        fontsize=9, color="white",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.45, linewidth=0),
    )
    ax1.axis("off")

    ax2.imshow(after_overlay)
    draw_profile_line_on_axis(ax2, line_xyxy, color="yellow", linewidth=2.4)
    add_panel_label(ax2, "(b) After")
    ax2.text(
        0.02, 0.05, f"Edge IoU: {iou_after:.3f}",
        transform=ax2.transAxes, ha="left", va="bottom",
        fontsize=9, color="white",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.45, linewidth=0),
    )
    ax2.axis("off")

    ax3.set_box_aspect(H / W)
    ax3.plot(x_axis, prof_before, label="Before", linewidth=2.0)
    ax3.plot(x_axis, prof_after, label="After", linewidth=2.0)
    ax3.plot(x_axis, prof_real, label="Reference", linewidth=2.0)

    finite_all = np.concatenate([
        prof_before[np.isfinite(prof_before)],
        prof_after[np.isfinite(prof_after)],
        prof_real[np.isfinite(prof_real)],
    ])
    if finite_all.size > 0:
        ymin = np.percentile(finite_all, 1)
        ymax = np.percentile(finite_all, 99)
        if ymax > ymin:
            pad = 0.05 * (ymax - ymin)
            ax3.set_ylim(ymin - pad, ymax + pad)

    add_panel_label(ax3, "(c) Depth profile")
    ax3.set_xlabel("Profile position")
    ax3.set_ylabel("Depth")
    ax3.grid(True, alpha=0.25, linewidth=0.6)
    # ax3.legend(frameon=False, loc="upper right")
    ax3.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.03, 1.0))

    for spine in ["top", "right"]:
        ax3.spines[spine].set_visible(False)

    fig.savefig(str(out_path), dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved combined figure to: {out_path}")
    print(f"Saved single panel to: {before_single}")
    print(f"Saved single panel to: {after_single}")
    print(f"Saved single panel to: {profile_single}")
    print(f"Copied original inputs to: {out_path.parent / 'inputs'}")


if __name__ == "__main__":
    main()


"""

python plot_alignment_figure.py \
    --before-rgb lidar/nanfang/images/a3cf12d127044b038f231ccdc5ece529ba9af165.jpg \
    --after-rgb lidar/nanfang_final/images/a3cf12d127044b038f231ccdc5ece529ba9af165.jpg \
    --real-rgb recon/nanfang_render/images/a3cf12d127044b038f231ccdc5ece529ba9af165.png \
    --before-depth lidar/nanfang/depth_npy/a3cf12d127044b038f231ccdc5ece529ba9af165.npy \
    --after-depth lidar/nanfang_final/depth_npy/a3cf12d127044b038f231ccdc5ece529ba9af165.npy \
    --real-depth recon/nanfang_render/depth_npy/a3cf12d127044b038f231ccdc5ece529ba9af165.npy \
    --out ./vis/nanfang/alignment_vis.png \
    --line-mode horizontal \
    --line-y 700



python plot_alignment_figure.py \
    --before-rgb lidar/xiaoxiang/images/1e4231ddb0484349c51820eff1dfb82275b04a0b.jpg \
    --after-rgb lidar/xiaoxiang_final/images/1e4231ddb0484349c51820eff1dfb82275b04a0b.jpg \
    --real-rgb recon/xiaoxiang_render/images/1e4231ddb0484349c51820eff1dfb82275b04a0b.png \
    --before-depth lidar/xiaoxiang/depth_npy/1e4231ddb0484349c51820eff1dfb82275b04a0b.npy \
    --after-depth lidar/xiaoxiang_final/depth_npy/1e4231ddb0484349c51820eff1dfb82275b04a0b.npy \
    --real-depth recon/xiaoxiang_render/depth_npy/1e4231ddb0484349c51820eff1dfb82275b04a0b.npy \
    --out ./vis/xiaoxiang/alignment_vis.png \
    --line-mode horizontal \
    --line-y 300


python plot_alignment_figure.py \
    --before-rgb lidar/yanghaitang/images/5f10b891dad1872196860f7646d73146da747bd0.jpg \
    --after-rgb lidar/yanghaitang_final/images/5f10b891dad1872196860f7646d73146da747bd0.jpg \
    --real-rgb recon/yanghaitang_render/images/5f10b891dad1872196860f7646d73146da747bd0.png \
    --before-depth lidar/yanghaitang/depth_npy/5f10b891dad1872196860f7646d73146da747bd0.npy \
    --after-depth lidar/yanghaitang_final/depth_npy/5f10b891dad1872196860f7646d73146da747bd0.npy \
    --real-depth recon/yanghaitang_render/depth_npy/5f10b891dad1872196860f7646d73146da747bd0.npy \
    --out ./vis/yanghaitang/alignment_vis.png \
    --line-mode horizontal \
    --line-y 500


"""