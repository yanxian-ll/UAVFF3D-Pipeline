import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np


def read_exr_depth(exr_path):
    """
    读取 EXR 深度图，返回 (H, W) float32
    """
    try:
        import cv2
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        depth = cv2.imread(str(exr_path), cv2.IMREAD_UNCHANGED)
        if depth is not None:
            if depth.ndim == 3:
                depth = depth[..., 0]
            return depth.astype(np.float32)
    except Exception:
        pass

    try:
        import imageio.v2 as imageio
        depth = imageio.imread(str(exr_path))
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"读取 EXR 失败: {exr_path}\n原始错误: {e}")


def write_exr_depth(exr_path, depth):
    """
    直接覆盖写回 EXR 深度图
    """
    import cv2

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    exr_path = Path(exr_path)

    depth = np.asarray(depth, dtype=np.float32).copy()
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 0] = 0.0

    ok = cv2.imwrite(str(exr_path), depth)
    if not ok:
        raise RuntimeError(f"写入 EXR 失败: {exr_path}")


def remove_small_valid_components(valid_mask, min_area=16):
    """
    去掉很小的有效连通域
    """
    import cv2

    valid_u8 = valid_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(valid_u8, connectivity=8)

    out = np.zeros_like(valid_u8)
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= int(min_area):
            out[labels == lab] = 1
    return out.astype(bool)


def depth_gradient_sobel(depth):
    """
    计算深度梯度幅值
    """
    import cv2

    depth = np.asarray(depth, dtype=np.float32).copy()
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 0] = 0.0

    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    return grad.astype(np.float32)


def compute_finite_min_plus_offset_threshold(depth, offset=1.0):
    """
    直接使用 finite_min + offset 作为阈值
    仅在正的 finite 深度上统计 finite_min
    """
    depth = np.asarray(depth, dtype=np.float32)

    finite_mask = np.isfinite(depth) & (depth > 1e-6)
    finite_count = int(np.sum(finite_mask))
    if finite_count == 0:
        return None, {
            "finite_count": 0,
            "finite_min": None,
            "threshold": None,
            "offset": float(offset),
        }

    finite_vals = depth[finite_mask]
    finite_min = float(np.min(finite_vals))
    threshold = float(finite_min + float(offset))

    return threshold, {
        "finite_count": finite_count,
        "finite_min": finite_min,
        "threshold": threshold,
        "offset": float(offset),
    }


def apply_finite_min_threshold_cleanup(depth, threshold):
    """
    将所有 finite 且 <= threshold 的深度置 0
    """
    depth = np.asarray(depth, dtype=np.float32).copy()

    if threshold is None:
        removed_mask = np.zeros_like(depth, dtype=bool)
        return depth, removed_mask, 0

    removed_mask = np.isfinite(depth) & (depth <= float(threshold))
    removed_count = int(np.sum(removed_mask))
    depth[removed_mask] = 0.0
    return depth, removed_mask, removed_count


def invalidate_depth_discontinuity_band(
    depth,
    abs_grad_thr=1.0,
    rel_grad_thr=0.02,
    grad_percentile=98,
    dilate_ksize=5,
    dilate_iter=1,
    only_near_invalid=False,
    invalid_band_ksize=7,
    remove_small_cc=True,
    min_cc_area=16,
):
    """
    针对深度变化区域，将边界带直接置 0
    返回：
      depth_clean
      edge_mask
      band_mask
    """
    import cv2

    depth = np.asarray(depth, dtype=np.float32).copy()
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 0] = 0.0

    valid = depth > 0
    if not np.any(valid):
        edge_mask = np.zeros_like(valid, dtype=bool)
        band_mask = np.zeros_like(valid, dtype=bool)
        return depth, edge_mask, band_mask

    grad = depth_gradient_sobel(depth)

    grad = np.asarray(grad, dtype=np.float32)
    valid_grad = grad[np.isfinite(grad)]

    if valid_grad.size == 0:
        return np.inf

    thr = float(np.percentile(valid_grad, float(grad_percentile)))

    # thr = float(abs_grad_thr) + float(rel_grad_thr) * depth
    edge_mask = valid & (grad > thr)

    if only_near_invalid:
        invalid = ~valid
        k_invalid = np.ones((int(invalid_band_ksize), int(invalid_band_ksize)), np.uint8)
        invalid_band = cv2.dilate(invalid.astype(np.uint8), k_invalid, iterations=1) > 0
        edge_mask &= invalid_band

    if dilate_ksize is not None and int(dilate_ksize) > 1:
        k_edge = np.ones((int(dilate_ksize), int(dilate_ksize)), np.uint8)
        band_mask = cv2.dilate(
            edge_mask.astype(np.uint8),
            k_edge,
            iterations=int(dilate_iter),
        ) > 0
    else:
        band_mask = edge_mask.copy()

    depth_clean = depth.copy()
    depth_clean[band_mask] = 0.0

    if remove_small_cc:
        valid2 = depth_clean > 0
        valid2 = remove_small_valid_components(valid2, min_area=min_cc_area)
        depth_clean[~valid2] = 0.0

    depth_clean[~np.isfinite(depth_clean)] = 0.0
    depth_clean[depth_clean <= 0] = 0.0

    return depth_clean.astype(np.float32), edge_mask, band_mask


def get_done_dir(scene_dir, done_dirname):
    return Path(scene_dir) / done_dirname


def get_done_file(exr_path, scene_dir, done_dirname):
    """
    对 scene/depth/xxx.exr 生成 scene/_depth_filter_done/xxx.done
    """
    exr_path = Path(exr_path)
    done_dir = get_done_dir(scene_dir, done_dirname)
    return done_dir / f"{exr_path.stem}.done"


def mark_done(done_file, payload):
    done_file = Path(done_file)
    done_file.parent.mkdir(parents=True, exist_ok=True)
    with open(done_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def is_done(done_file):
    return Path(done_file).exists()


def process_one_depth_file(
    exr_path,
    scene_dir,
    done_dirname="_depth_filter_done",
    apply_finite_min_threshold=True,
    threshold_offset=1.0,
    abs_grad_thr=1.0,
    rel_grad_thr=0.02,
    grad_percentile=98,
    dilate_ksize=5,
    dilate_iter=1,
    only_near_invalid=False,
    invalid_band_ksize=7,
    remove_small_cc=True,
    min_cc_area=16,
):
    """
    处理单张 EXR，并直接覆盖原文件
    若已存在 done 标记，则直接跳过
    """
    exr_path = Path(exr_path)
    scene_dir = Path(scene_dir)
    done_file = get_done_file(exr_path, scene_dir, done_dirname)

    if is_done(done_file):
        return {
            "file": str(exr_path),
            "scene_dir": str(scene_dir),
            "skipped": True,
            "threshold": None,
            "threshold_removed": 0,
            "valid_before": 0,
            "valid_after_threshold": 0,
            "valid_after": 0,
            "total_pixels": 0,
            "edge_pixels": 0,
            "band_pixels": 0,
        }

    depth_raw = read_exr_depth(exr_path)

    n_total = int(depth_raw.size)
    n_valid_before = int(np.sum(np.isfinite(depth_raw) & (depth_raw > 0)))

    threshold = None
    n_thr_removed = 0

    # 1) 先做 finite_min + offset 阈值处理
    if apply_finite_min_threshold:
        threshold, threshold_info = compute_finite_min_plus_offset_threshold(depth_raw, offset=threshold_offset)
        depth_after_thr, _, n_thr_removed = apply_finite_min_threshold_cleanup(depth_raw, threshold)
    else:
        threshold_info = {
            "finite_count": int(np.sum(np.isfinite(depth_raw) & (depth_raw > 1e-6))),
            "finite_min": None,
            "threshold": None,
            "offset": float(threshold_offset),
        }
        depth_after_thr = np.asarray(depth_raw, dtype=np.float32).copy()

    n_valid_after_thr = int(np.sum(np.isfinite(depth_after_thr) & (depth_after_thr > 0)))

    # 2) 再做深度断层边界处理
    depth_clean, edge_mask, band_mask = invalidate_depth_discontinuity_band(
        depth=depth_after_thr,
        abs_grad_thr=abs_grad_thr,
        rel_grad_thr=rel_grad_thr,
        grad_percentile=grad_percentile,
        dilate_ksize=dilate_ksize,
        dilate_iter=dilate_iter,
        only_near_invalid=only_near_invalid,
        invalid_band_ksize=invalid_band_ksize,
        remove_small_cc=remove_small_cc,
        min_cc_area=min_cc_area,
    )

    n_edge = int(np.sum(edge_mask))
    n_band = int(np.sum(band_mask))
    n_valid_after = int(np.sum(depth_clean > 0))

    # 直接覆盖原始 exr
    write_exr_depth(exr_path, depth_clean)

    # 成功后写 done 标记
    mark_done(done_file, {
        "file": str(exr_path),
        "scene_dir": str(scene_dir),
        "threshold": threshold,
        "threshold_info": threshold_info,
        "threshold_removed": int(n_thr_removed),
        "valid_before": n_valid_before,
        "valid_after_threshold": n_valid_after_thr,
        "valid_after": n_valid_after,
        "total_pixels": n_total,
        "edge_pixels": n_edge,
        "band_pixels": n_band,
    })

    return {
        "file": str(exr_path),
        "scene_dir": str(scene_dir),
        "skipped": False,
        "threshold": threshold,
        "threshold_removed": int(n_thr_removed),
        "valid_before": n_valid_before,
        "valid_after_threshold": n_valid_after_thr,
        "valid_after": n_valid_after,
        "total_pixels": n_total,
        "edge_pixels": n_edge,
        "band_pixels": n_band,
    }


def collect_scene_depth_files(root):
    """
    收集所有 scene 及其 depth/*.exr
    约定：
      root/
        scene_xxx/
          depth/*.exr
    返回：
      [(scene_name, scene_dir, [exr1, exr2, ...]), ...]
    """
    root = Path(root)
    scene_dirs = sorted([p for p in root.iterdir() if p.is_dir()], reverse=False)
    if not scene_dirs:
        raise FileNotFoundError(f"未在目录下找到任何场景子文件夹: {root}")

    results = []
    for scene_dir in scene_dirs:
        depth_dir = scene_dir / "depth"
        if not depth_dir.exists() or not depth_dir.is_dir():
            print(f"[WARN] 缺少 depth 目录，跳过: {scene_dir}")
            continue

        exr_files = sorted(depth_dir.glob("*.exr"))
        if not exr_files:
            print(f"[WARN] depth 目录下没有 EXR，跳过: {depth_dir}")
            continue

        results.append((scene_dir.name, scene_dir, exr_files))

    if not results:
        raise FileNotFoundError(f"所有场景中都没有找到可处理的 EXR: {root}")

    return results


def process_one_scene(
    scene_name,
    scene_dir,
    exr_files,
    done_dirname="_depth_filter_done",
    apply_finite_min_threshold=True,
    threshold_offset=1.0,
    abs_grad_thr=1.0,
    rel_grad_thr=0.02,
    grad_percentile=98,
    dilate_ksize=5,
    dilate_iter=1,
    only_near_invalid=False,
    invalid_band_ksize=7,
    remove_small_cc=True,
    min_cc_area=16,
):
    """
    逐 scene 处理所有 EXR
    """
    file_results = []
    ok_count = 0
    err_count = 0
    skip_count = 0

    sum_valid_before = 0
    sum_valid_after_thr = 0
    sum_valid_after = 0
    sum_thr_removed = 0

    # 确保 scene 下的 done 文件夹存在
    get_done_dir(scene_dir, done_dirname).mkdir(parents=True, exist_ok=True)

    for idx, exr_path in enumerate(exr_files, start=1):
        try:
            r = process_one_depth_file(
                exr_path=exr_path,
                scene_dir=scene_dir,
                done_dirname=done_dirname,
                apply_finite_min_threshold=apply_finite_min_threshold,
                threshold_offset=threshold_offset,
                abs_grad_thr=abs_grad_thr,
                rel_grad_thr=rel_grad_thr,
                grad_percentile=grad_percentile,
                dilate_ksize=dilate_ksize,
                dilate_iter=dilate_iter,
                only_near_invalid=only_near_invalid,
                invalid_band_ksize=invalid_band_ksize,
                remove_small_cc=remove_small_cc,
                min_cc_area=min_cc_area,
            )
            file_results.append(r)

            if r["skipped"]:
                skip_count += 1
            else:
                ok_count += 1
                sum_valid_before += r["valid_before"]
                sum_valid_after_thr += r["valid_after_threshold"]
                sum_valid_after += r["valid_after"]
                sum_thr_removed += r["threshold_removed"]
        except Exception as e:
            err_count += 1
            file_results.append({
                "file": str(exr_path),
                "error": str(e),
            })

    return {
        "scene": scene_name,
        "scene_dir": str(scene_dir),
        "num_files": len(exr_files),
        "ok_files": ok_count,
        "skip_files": skip_count,
        "error_files": err_count,
        "sum_valid_before": int(sum_valid_before),
        "sum_valid_after_threshold": int(sum_valid_after_thr),
        "sum_valid_after": int(sum_valid_after),
        "sum_threshold_removed": int(sum_thr_removed),
        "file_results": file_results,
    }


def count_already_done(scene_items, done_dirname):
    """
    启动前统计已有 done 数量
    """
    done_count = 0
    for _, scene_dir, exr_files in scene_items:
        for exr_path in exr_files:
            if is_done(get_done_file(exr_path, scene_dir, done_dirname)):
                done_count += 1
    return done_count


def main():
    parser = argparse.ArgumentParser(
        description="处理所有场景下的所有 depth EXR：先做 finite_min+offset 阈值处理，再做深度边缘处理，直接覆盖原始 EXR；支持中断续跑"
    )
    parser.add_argument("--input", type=str, required=True, help="输入根目录；每个一级子文件夹视作一个场景")
    parser.add_argument("--workers", type=int, default=16, help="并行进程数；0 表示自动")
    parser.add_argument("--done_dirname", type=str, default="_depth_filter_done",
                        help="每个场景下用于记录已处理文件的标记文件夹名")

    parser.add_argument("--no_finite_min_threshold", action="store_true", help="关闭 finite_min + offset 阈值处理")
    parser.add_argument("--threshold_offset", type=float, default=1.0, help="阈值偏移，threshold = finite_min + threshold_offset")

    parser.add_argument("--abs_grad_thr", type=float, default=1.0, help="绝对梯度阈值")
    parser.add_argument("--rel_grad_thr", type=float, default=0.02, help="相对梯度阈值")
    parser.add_argument("--dilate_ksize", type=int, default=3, help="边界带膨胀核大小")
    parser.add_argument("--dilate_iter", type=int, default=1, help="边界带膨胀次数")
    parser.add_argument("--only_near_invalid", action="store_true", help="只处理靠近无效区的深度边缘")
    parser.add_argument("--invalid_band_ksize", type=int, default=0, help="原始无效区边界带核大小")
    parser.add_argument("--keep_small_cc", action="store_true", help="保留小有效连通域")
    parser.add_argument("--min_cc_area", type=int, default=16, help="最小有效连通域面积")
    parser.add_argument("--grad_percentile", default=98)
    args = parser.parse_args()

    input_root = Path(args.input)
    scene_items = collect_scene_depth_files(input_root)

    total_scenes = len(scene_items)
    total_files = sum(len(files) for _, _, files in scene_items)
    already_done_files = count_already_done(scene_items, args.done_dirname)

    if args.workers <= 0:
        workers = min(max(1, mp.cpu_count() // 2), total_scenes)
    else:
        workers = min(args.workers, total_scenes)

    print(f"[START] root={input_root}")
    print(f"[START] scenes={total_scenes}, total_exr={total_files}, workers={workers}")
    print(f"[START] inplace overwrite enabled")
    print(f"[START] resume enabled | done_dirname={args.done_dirname} | already_done={already_done_files}")
    print(
        f"[CONFIG] finite_min_threshold={'OFF' if args.no_finite_min_threshold else 'ON'} | "
        f"threshold_offset={args.threshold_offset} | "
        f"abs_grad_thr={args.abs_grad_thr} | rel_grad_thr={args.rel_grad_thr} | "
        f"dilate_ksize={args.dilate_ksize} | dilate_iter={args.dilate_iter}"
    )

    done_scenes = 0
    done_files = 0
    total_ok_files = 0
    total_skip_files = 0
    total_err_files = 0
    total_sum_valid_before = 0
    total_sum_valid_after_thr = 0
    total_sum_valid_after = 0
    total_sum_thr_removed = 0

    if workers == 1:
        for scene_idx, (scene_name, scene_dir, exr_files) in enumerate(scene_items, start=1):
            print(f"[SCENE {scene_idx}/{total_scenes}] start | scene={scene_name} | files={len(exr_files)}")

            scene_result = process_one_scene(
                scene_name=scene_name,
                scene_dir=scene_dir,
                exr_files=exr_files,
                done_dirname=args.done_dirname,
                apply_finite_min_threshold=not args.no_finite_min_threshold,
                threshold_offset=args.threshold_offset,
                abs_grad_thr=args.abs_grad_thr,
                rel_grad_thr=args.rel_grad_thr,
                dilate_ksize=args.dilate_ksize,
                grad_percentile=args.grad_percentile,
                dilate_iter=args.dilate_iter,
                only_near_invalid=args.only_near_invalid,
                invalid_band_ksize=args.invalid_band_ksize,
                remove_small_cc=not args.keep_small_cc,
                min_cc_area=args.min_cc_area,
            )

            done_scenes += 1
            done_files += scene_result["num_files"]
            total_ok_files += scene_result["ok_files"]
            total_skip_files += scene_result["skip_files"]
            total_err_files += scene_result["error_files"]
            total_sum_valid_before += scene_result["sum_valid_before"]
            total_sum_valid_after_thr += scene_result["sum_valid_after_threshold"]
            total_sum_valid_after += scene_result["sum_valid_after"]
            total_sum_thr_removed += scene_result["sum_threshold_removed"]

            print(
                f"[SCENE {scene_idx}/{total_scenes}] done | scene={scene_name} | "
                f"ok={scene_result['ok_files']}/{scene_result['num_files']} | "
                f"skip={scene_result['skip_files']} | "
                f"err={scene_result['error_files']} | "
                f"thr_removed={scene_result['sum_threshold_removed']} | "
                f"valid_before={scene_result['sum_valid_before']} | "
                f"after_thr={scene_result['sum_valid_after_threshold']} | "
                f"after_band={scene_result['sum_valid_after']}"
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_scene = {}
            for scene_name, scene_dir, exr_files in scene_items:
                fut = executor.submit(
                    process_one_scene,
                    scene_name,
                    scene_dir,
                    exr_files,
                    args.done_dirname,
                    not args.no_finite_min_threshold,
                    args.threshold_offset,
                    args.abs_grad_thr,
                    args.rel_grad_thr,
                    args.grad_percentile,
                    args.dilate_ksize,
                    args.dilate_iter,
                    args.only_near_invalid,
                    args.invalid_band_ksize,
                    not args.keep_small_cc,
                    args.min_cc_area,
                )
                future_to_scene[fut] = (scene_name, len(exr_files))

            for fut in as_completed(future_to_scene):
                scene_name, num_files = future_to_scene[fut]
                done_scenes += 1
                try:
                    scene_result = fut.result()
                    done_files += scene_result["num_files"]
                    total_ok_files += scene_result["ok_files"]
                    total_skip_files += scene_result["skip_files"]
                    total_err_files += scene_result["error_files"]
                    total_sum_valid_before += scene_result["sum_valid_before"]
                    total_sum_valid_after_thr += scene_result["sum_valid_after_threshold"]
                    total_sum_valid_after += scene_result["sum_valid_after"]
                    total_sum_thr_removed += scene_result["sum_threshold_removed"]

                    print(
                        f"[PROGRESS] scenes={done_scenes}/{total_scenes} | files={done_files}/{total_files} | "
                        f"scene={scene_name} | ok={scene_result['ok_files']}/{scene_result['num_files']} | "
                        f"skip={scene_result['skip_files']} | "
                        f"err={scene_result['error_files']} | "
                        f"thr_removed={scene_result['sum_threshold_removed']} | "
                        f"valid_before={scene_result['sum_valid_before']} | "
                        f"after_thr={scene_result['sum_valid_after_threshold']} | "
                        f"after_band={scene_result['sum_valid_after']}"
                    )
                except Exception as e:
                    done_files += num_files
                    total_err_files += num_files
                    print(
                        f"[ERROR] scenes={done_scenes}/{total_scenes} | files={done_files}/{total_files} | "
                        f"scene={scene_name} | files_in_scene={num_files} | error={e}"
                    )

    print("[DONE]")
    print(f"  scenes_done            : {done_scenes}/{total_scenes}")
    print(f"  files_done             : {done_files}/{total_files}")
    print(f"  ok_files               : {total_ok_files}")
    print(f"  skip_files             : {total_skip_files}")
    print(f"  error_files            : {total_err_files}")
    print(f"  total_threshold_removed: {total_sum_thr_removed}")
    print(f"  total_valid_before     : {total_sum_valid_before}")
    print(f"  total_valid_after_thr  : {total_sum_valid_after_thr}")
    print(f"  total_valid_after_band : {total_sum_valid_after}")


if __name__ == "__main__":
    main()


"""
# 只需要处理 Syn-L 和 blendedmvs

python filter_depth.py \
  --input UAVFF3D-Syn-L \
  --workers 16 \
  --done_dirname _depth_filter_done \
  --threshold_offset 1.0 \
  --grad_percentile 98 \
  --dilate_ksize 3 \
  --dilate_iter 1

python filter_depth.py \
  --input blendedmvs \
  --workers 16 \
  --done_dirname _depth_filter_done \
  --threshold_offset 0.0 \
  --grad_percentile 99 \
  --dilate_ksize 3 \
  --dilate_iter 1

"""