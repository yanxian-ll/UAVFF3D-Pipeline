import argparse
from pathlib import Path
import numpy as np
import laspy


def has_rgb(point_format) -> bool:
    dims = set(point_format.dimension_names)
    return {"red", "green", "blue"}.issubset(dims)


def normalize_rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    """
    将 LAS 中的 RGB 转成 PLY 常用的 uint8 [0,255]
    LAS 中颜色常见为 uint16 [0,65535]
    """
    if rgb.size == 0:
        return rgb.astype(np.uint8)

    if rgb.dtype == np.uint16 or rgb.max() > 255:
        rgb = np.clip(rgb / 256.0, 0, 255).astype(np.uint8)
    else:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def write_binary_ply(ply_path: str, xyz: np.ndarray, rgb: np.ndarray = None):
    """
    写出 binary_little_endian 的 PLY
    xyz: (N, 3), float32/float64
    rgb: (N, 3), uint8 or None
    """
    xyz = np.asarray(xyz, dtype=np.float32)

    if rgb is not None:
        rgb = np.asarray(rgb, dtype=np.uint8)
        if len(xyz) != len(rgb):
            raise ValueError("xyz 和 rgb 点数不一致。")

        vertex_data = np.empty(
            len(xyz),
            dtype=[
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        vertex_data["x"] = xyz[:, 0]
        vertex_data["y"] = xyz[:, 1]
        vertex_data["z"] = xyz[:, 2]
        vertex_data["red"] = rgb[:, 0]
        vertex_data["green"] = rgb[:, 1]
        vertex_data["blue"] = rgb[:, 2]

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(vertex_data)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
    else:
        vertex_data = np.empty(
            len(xyz),
            dtype=[
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
            ],
        )
        vertex_data["x"] = xyz[:, 0]
        vertex_data["y"] = xyz[:, 1]
        vertex_data["z"] = xyz[:, 2]

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(vertex_data)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        )

    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))
        vertex_data.tofile(f)


def downsample_las_to_ply(
    input_path: str,
    output_path: str,
    voxel_size: float,
    chunk_size: int = 2_000_000,
    keep_color: bool = True,
):
    """
    分块读取 LAS/LAZ，按体素降采样后保存为 PLY。
    当前策略：每个体素保留第一次遇到的点。
    """
    input_path = str(input_path)
    output_path = str(output_path)

    if voxel_size <= 0:
        raise ValueError("voxel_size 必须大于 0。")

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    kept_xyz = []
    kept_rgb = [] if keep_color else None
    seen_voxels = set()

    voxel_dtype = np.dtype([("ix", "<i8"), ("iy", "<i8"), ("iz", "<i8")])

    with laspy.open(input_path) as reader:
        total_points = reader.header.point_count
        file_has_rgb = has_rgb(reader.header.point_format)
        use_rgb = keep_color and file_has_rgb

        print(f"[INFO] 输入文件: {input_path}")
        print(f"[INFO] 总点数: {total_points}")
        print(f"[INFO] 是否包含 RGB: {file_has_rgb}")
        print(f"[INFO] 是否输出 RGB: {use_rgb}")
        print(f"[INFO] 体素大小: {voxel_size}")
        print(f"[INFO] 分块大小: {chunk_size}")

        processed = 0
        chunk_id = 0

        for points in reader.chunk_iterator(chunk_size):
            chunk_id += 1

            x = np.asarray(points.x)
            y = np.asarray(points.y)
            z = np.asarray(points.z)

            if x.size == 0:
                continue

            xyz = np.column_stack((x, y, z)).astype(np.float64, copy=False)

            # 体素索引（注意单位与坐标单位一致，很多 LAS 是米）
            voxel_idx = np.floor(xyz / voxel_size).astype(np.int64)

            # 先在当前块内去重，减少后续 Python 循环压力
            voxel_idx_view = np.ascontiguousarray(voxel_idx).view(voxel_dtype).reshape(-1)
            _, first_indices = np.unique(voxel_idx_view, return_index=True)
            first_indices.sort()

            local_xyz = xyz[first_indices]
            local_voxels = voxel_idx[first_indices]

            if use_rgb:
                rgb = np.column_stack((
                    np.asarray(points.red),
                    np.asarray(points.green),
                    np.asarray(points.blue),
                ))
                local_rgb = normalize_rgb_to_uint8(rgb[first_indices])
            else:
                local_rgb = None

            new_count = 0

            # 全局去重：跨 chunk 的重复体素只保留一个点
            for i in range(local_voxels.shape[0]):
                key = tuple(local_voxels[i])
                if key in seen_voxels:
                    continue

                seen_voxels.add(key)
                kept_xyz.append(local_xyz[i].astype(np.float32))
                if use_rgb:
                    kept_rgb.append(local_rgb[i])
                new_count += 1

            processed += x.size
            ratio = processed / total_points * 100 if total_points > 0 else 100.0
            print(
                f"[INFO] chunk={chunk_id:04d} | "
                f"已处理={processed}/{total_points} ({ratio:.2f}%) | "
                f"当前保留点数={len(kept_xyz)} | "
                f"本块新增={new_count}"
            )

    if len(kept_xyz) == 0:
        raise RuntimeError("降采样后没有点，请检查输入文件或 voxel_size。")

    kept_xyz = np.asarray(kept_xyz, dtype=np.float32)
    if kept_rgb is not None and len(kept_rgb) > 0:
        kept_rgb = np.asarray(kept_rgb, dtype=np.uint8)
    else:
        kept_rgb = None

    print(f"[INFO] 开始写出 PLY: {output_path}")
    write_binary_ply(output_path, kept_xyz, kept_rgb)

    print("[INFO] 完成")
    print(f"[INFO] 输出点数: {len(kept_xyz)}")
    print(f"[INFO] 输出文件: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="对超大 LAS/LAZ 文件进行分块体素降采样，并保存为 PLY。"
    )
    parser.add_argument("input", type=str, help="输入 LAS/LAZ 文件路径")
    parser.add_argument("output", type=str, help="输出 PLY 文件路径")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.3,
        help="体素大小，单位与 LAS 坐标单位一致（通常为米）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000_000,
        help="每次读取的点数，默认 2000000",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="不保留 RGB",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    downsample_las_to_ply(
        input_path=args.input,
        output_path=args.output,
        voxel_size=args.voxel_size,
        chunk_size=args.chunk_size,
        keep_color=not args.no_color,
    )


"""

python downsample_las_to_ply.py nanfang/cloud_merged.las nanfang/downsample_lidar.ply
python downsample_las_to_ply.py yanghaitang/cloud_merged.las yanghaitang/downsample_lidar.ply
python downsample_las_to_ply.py xiaoxiang/cloud_merged.las xiaoxiang/downsample_lidar.ply

"""

