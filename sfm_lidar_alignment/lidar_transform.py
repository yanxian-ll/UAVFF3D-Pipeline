"""Apply the LiDAR-to-SfM transform chain to LAS/LAZ/PLY point clouds.

Transform files are composed in this order when present:

    transform_manual.txt -> transform_icp.txt -> transform_refine.txt

If the point cloud is first transformed by T1, then T2, then T3, the composed
matrix is ``T = T3 @ T2 @ T1``. The output is always a binary PLY file.
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


TRANSFORM_CHAIN = [
    ("transform_manual.txt", "manual coarse alignment"),
    ("transform_icp.txt", "coarse ICP alignment"),
    ("transform_refine.txt", "view/depth refinement"),
]


def read_transformation_matrix(file_path):
    """Read a 4x4 transform matrix, returning None when the file is absent."""
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    matrix = np.array([list(map(float, line.split())) for line in lines], dtype=np.float64)
    if matrix.shape[0] > 4:
        matrix = matrix[:4, :]
    if matrix.shape != (4, 4):
        raise ValueError(f"Transform must be 4x4: {file_path}, got {matrix.shape}")
    return matrix


def compose_transformations(transform_dir):
    """Compose all available transforms in the documented chain order."""
    transform_dir = Path(transform_dir)
    composed = np.eye(4, dtype=np.float64)

    for filename, label in TRANSFORM_CHAIN:
        matrix = read_transformation_matrix(transform_dir / filename)
        if matrix is None:
            continue
        print(f"apply {label}: {filename}")
        composed = matrix @ composed

    return composed


def has_las_rgb(header):
    dim_names = set(header.point_format.dimension_names)
    return {"red", "green", "blue"}.issubset(dim_names)


def convert_las_rgb_to_u8(r, g, b):
    """Normalize LAS RGB channels, which are commonly uint16, to uint8."""
    r = np.asarray(r)
    g = np.asarray(g)
    b = np.asarray(b)

    if r.dtype == np.uint8 and g.dtype == np.uint8 and b.dtype == np.uint8:
        return r, g, b

    max_val = max(
        np.max(r) if r.size > 0 else 0,
        np.max(g) if g.size > 0 else 0,
        np.max(b) if b.size > 0 else 0,
    )
    if max_val <= 255:
        return r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)

    return (
        np.clip(r / 256.0, 0, 255).astype(np.uint8),
        np.clip(g / 256.0, 0, 255).astype(np.uint8),
        np.clip(b / 256.0, 0, 255).astype(np.uint8),
    )


def convert_o3d_rgb_to_u8(colors):
    """Normalize Open3D color arrays to uint8 RGB."""
    colors = np.asarray(colors)
    if colors.size == 0:
        return None
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"PLY colors must have shape Nx3, got {colors.shape}")

    if np.issubdtype(colors.dtype, np.integer):
        max_val = int(colors.max()) if colors.size > 0 else 0
        scale = 1.0 if max_val <= 255 else 1.0 / 256.0
        return np.clip(colors * scale, 0, 255).astype(np.uint8)

    max_val = float(colors.max()) if colors.size > 0 else 0.0
    if max_val <= 1.0 + 1e-6:
        return np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    if max_val <= 255.0 + 1e-6:
        return np.clip(colors, 0, 255).astype(np.uint8)
    return np.clip(colors / 256.0, 0, 255).astype(np.uint8)


def write_binary_ply_header(f, num_points, with_color=True):
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if with_color:
        header.extend(
            [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        )
    header.append("end_header\n")
    f.write("\n".join(header).encode("ascii"))


def transform_xyz_chunk(x, y, z, transform):
    xyz = np.stack([x, y, z], axis=1).astype(np.float64)
    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    xyz_h = np.concatenate([xyz, ones], axis=1)
    xyz_t = xyz_h @ transform.T
    return (
        xyz_t[:, 0].astype(np.float32),
        xyz_t[:, 1].astype(np.float32),
        xyz_t[:, 2].astype(np.float32),
    )


def write_points_to_binary_ply(out_ply_path, x, y, z, rgb=None, default_color=(180, 180, 180)):
    out_ply_path = Path(out_ply_path)
    out_ply_path.parent.mkdir(parents=True, exist_ok=True)

    num_points = len(x)
    with open(out_ply_path, "wb") as f:
        write_binary_ply_header(f, num_points, with_color=True)

        if rgb is None:
            r = np.full((num_points,), default_color[0], dtype=np.uint8)
            g = np.full((num_points,), default_color[1], dtype=np.uint8)
            b = np.full((num_points,), default_color[2], dtype=np.uint8)
        else:
            rgb = np.asarray(rgb)
            if rgb.shape != (num_points, 3):
                raise ValueError(f"rgb must have shape ({num_points}, 3), got {rgb.shape}")
            r = rgb[:, 0].astype(np.uint8)
            g = rgb[:, 1].astype(np.uint8)
            b = rgb[:, 2].astype(np.uint8)

        ply_dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        arr = np.empty(num_points, dtype=ply_dtype)
        arr["x"] = np.asarray(x, dtype=np.float32)
        arr["y"] = np.asarray(y, dtype=np.float32)
        arr["z"] = np.asarray(z, dtype=np.float32)
        arr["red"] = r
        arr["green"] = g
        arr["blue"] = b
        arr.tofile(f)


def save_las_to_ply_chunked(
    las_path,
    out_ply_path,
    transform,
    chunk_size=2_000_000,
    default_color=(180, 180, 180),
):
    import laspy

    las_path = Path(las_path)
    out_ply_path = Path(out_ply_path)
    out_ply_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"reading LAS/LAZ header: {las_path}")
    with laspy.open(str(las_path)) as reader:
        total_points = reader.header.point_count
        use_rgb = has_las_rgb(reader.header)

        print(f"total points: {total_points}")
        print(f"has rgb: {use_rgb}")
        print(f"writing PLY: {out_ply_path}")

        with open(out_ply_path, "wb") as f:
            write_binary_ply_header(f, total_points, with_color=True)
            processed = 0

            for chunk_id, chunk in enumerate(reader.chunk_iterator(chunk_size), start=1):
                x = np.asarray(chunk.x, dtype=np.float64)
                y = np.asarray(chunk.y, dtype=np.float64)
                z = np.asarray(chunk.z, dtype=np.float64)
                xt, yt, zt = transform_xyz_chunk(x, y, z, transform)
                num_chunk_points = xt.shape[0]

                if use_rgb:
                    r, g, b = convert_las_rgb_to_u8(chunk.red, chunk.green, chunk.blue)
                else:
                    r = np.full((num_chunk_points,), default_color[0], dtype=np.uint8)
                    g = np.full((num_chunk_points,), default_color[1], dtype=np.uint8)
                    b = np.full((num_chunk_points,), default_color[2], dtype=np.uint8)

                ply_dtype = np.dtype(
                    [
                        ("x", "<f4"),
                        ("y", "<f4"),
                        ("z", "<f4"),
                        ("red", "u1"),
                        ("green", "u1"),
                        ("blue", "u1"),
                    ]
                )
                arr = np.empty(num_chunk_points, dtype=ply_dtype)
                arr["x"] = xt
                arr["y"] = yt
                arr["z"] = zt
                arr["red"] = r
                arr["green"] = g
                arr["blue"] = b
                arr.tofile(f)

                processed += num_chunk_points
                print(f"[chunk {chunk_id}] processed {processed}/{total_points}")

    print(f"transformed point cloud saved to: {out_ply_path}")


def save_ply_to_ply(ply_path, out_ply_path, transform, default_color=(180, 180, 180)):
    if o3d is None:
        raise RuntimeError("Reading PLY input requires open3d, but it is not installed.")

    ply_path = Path(ply_path)
    print(f"reading PLY: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd is None or pcd.is_empty():
        raise RuntimeError(f"Failed to read a non-empty point cloud: {ply_path}")

    points = np.asarray(pcd.points, dtype=np.float64)
    print(f"total points: {points.shape[0]}")
    print(f"has rgb: {pcd.has_colors()}")

    xt, yt, zt = transform_xyz_chunk(points[:, 0], points[:, 1], points[:, 2], transform)
    rgb = convert_o3d_rgb_to_u8(np.asarray(pcd.colors)) if pcd.has_colors() else None

    write_points_to_binary_ply(
        out_ply_path=out_ply_path,
        x=xt,
        y=yt,
        z=zt,
        rgb=rgb,
        default_color=default_color,
    )
    print(f"transformed point cloud saved to: {out_ply_path}")


def save_point_cloud_to_ply(cloud_path, out_ply_path, transform, chunk_size=2_000_000, default_color=(180, 180, 180)):
    cloud_path = Path(cloud_path)
    suffix = cloud_path.suffix.lower()

    if suffix in {".las", ".laz"}:
        save_las_to_ply_chunked(
            las_path=cloud_path,
            out_ply_path=out_ply_path,
            transform=transform,
            chunk_size=chunk_size,
            default_color=default_color,
        )
        return

    if suffix == ".ply":
        save_ply_to_ply(
            ply_path=cloud_path,
            out_ply_path=out_ply_path,
            transform=transform,
            default_color=default_color,
        )
        return

    raise ValueError(f"Unsupported input format: {cloud_path.suffix}. Expected .las, .laz, or .ply.")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Apply transform_manual/transform_icp/transform_refine to a LAS/LAZ/PLY point cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lidar", type=str, required=True, help="Input LAS/LAZ/PLY point cloud path.")
    parser.add_argument("--transform", type=str, required=True, help="Directory containing transform txt files.")
    parser.add_argument("--out_lidar_name", type=str, default="lidar_full.ply", help="Output PLY filename.")
    parser.add_argument("--chunk_size", type=int, default=100_000_000, help="LAS/LAZ chunk size.")
    parser.add_argument(
        "--default_color",
        type=int,
        nargs=3,
        default=[180, 180, 180],
        help="Fallback RGB color for uncolored input points.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    transform = compose_transformations(args.transform)
    out_ply_path = Path(args.transform) / args.out_lidar_name

    save_point_cloud_to_ply(
        cloud_path=args.lidar,
        out_ply_path=out_ply_path,
        transform=transform,
        chunk_size=args.chunk_size,
        default_color=tuple(args.default_color),
    )


if __name__ == "__main__":
    main()
