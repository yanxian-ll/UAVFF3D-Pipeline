# -*- coding: utf-8 -*-

import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import laspy


def get_local_origin_from_header(header, mode="min"):
    """
    根据 LAS header 自动确定局部坐标系原点

    mode:
        - min    : 用包围盒最小值作为原点，结果通常非负，推荐
        - center : 用包围盒中心作为原点
    """
    mins = np.asarray(header.mins, dtype=np.float64)
    maxs = np.asarray(header.maxs, dtype=np.float64)

    if mode == "min":
        return mins
    elif mode == "center":
        return (mins + maxs) / 2.0
    else:
        raise ValueError(f"Unsupported origin mode: {mode}")


def parse_metadata_xml_origin(metadata_xml_path: str):
    """
    从 metadata.xml 中读取:
        <SRSOrigin>x,y,z</SRSOrigin>

    返回:
        origin: np.ndarray shape=(3,)
        srs: str or None
    """
    metadata_xml_path = Path(metadata_xml_path)
    if not metadata_xml_path.exists():
        raise FileNotFoundError(f"metadata.xml not found: {metadata_xml_path}")

    tree = ET.parse(str(metadata_xml_path))
    root = tree.getroot()

    srs = root.findtext("SRS", default=None)
    srs_origin_text = root.findtext("SRSOrigin", default=None)

    if srs_origin_text is None:
        raise ValueError(f"SRSOrigin not found in metadata xml: {metadata_xml_path}")

    parts = [x.strip() for x in srs_origin_text.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"SRSOrigin format error in metadata xml: {metadata_xml_path}\n"
            f"Expected: x,y,z\n"
            f"Got: {srs_origin_text}"
        )

    origin = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
    return origin, srs


def has_rgb_dimensions(point_format) -> bool:
    dim_names = set(point_format.dimension_names)
    return {"red", "green", "blue"}.issubset(dim_names)


def create_output_header(input_header, keep_rgb: bool):
    """
    只保留 xyz 或 xyz+rgb
    """
    if keep_rgb:
        # point_format=2: XYZ + RGB
        out_header = laspy.LasHeader(point_format=2, version="1.2")
    else:
        # point_format=0: XYZ
        out_header = laspy.LasHeader(point_format=0, version="1.2")

    out_header.scales = np.asarray(input_header.scales, dtype=np.float64)
    out_header.offsets = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    try:
        out_header.system_identifier = input_header.system_identifier
    except Exception:
        pass

    try:
        out_header.generating_software = input_header.generating_software
    except Exception:
        pass

    return out_header


def shift_las_to_local_keep_xyz_rgb(
    input_path: str,
    output_path: str = None,
    origin_xyz=None,
    metadata_xml_path: str = None,
    origin_mode: str = "min",
    chunk_size: int = 2_000_000,
    replace_original: bool = False,
):
    """
    分块读取 LAS/LAZ，将点云平移到局部坐标系，只保留 XYZ 和 RGB

    原点优先级:
        1. origin_xyz
        2. metadata_xml_path 中的 SRSOrigin
        3. LAS header 自动推断 origin_mode

    参数：
        input_path: 输入 LAS/LAZ 路径
        output_path: 输出 LAS/LAZ 路径
        origin_xyz: 手动指定局部原点，如 (x0, y0, z0)
        metadata_xml_path: metadata.xml 路径，从中读取 SRSOrigin
        origin_mode:
            - min
            - center
        chunk_size: 分块大小
        replace_original:
            - False: 输出新文件
            - True : 完成后替换原文件

    返回：
        final_output_path
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_local{input_path.suffix}")
    else:
        output_path = Path(output_path)

    if replace_original:
        temp_output_path = input_path.with_name(f"{input_path.stem}._tmp_local{input_path.suffix}")
    else:
        temp_output_path = output_path

    temp_output_path.parent.mkdir(parents=True, exist_ok=True)

    with laspy.open(str(input_path)) as reader:
        total_points = reader.header.point_count
        keep_rgb = has_rgb_dimensions(reader.header.point_format)

        # -------------------------
        # 决定原点
        # -------------------------
        origin_source = None
        srs_name = None

        if origin_xyz is not None:
            origin = np.asarray(origin_xyz, dtype=np.float64)
            if origin.shape != (3,):
                raise ValueError("origin_xyz must be a 3-element coordinate, e.g. (x0, y0, z0)")
            origin_source = "manual origin_xyz"

        elif metadata_xml_path is not None:
            origin, srs_name = parse_metadata_xml_origin(metadata_xml_path)
            origin_source = f"metadata.xml SRSOrigin ({metadata_xml_path})"

        else:
            origin = get_local_origin_from_header(reader.header, mode=origin_mode)
            origin_source = f"header {origin_mode}"

        out_header = create_output_header(reader.header, keep_rgb=keep_rgb)

        print(f"[INFO] Input: {input_path}")
        print(f"[INFO] Total points: {total_points}")
        print(f"[INFO] Keep RGB: {keep_rgb}")
        print(f"[INFO] Origin source: {origin_source}")
        if srs_name is not None:
            print(f"[INFO] SRS: {srs_name}")
        print(f"[INFO] Origin: {origin.tolist()}")
        print(f"[INFO] Output: {temp_output_path}")

        processed = 0
        chunk_id = 0

        with laspy.open(str(temp_output_path), mode="w", header=out_header) as writer:
            for points in reader.chunk_iterator(chunk_size):
                chunk_id += 1
                n = len(points)
                if n == 0:
                    continue

                x = np.asarray(points.x, dtype=np.float64)
                y = np.asarray(points.y, dtype=np.float64)
                z = np.asarray(points.z, dtype=np.float64)

                # 平移到局部坐标系
                local_x = x - origin[0]
                local_y = y - origin[1]
                local_z = z - origin[2]

                out_points = laspy.ScaleAwarePointRecord.zeros(n, header=out_header)
                out_points.x = local_x
                out_points.y = local_y
                out_points.z = local_z

                if keep_rgb:
                    out_points.red = points.red
                    out_points.green = points.green
                    out_points.blue = points.blue

                writer.write_points(out_points)

                processed += n
                ratio = processed / total_points * 100 if total_points > 0 else 100.0
                print(
                    f"[INFO] chunk={chunk_id:04d} | "
                    f"processed={processed}/{total_points} ({ratio:.2f}%)"
                )

    if replace_original:
        os.replace(str(temp_output_path), str(input_path))
        final_output = input_path
    else:
        final_output = temp_output_path

    print(f"[INFO] Done: {final_output}")
    return str(final_output)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shift LAS/LAZ to a local coordinate system and keep only XYZ/RGB"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input LAS/LAZ file path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output LAS/LAZ file path. If omitted, *_local.xxx will be used",
    )
    parser.add_argument(
        "--metadata_xml_path",
        type=str,
        default=None,
        help="metadata.xml path. If provided, SRSOrigin will be used as local origin",
    )
    parser.add_argument(
        "--origin_mode",
        type=str,
        default="min",
        choices=["min", "center"],
        help="How to choose local origin automatically when origin_xyz / metadata_xml_path are not provided",
    )
    parser.add_argument(
        "--origin_xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("X0", "Y0", "Z0"),
        help="Manually specify local origin. Higher priority than metadata_xml_path",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50_000_000,
        help="Chunk size for streaming",
    )
    parser.add_argument(
        "--replace_original",
        action="store_true",
        help="Replace the original file after conversion",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    shift_las_to_local_keep_xyz_rgb(
        input_path=args.input_path,
        output_path=args.output_path,
        origin_xyz=args.origin_xyz,
        metadata_xml_path=args.metadata_xml_path,
        origin_mode=args.origin_mode,
        chunk_size=args.chunk_size,
        replace_original=args.replace_original,
    )


if __name__ == "__main__":
    main()


"""
python shift_las_to_local.py --input_path nanfang/cloud_merged.las --replace_original \
    --metadata_xml_path ../recon/nanfang/models/pc/0/terra_obj/metadata.xml

python shift_las_to_local.py --input_path yanghaitang/cloud_merged.las --replace_original \
    --metadata_xml_path ../recon/yanghaitang/models/pc/0/terra_obj/metadata.xml

python shift_las_to_local.py --input_path xiaoxiang/cloud_merged.las --replace_original \
    --metadata_xml_path ../recon/xiaoxiang/models/pc/0/terra_obj/metadata.xml
"""
