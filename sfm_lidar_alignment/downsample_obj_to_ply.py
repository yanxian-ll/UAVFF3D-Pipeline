import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from tqdm import tqdm

def find_obj_files(root_dir: Path):
    return sorted(root_dir.rglob("*.obj"))


def load_obj_vertices_as_pcd(obj_path: Path, default_color=(180, 180, 180)):
    """
    使用 open3d 直接读取 OBJ 网格，
    直接取 mesh.vertices 作为点云点，
    如果有 vertex_colors 就保留，否则使用默认颜色。
    """
    try:
        mesh = o3d.io.read_triangle_mesh(str(obj_path), enable_post_processing=False)
    except Exception as e:
        print(f"[WARN] 读取失败: {obj_path} | {e}")
        return None

    if mesh is None or mesh.is_empty():
        print(f"[WARN] 空网格: {obj_path}")
        return None

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.shape[0] == 0:
        print(f"[WARN] 顶点为空: {obj_path}")
        return None

    # 顶点颜色
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        if colors.shape[0] != vertices.shape[0]:
            colors = np.full((vertices.shape[0], 3), default_color, dtype=np.uint8)
        else:
            # open3d 一般是 0~1 浮点
            if colors.dtype != np.uint8:
                colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
            else:
                colors = colors[:, :3]
    else:
        colors = np.full((vertices.shape[0], 3), default_color, dtype=np.uint8)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    return pcd


def pcd_num_points(pcd: o3d.geometry.PointCloud) -> int:
    return np.asarray(pcd.points).shape[0]


def main():
    parser = argparse.ArgumentParser(
        description="递归查找文件夹下所有 OBJ，直接取顶点生成带颜色点云，降采样后合并输出为一个 PLY。"
    )
    parser.add_argument("input_dir", type=str, help="输入根文件夹")
    parser.add_argument("output_ply", type=str, help="输出 PLY 文件路径")

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.5,
        help="每个 OBJ 单独降采样体素大小，默认 0.5",
    )
    parser.add_argument(
        "--default-color",
        type=int,
        nargs=3,
        default=[180, 180, 180],
        help="没有 vertex color 时使用的默认 RGB，默认 180 180 180",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_ply = Path(args.output_ply)
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    obj_files = find_obj_files(input_dir)
    if len(obj_files) == 0:
        print(f"[ERROR] 没有找到 OBJ 文件: {input_dir}")
        return

    print(f"[INFO] 共找到 {len(obj_files)} 个 OBJ 文件")

    merged_pcd = o3d.geometry.PointCloud()

    for idx, obj_path in tqdm(enumerate(obj_files, 1)):
        print(f"\n[INFO] [{idx}/{len(obj_files)}] 处理: {obj_path}")

        pcd = load_obj_vertices_as_pcd(
            obj_path,
            default_color=tuple(args.default_color),
        )
        if pcd is None:
            print("[WARN] 跳过")
            continue

        raw_n = pcd_num_points(pcd)
        print(f"[INFO] 原始顶点数: {raw_n}")

        # 单 OBJ 降采样
        if args.voxel_size > 0:
            pcd = pcd.voxel_down_sample(args.voxel_size)

        ds_n = pcd_num_points(pcd)
        print(f"[INFO] 单模型降采样后点数: {ds_n}")

        if ds_n == 0:
            print("[WARN] 降采样后为空，跳过")
            continue

        # 增量合并，避免先存所有 points/colors list
        merged_pcd += pcd
        cur_n = pcd_num_points(merged_pcd)
        print(f"[INFO] 当前累计点数: {cur_n}")


    final_n = pcd_num_points(merged_pcd)
    if final_n == 0:
        print("[ERROR] 没有成功生成任何点云")
        return

    print(f"\n[INFO] 合并后点数: {final_n}")

    ok = o3d.io.write_point_cloud(str(output_ply), merged_pcd, write_ascii=False)
    if ok:
        print(f"[INFO] 输出完成: {output_ply}")
    else:
        print("[ERROR] PLY 写出失败")


if __name__ == "__main__":
    main()

"""

python downsample_obj_to_ply.py yanghaitang/models/pc/0/terra_obj yanghaitang/models/pc/downsample_recon.ply
python downsample_obj_to_ply.py nanfang/models/pc/0/terra_obj nanfang/models/pc/downsample_recon.ply
python downsample_obj_to_ply.py xiaoxiang/models/pc/0/terra_obj xiaoxiang/models/pc/downsample_recon.ply

"""
