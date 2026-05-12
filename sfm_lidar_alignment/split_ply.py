import os
import argparse
import numpy as np
import open3d as o3d


def split_ply_by_longest_xy_edge(ply_path, voxel_size=0.1):
    """
    读取 ply 点云。
    降采样仅用于提高计算切分线的效率；
    最终保存时，仍然使用原始点云进行切分并保存。
    保存格式为二进制 PLY。

    参数：
        ply_path (str): 输入 ply 文件路径
        voxel_size (float): 体素降采样大小，仅用于计算切分线
    """
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"找不到文件: {ply_path}")

    # =========================
    # 读取原始点云
    # =========================
    pcd_raw = o3d.io.read_point_cloud(ply_path)
    if pcd_raw.is_empty():
        raise ValueError(f"读取失败或点云为空: {ply_path}")

    raw_pts = np.asarray(pcd_raw.points)
    print(f"[Info] 原始点数: {len(raw_pts)}")

    # =========================
    # 降采样，仅用于估计切分轴和切分位置
    # =========================
    if voxel_size > 0:
        pcd_ds = pcd_raw.voxel_down_sample(voxel_size=voxel_size)
        if pcd_ds.is_empty():
            raise ValueError("降采样后点云为空，请减小 voxel_size")
        ds_pts = np.asarray(pcd_ds.points)
        print(f"[Info] 降采样后点数: {len(ds_pts)}")
    else:
        ds_pts = raw_pts
        print("[Info] voxel_size <= 0，跳过降采样，直接使用原始点云计算切分线")

    # =========================
    # 在降采样点云上计算 xy 平面最长边
    # =========================
    min_xy = ds_pts[:, :2].min(axis=0)   # [min_x, min_y]
    max_xy = ds_pts[:, :2].max(axis=0)   # [max_x, max_y]
    extent_xy = max_xy - min_xy          # [range_x, range_y]

    range_x, range_y = extent_xy[0], extent_xy[1]

    if range_x >= range_y:
        split_axis = 0   # x
        split_value = (min_xy[0] + max_xy[0]) / 2.0
        axis_name = "x"
    else:
        split_axis = 1   # y
        split_value = (min_xy[1] + max_xy[1]) / 2.0
        axis_name = "y"

    print(f"[Info] 按 {axis_name} 方向切分")
    print(f"[Info] 切分位置: {split_value:.6f}")

    # =========================
    # 用原始点云进行真正切分
    # =========================
    idx_part1 = np.where(raw_pts[:, split_axis] <= split_value)[0]
    idx_part2 = np.where(raw_pts[:, split_axis] > split_value)[0]

    if len(idx_part1) == 0 or len(idx_part2) == 0:
        raise ValueError("切分后有一部分为空，请检查点云分布或切分逻辑")

    pcd_part1 = pcd_raw.select_by_index(idx_part1)
    pcd_part2 = pcd_raw.select_by_index(idx_part2)

    print(f"[Info] part1 原始点数: {len(pcd_part1.points)}")
    print(f"[Info] part2 原始点数: {len(pcd_part2.points)}")
    print(f"[Info] 切分后总点数: {len(pcd_part1.points) + len(pcd_part2.points)}")

    # =========================
    # 输出路径
    # =========================
    in_dir = os.path.dirname(ply_path)
    stem = os.path.splitext(os.path.basename(ply_path))[0]

    out_path1 = os.path.join(in_dir, f"{stem}_part1.ply")
    out_path2 = os.path.join(in_dir, f"{stem}_part2.ply")

    # =========================
    # 保存为二进制 PLY
    # =========================
    ok1 = o3d.io.write_point_cloud(
        out_path1,
        pcd_part1,
        write_ascii=False,
        compressed=False
    )
    ok2 = o3d.io.write_point_cloud(
        out_path2,
        pcd_part2,
        write_ascii=False,
        compressed=False
    )

    if not ok1 or not ok2:
        raise IOError("保存 ply 文件失败")

    print("[Done] 已保存（二进制 PLY）：")
    print(f"  {out_path1}")
    print(f"  {out_path2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_path", type=str, help="输入 ply 文件路径")
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=5,
        help="降采样体素大小，仅用于计算切分线，默认 5"
    )
    args = parser.parse_args()

    split_ply_by_longest_xy_edge(args.ply_path, args.voxel_size)


if __name__ == "__main__":
    main()

"""
python split_ply.py yanghaitang/models/pc/sampled_recon.ply

python split_ply.py xiaoxiang/models/pc/sampled_recon.ply

python split_ply.py nanfang/models/pc/sampled_recon.ply

"""