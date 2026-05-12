import argparse
import os
import shutil
import cv2
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import itertools
from collections import defaultdict

from reorganize_utils import save_cam_txt, cam2world_whu_to_opencv

def read_luojiamvs_pose(path):
    """Load camera pose for Luojiamvs cam2world"""
    f = open(path)
    RT = np.loadtxt(f, skiprows=1, max_rows=4, dtype=np.float32)
    assert RT.shape == (4, 4)
    info = np.loadtxt(f, skiprows=1, max_rows=1, dtype=np.float32)
    fx = fy = info[0]
    cx = info[1]
    cy = info[2]
    RT = cam2world_whu_to_opencv(RT)
    return RT, fx, fy, cx, cy

def process_viewport_for_scene(viewport_data, out_scene_dir):
    """处理单个视点，将结果保存到指定的场景目录中"""
    viewport, img_dir, depth_dir, cam_dir = viewport_data
    
    strip_id = int(viewport.split("_")[0])
    view_id = int(viewport.split("_")[1])
    
    # 5个相机中只使用第0个
    for camera_id in range(1):  
        # 构建输出文件名: strip_view_camera
        name = f"{strip_id:04d}_{view_id:08d}_{camera_id}"
        
        # 检查是否已存在该文件，避免重复处理
        img_output_path = os.path.join(out_scene_dir, "images", name + ".png")
            
        # 源文件路径
        src_img_path = os.path.join(img_dir, viewport, str(camera_id), '000000.png')
        src_cam_path = os.path.join(cam_dir, viewport, str(camera_id), '000000.txt')
        src_depth_path = os.path.join(depth_dir, viewport, str(camera_id), '000000.png')

        if (not os.path.exists(src_img_path)) or (not os.path.exists(src_cam_path)) or (not os.path.exists(src_depth_path)):
            continue
        
        # 1. 复制/重命名图像
        if os.path.exists(src_img_path) and (not os.path.exists(img_output_path)):
            shutil.copy2(src_img_path, img_output_path)
        else:
            continue
        
        # 2. 处理相机参数
        if os.path.exists(src_cam_path):
            RT, fx, fy, cx, cy = read_luojiamvs_pose(src_cam_path)
            with Image.open(src_img_path) as img:
                width, height = img.size
            
            # luojiamvs 的 cx 和 cy 是错误的
            cx = width / 2.0
            cy = height / 2.0

            save_cam_txt(
                os.path.join(out_scene_dir, "cams", name + ".txt"),
                RT, fx, fy, cx, cy, height, width
            )
        else:
            print(f"Warning: Camera file not found: {src_cam_path}")
        
        # 3. 处理深度图
        depth_output_path = os.path.join(out_scene_dir, "depth", name + ".exr")
        if os.path.exists(src_depth_path) and (not os.path.exists(depth_output_path)):
            depth = cv2.imread(src_depth_path, cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / 64.0  # true depth
            imageio.imwrite(depth_output_path, depth)
        else:
            continue

    return viewport

def process_strip_group(data_dir, strip_group, scene_id, out_dir):
    """处理一组航带（一个场景）"""
    # 创建场景目录
    scene_dir = os.path.join(out_dir, f"scene_{scene_id:04d}")
    os.makedirs(os.path.join(scene_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "cams"), exist_ok=True)
    
    print(f"Processing scene_{scene_id:04d} with {len(strip_group)} strips...")
    
    # 收集所有视点
    all_viewport_data = []
    for strip_data in strip_group:
        strip_id, viewport_info_list = strip_data
        
        # viewport_info_list 中的每个元素是 (viewport, split)
        for viewport_info in viewport_info_list:
            viewport, split = viewport_info
            img_dir = os.path.join(data_dir, split, "Images")
            depth_dir = os.path.join(data_dir, split, "Depths")
            cam_dir = os.path.join(data_dir, split, "Cams")
            
            all_viewport_data.append((viewport, img_dir, depth_dir, cam_dir))
    
    # 使用多线程处理所有视点
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for viewport_data in all_viewport_data:
            futures.append(
                executor.submit(
                    process_viewport_for_scene,
                    viewport_data, scene_dir
                )
            )
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Scene {scene_id}"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing viewport: {e}")
    
    print(f"Scene_{scene_id:04d} completed. Total images: {len(os.listdir(os.path.join(scene_dir, 'images')))}")
    return scene_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="../raw_data/LuoJia_MVS_dataset")
    ap.add_argument("--out_dir", type=str, default="../data/luojiamvs")
    ap.add_argument("--num_strip_per_scene", type=int, default=18)
    args = ap.parse_args()
    
    splits = ["train", "test"]
    
    # 使用defaultdict来存储航带信息
    # 结构: strip_id -> [(viewport1, split1), (viewport2, split2), ...]
    all_strips = defaultdict(list)
    
    for split in splits:
        print(f"Loading {split} split...")        
        img_dir = os.path.join(args.data_dir, split, "Images")
        
        if not os.path.exists(img_dir):
            print(f"Warning: {img_dir} does not exist, skipping...")
            continue
            
        list_viewport = os.listdir(img_dir)
        list_viewport = sorted(list_viewport, key=lambda x: int(x.split("_")[0]))
        
        from itertools import groupby
        for key, group in groupby(list_viewport, key=lambda x: int(x.split("_")[0])):
            strip_id = key
            viewports = list(group)
            
            # 将每个视点及其所属的split添加到列表中
            for viewport in viewports:
                all_strips[strip_id].append((viewport, split))
    
    strip_ids = sorted(all_strips.keys())
    print(f"Total unique strips: {len(strip_ids)}")
    
    if not strip_ids:
        print("No strips found! Check your data directory.")
        return
    
    # 打印一些统计信息
    for strip_id in list(strip_ids)[:5]:  # 只打印前5个航带的信息
        print(f"Strip {strip_id}: {len(all_strips[strip_id])} viewports")
    
    strip_groups = []
    current_group = []
    last_strip_id = -10
    
    for strip_id in strip_ids:
        # 如果当前航带与前一个航带不连续，或者当前组已有指定数量的航带，则开始新组
        if (strip_id - last_strip_id > 1) or len(current_group) >= args.num_strip_per_scene:
            if current_group:  # 保存当前组
                strip_groups.append([(sid, all_strips[sid]) for sid in current_group])
            current_group = []  # 开始新组
        
        current_group.append(strip_id)
        last_strip_id = strip_id
    
    # 添加最后一组
    if current_group:
        strip_groups.append([(sid, all_strips[sid]) for sid in current_group])
    
    print(f"Grouped into {len(strip_groups)} scenes")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 处理每个场景
    scene_count = 0
    for i, strip_group in enumerate(strip_groups):
        # 为每组创建一个场景
        scene_dir = process_strip_group(
            args.data_dir,
            strip_group, 
            scene_count, 
            args.out_dir
        )
        
        scene_count += 1
    
    print(f"Finished processing. Created {scene_count} scenes.")
    
    # 统计每个场景的图像数量
    print("\nScene statistics:")
    for i in range(scene_count):
        scene_dir = os.path.join(args.out_dir, f"scene_{i:04d}")
        images_dir = os.path.join(scene_dir, "images")
        if os.path.exists(images_dir):
            num_images = len(os.listdir(images_dir))
            print(f"  scene_{i:04d}: {num_images} images")

if __name__ == "__main__":
    main()
