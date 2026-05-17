import argparse
import os
import cv2
import imageio.v2 as imageio
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from itertools import groupby

from reorganize_utils import save_cam_txt, cam2world_whu_to_opencv

def load_whumvs_pose(path):
    "Load camera pose for WHUMVS cam2world"
    f = open(path)
    k_info = np.loadtxt(f, skiprows=1, max_rows=1, dtype=np.float32)
    fx = k_info[0]
    fy = k_info[0]
    cx = k_info[1]
    cy = k_info[2]
    h = k_info[3]
    w = k_info[4]
    RT = np.loadtxt(f, skiprows=2, max_rows=4, dtype=np.float32)
    assert RT.shape == (4, 4)
    RT = cam2world_whu_to_opencv(RT)
    return RT, fx, fy, cx, cy, h, w

def process_viewport_for_scene(viewport_data, scene_dir, scale):
    """处理单个视点，将结果保存到指定的场景目录中"""
    filename, img_dir, depth_dir, cam_dir = viewport_data
    
    base = filename.split('.')[0]
    strip_id = base.split('_')[0]  # 航带ID
    view_id = int(base.split('_')[1])  # 视点ID
    
    # 构建输出文件名: strip_view
    name = f"{int(strip_id):04d}_{view_id:08d}"
    
    # 检查是否已存在该文件，避免重复处理
    img_output_path = os.path.join(scene_dir, "images", name + ".png")
    if os.path.exists(img_output_path):
        return
    
    # 1. 读取并调整图像大小
    img_path = os.path.join(img_dir, filename)
    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")
        return
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Failed to read image: {img_path}")
        return
    
    h, w = img.shape[:2]
    max_size = 1024
    scale = max_size / max(h, w)
    new_h = int(h * scale + 0.5)
    new_w = int(w * scale + 0.5)
    
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(img_output_path, resized_img)
    
    # 2. 处理深度图
    depth_path = os.path.join(depth_dir, base + ".png")
    if os.path.exists(depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is not None:
            depth = depth.astype(np.float32) / 64.0  # true depth
            resized_depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            depth_output_path = os.path.join(scene_dir, "depth", name + ".exr")
            imageio.imwrite(depth_output_path, resized_depth)
    
    # 3. 处理相机参数
    cam_path = os.path.join(cam_dir, base + ".txt")
    if os.path.exists(cam_path):
        RT, fx, fy, cx, cy, h_orig, w_orig = load_whumvs_pose(cam_path)
        
        # 调整相机参数
        fx *= scale
        fy *= scale
        cx *= scale
        cy *= scale
        h = new_h
        w = new_w
        
        save_cam_txt(
            os.path.join(scene_dir, "cams", name + ".txt"),
            RT, fx, fy, cx, cy, h, w
        )

def process_strip_group(strip_group, scene_id, img_dir, depth_dir, cam_dir, out_dir, scale):
    """处理一组航带（一个场景）"""
    # 创建场景目录
    scene_dir = os.path.join(out_dir, f"scene_{scene_id:04d}")
    os.makedirs(os.path.join(scene_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "cams"), exist_ok=True)
    
    print(f"Processing scene_{scene_id:04d} with {len(strip_group)} strips...")
    
    # 收集所有视点
    all_viewport_data = []
    for strip_id, filenames in strip_group:
        for filename in filenames:
            all_viewport_data.append((filename, img_dir, depth_dir, cam_dir))
    
    # 使用多线程处理所有视点
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for viewport_data in all_viewport_data:
            futures.append(
                executor.submit(
                    process_viewport_for_scene,
                    viewport_data, scene_dir, scale
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
    ap.add_argument("--data_dir", type=str, default="../raw_data/WHU-dataset")
    ap.add_argument("--out_dir", type=str, default="../data/whumvs")
    ap.add_argument("--num_strip_per_scene", type=int, default=4)
    ap.add_argument("--max_size", type=int, default=1024)
    args = ap.parse_args()

    img_dir = os.path.join(args.data_dir, "Images")
    cam_dir = os.path.join(args.data_dir, "Cams")
    depth_dir = os.path.join(args.data_dir, "Depths")
    
    # 获取所有图像文件名
    list_filenames = os.listdir(img_dir)
    
    # 按航带ID分组
    list_filenames = sorted(list_filenames, key=lambda x: int(x.split('_')[0]))
    
    # 按航带ID进行分组
    strips = {}
    for key, group in groupby(list_filenames, key=lambda x: int(x.split('_')[0])):
        strips[key] = list(group)
    
    strip_ids = sorted(strips.keys())
    print(f"Total unique strips: {len(strip_ids)}")
    
    # 将航带按相邻的num_strip_per_scene个一组分组
    strip_groups = []
    current_group = []
    last_strip_id = -10
    
    for strip_id in strip_ids:
        # 如果当前航带与前一个航带不连续，或者当前组已有指定数量的航带，则开始新组
        if (strip_id - last_strip_id > 1) or len(current_group) >= args.num_strip_per_scene:
            if current_group:  # 保存当前组
                strip_groups.append([(sid, strips[sid]) for sid in current_group])
            current_group = []  # 开始新组
        
        current_group.append(strip_id)
        last_strip_id = strip_id
    
    # 添加最后一组
    if current_group:
        strip_groups.append([(sid, strips[sid]) for sid in current_group])
    
    print(f"Grouped into {len(strip_groups)} scenes")
    
    # 计算缩放比例（基于第一个图像的尺寸）
    if list_filenames:
        first_img_path = os.path.join(img_dir, list_filenames[0])
        img = cv2.imread(first_img_path)
        if img is not None:
            h, w = img.shape[:2]
            scale = args.max_size / max(h, w)
        else:
            scale = 1.0
    else:
        scale = 1.0
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 处理每个场景
    scene_count = 0
    for i, strip_group in enumerate(strip_groups):
        # 为每组创建一个场景
        scene_dir = process_strip_group(
            strip_group, 
            scene_count, 
            img_dir, 
            depth_dir, 
            cam_dir, 
            args.out_dir,
            scale
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
