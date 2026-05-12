# SfM--LiDAR Alignment

This directory contains the real-scene alignment branch used for LiDAR-grounded
A3D-Real scenes. The scripts prepare LiDAR and SfM point clouds, refine
registration, render image-aligned depth maps, evaluate alignment quality, and
optionally fuse LiDAR/SfM depths.

## Main Steps

1. Shift raw LAS/LAZ into a local coordinate frame:

```bash
python shift_las_to_local.py --input_path scene/cloud_merged.las --replace_original
```

2. Create lightweight point clouds for manual or CloudCompare alignment:

```bash
python downsample_las_to_ply.py scene/cloud_merged.las scene/downsample_lidar.ply --voxel-size 0.3
python downsample_obj_to_ply.py recon/models/pc/0/terra_obj recon/downsample_recon.ply --voxel-size 0.5
```

3. Store coarse transforms in:

```text
scene/transform/
  transform_manual.txt
  transform_icp.txt
  transform_refine.txt
```

4. Apply the transform chain:

```bash
python lidar_transform.py --lidar scene/cloud_merged.las --transform scene/transform --out_lidar_name lidar_final.ply
```

5. Render and refine:

```bash
python random_select_downward_views.py recon scene/selected_views.txt
python render_depth_from_lidar.py scene/transform/lidar_final.ply recon scene/lidar_render --save-depth-npy --save-depth-vis
python render_depth_from_ply.py recon/models/pc/0 recon recon/recon_render --save-depth-npy --save-depth-vis
python refine_transform_v2.py --root_a scene/lidar_render --root_b recon/recon_render --out_dir scene/transform_refine --mode match_then_icp
```

6. Evaluate or fuse outputs:

```bash
python evaluate_pair_alignment_metrics.py --pair_dir recon/pair_selected --cam_dir recon/recon_render/cams --depth_dir scene/lidar_render/depth_npy --out_dir scene/eval_pair
python depth_world_cloud_metric.py --depth_dir_a scene/lidar_render/depth_npy --depth_dir_b recon/recon_render/depth_npy --cam_dir recon/recon_render/cams --out_dir scene/eval_world
python render_fused_depth_lidar_obj.py scene/transform/lidar_final.ply recon/models/pc/0 recon/cams scene/fused_render --save-depth-npy --save-depth-vis
```

## Transform Convention

Transforms are left-multiplied in the same order they are applied to points:

```text
T = transform_refine @ transform_icp @ transform_manual
```

Missing transform files are skipped.

