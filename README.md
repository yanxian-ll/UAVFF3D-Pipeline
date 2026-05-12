# A3D-Bench Pipeline

This repository contains the data-construction code used for the A3D-Bench
paper pipeline. It covers synthetic data rendering, LiDAR--SfM alignment,
conversion of heterogeneous UAV/MVS datasets into a shared format, scene
metadata generation, covisibility graph construction, and split preparation.

## Repository Layout

```text
a3dbench_pipeline/
  synthetic_generation/        # A3D-Syn-L, A3D-Syn-S, and A3D-FA renderers
  sfm_lidar_alignment/         # LiDAR/SfM registration, depth rendering, fusion, metrics
  uav_data_processing/         # Unified-format conversion, metadata, covisibility, splits
  docs/                        # Format and workflow notes
```

The three original root-level renderer names are kept as compatibility wrappers:

- `multi_obj_tile_renderer_en_standardized.py`
- `view_selector_renderer_standardized.py`
- `view_selector_renderer_en_standardized.py`

New work should use the clearer paths under `synthetic_generation/`.

## Pipeline Overview

1. **Synthetic generation**
   - `synthetic_generation/a3d_syn_l_tile_renderer.py` renders large structured
     tiled synthetic scenes for A3D-Syn-L.
   - `synthetic_generation/a3d_syn_s_view_selector_renderer.py` supports
     interactive viewpoint/trajectory selection for smaller or irregular
     synthetic scene units used by A3D-Syn-S.
   - `synthetic_generation/a3d_fa_view_selector_renderer.py` renders controlled
     hFOV--height groups for A3D-FA.
2. **LiDAR-grounded real scenes**
   - `sfm_lidar_alignment/` prepares LAS/LAZ/PLY and SfM point clouds, applies
     coarse transforms, renders LiDAR/SfM depths, refines alignment, evaluates
     alignment quality, and fuses depth.
3. **Unified data format**
   - `uav_data_processing/scripts/convert/` converts external datasets and
     reconstructed scenes into the common `images/`, `cams/`, `depth/` layout.
   - `uav_data_processing/generate_scene_meta.py` writes `scene_meta.json`.
   - `uav_data_processing/covisibility.py` builds sparse covisibility graphs.
   - `uav_data_processing/split_scene.py` writes train/val/test split files and
     optional hFOV summaries.

## Common Scene Format

Each processed scene should follow this minimal layout:

```text
scene/
  images/<frame>.png|jpg
  cams/<frame>.txt
  depth/<frame>.exr
  scene_meta.json
```

Optional modalities can be added with matching frame stems:

```text
mask/
depth_da3/
depth_complete/
semantic_mask/
covisibility/view_covis_graph_csr_mmap/
```

Camera files use OpenCV coordinates: x right, y down, z forward. The extrinsic
matrix is world-to-camera; `scene_meta.json` stores the inverse camera-to-world
matrix.

## Installation

Create an environment with Python 3.10 or newer. The core scripts use:

```bash
pip install -r requirements.txt
```

Some optional scripts require extra local packages or checkpoints, for example
SAM3 for exclusion masks, `uniception` for specific downstream loaders, and a
GUI/off-screen rendering setup for Open3D visualization.

## Typical Commands

Generate `scene_meta.json` for processed scenes:

```bash
cd uav_data_processing
python generate_scene_meta.py --root_dir ../data --dataset A3D-Real A3D-Syn-L A3D-Syn-S --overwrite
```

Build covisibility graphs:

```bash
cd uav_data_processing
python covisibility.py --config configs/covisibility_config.yaml --root ../data/A3D-Syn-L
```

Apply a LiDAR transform chain:

```bash
cd sfm_lidar_alignment
python lidar_transform.py --lidar scene/cloud_merged.las --transform scene/transform --out_lidar_name lidar_final.ply
```

Render a controlled A3D-FA hFOV group:

```bash
python synthetic_generation/a3d_fa_view_selector_renderer.py --mesh path/to/mesh.obj --out-dir outputs/a3d_fa_scene
```

## Notes

- Large scenes should be rendered and converted scene-by-scene; generated data
  are intentionally not stored in this code repository.
- `__pycache__`, rendered outputs, point clouds, and raw datasets are ignored by
  `.gitignore`.
- Existing dataset assets, LiDAR data, textured meshes, and third-party model
  checkpoints keep their original licenses. The MIT license in this repository
  applies only to the code written here.

