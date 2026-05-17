# UAVFF3D/WAI common scene format

This document describes the minimal processed scene format consumed by the UAVFF3D pipeline.

## Directory layout

```text
<dataset>/<scene>/
  images/
    <frame>.png|jpg|jpeg
  cams/
    <frame>.txt
  depth/
    <frame>.exr
  mask/
    <frame>.png              # optional binary valid mask
  scene_meta.json            # generated
```

File stems must match across `images/`, `cams/`, and `depth/`. Optional modalities such as `mask/` are added to `scene_meta.json` only when present.

## Camera text format

`generate_scene_meta.py` expects camera files with the same structure used by BlendedMVS-style text cameras:

```text
extrinsic
<4 numbers>
<4 numbers>
<4 numbers>
<4 numbers>

intrinsic
<3 numbers>
<3 numbers>
<3 numbers>

h w hfov
<height> <width> <horizontal_fov_degrees>
```

The extrinsic matrix is interpreted as world-to-camera. The metadata generator stores the inverse as `transform_matrix` in camera-to-world OpenCV convention.

## Generated metadata

`scene_meta.json` contains:

- `scene_name`
- `dataset_name`
- `camera_model`
- `camera_convention`
- per-frame image/depth paths
- intrinsics (`fl_x`, `fl_y`, `cx`, `cy`)
- image size (`h`, `w`)
- camera-to-world transform matrix
- optional mask paths when available

## Split metadata

`split_scene.py` writes both NumPy and text scene lists:

```text
<metadata>/<split>/<dataset>_scene_list_<split>.npy
<metadata>/<split>/<dataset>_scene_list_<split>.txt
```

It also writes hFOV summaries:

```text
<metadata>/<split>/<dataset>_scene_hfov_<split>.json
```
