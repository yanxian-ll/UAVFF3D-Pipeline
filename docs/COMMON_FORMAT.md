# Common A3D Scene Format

This format is the bridge between synthetic rendering, LiDAR-grounded real
scenes, external dataset conversion, training samplers, and evaluation scripts.

## Required Files

```text
<dataset>/<scene>/
  images/<frame>.png|jpg|jpeg
  cams/<frame>.txt
  depth/<frame>.exr
  scene_meta.json
```

Frame stems must match across modalities. For example, `00000012.png`,
`00000012.txt`, and `00000012.exr` describe the same view.

## Camera Text

```text
extrinsic opencv(x Right, y Down, z Forward) world2camera
4x4 matrix

intrinsic: fx fy cx cy (pixel)
3x3 matrix

h w fov
height width horizontal_or_vertical_fov
```

Most converters write horizontal FOV. Some legacy renderers wrote vertical FOV;
downstream geometry uses the full intrinsic matrix, so the matrix values are the
authoritative source.

## Scene Metadata

`scene_meta.json` stores:

- `scene_name`, `dataset_name`, and `version`
- `camera_convention: opencv`
- one frame record per stem
- `transform_matrix` as camera-to-world
- per-frame intrinsics and image size
- modality declarations in `frame_modalities`

Generate or refresh it with:

```bash
cd uav_data_processing
python generate_scene_meta.py --root_dir ../data --dataset A3D-Real A3D-Syn-L A3D-Syn-S --overwrite
```

