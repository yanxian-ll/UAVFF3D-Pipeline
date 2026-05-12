# Synthetic Generation

This directory contains the renderers for the synthetic components of
A3D-Bench.

## Scripts

- `a3d_syn_l_tile_renderer.py`  
  Large-scene renderer for A3D-Syn-L. It scans OBJ tiles, groups them manually,
  merges each selected group, plans a lawnmower route, and renders nadir or
  five-camera UAV observations.

- `a3d_syn_s_view_selector_renderer.py`  
  Interactive Open3D viewpoint selector for A3D-Syn-S. It is useful for smaller,
  irregular, or locally interesting synthetic assets where manual viewpoint
  selection is preferable to a strict grid route.

- `a3d_fa_view_selector_renderer.py`  
  Controlled hFOV--height renderer for A3D-FA. It reuses a reference pose or
  trajectory and renders multiple target hFOV settings while approximately
  preserving image footprint.

## Output Layout

The renderers write the common scene layout:

```text
scene/
  images/
  depth/
  cams/
  meta/
```

Run `uav_data_processing/generate_scene_meta.py` after rendering if the scene
will be consumed by the unified dataset loaders.

