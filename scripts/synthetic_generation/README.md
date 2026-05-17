# Synthetic Generation

This directory contains the renderers for the synthetic components of
UAVFF3D.

## Scripts

- `uavff3d_syn_large_tile_renderer.py`  
  Large-scene renderer for UAVFF3D-Syn-L. It scans OBJ tiles, groups them manually,
  merges each selected group, plans a lawnmower route, and renders nadir or
  five-camera UAV observations.

- `uavff3d_syn_small_view_selector_renderer.py`  
  Interactive Open3D viewpoint selector for UAVFF3D-Syn-S. It is useful for smaller,
  irregular, or locally interesting synthetic assets where manual viewpoint
  selection is preferable to a strict grid route.

- `uavff3d_fa_view_selector_renderer.py`  
  Controlled hFOV--height renderer for UAVFF3D-FA. It reuses a reference pose or
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

Run `generate_scene_meta.py` after rendering if the scene
will be consumed by the unified dataset loaders.

