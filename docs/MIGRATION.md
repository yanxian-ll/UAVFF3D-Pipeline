# Migration notes

This codebase now uses the UAVFF3D paper naming consistently across dataset folders, Python modules, class names, CLI examples, and metadata filenames.

## Current dataset loaders

| Module | Class |
| --- | --- |
| `dataset.vis.uavff3d_real` | `UAVFF3DRealWAI` |
| `dataset.vis.uavff3d_syn_large` | `UAVFF3DSynLargeWAI` |
| `dataset.vis.uavff3d_syn_small` | `UAVFF3DSynSmallWAI` |
| `dataset.vis.uavff3d_scenes` | `UAVFF3DScenesWAI` |

## Current synthetic scripts

- `uavff3d_syn_large_tile_renderer.py`
- `uavff3d_syn_small_view_selector_renderer.py`
- `uavff3d_fa_view_selector_renderer.py`

## Current dataset folder names

Use these dataset names in commands and metadata filenames:

- `UAVFF3D-Real`
- `UAVFF3D-Syn-L`
- `UAVFF3D-Syn-S`
- `UAVFF3D-FA`
