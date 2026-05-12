# UAV Data Processing

This directory converts A3D and external UAV/MVS datasets into the common
A3D/WAI scene format used by training and evaluation.

## Main Entry Points

- `generate_scene_meta.py`  
  Scans processed scenes with `images/`, `cams/`, and `depth/`, then writes
  `scene_meta.json`.

- `covisibility.py`  
  Builds sparse view-covisibility graphs from depth reprojection consistency.

- `split_scene.py`  
  Creates dataset split files and optional hFOV summaries.

- `scripts/convert/`  
  Dataset-specific converters and reorganizers for COLMAP, BlendedMVS, UseGeo,
  UAVScenes, WHU/WHU-OMVS, LuoJia-MVS, ENRICH, Ortholoc, and related sources.

- `scripts/download/`  
  Download helpers for public datasets where supported.

- `dataset/`  
  Dataset loader, geometry, I/O, camera, and visualization utilities.

## Minimal Workflow

```bash
python scripts/convert/<dataset_converter>.py --help
python generate_scene_meta.py --root_dir ../data --dataset A3D-Real A3D-Syn-L A3D-Syn-S --overwrite
python covisibility.py --config configs/covisibility_config.yaml --root ../data/A3D-Syn-L
python split_scene.py --root ../data --metadata split_outputs --dataset A3D-Syn-L
```

Many converters expect dataset-specific raw layouts, so check `--help` for the
exact input arguments before running a converter.

