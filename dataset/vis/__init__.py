"""Dataset loaders for UAVFF3D and related UAV/MVS datasets.

The loaders are imported lazily so that importing ``dataset.vis`` does not
force heavy optional dependencies until a concrete loader is requested.
"""

from importlib import import_module

__all__ = [
    "UAVFF3DRealWAI",
    "UAVFF3DScenesWAI",
    "UAVFF3DSynLargeWAI",
    "UAVFF3DSynSmallWAI",
]

_LOADER_MODULES = {
    "UAVFF3DRealWAI": "dataset.vis.uavff3d_real",
    "UAVFF3DScenesWAI": "dataset.vis.uavff3d_scenes",
    "UAVFF3DSynLargeWAI": "dataset.vis.uavff3d_syn_large",
    "UAVFF3DSynSmallWAI": "dataset.vis.uavff3d_syn_small",
}


def __getattr__(name: str):
    if name not in _LOADER_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_LOADER_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
