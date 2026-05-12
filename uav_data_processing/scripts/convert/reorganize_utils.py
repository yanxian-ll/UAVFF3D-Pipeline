"""Shared camera-format helpers for dataset reorganization scripts."""

import math

import numpy as np


def compute_hfov_deg(width_px: int, fx: float) -> float:
    """Compute horizontal field of view in degrees from image width and fx."""
    return float(2.0 * math.degrees(math.atan(width_px / (2.0 * fx + 1e-12))))


def save_cam_txt(
    out_txt_path: str,
    T_cam2world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    H: int,
    W: int,
):
    """Write one camera file in the common A3D OpenCV camera convention."""
    T_world2cam = np.linalg.inv(T_cam2world).astype(np.float64)

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    fov = compute_hfov_deg(W, fx)

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("extrinsic opencv(x Right, y Down, z Forward) world2camera\n")
        for row in range(4):
            f.write(
                f"{T_world2cam[row,0]:.12f} {T_world2cam[row,1]:.12f} "
                f"{T_world2cam[row,2]:.12f} {T_world2cam[row,3]:.12f}\n"
            )
        f.write("\n")
        f.write("intrinsic: fx fy cx cy (pixel)\n")
        for row in range(3):
            f.write(f"{K[row,0]:.12f} {K[row,1]:.12f} {K[row,2]:.12f}\n")
        f.write("\n")
        f.write("h w fov\n")
        f.write(f"{H} {W} {fov:.12f}\n")


def cam2world_whu_to_opencv(Twc):
    """Convert WHU camera-to-world matrices to the OpenCV camera convention."""
    S = np.eye(4, dtype=np.float32)
    S[1, 1] = -1.0
    S[2, 2] = -1.0
    return Twc @ S
