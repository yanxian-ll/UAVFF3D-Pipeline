import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_flow(flow, normalize=True):
    """
    Visualize forward flow as a color-coded RGB image (similar to optical flow visualization).

    Args:
        flow: torch.Tensor or numpy array
              shape [H, W, 2] or [B, H, W, 2]
        normalize: bool
              whether to normalize magnitude for better visibility

    Returns:
        vis: numpy array, shape [H, W, 3] or [B, H, W, 3], uint8
    """

    # Convert to numpy
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()

    single = False
    if flow.ndim == 3:  # [H,W,2]
        flow = flow[None]  # → [1,H,W,2]
        single = True

    B, H, W, _ = flow.shape

    # Allocate output
    vis_list = []

    for b in range(B):
        u = flow[b, :, :, 0]
        v = flow[b, :, :, 1]

        # compute angle and magnitude
        angle = np.arctan2(v, u)  # [-pi, pi]
        angle = (angle + np.pi) / (2 * np.pi)  # normalize to [0,1]

        magnitude = np.sqrt(u*u + v*v)
        if normalize:
            mag_norm = magnitude / (magnitude.max() + 1e-6)
        else:
            mag_norm = magnitude / (np.percentile(magnitude, 95) + 1e-6)
            mag_norm = np.clip(mag_norm, 0, 1)

        # HSV map
        hsv = np.zeros((H, W, 3), dtype=np.float32)
        hsv[:, :, 0] = angle         # hue = motion direction
        hsv[:, :, 1] = 1.0           # full saturation
        hsv[:, :, 2] = mag_norm      # value = magnitude

        # convert hsv → rgb
        rgb = hsv_to_rgb(hsv)
        vis_list.append((rgb * 255).astype(np.uint8))

    vis = np.stack(vis_list)

    return vis[0] if single else vis


def hsv_to_rgb(hsv):
    """Convert HSV image to RGB. hsv ∈ [0,1]"""
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    i = np.floor(h * 6).astype(np.int32)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    i = i % 6

    rgb = np.zeros_like(hsv)

    mask = (i == 0)
    rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=-1)

    mask = (i == 1)
    rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=-1)

    mask = (i == 2)
    rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=-1)

    mask = (i == 3)
    rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=-1)

    mask = (i == 4)
    rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=-1)

    mask = (i == 5)
    rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=-1)

    return rgb
