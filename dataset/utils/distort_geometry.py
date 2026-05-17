import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import random


def generate_distort_params(m, distortion_strength, rng,
                            w, h, fx, fy, cx, cy):
    models = ["brown", "division", "fisheye", "affine"]

    if m is None:
        m = models[int(rng.integers(0, len(models)))]

    if m == "brown":
        params = {
            "k1": float(rng.uniform(-0.2, -0.01)),
            "k2": float(rng.uniform(0.01, 0.05)),
            "k3": float(rng.uniform(-0.01, 0.01)),
            "p1": float(rng.uniform(-0.001, 0.001)),
            "p2": float(rng.uniform(-0.001, 0.001))
        }
    elif m == "division":
        # λ 量级要很小（像素坐标 r^2 很大）
        params = {
            "lambda": float(rng.uniform(-8e-8, 8e-8)),
            "x0": float(cx + rng.uniform(-0.05*w, 0.05*w)),
            "y0": float(cy + rng.uniform(-0.05*h, 0.05*h)),
        }
    elif m == "fisheye":
        params = {
            "fisheye_type": rng.choice(["equidistant", "equisolid", "stereographic", "orthographic"]),
            "alpha": float(rng.uniform(0.8, 1.3)),  # 强度
        }
    elif m == "affine":
        params = {
            "rot_deg": float(rng.uniform(-8, 8)),
            "scale": float(rng.uniform(0.95, 1.05)),
            "shear_x": float(rng.uniform(-0.03, 0.03)),
            "shear_y": float(rng.uniform(-0.03, 0.03)),
        }
    return m, params

# ----------------------------
# # 畸变映射：返回 grid / x_map / y_map
# # ----------------------------
# @torch.no_grad()
# def apply_lens_distortion_approximate_return_map(
#     image: torch.Tensor,                 # (1,3,H,W)
#     mask: Optional[torch.Tensor],         # (H,W) or (1,H,W) or (1,1,H,W)
#     intrinsics: torch.Tensor,             # (3,3) or (1,3,3)
#     params: Dict,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

#     assert image.dim() == 4 and image.shape[0] == 1, "image must be (1,3,H,W)"
#     _, _, h, w = image.shape
#     device, dtype = image.device, image.dtype

#     yy, xx = torch.meshgrid(
#         torch.arange(h, device=device, dtype=dtype),
#         torch.arange(w, device=device, dtype=dtype),
#         indexing="ij",
#     )

#     if intrinsics.dim() == 3:
#         intrinsics = intrinsics[0]
#     fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]

#     k1 = torch.tensor(params["k1"], device=device, dtype=dtype)
#     k2 = torch.tensor(params["k2"], device=device, dtype=dtype)
#     k3 = torch.tensor(params["k3"], device=device, dtype=dtype)
#     p1 = torch.tensor(params["p1"], device=device, dtype=dtype)
#     p2 = torch.tensor(params["p2"], device=device, dtype=dtype)

#     x_dist_norm = (xx - cx) / fx
#     y_dist_norm = (yy - cy) / fy

#     x_norm = x_dist_norm
#     y_norm = y_dist_norm

#     r2 = x_norm * x_norm + y_norm * y_norm
#     r4 = r2 * r2
#     r6 = r4 * r2

#     x_dist_norm = x_dist_norm - 2.0 * p1 * x_norm * y_norm - p2 * (r2 + 2.0 * x_norm * x_norm)
#     y_dist_norm = y_dist_norm - p1 * (r2 + 2.0 * y_norm * y_norm) - 2.0 * p2 * x_norm * y_norm

#     radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
#     x_und_norm = x_dist_norm / radial
#     y_und_norm = y_dist_norm / radial

#     x_map = x_und_norm * fx + cx
#     y_map = y_und_norm * fy + cy

#     dx = x_map - xx
#     dy = y_map - yy
#     flow = torch.stack([dx, dy], dim=0)

#     denom_w = float(max(w - 1, 1))
#     denom_h = float(max(h - 1, 1))
#     grid_x = (x_map / denom_w) * 2.0 - 1.0
#     grid_y = (y_map / denom_h) * 2.0 - 1.0
#     grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1,H,W,2)

#     distorted = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

#     # mask 兼容多种形状
#     if mask is None:
#         mask4 = torch.ones((1, 1, h, w), device=device, dtype=dtype)
#     else:
#         if mask.dim() == 2:
#             mask4 = mask[None, None].to(device=device, dtype=dtype)
#         elif mask.dim() == 3:
#             mask4 = mask[None].to(device=device, dtype=dtype)  # (1,1,H,W) if (1,H,W)
#         elif mask.dim() == 4:
#             mask4 = mask.to(device=device, dtype=dtype)
#         else:
#             raise ValueError(f"mask shape not supported: {mask.shape}")

#     mask_f = F.grid_sample(mask4, grid, mode="nearest", padding_mode="zeros", align_corners=True)
#     distorted_mask = (mask_f[0, 0] > 0.5).to(torch.uint8) * 255
#     return distorted, distorted_mask, flow, x_map, y_map, grid



# ----------------------------
# (xd,yd) -> (xu,yu) 的多模型映射
# 输出 xu,yu 都是“无畸变域像素坐标”（用于采样原图）
# ----------------------------
def _map_brown_conrady_dist2und(
    xx: torch.Tensor, yy: torch.Tensor, K: torch.Tensor, params: Dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 你的原实现（Brown-Conrady：k1,k2,k3,p1,p2）
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    dtype, device = xx.dtype, xx.device

    k1 = torch.tensor(params.get("k1", 0.0), device=device, dtype=dtype)
    k2 = torch.tensor(params.get("k2", 0.0), device=device, dtype=dtype)
    k3 = torch.tensor(params.get("k3", 0.0), device=device, dtype=dtype)
    p1 = torch.tensor(params.get("p1", 0.0), device=device, dtype=dtype)
    p2 = torch.tensor(params.get("p2", 0.0), device=device, dtype=dtype)

    x_dist_norm = (xx - cx) / fx
    y_dist_norm = (yy - cy) / fy

    x_norm = x_dist_norm
    y_norm = y_dist_norm

    r2 = x_norm * x_norm + y_norm * y_norm
    r4 = r2 * r2
    r6 = r4 * r2

    # tangential compensation (same as your code)
    x_dist_norm = x_dist_norm - 2.0 * p1 * x_norm * y_norm - p2 * (r2 + 2.0 * x_norm * x_norm)
    y_dist_norm = y_dist_norm - p1 * (r2 + 2.0 * y_norm * y_norm) - 2.0 * p2 * x_norm * y_norm

    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    x_und_norm = x_dist_norm / radial
    y_und_norm = y_dist_norm / radial

    xu = x_und_norm * fx + cx
    yu = y_und_norm * fy + cy
    return xu, yu

def _map_division_dist2und(
    xx: torch.Tensor, yy: torch.Tensor, K: torch.Tensor, params: Dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 论文代码那种：xu = (xd-x0)/(1+λ r^2)+x0
    # 注意：这里 r^2 用像素坐标平方，所以 λ 的量级会很小（e-8 ~ e-10 级别常见）
    dtype, device = xx.dtype, xx.device

    lam = torch.tensor(params.get("lambda", 0.0), device=device, dtype=dtype)
    x0 = torch.tensor(params.get("x0", float(K[0,2])), device=device, dtype=dtype)
    y0 = torch.tensor(params.get("y0", float(K[1,2])), device=device, dtype=dtype)

    dx = xx - x0
    dy = yy - y0
    r2 = dx * dx + dy * dy
    coeff = 1.0 + lam * r2

    # 避免 coeff=0
    coeff = torch.where(coeff.abs() < 1e-8, coeff.sign() * 1e-8, coeff)
    xu = dx / coeff + x0
    yu = dy / coeff + y0
    return xu, yu

def _map_fisheye_dist2und(
    xx: torch.Tensor, yy: torch.Tensor, K: torch.Tensor, params: Dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    fisheye_type:
      - 'equidistant'   : r_d = theta
      - 'equisolid'     : r_d = 2 sin(theta/2)
      - 'stereographic' : r_d = 2 tan(theta/2)
      - 'orthographic'  : r_d = sin(theta)

    输出：把“鱼眼畸变域(以 fisheye 模型投影得到的像素)”反投到“针孔无畸变域(透视投影)”
    """
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    dtype, device = xx.dtype, xx.device

    fisheye_type = params.get("fisheye_type", "equidistant")
    # 强度缩放：相当于改变 FOV/投影程度，>1 更强畸变
    alpha = torch.tensor(params.get("alpha", 1.0), device=device, dtype=dtype)

    xdn = (xx - cx) / fx
    ydn = (yy - cy) / fy
    r = torch.sqrt(xdn * xdn + ydn * ydn).clamp_min(1e-8)   # distorted radius (normalized)
    rd = (r * alpha).clamp_max(1.999)  # 防止 asin/tan 爆掉

    # 从 rd 反解 theta
    if fisheye_type == "equidistant":
        theta = rd
    elif fisheye_type == "equisolid":
        theta = 2.0 * torch.asin((rd / 2.0).clamp_max(0.999999))
    elif fisheye_type == "stereographic":
        theta = 2.0 * torch.atan(rd / 2.0)
    elif fisheye_type == "orthographic":
        theta = torch.asin(rd.clamp_max(0.999999))
    else:
        raise ValueError(f"Unknown fisheye_type: {fisheye_type}")

    # 无畸变针孔：r_u = tan(theta)
    ru = torch.tan(theta).clamp_max(1e6)
    scale = ru / r
    xun = xdn * scale
    yun = ydn * scale

    xu = xun * fx + cx
    yu = yun * fy + cy
    return xu, yu

def _map_affine_dist2und(
    xx: torch.Tensor, yy: torch.Tensor, K: torch.Tensor, params: Dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    非镜头畸变，但很实用的增强：仿射（旋转/缩放/剪切）
    params:
      - rot_deg, scale, shear_x, shear_y
    """
    dtype, device = xx.dtype, xx.device
    cx, cy = K[0,2], K[1,2]

    rot = torch.tensor(params.get("rot_deg", 0.0), device=device, dtype=dtype) * (3.14159265 / 180.0)
    s   = torch.tensor(params.get("scale", 1.0), device=device, dtype=dtype)
    shx = torch.tensor(params.get("shear_x", 0.0), device=device, dtype=dtype)
    shy = torch.tensor(params.get("shear_y", 0.0), device=device, dtype=dtype)

    cosr = torch.cos(rot)
    sinr = torch.sin(rot)

    # 组合矩阵：A = R * Sh * S（你也可以改顺序）
    S = torch.stack([torch.stack([s, 0.0*s]), torch.stack([0.0*s, s])])  # 2x2
    Sh = torch.stack([torch.stack([1.0 + 0.0*shx, shx]), torch.stack([shy, 1.0 + 0.0*shy])])
    R = torch.stack([torch.stack([cosr, -sinr]), torch.stack([sinr, cosr])])

    A = R @ (Sh @ S)  # 2x2
    Ainv = torch.inverse(A)

    dx = xx - cx
    dy = yy - cy
    vec = torch.stack([dx, dy], dim=0)               # (2,H,W)
    vec_u = (Ainv @ vec.flatten(1)).view_as(vec)     # (2,H,W)

    xu = vec_u[0] + cx
    yu = vec_u[1] + cy
    return xu, yu

# ----------------------------
# 通用畸变函数：复用 grid_sample + flow + mask
# model: 'brown' | 'division' | 'fisheye' | 'affine'
# ----------------------------
@torch.no_grad()
def apply_distortion_return_map(
    image: torch.Tensor,                 # (1,3,H,W)
    mask: Optional[torch.Tensor],         # (H,W) or (1,H,W) or (1,1,H,W)
    intrinsics: torch.Tensor,             # (3,3) or (1,3,3)
    params: Dict,
    model: str = "brown",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert image.dim() == 4 and image.shape[0] == 1, "image must be (1,3,H,W)"
    _, _, h, w = image.shape
    device, dtype = image.device, image.dtype

    if intrinsics.dim() == 3:
        K = intrinsics[0].to(device=device, dtype=dtype)
    else:
        K = intrinsics.to(device=device, dtype=dtype)

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )

    # (xd,yd)->(xu,yu)
    if model == "brown":
        x_map, y_map = _map_brown_conrady_dist2und(xx, yy, K, params)
    elif model == "division":
        x_map, y_map = _map_division_dist2und(xx, yy, K, params)
    elif model == "fisheye":
        x_map, y_map = _map_fisheye_dist2und(xx, yy, K, params)
    elif model == "affine":
        x_map, y_map = _map_affine_dist2und(xx, yy, K, params)
    else:
        raise ValueError(f"Unknown model: {model}")

    # forward flow: distorted -> undistorted
    dx = x_map - xx
    dy = y_map - yy
    flow = torch.stack([dx, dy], dim=0)  # (2,H,W)

    # grid for sampling source (undist image)
    denom_w = float(max(w - 1, 1))
    denom_h = float(max(h - 1, 1))
    grid_x = (x_map / denom_w) * 2.0 - 1.0
    grid_y = (y_map / denom_h) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1,H,W,2)

    distorted = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    if mask is None:
        mask4 = torch.ones((1, 1, h, w), device=device, dtype=dtype)
    else:
        if mask.dim() == 2:
            mask4 = mask[None, None].to(device=device, dtype=dtype)
        elif mask.dim() == 3:
            mask4 = mask[None].to(device=device, dtype=dtype)
        elif mask.dim() == 4:
            mask4 = mask.to(device=device, dtype=dtype)
        else:
            raise ValueError(f"mask shape not supported: {mask.shape}")

    mask_f = F.grid_sample(mask4, grid, mode="nearest", padding_mode="zeros", align_corners=True)
    distorted_mask = (mask_f[0, 0] > 0.5).to(torch.uint8) * 255
    return distorted, distorted_mask, flow, x_map, y_map, grid

@torch.no_grad()
def rectify_from_forward_flow(
    distorted: torch.Tensor,          # (B,C,H,W) or (1,3,H,W)
    flow: torch.Tensor,               # (B,2,H,W) or (2,H,W)
    max_iter: int = 100,
    precision: float = 1e-2,
    fill_value: Optional[float] = None,   # invalid 区域填充值；None 时 float->1.0, uint8->255
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
      rectified: (B,C,H,W)  去畸变图
      invalid_mask: (B,H,W) uint8, invalid=255, valid=0   （与论文代码 resultMsk 对齐）
    """

    # ---------- shape 统一 ----------
    squeeze_batch = False
    if distorted.dim() == 3:  # (C,H,W)
        distorted = distorted.unsqueeze(0)
        squeeze_batch = True
    elif distorted.dim() == 2:  # (H,W)
        distorted = distorted.unsqueeze(0).unsqueeze(0)
        squeeze_batch = True
    assert distorted.dim() == 4, "distorted must be (B,C,H,W) or (C,H,W) or (H,W)"

    if flow.dim() == 3:  # (2,H,W)
        flow = flow.unsqueeze(0)
    assert flow.dim() == 4 and flow.size(1) == 2, "flow must be (B,2,H,W) or (2,H,W)"

    B, C, H, W = distorted.shape
    assert flow.shape[0] == B and flow.shape[2] == H and flow.shape[3] == W, "flow size must match image"

    device = distorted.device

    # ---------- dtype 处理 ----------
    orig_dtype = distorted.dtype
    if fill_value is None:
        fill_value = 255.0 if orig_dtype == torch.uint8 else 1.0

    distorted_f = distorted.float() if orig_dtype != torch.float32 else distorted
    flow_f = flow.float() if flow.dtype != torch.float32 else flow

    # ---------- padding（对齐论文的 H+1, W+1 replicate padding） ----------
    # 右+1、下+1，边界复制
    pad_img = F.pad(distorted_f, (0, 1, 0, 1), mode="replicate")   # (B,C,H+1,W+1)
    pad_flow = F.pad(flow_f,      (0, 1, 0, 1), mode="replicate") # (B,2,H+1,W+1)

    # 取出整像素格点上的 u,v（仅 H,W 区域，和论文 padu[0:H,0:W] 一样）
    u_ref = pad_flow[:, 0, :H, :W]  # (B,H,W)
    v_ref = pad_flow[:, 1, :H, :W]

    # ---------- 目标像素网格 (xr,yr) ----------
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xx = xx.unsqueeze(0).expand(B, -1, -1)  # (B,H,W)
    yy = yy.unsqueeze(0).expand(B, -1, -1)

    # ---------- 论文里的初始化：u_next = u/(1+dif), v_next = v/(1+dif) ----------
    difx = torch.zeros_like(u_ref)
    dify = torch.zeros_like(v_ref)

    if W > 1:
        difx[:, :, :-1] = u_ref[:, :, 1:] - u_ref[:, :, :-1]
        difx[:, :, -1]  = u_ref[:, :, -1] - u_ref[:, :, -2]
    # W==1 时 difx 维持 0

    if H > 1:
        dify[:, :-1, :] = v_ref[:, 1:, :] - v_ref[:, :-1, :]
        dify[:, -1,  :] = v_ref[:, -1, :] - v_ref[:, -2, :]
    # H==1 时 dify 维持 0

    # 避免 (1+dif) 过小导致数值炸裂
    def safe_div(n, d, eps=1e-6):
        d2 = torch.where(d.abs() < eps, d + (eps * torch.sign(d).clamp(min=1.0)), d)
        return n / d2

    u_init = safe_div(u_ref, 1.0 + difx)
    v_init = safe_div(v_ref, 1.0 + dify)

    # 如果该像素 flow 本来就很小，直接返回 (xr,yr)
    done = (u_ref.abs() < precision) & (v_ref.abs() < precision)

    i = xx - u_init
    j = yy - v_init
    i = torch.where(done, xx, i)
    j = torch.where(done, yy, j)

    # valid 域：i/j 必须落在 [0, W-1]、[0, H-1]
    valid = (i >= 0.0) & (i <= (W - 1)) & (j >= 0.0) & (j <= (H - 1))
    done = done | (~valid)   # invalid 直接结束

    # ---------- 迭代反解：i_next = xr - u(i,j), j_next = yr - v(i,j) ----------
    # 注意：grid_sample 的输入是 pad_flow (H+1,W+1)，align_corners=True
    # 所以归一化分母是 (Wp-1)=W、(Hp-1)=H
    denom_w = float(W)
    denom_h = float(H)

    for _ in range(max_iter):
        if bool(done.all()):
            break

        grid_x = (i / denom_w) * 2.0 - 1.0
        grid_y = (j / denom_h) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (B,H,W,2)

        uv = F.grid_sample(
            pad_flow, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (B,2,H,W)
        u = uv[:, 0]
        v = uv[:, 1]

        i_next = xx - u
        j_next = yy - v

        # 越界直接标 invalid
        in_bounds = (i_next >= 0.0) & (i_next <= (W - 1)) & (j_next >= 0.0) & (j_next <= (H - 1))
        newly_invalid = (~done) & (~in_bounds)

        # 收敛判定：论文里是“如果 abs(i-i_next)<precision 则返回当前 i,j（不更新）”
        conv = (~done) & in_bounds & ((i - i_next).abs() < precision) & ((j - j_next).abs() < precision)

        # 仅更新那些：还没 done、且没收敛、且在界内
        upd = (~done) & in_bounds & (~conv)
        i = torch.where(upd, i_next, i)
        j = torch.where(upd, j_next, j)

        # 更新 done/valid
        done = done | conv | newly_invalid
        valid = valid & (~newly_invalid)

    # ---------- 最终用 (i,j) 从畸变图采样，得到 rectified ----------
    grid_x = (i / denom_w) * 2.0 - 1.0
    grid_y = (j / denom_h) * 2.0 - 1.0
    grid_final = torch.stack([grid_x, grid_y], dim=-1)  # (B,H,W,2)

    rectified = F.grid_sample(
        pad_img, grid_final,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (B,C,H,W)

    # invalid 区域填白（对齐论文 resultImg.fill(255) 的语义）
    if isinstance(fill_value, (int, float)):
        fill = torch.full_like(rectified, float(fill_value))
        rectified = torch.where(valid.unsqueeze(1), rectified, fill)

    invalid_mask = (~valid).to(torch.uint8) * 255  # (B,H,W)

    # ---------- 恢复 dtype / shape ----------
    if orig_dtype == torch.uint8:
        rectified = rectified.clamp(0, 255).round().to(torch.uint8)
    elif orig_dtype != torch.float32:
        rectified = rectified.to(orig_dtype)

    if squeeze_batch:
        rectified = rectified.squeeze(0)
        invalid_mask = invalid_mask.squeeze(0)
    return rectified, invalid_mask

# ----------------------------
# 无畸变域：由 depth_z + K 计算 pts_cam_undist
# ----------------------------
@torch.no_grad()
def build_pts_cam_from_depth_z(
    depth_z_undist: torch.Tensor,  # (H,W) or (1,1,H,W), z-depth
    K: torch.Tensor,               # (3,3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    returns:
      pts_cam_undist: (H,W,3)
      valid_mask_u : (H,W) bool  depth_z>0
    """
    if depth_z_undist.dim() == 2:
        depth = depth_z_undist[None, None]  # (1,1,H,W)
    else:
        depth = depth_z_undist
    device, dtype = depth.device, depth.dtype
    _, _, H, W = depth.shape

    fu = K[0, 0].to(device=device, dtype=dtype)
    fv = K[1, 1].to(device=device, dtype=dtype)
    cu = K[0, 2].to(device=device, dtype=dtype)
    cv = K[1, 2].to(device=device, dtype=dtype)

    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )

    x_cam = (u - cu) / fu
    y_cam = (v - cv) / fv
    z_cam = torch.ones_like(x_cam)

    ray_dirs_on_unit_plane = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (H,W,3)

    depth_hw = depth[0, 0]  # (H,W)
    pts_cam = depth_hw[..., None] * ray_dirs_on_unit_plane  # (H,W,3)

    valid_mask_u = depth_hw > 0
    return pts_cam, valid_mask_u


# ----------------------------
# warp pts_cam：直接 grid_sample(pts_cam)
# ----------------------------
@torch.no_grad()
def warp_pts_cam_to_distorted(
    pts_cam_undist: torch.Tensor,   # (H,W,3)
    valid_mask_u: torch.Tensor,     # (H,W) bool
    grid: torch.Tensor,             # (1,H,W,2) distorted->undist
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    returns:
      pts_cam_distorted: (H,W,3)
      valid_mask_d     : (H,W) bool
    """
    device = grid.device
    dtype = grid.dtype

    H, W = pts_cam_undist.shape[:2]

    # (H,W,3) -> (1,3,H,W)
    pts_u = pts_cam_undist.to(device=device, dtype=dtype).permute(2, 0, 1).unsqueeze(0)

    # 直接 warp pts_cam（用 bilinear 足够）
    pts_d = F.grid_sample(
        pts_u, grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0]  # (3,H,W)

    pts_cam_d = pts_d.permute(1, 2, 0).contiguous()  # (H,W,3)

    # warp valid mask（建议 nearest，避免边界被平滑）
    mask_u = valid_mask_u.to(device=device, dtype=dtype)[None, None]  # (1,1,H,W)
    mask_d = F.grid_sample(
        mask_u, grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]  # (H,W)
    valid_mask_d = mask_d > 0.5
    return pts_cam_d, valid_mask_d


# ----------------------------
# 由 pts_cam 推 ray / depth_along_ray，并可转世界系
# ----------------------------
@torch.no_grad()
def pts_cam_to_rays_and_world(
    pts_cam: torch.Tensor,                 # (H,W,3) in camera coordinates
    valid_mask: torch.Tensor,              # (H,W) bool
    cam2world: torch.Tensor,               # (4,4) 
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    输出对齐原 numpy 函数：
      pts_world (H,W,3),
      valid_mask (H,W),
      ray_origins_world (H,W,3),
      ray_directions_world (H,W,3),
      depth_along_ray (H,W,1),
      ray_directions_cam (H,W,3),
      pts_cam (H,W,3)
    """
    device, dtype = pts_cam.device, pts_cam.dtype
    H, W = pts_cam.shape[:2]

    depth_along_ray = torch.linalg.norm(pts_cam, dim=-1, keepdims=True)  # (H,W,1)
    ray_directions_cam = pts_cam / depth_along_ray.clamp_min(1e-8)       # (H,W,3)

    # 默认世界系=相机系
    pts_world = pts_cam
    ray_dirs_world = ray_directions_cam
    ray_origins_world = torch.zeros((H, W, 3), device=device, dtype=dtype)

    R = cam2world[:3, :3].to(device=device, dtype=dtype)
    t = cam2world[:3, 3].to(device=device, dtype=dtype)

    # 点从相机系到世界系：Xw = R * Xc + t
    pts_world = pts_cam @ R.T + t.view(1, 1, 3)

    # 射线方向：dw = R * dc
    ray_dirs_world = ray_directions_cam @ R.T

    # 射线原点：所有像素同一光心（相机中心）= t
    ray_origins_world = t.view(1, 1, 3).expand(H, W, 3)

    return (
        pts_world,              # (H,W,3)
        valid_mask,             # (H,W)
        ray_origins_world,      # (H,W,3)
        ray_dirs_world,         # (H,W,3)
        depth_along_ray,        # (H,W,1)
        ray_directions_cam,     # (H,W,3)
        pts_cam,                # (H,W,3)
    )


# ----------------------------
# 给畸变图像的 per-pixel pts/rays (warp pts_cam 方案)
# ----------------------------
@torch.no_grad()
def get_absolute_pointmaps_and_rays_info_for_distorted_by_warp_pts(
    depth_z_undist: torch.Tensor,    # (H,W) or (1,1,H,W)  原图无畸变深度(z-depth)
    K: torch.Tensor,                 # (3,3)
    cam2world: Optional[torch.Tensor],
    grid: torch.Tensor,              # (1,H,W,2) 来自畸变函数
):
    # 1) 无畸变域算 pts_cam_u
    pts_cam_u, valid_u = build_pts_cam_from_depth_z(depth_z_undist, K)

    # 2) 直接 warp pts_cam -> 畸变域
    pts_cam_d, valid_d = warp_pts_cam_to_distorted(pts_cam_u, valid_u, grid)

    # 3) 从 pts_cam_d 推 ray/depth_along_ray，并转世界系
    return pts_cam_to_rays_and_world(pts_cam_d, valid_d, cam2world)


if __name__ == "__main__":
    import os
    import cv2
    import numpy as np
    
    # ----------------------------
    # 0) 路径 & 设备
    # ----------------------------
    output_path = "./outputs/simulate_distort/distort_test"
    os.makedirs(output_path, exist_ok=True)
    
    undistort_image_path = "./examples/simulate_distort/undistort/d.jpeg"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ----------------------------
    # 1) 读取原图（无畸变域）
    # ----------------------------
    img_bgr = cv2.imread(undistort_image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {undistort_image_path}")
    
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # uint8 -> float [0,1], (1,3,H,W)
    img_t = torch.from_numpy(img_rgb).to(device=device).float() / 255.0
    img_t = img_t.permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,3,H,W)
    
    # 全 1 mask（可选）
    mask = torch.ones((h, w), device=device, dtype=torch.float32)
    
    # ----------------------------
    # 2) 构造内参 K
    # ----------------------------
    fx = 3713.0
    fy = 3713.0
    cx = w / 2.0
    cy = h / 2.0
    
    K = torch.tensor(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32
    )
    
    # 设置随机种子以保证可重复性
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    
    # 所有模型列表
    models = ["brown", "division", "fisheye", "affine"]
    
    for model_type in models:
        print(f"\n=== 测试模型: {model_type} ===")
        
        # 为每种模型创建子目录
        model_output_path = os.path.join(output_path, model_type)
        os.makedirs(model_output_path, exist_ok=True)
        
        # 生成该模型的畸变参数
        m, params = generate_distort_params(model_type, distortion_strength=1.0, rng=rng)
        print(f"模型参数: {params}")
        
        # 应用畸变
        distorted_t, distorted_mask_u8, flow_2hw, x_map, y_map, grid = apply_distortion_return_map(
            image=img_t,
            mask=mask,
            intrinsics=K,
            params=params,
            model=m
        )
        
        # 用 forward-flow-resampling 去畸变
        rectified_t, invalid_mask_u8 = rectify_from_forward_flow(
            distorted=distorted_t,   # (1,3,H,W)
            flow=flow_2hw,           # (2,H,W) / (1,2,H,W) 都行
            max_iter=100,
            precision=1e-2,
            fill_value=1.0,          # float图填白（1.0）；uint8的话会自动用255
        )
        
        # ----------------------------
        # 保存结果
        # ----------------------------
        def to_u8_rgb(t: torch.Tensor) -> np.ndarray:
            """(1,3,H,W) or (3,H,W) float [0,1] -> uint8 RGB(H,W,3)"""
            if t.dim() == 4:
                t = t[0]
            t = t.detach().clamp(0, 1).cpu()
            img = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
            return img
        
        undist_u8 = to_u8_rgb(img_t)
        dist_u8 = to_u8_rgb(distorted_t)
        rect_u8 = to_u8_rgb(rectified_t)
        
        # 保存图（OpenCV写BGR）
        cv2.imwrite(os.path.join(model_output_path, "undist_input.png"), cv2.cvtColor(undist_u8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(model_output_path, "distorted.png"), cv2.cvtColor(dist_u8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(model_output_path, "rectified.png"), cv2.cvtColor(rect_u8, cv2.COLOR_RGB2BGR))
        
        # 保存 mask（论文 resultMsk：invalid=255）
        invalid_mask_np = invalid_mask_u8[0].cpu().numpy() if invalid_mask_u8.dim()==3 else invalid_mask_u8.cpu().numpy()
        cv2.imwrite(os.path.join(model_output_path, "invalid_mask.png"), invalid_mask_np)
        cv2.imwrite(os.path.join(model_output_path, "distorted_mask.png"), distorted_mask_u8.cpu().numpy())
        
        # 拼接对比：上=原图/畸变图，下=去畸变/差异
        diff = cv2.absdiff(rect_u8, undist_u8)
        vis = np.concatenate([undist_u8, dist_u8], axis=1)
        vis2 = np.concatenate([rect_u8, diff], axis=1)
        vis_all = np.concatenate([vis, vis2], axis=0)
        cv2.imwrite(os.path.join(model_output_path, "compare.png"), cv2.cvtColor(vis_all, cv2.COLOR_RGB2BGR))
        
        # 保存 flow.npy（对齐论文：flow[0]=u, flow[1]=v）
        flow_np = flow_2hw.detach().float().cpu().numpy()  # (2,H,W)
        np.save(os.path.join(model_output_path, "flow.npy"), flow_np)
        
        # 保存参数为文本文件
        with open(os.path.join(model_output_path, "params.txt"), "w") as f:
            f.write(f"Model: {m}\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
        
        # ----------------------------
        # 评估：只在 valid 区域算误差
        # ----------------------------
        # invalid_mask_u8: (B,H,W) uint8
        inv = invalid_mask_u8
        if inv.dim() == 2:
            inv = inv.unsqueeze(0)
        valid = (inv == 0).to(torch.bool)  # (1,H,W)
        
        und = img_t.detach().float()
        rec = rectified_t.detach().float()
        
        # valid 上的 MAE / PSNR
        valid3 = valid.unsqueeze(1).expand_as(und)  # (1,3,H,W)
        if valid3.any():
            mae = (und[valid3] - rec[valid3]).abs().mean().item()
            mse = ((und[valid3] - rec[valid3]) ** 2).mean().item()
            psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
        else:
            mae = float('inf')
            psnr = 0.0
        
        print(f"  valid MAE  : {mae:.6f}")
        print(f"  valid PSNR : {psnr:.2f} dB")
        print(f"  flow range u: [{flow_np[0].min():.3f}, {flow_np[0].max():.3f}]  v: [{flow_np[1].min():.3f}, {flow_np[1].max():.3f}]")
        
        # 保存评估结果
        with open(os.path.join(model_output_path, "evaluation.txt"), "w") as f:
            f.write(f"Valid MAE: {mae:.6f}\n")
            f.write(f"Valid PSNR: {psnr:.2f} dB\n")
            f.write(f"Flow range u: [{flow_np[0].min():.3f}, {flow_np[0].max():.3f}]\n")
            f.write(f"Flow range v: [{flow_np[1].min():.3f}, {flow_np[1].max():.3f}]\n")
    
    print(f"\n所有模型测试完成！结果保存在: {output_path}")
    