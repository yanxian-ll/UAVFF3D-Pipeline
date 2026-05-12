# drone_distortion_torch.py
import os
import argparse
import random
import warnings
from glob import glob
from typing import Tuple

import cv2
import numpy as np

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def read_camera_params(file_path):
    """
    从txt文件读取相机内参和外参矩阵
    """
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

        extrinsic = []
        for i in range(1, 5):
            extrinsic.append(list(map(float, lines[i].split())))
        extrinsic_matrix = np.array(extrinsic)

        intrinsic = []
        for i in range(5, 8):
            intrinsic.append(list(map(float, lines[i].split())))
        intrinsic_matrix = np.array(intrinsic)

    return {
        "intrinsic": intrinsic_matrix,
        "extrinsic": extrinsic_matrix,
        "fx": intrinsic_matrix[0, 0],
        "fy": intrinsic_matrix[1, 1],
        "cx": intrinsic_matrix[0, 2],
        "cy": intrinsic_matrix[1, 2],
    }


def bgr_uint8_to_torch(img_bgr: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """
    (H,W,3) uint8 BGR -> (1,3,H,W) float32 [0,1]
    """
    t = torch.from_numpy(img_bgr).to(device=device)
    t = t.permute(2, 0, 1).contiguous().to(dtype=dtype) / 255.0
    return t.unsqueeze(0)


def torch_to_bgr_uint8(img: torch.Tensor) -> np.ndarray:
    """
    (1,3,H,W) or (3,H,W) float [0,1] -> (H,W,3) uint8 BGR
    """
    if img.dim() == 4:
        img = img[0]
    img = img.clamp(0, 1) * 255.0
    img = img.to(torch.uint8).permute(1, 2, 0).contiguous()
    return img.detach().cpu().numpy()


class WideAngleDroneDistortionTorch:
    """
    PyTorch版本：使用 grid_sample 实现 remap
    - 输出 distorted_img (torch)
    - 输出 distorted_mask (torch uint8 0/255)
    - 输出 flow (torch) shape=(2,H,W), 像素位移 (dx,dy) = (x_map-xx, y_map-yy)
    """

    def __init__(self, fx=None, fy=None, cx=None, cy=None, device="cuda", dtype=torch.float32):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.device = torch.device(device)
        self.dtype = dtype

        # 桶形畸变为主（k1负）
        self.k1_range = (-0.2, -0.01)
        self.k2_range = (0.01, 0.05)
        self.k3_range = (-0.01, 0.01)

        self.p1_range = (-0.001, 0.001)
        self.p2_range = (-0.001, 0.001)

        # cache：避免反复创建网格
        self._grid_cache = {}

    def _get_meshgrid_xy(self, h: int, w: int, device: torch.device, dtype: torch.dtype):
        """
        返回 (yy, xx) 形状 (H,W)，其中：
        - xx: x坐标 0..w-1
        - yy: y坐标 0..h-1
        """
        key = (h, w, device.type, device.index, dtype)
        if key in self._grid_cache:
            return self._grid_cache[key]

        y = torch.arange(h, device=device, dtype=dtype)
        x = torch.arange(w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self._grid_cache[key] = (yy, xx)
        return yy, xx

    def generate_distortion_parameters(self, distortion_strength: float = 1.0) -> dict:
        k1 = random.uniform(*self.k1_range) * distortion_strength
        k2 = random.uniform(*self.k2_range) * distortion_strength
        k3 = random.uniform(*self.k3_range) * distortion_strength
        p1 = random.uniform(*self.p1_range) * distortion_strength
        p2 = random.uniform(*self.p2_range) * distortion_strength

        print(f"Generated distortion params: k1={k1:.6f}, k2={k2:.6f}, k3={k3:.6f}, p1={p1:.6f}, p2={p2:.6f}")

        return {
            "k1": k1, "k2": k2, "k3": k3,
            "p1": p1, "p2": p2,
            "cx": self.cx, "cy": self.cy,
            "fx": self.fx, "fy": self.fy
        }

    @torch.no_grad()
    def apply_lens_distortion_approximate(
        self, image: torch.Tensor, params: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        image: (1,3,H,W) float [0,1]
        returns:
          distorted_img: (1,3,H,W)
          distorted_mask: (H,W) uint8 {0,255}
          flow: (2,H,W) float (dx,dy)
        """
        assert image.dim() == 4 and image.shape[0] == 1, "image must be (1,3,H,W)"

        _, _, h, w = image.shape
        device = image.device
        dtype = image.dtype

        yy, xx = self._get_meshgrid_xy(h, w, device, dtype)  # (H,W)

        fx = torch.tensor(params["fx"], device=device, dtype=dtype)
        fy = torch.tensor(params["fy"], device=device, dtype=dtype)
        cx = torch.tensor(params["cx"], device=device, dtype=dtype)
        cy = torch.tensor(params["cy"], device=device, dtype=dtype)

        k1 = torch.tensor(params["k1"], device=device, dtype=dtype)
        k2 = torch.tensor(params["k2"], device=device, dtype=dtype)
        k3 = torch.tensor(params["k3"], device=device, dtype=dtype)
        p1 = torch.tensor(params["p1"], device=device, dtype=dtype)
        p2 = torch.tensor(params["p2"], device=device, dtype=dtype)

        # 归一化（把“输出像素(=畸变图坐标)”映射到输入图采样坐标）
        x_dist_norm = (xx - cx) / fx
        y_dist_norm = (yy - cy) / fy

        x_norm = x_dist_norm
        y_norm = y_dist_norm

        r2 = x_norm * x_norm + y_norm * y_norm
        r4 = r2 * r2
        r6 = r4 * r2

        # tangential “反推”近似（保持与你原代码一致）
        x_dist_norm = x_dist_norm - 2.0 * p1 * x_norm * y_norm - p2 * (r2 + 2.0 * x_norm * x_norm)
        y_dist_norm = y_dist_norm - p1 * (r2 + 2.0 * y_norm * y_norm) - 2.0 * p2 * x_norm * y_norm

        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        x_und_norm = x_dist_norm / radial
        y_und_norm = y_dist_norm / radial

        x_map = x_und_norm * fx + cx  # (H,W)
        y_map = y_und_norm * fy + cy  # (H,W)

        # flow（像素位移）
        dx = x_map - xx
        dy = y_map - yy
        flow = torch.stack([dx, dy], dim=0)  # (2,H,W)

        # grid_sample 的 grid 归一化到 [-1,1]
        # align_corners=True 时：x_norm = x/(w-1)*2-1
        denom_w = float(max(w - 1, 1))
        denom_h = float(max(h - 1, 1))
        grid_x = (x_map / denom_w) * 2.0 - 1.0
        grid_y = (y_map / denom_h) * 2.0 - 1.0

        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1,H,W,2)

        distorted = F.grid_sample(
            image, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        ones = torch.ones((1, 1, h, w), device=device, dtype=dtype)
        mask_f = F.grid_sample(
            ones, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        distorted_mask = (mask_f[0, 0] > 0.5).to(torch.uint8) * 255

        return distorted, distorted_mask, flow

    @torch.no_grad()
    def apply_vignetting(self, image: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
        """
        image: (1,3,H,W) float [0,1]
        """
        _, _, h, w = image.shape
        device = image.device
        dtype = image.dtype

        yy, xx = self._get_meshgrid_xy(h, w, device, dtype)

        center_x = (w - 1) / 2.0
        center_y = (h - 1) / 2.0
        dist = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        max_dist = torch.sqrt(torch.tensor(center_x**2 + center_y**2, device=device, dtype=dtype)).clamp(min=1e-6)
        norm_dist = dist / max_dist

        vignette = 1.0 - strength * (norm_dist ** 2)
        vignette = vignette.clamp(0.01, 1.0)  # 防止全黑
        vignette = vignette.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        return (image * vignette).clamp(0, 1)

    def apply_distortion_pipeline(self, img_bgr_uint8: np.ndarray, distortion_strength: float = 1.0, vignetting: bool = True):
        """
        输入/输出仍然用 numpy+cv2，内部用 torch
        Returns:
          distorted_img_bgr_uint8, mask_uint8, flow_np (2,H,W float32)
        """
        params = self.generate_distortion_parameters(distortion_strength)

        image_t = bgr_uint8_to_torch(img_bgr_uint8, device=self.device, dtype=self.dtype)

        print("\nApplying approximate distortion method (PyTorch)...")
        distorted_t, mask_t, flow_t = self.apply_lens_distortion_approximate(image_t, params)
        print("Done...")

        if vignetting:
            vignette_strength = random.uniform(0.1, 0.3) * distortion_strength
            distorted_t = self.apply_vignetting(distorted_t, vignette_strength)

        distorted_np = torch_to_bgr_uint8(distorted_t)
        mask_np = mask_t.detach().cpu().numpy()
        flow_np = flow_t.detach().cpu().to(torch.float32).numpy()

        return distorted_np, mask_np, flow_np


def batch_process_images(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_samples = args.num_samples
    os.makedirs(output_dir, exist_ok=True)

    image_files = (
        glob(os.path.join(input_dir, "*.jpg")) +
        glob(os.path.join(input_dir, "*.png")) +
        glob(os.path.join(input_dir, "*.jpeg")) +
        glob(os.path.join(input_dir, "*.JPG"))
    )

    if not image_files:
        print(f"在目录 {input_dir} 中未找到图像文件")
        return

    camera_params = read_camera_params(args.cam_file)

    # 读取第一张图像获取尺寸
    img0 = cv2.imread(image_files[0])
    if img0 is None:
        print("无法读取第一张图像，退出")
        return
    h, w = img0.shape[:2]

    # 你原来的 DEBUG 覆盖逻辑（保留）
    camera_params["fx"] = 3713.209473
    camera_params["fy"] = 3713.209473
    camera_params["cx"] = w / 2
    camera_params["cy"] = h / 2

    # device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    distorter = WideAngleDroneDistortionTorch(
        fx=camera_params["fx"],
        fy=camera_params["fy"],
        cx=camera_params["cx"],
        cy=camera_params["cy"],
        device=device,
        dtype=torch.float32
    )

    total_images = len(image_files) * num_samples
    processed = 0

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        for i in range(num_samples):
            strength = random.uniform(args.distortion_strength_min, args.distortion_strength_max)

            print(f"\nProcessing {os.path.basename(img_path)} - sample {i+1}/{num_samples}")
            print(f"Distortion strength: {strength:.2f}")

            distorted_img, mask, flow = distorter.apply_distortion_pipeline(
                img,
                distortion_strength=strength,
                vignetting=args.vignetting
            )

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_img = os.path.join(output_dir, f"{base_name}_distorted_{i:03d}.jpg")
            out_msk = os.path.join(output_dir, f"{base_name}_mask_{i:03d}.jpg")

            cv2.imwrite(out_img, distorted_img)
            cv2.imwrite(out_msk, mask)

            if args.save_flow:
                out_flow = os.path.join(output_dir, f"{base_name}_flow_{i:03d}.npy")
                np.save(out_flow, flow.astype(np.float32))

            processed += 1
            print(f"进度: {processed}/{total_images} - 已生成: {out_img}")


def main():
    parser = argparse.ArgumentParser(
        description="广角无人机图像畸变模拟工具（PyTorch版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input_dir", type=str,
                        default="/home/csuzhang/disk/vggt_test/testdata/simulate_distort/undistort",
                        help="输入图像目录路径")
    parser.add_argument("--cam_file", type=str,
                        default="/home/csuzhang/disk/vggt_test/testdata/simulate_distort/undistort/001_001.txt",
                        help="输入相机参数文件路径")
    parser.add_argument("--output_dir", type=str,
                        default="/home/csuzhang/disk/vggt_test/testdata/simulate_distort/distort",
                        help="输出图像目录路径")

    parser.add_argument("--num_samples", type=int, default=3,
                        help="每个输入图像生成的样本数（默认: 3）")
    parser.add_argument("--distortion_strength_min", type=float, default=0.9,
                        help="畸变强度最小值（默认: 0.9）")
    parser.add_argument("--distortion_strength_max", type=float, default=1.5,
                        help="畸变强度最大值（默认: 1.5）")
    parser.add_argument("--vignetting", action="store_false", default=True,
                        help="包含渐晕效应（默认: 包含）")

    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="运行设备：auto/cpu/cuda")
    parser.add_argument("--save_flow", action="store_true", default=False,
                        help="是否保存flow为npy（默认不保存）")

    args = parser.parse_args()

    if args.distortion_strength_min >= args.distortion_strength_max:
        print("错误: distortion_strength_min 必须小于 distortion_strength_max")
        return
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    if not os.path.exists(args.cam_file):
        print(f"错误: 相机参数文件不存在: {args.cam_file}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("开始批量处理图像（PyTorch版）...")
    batch_process_images(args)
    print("批量处理完成!")


if __name__ == "__main__":
    main()

