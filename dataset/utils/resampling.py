import numpy as np
import skimage.io as io
from numba import cuda
import math


# =========================================================
#  CUDA Device Functions
# =========================================================

@cuda.jit(device=True)
def iterSearchShader(padu, padv, xr, yr, maxIter, precision):
    """
    CUDA 设备函数：对单个像素 (xr, yr) 进行前向流的反向 iterative search，
    求解畸变图像中对应的像素坐标 (i, j)。

    输入:
        padu, padv : (H+1, W+1) 扩展 flow 的 u,v 分量
        xr, yr     : 无畸变图像坐标
        maxIter    : 最大迭代次数
        precision  : 迭代停止阈值

    输出:
        (i, j) : 畸变图像坐标，如果失败返回 (-1, -1)
    """

    H = padu.shape[0] - 1
    W = padu.shape[1] - 1
    
    # 如果本地flow已接近零，直接返回原位置
    if abs(padu[yr, xr]) < precision and abs(padv[yr, xr]) < precision:
        return xr, yr

    else:
        # ---- CVPR 2019: Improved Initialization ----
        if (xr + 1) <= (W - 1):
            dif = padu[yr, xr + 1] - padu[yr, xr]
            u_next = padu[yr, xr] / (1 + dif)
        else:
            dif = padu[yr, xr] - padu[yr, xr - 1]
            u_next = padu[yr, xr] / (1 + dif)

        if (yr + 1) <= (H - 1):
            dif = padv[yr + 1, xr] - padv[yr, xr]
            v_next = padv[yr, xr] / (1 + dif)
        else:
            dif = padv[yr, xr] - padv[yr - 1, xr]
            v_next = padv[yr, xr] / (1 + dif)

        i = xr - u_next
        j = yr - v_next

        # ---- 迭代反向搜索 ----
        for iter in range(maxIter):

            if 0 <= i <= (W - 1) and 0 <= j <= (H - 1):

                # 双线性插值读取 forward flow(u, v)
                u11 = padu[int(j), int(i)]
                v11 = padv[int(j), int(i)]

                u12 = padu[int(j), int(i) + 1]
                v12 = padv[int(j), int(i) + 1]

                u21 = padu[int(j) + 1, int(i)]
                v21 = padv[int(j) + 1, int(i)]

                u22 = padu[int(j) + 1, int(i) + 1]
                v22 = padv[int(j) + 1, int(i) + 1]

                u = (
                    u11 * (int(i) + 1 - i) * (int(j) + 1 - j)
                    + u12 * (i - int(i)) * (int(j) + 1 - j)
                    + u21 * (int(i) + 1 - i) * (j - int(j))
                    + u22 * (i - int(i)) * (j - int(j))
                )

                v = (
                    v11 * (int(i) + 1 - i) * (int(j) + 1 - j)
                    + v12 * (i - int(i)) * (int(j) + 1 - j)
                    + v21 * (int(i) + 1 - i) * (j - int(j))
                    + v22 * (i - int(i)) * (j - int(j))
                )

                i_next = xr - u
                j_next = yr - v

                # 收敛判断
                if abs(i - i_next) < precision and abs(j - j_next) < precision:
                    return i, j

                i = i_next
                j = j_next

            else:
                return -1, -1

        # 超过迭代次数仍未收敛
        if 0 <= i_next <= (W - 1) and 0 <= j_next <= (H - 1):
            return i_next, j_next
        elif 0 <= i <= (W - 1) and 0 <= j <= (H - 1):
            return i, j
        else:
            return -1, -1


@cuda.jit(device=True)
def biInterpolation(distorted, i, j):
    """
    CUDA 双线性插值计算 (i,j) 处图像像素值
    用于根据求得的反向坐标提取畸变图的颜色。
    """
    Q11 = distorted[int(j), int(i)]
    Q12 = distorted[int(j), int(i) + 1]
    Q21 = distorted[int(j) + 1, int(i)]
    Q22 = distorted[int(j) + 1, int(i) + 1]

    pixel = (
        Q11 * (int(i) + 1 - i) * (int(j) + 1 - j)
        + Q12 * (i - int(i)) * (int(j) + 1 - j)
        + Q21 * (int(i) + 1 - i) * (j - int(j))
        + Q22 * (i - int(i)) * (j - int(j))
    )
    return pixel


# =========================================================
#  CUDA Kernel: RGB
# =========================================================

@cuda.jit
def iterSearch(padu, padv, paddistorted, resultImg, maxIter, precision, resultMsk):
    """
    CUDA Kernel：对彩色图像执行迭代反向搜索。
    每个 pixel (xr, yr) 独立求解对应畸变图像位置 (i,j)，并完成采样。
    """
    H = padu.shape[0] - 1
    W = padu.shape[1] - 1

    start_x, start_y = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)
    
    for xr in range(start_x, W, stride_x):
        for yr in range(start_y, H, stride_y):

            i, j = iterSearchShader(padu, padv, xr, yr, maxIter, precision)

            if (i != -1) and (j != -1):
                resultImg[yr, xr, 0] = biInterpolation(paddistorted[:, :, 0], i, j)
                resultImg[yr, xr, 1] = biInterpolation(paddistorted[:, :, 1], i, j)
                resultImg[yr, xr, 2] = biInterpolation(paddistorted[:, :, 2], i, j)
            else:
                resultMsk[yr, xr] = 255


# =========================================================
#  CUDA Kernel: Grayscale
# =========================================================

@cuda.jit
def iterSearchGrey(padu, padv, paddistorted, resultImg, maxIter, precision, resultMsk):
    """
    灰度图像版本
    """
    H = padu.shape[0] - 1
    W = padu.shape[1] - 1

    start_x, start_y = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for xr in range(start_x, W, stride_x):
        for yr in range(start_y, H, stride_y):

            i, j = iterSearchShader(padu, padv, xr, yr, maxIter, precision)

            if (i != -1) and (j != -1):
                resultImg[yr, xr] = biInterpolation(paddistorted[:, :], i, j)
            else:
                resultMsk[yr, xr] = 255


# =========================================================
#  Public Function: rectification()
# =========================================================

def rectification(distorted, flow):
    """
    对输入图像进行去畸变 (Rectification)
    使用前向 flow，通过 CUDA iterative search 完成反向采样。

    输入:
        distorted: (H, W, 3) 或 (H, W) numpy 图像 (uint8)
        flow:      (2, H, W) forward flow:
                       flow[0] = u(x,y)
                       flow[1] = v(x,y)

    输出:
        resultImg: 去畸变图
        resultMsk: 掩码（未能收敛的像素标记为 255）
    """

    H = distorted.shape[0]
    W = distorted.shape[1]

    maxIter = 100
    precision = 1e-2

    # ---------------- prepare padded images ----------------
    isGrey = (len(distorted.shape) == 2)

    resultMsk = np.zeros((H, W), dtype=np.uint8)

    if not isGrey:
        resultImg = np.zeros((H, W, 3), dtype=np.uint8)
        paddistorted = np.zeros((H + 1, W + 1, 3), dtype=np.uint8)
    else:
        resultImg = np.zeros((H, W), dtype=np.uint8)
        paddistorted = np.zeros((H + 1, W + 1), dtype=np.uint8)

    resultImg.fill(255)

    # pad distorted image
    distorted = (distorted * 255).astype(np.uint8)
    paddistorted[0:H, 0:W] = distorted
    paddistorted[H, 0:W] = distorted[H - 1, 0:W]
    paddistorted[0:H, W] = distorted[0:H, W - 1]
    paddistorted[H, W] = distorted[H - 1, W - 1]

    # pad flow
    padu = np.zeros((H + 1, W + 1), dtype=np.float32)
    padv = np.zeros((H + 1, W + 1), dtype=np.float32)

    padu[0:H, 0:W] = flow[0]
    padu[H, 0:W] = flow[0][H - 1]
    padu[0:H, W] = flow[0][:, W - 1]
    padu[H, W] = flow[0][H - 1, W - 1]

    padv[0:H, 0:W] = flow[1]
    padv[H, 0:W] = flow[1][H - 1]
    padv[0:H, W] = flow[1][:, W - 1]
    padv[H, W] = flow[1][H - 1, W - 1]

    # copy to GPU
    padu = cuda.to_device(padu)
    padv = cuda.to_device(padv)
    paddistorted = cuda.to_device(paddistorted)
    resultImg_d = cuda.to_device(resultImg)
    resultMsk_d = cuda.to_device(resultMsk)

    # launch config
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(W / threadsperblock[0])
    blockspergrid_y = math.ceil(H / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # call kernel
    if isGrey:
        iterSearchGrey[blockspergrid, threadsperblock](padu, padv, paddistorted,
                                                       resultImg_d, maxIter,
                                                       precision, resultMsk_d)
    else:
        iterSearch[blockspergrid, threadsperblock](padu, padv, paddistorted,
                                                   resultImg_d, maxIter,
                                                   precision, resultMsk_d)

    # copy back
    resultImg = resultImg_d.copy_to_host()
    resultMsk = resultMsk_d.copy_to_host()

    return (resultImg / 255.0).astype(float), resultMsk
