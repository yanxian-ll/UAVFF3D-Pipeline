# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Base class for MapAnything datasets.
"""

from typing import List, Tuple, Union

import numpy as np
import PIL
import torch
import torchvision.transforms as tvf
from scipy.spatial.transform import Rotation
import random
import inspect

from dataset.base.easy_dataset import EasyDataset
from dataset.utils.cropping import (
    bbox_from_intrinsics_in_out,
    camera_matrix_of_crop,
    crop_image_and_other_optional_info,
    rescale_image_and_other_optional_info,
)
from dataset.utils.geometry import (
    depthmap_to_camera_coordinates,
    get_absolute_pointmaps_and_rays_info,
)
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT


class BaseDataset(EasyDataset):
    """
    Base class for MapAnything datasets. Defines basic functionality for loading and processing multi-view data.

    Args:
        num_views (int): Number of views to sample for each data instance.
        variable_num_views (bool): If True, number of views can vary from batch to batch. Default: False.
        split (str): Split of the dataset, e.g., 'train', 'val', 'test'.
        covisibility_thres (float): Minimum percentage of visibility between two views for them to be considered neighbors.
        resolution (int or tuple): Resolution of the images.
        principal_point_centered (bool): Whether to center the principal point in the image.
        transform (str): Image transformation type.
        data_norm_type (str): Data normalization type for image normalization.
        aug_crop (int): Augmentation crop size.
        seed (int): Random seed for reproducibility.
        max_num_retries (int): Maximum number of retries when loading data.
        sampling_mode (str): Sampling strategy for selecting views, e.g., "random_walk", "two_strips".
        use_frame_graph (bool): Whether to use frame graph for multi-camera setups.
        cam_policy (str): Camera selection policy. Options: 'random', 'all', 'cycle', 'same_camera_for_all_frames'.
    """

    def __init__(
        self,
        num_views: int,
        variable_num_views: bool = False,
        split: str = None,
        covisibility_thres: float = None,
        resolution: Union[int, Tuple[int, int], List[Tuple[int, int]]] = None,
        principal_point_centered: bool = False,
        transform: str = None,
        data_norm_type: str = None,
        aug_crop: int = 0,
        seed: int = 42,
        max_num_retries: int = 5,
        covisibility_thres_max: float = 1.0, 
        interval: Union[int, Tuple[int], List[int]] = 1,
        sampling_mode: str = "random_walk",  # random_walk, two_strips, greedy_chain
        use_frame_graph: bool = True,
        cam_policy: str = "random",  # "random", "all", "cycle", "same"
        walk_restart_prob: float = 0.10,
        walk_temperature: float = 1.0,
        walk_topk_step: int = 50,
        two_strip_interleave: bool = True,
        two_strip_same_extent: bool = False,
        two_strip_extent_margin: float = 0.2,
        two_strip_pair_ratio: float = 0.2,
        two_strip_cross_min: float = 0.1,
        two_strip_along_min: float = 0.3,
        **kwargs,
    ):
        """
        Initializes the dataset parameters, transforms, and augmentation options.

        Args:
            num_views, variable_num_views, etc. as described above.
        """
        self.num_views = num_views
        self.variable_num_views = variable_num_views
        self.num_views_min = 2
        self.split = split
        self.covisibility_thres = covisibility_thres
        self.covisibility_thres_max = covisibility_thres_max
        assert float(self.covisibility_thres_max) >= float(self.covisibility_thres), \
            f"covisibility_thres_max({self.covisibility_thres_max}) must be >= covisibility_thres({self.covisibility_thres})"
        self._set_resolutions(resolution)
        self.principal_point_centered = principal_point_centered
        
        self.interval = interval
        self.sampling_mode = sampling_mode
        self.use_frame_graph = use_frame_graph
        self.cam_policy = cam_policy
        self.walk_restart_prob = walk_restart_prob
        self.walk_temperature = walk_temperature
        self.walk_topk_step = walk_topk_step

        self.two_strip_interleave = two_strip_interleave
        self.two_strip_same_extent = two_strip_same_extent
        self.two_strip_extent_margin = two_strip_extent_margin
        self.two_strip_pair_ratio = two_strip_pair_ratio
        self.two_strip_cross_min = two_strip_cross_min
        self.two_strip_along_min = two_strip_along_min

        # Update the number of views if necessary and make it a list if variable_num_views is True
        if self.variable_num_views and self.num_views > self.num_views_min:
            self.num_views = list(range(self.num_views_min, self.num_views + 1))

        # Initialize the image normalization type
        if data_norm_type in IMAGE_NORMALIZATION_DICT.keys():
            self.data_norm_type = data_norm_type
            image_norm = IMAGE_NORMALIZATION_DICT[data_norm_type]
            ImgNorm = tvf.Compose(
                [
                    tvf.ToTensor(),
                    tvf.Normalize(mean=image_norm.mean, std=image_norm.std),
                ]
            )
        elif data_norm_type == "identity":
            self.data_norm_type = data_norm_type
            ImgNorm = tvf.Compose([tvf.ToTensor()])
        else:
            raise ValueError(
                f"Unknown data_norm_type: {data_norm_type}. Available options: identity or {list(IMAGE_NORMALIZATION_DICT.keys())}"
            )

        # Initialize torchvision transforms
        if transform == "imgnorm":
            self.transform = ImgNorm
        elif transform == "colorjitter":
            self.transform = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])
        elif transform == "colorjitter+grayscale+gaublur":
            self.transform = tvf.Compose(
                [
                    tvf.RandomApply([tvf.ColorJitter(0.3, 0.4, 0.2, 0.1)], p=0.75),
                    tvf.RandomGrayscale(p=0.05),
                    tvf.RandomApply([tvf.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05),
                    ImgNorm,
                ]
            )
        else:
            raise ValueError(
                'Unknown transform. Available options: "imgnorm", "colorjitter", "colorjitter+grayscale+gaublur"'
            )

        # Initialize the augmentation parameters
        self.aug_crop = aug_crop

        # Initialize the seed for the random number generator
        self.seed = seed
        self._seed_offset = 0

        # Initialize the maximum number of retries for loading a different sample from the dataset, if the first idx fails
        self.max_num_retries = max_num_retries

        # Initialize the dataset type flags
        self.is_metric_scale = False  # by default a dataset is not metric scale, subclasses can overwrite this
        self.is_synthetic = False  # by default a dataset is not synthetic, subclasses can overwrite this

    def _load_data(self):
        self.scenes = []
        self.num_of_scenes = len(self.scenes)

    def __len__(self):
        "Length of the dataset is determined by the number of scenes in the dataset split"
        return self.num_of_scenes

    def get_stats(self):
        "Get the number of scenes in the dataset split"
        return f"{self.num_of_scenes} scenes"

    def __repr__(self):
        resolutions_str = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.num_views=}
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace("self.", "")
            .replace("\n", "")
            .replace("   ", "")
        )

    def _get_views(self, idx, num_views_to_sample, resolution):
        raise NotImplementedError()
    
    def _call_get_views_compat(self, idx, num_views_to_sample, resolution, **extra_kwargs):
        """
        Call subclass _get_views with best-effort backward compatibility.

        Supports (common cases):
        - _get_views(self, idx)
        - _get_views(self, idx, num_views_to_sample, resolution)
        - _get_views(self, idx, num_views_to_sample, resolution, **kwargs)
        - _get_views(self, idx, num_views_to_sample, resolution, *, some_kw=None, ...)
        """
        fn = self._get_views

        # Try signature-based filtering first
        try:
            sig = inspect.signature(fn)
            params = sig.parameters
            names = set(params.keys())
            names.discard("self")

            # If subclass accepts **kwargs, pass everything
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if has_varkw:
                return fn(idx, num_views_to_sample, resolution, **extra_kwargs)

            # Otherwise, only pass supported kwargs (by name)
            filtered = {k: v for k, v in extra_kwargs.items() if k in names}

            # Decide how many positional args it likely expects
            # Prefer name-based decision to support old implementations.
            if {"num_views_to_sample", "resolution"}.issubset(names) or len(names) >= 3:
                return fn(idx, num_views_to_sample, resolution, **filtered)
            elif len(names) == 2:
                # ambiguous; most common legacy is (idx, num_views_to_sample) or (idx, resolution)
                # choose by name if possible
                if "resolution" in names:
                    return fn(idx, resolution, **filtered)
                else:
                    return fn(idx, num_views_to_sample, **filtered)
            else:
                return fn(idx)

        except Exception:
            # Fallback: try calling patterns (avoid breaking on environments where signature introspection fails)
            try:
                return fn(idx, num_views_to_sample, resolution, **extra_kwargs)
            except TypeError:
                try:
                    return fn(idx, num_views_to_sample, resolution)
                except TypeError:
                    return fn(idx)

    def _set_seed_offset(self, idx):
        """
        Set the seed offset. This is directly added to self.seed when setting the random seed.
        """
        self._seed_offset = idx

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, "undefined resolution"

        if isinstance(resolutions, int):
            resolutions = [resolutions]
        elif isinstance(resolutions, tuple):
            resolutions = [resolutions]
        elif isinstance(resolutions, list):
            assert all(isinstance(res, tuple) for res in resolutions), (
                f"Bad type for {resolutions=}, should be int or tuple of ints or list of tuples of ints"
            )
        else:
            raise ValueError(
                f"Bad type for {resolutions=}, should be int or tuple of ints or list of tuples of ints"
            )

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), (
                f"Bad type for {width=} {type(width)=}, should be int"
            )
            assert isinstance(height, int), (
                f"Bad type for {height=} {type(height)=}, should be int"
            )
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(
        self,
        image,
        resolution,
        depthmap,
        intrinsics,
        additional_quantities=None,
    ):
        """
        Process an image by downsampling and cropping as needed to match the target resolution.

        This method performs the following operations:
        1. Converts the image to PIL.Image if necessary
        2. Crops the image centered on the principal point if requested
        3. Downsamples the image using high-quality Lanczos filtering
        4. Performs final cropping to match the target resolution

        Args:
            image (numpy.ndarray or PIL.Image.Image): Input image to be processed
            resolution (tuple): Target resolution as (width, height)
            depthmap (numpy.ndarray): Depth map corresponding to the image
            intrinsics (numpy.ndarray): Camera intrinsics matrix (3x3)
            additional_quantities (dict, optional): Additional image-related data to be processed
                                                   alongside the main image with nearest interpolation. Defaults to None.

        Returns:
            tuple: Processed image, depthmap, and updated intrinsics matrix.
                  If additional_quantities is provided, it returns those as well.
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # Cropping centered on the principal point if necessary
        if self.principal_point_centered:
            W, H = image.size
            cx, cy = intrinsics[:2, 2].round().astype(int)
            if cx < 0 or cx >= W or cy < 0 or cy >= H:
                # Skip centered cropping if principal point is outside image bounds
                pass
            else:
                min_margin_x = min(cx, W - cx)
                min_margin_y = min(cy, H - cy)
                left, top = cx - min_margin_x, cy - min_margin_y
                right, bottom = cx + min_margin_x, cy + min_margin_y
                crop_bbox = (left, top, right, bottom)
                # Only perform the centered crop if the crop_bbox is larger than the target resolution
                crop_width = right - left
                crop_height = bottom - top
                if crop_width > resolution[0] and crop_height > resolution[1]:
                    image, depthmap, intrinsics, additional_quantities = (
                        crop_image_and_other_optional_info(
                            image=image,
                            crop_bbox=crop_bbox,
                            depthmap=depthmap,
                            camera_intrinsics=intrinsics,
                            additional_quantities=additional_quantities,
                        )
                    )

        # Get the target resolution for re-scaling
        target_rescale_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_rescale_resolution += self._rng.integers(0, self.aug_crop)

        # High-quality Lanczos down-scaling if necessary
        image, depthmap, intrinsics, additional_quantities = (
            rescale_image_and_other_optional_info(
                image=image,
                output_resolution=target_rescale_resolution,
                depthmap=depthmap,
                camera_intrinsics=intrinsics,
                additional_quantities_to_be_resized_with_nearest=additional_quantities,
            )
        )

        # Actual cropping (if necessary)
        new_intrinsics = camera_matrix_of_crop(
            input_camera_matrix=intrinsics,
            input_resolution=image.size,
            output_resolution=resolution,
            offset_factor=0.5,
        )
        crop_bbox = bbox_from_intrinsics_in_out(
            input_camera_matrix=intrinsics,
            output_camera_matrix=new_intrinsics,
            output_resolution=resolution,
        )
        image, depthmap, new_intrinsics, additional_quantities = (
            crop_image_and_other_optional_info(
                image=image,
                crop_bbox=crop_bbox,
                depthmap=depthmap,
                camera_intrinsics=intrinsics,
                additional_quantities=additional_quantities,
            )
        )

        # Return the output
        if additional_quantities is not None:
            return image, depthmap, new_intrinsics, additional_quantities
        else:
            return image, depthmap, new_intrinsics

    # =========================
    # CSR graph helpers
    # =========================
    def _sort_by_axis(self, ids, t):
        return [int(x) for x in sorted(ids, key=lambda i: float(t[int(i)]))]

    def _clamp01(self, x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def _is_csr_graph(self, x):
        return isinstance(x, dict) and x.get("format", None) in ["csr", "csr_npz"]

    def _csr_row(self, g, i: int):
        """Return neighbors, weights for row i from CSR graph dict."""
        indptr = g["indptr"]
        indices = g["indices"]
        data = g["data"]
        s = int(indptr[i]); e = int(indptr[i + 1])
        return indices[s:e], data[s:e]
    
    def _keep_w_in_range(self, w: np.ndarray, w_min: float = None, w_max: float = None) -> np.ndarray:
        """Return boolean mask where w in [w_min, w_max]. None means no bound."""
        keep = np.ones_like(w, dtype=bool)
        if w_min is not None:
            keep &= (w >= float(w_min))
        if w_max is not None:
            keep &= (w <= float(w_max))
        return keep

    def _csr_edge(self, g, i: int, j: int) -> float:
        """返回 i->j 的边权；若不存在返回 0"""
        nbrs, w = self._csr_row(g, i)
        if nbrs.size == 0:
            return 0.0
        for n, ww in zip(nbrs, w):
            if int(n) == int(j):
                return float(ww)
        return 0.0
    
    def _edge_w(self, g, u: int, v: int, bidirectional: bool = True) -> float:
        """Edge weight u<->v (optionally bidirectional). Missing edge -> 0."""
        w = float(self._csr_edge(g, u, v))
        if bidirectional:
            w = max(w, float(self._csr_edge(g, v, u)))
        return w
    
    def _pca_axis_1(self, centers: np.ndarray) -> np.ndarray:
        """centers: [F,3]，返回第一主轴单位向量"""
        C = centers.astype(np.float64)
        C = C - C.mean(axis=0, keepdims=True)
        cov = C.T @ C / max(1, (C.shape[0] - 1))
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, np.argmax(vals)]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis.astype(np.float32)

    def _weighted_choice(self, nbrs, w, temperature=1.0):
        """Sample neighbor proportional to softmax(w/temperature)."""
        if len(nbrs) == 1:
            return int(nbrs[0])
        w = np.asarray(w, dtype=np.float32)
        t = max(float(temperature), 1e-8)
        z = (w / t) - (w / t).max()
        p = np.exp(z)
        p = p / (p.sum() + 1e-12)
        return int(self._rng.choice(nbrs, p=p))

    def _random_walk_sampling_csr(
        self,
        g,
        num_of_samples: int,
        max_retries: int = 4,
        min_covis: float = None,
        max_covis: float = None,
        restart_prob: float = 0.10,
        temperature: float = 1.0,
        topk_step: int = 50,
        avoid_set: set = None,
    ):
        """
        Random walk on CSR graph.
        g: dict {format:'csr', indptr, indices, data, shape=(N,N)}
        """
        if min_covis is None:
            min_covis = self.covisibility_thres
        if max_covis is None:
            max_covis = self.covisibility_thres_max

        N = int(g["shape"][0])
        excluded_nodes = set()
        best_walk = []

        for _ in range(max_retries):
            visited = set()
            walk = []
            stack = []

            all_nodes = set(range(N))
            available_nodes = list(all_nodes - excluded_nodes)
            if avoid_set:
                available_nodes = [x for x in available_nodes if x not in avoid_set]
            if not available_nodes:
                break

            start = int(self._rng.choice(available_nodes))
            walk.append(start); visited.add(start); stack.append(start)

            while len(walk) < num_of_samples and stack:
                cur = stack[-1]

                if self._rng.random() < restart_prob:
                    cur = start

                nbrs, w = self._csr_row(g, cur)
                if nbrs.size == 0:
                    stack.pop()
                    continue

                # filter threshold
                keep = self._keep_w_in_range(w, min_covis, max_covis)
                nbrs = nbrs[keep]; w = w[keep]
                if nbrs.size == 0:
                    stack.pop()
                    continue

                # step-topk (optional)
                if topk_step is not None and nbrs.size > int(topk_step):
                    # choose topk by weight
                    k = int(topk_step)
                    part = np.argpartition(w, -k)[-k:]
                    nbrs = nbrs[part]; w = w[part]

                # remove visited
                mask = np.array([int(n) not in visited for n in nbrs], dtype=bool)
                if mask.any():
                    nbrs2 = nbrs[mask]; w2 = w[mask]
                    nxt = self._weighted_choice(nbrs2, w2, temperature=temperature)
                    walk.append(nxt); visited.add(nxt); stack.append(nxt)
                else:
                    stack.pop()

            if len(walk) > len(best_walk):
                best_walk = walk
            if len(walk) >= num_of_samples:
                return np.array(walk, dtype=np.int64)

            excluded_nodes.update(visited)

        return np.array(best_walk, dtype=np.int64)

    def _greedy_chain_sampling_csr_once(
        self,
        g,
        num_of_samples: int,
        min_covis: float = None,
        max_covis: float = None,
        topk_step: int = 50,
        start: int = None,
        bidirectional_edge: bool = True,
        enforce_global_max: bool = True,
    ):
        """
        Greedy chain with constraints:
        - next must satisfy w(prev,next) in [min_covis, max_covis)
        - additionally, next must satisfy w(u,next) < max_covis for ALL historical u in chain
            (only upper bound; no min constraint to history)
        """
        if min_covis is None:
            min_covis = self.covisibility_thres
        if max_covis is None:
            max_covis = self.covisibility_thres_max

        N = int(g["shape"][0])
        if N <= 0:
            return np.array([], dtype=np.int64)

        if start is None:
            start = int(self._rng.integers(0, N))

        eps = 1e-12

        walk = [start]
        visited = set([start])
        cur = start

        while len(walk) < num_of_samples:
            nbrs, w_dir = self._csr_row(g, cur)
            if nbrs.size == 0:
                break

            # Effective weights (optional bidirectional)
            if bidirectional_edge:
                w_eff = np.empty_like(w_dir, dtype=np.float32)
                for k, (n, wd) in enumerate(zip(nbrs, w_dir)):
                    n = int(n)
                    w_eff[k] = max(float(wd), float(self._csr_edge(g, n, cur)))
            else:
                w_eff = w_dir.astype(np.float32, copy=False)

            # Filter: prev edge must be in [min, max)
            keep = np.ones_like(w_eff, dtype=bool)
            if min_covis is not None:
                keep &= (w_eff >= float(min_covis))
            if max_covis is not None:
                keep &= (w_eff < float(max_covis) - eps)

            nbrs = nbrs[keep]
            w_eff = w_eff[keep]
            if nbrs.size == 0:
                break

            # topk by weight (optional)
            if topk_step is not None and nbrs.size > int(topk_step):
                k = int(topk_step)
                part = np.argpartition(w_eff, -k)[-k:]
                nbrs = nbrs[part]
                w_eff = w_eff[part]

            # Choose best candidate by descending w_eff, but respecting GLOBAL max constraint
            order = np.argsort(-w_eff)
            chosen = None
            for j in order:
                cand = int(nbrs[j])
                if cand in visited:
                    continue

                # Global constraint: for ALL previous u, w(u,cand) < max_covis
                if enforce_global_max and (max_covis is not None) and (len(walk) >= 2):
                    ok = True
                    for u in walk:  # includes cur; that is fine
                        if u == cand:
                            ok = False
                            break
                        if self._edge_w(g, int(u), cand, bidirectional=bidirectional_edge) >= float(max_covis) - eps:
                            ok = False
                            break
                    if not ok:
                        continue

                chosen = cand
                break

            if chosen is None:
                break

            walk.append(chosen)
            visited.add(chosen)
            cur = chosen

        return np.array(walk, dtype=np.int64)


    def _greedy_chain_sampling_csr(
        self,
        g,
        num_of_samples: int,
        min_covis: float = None,
        max_covis: float = None,
        topk_step: int = 50,
        max_retries: int = 6,
        avoid_set: set = None,
        bidirectional_edge: bool = True,
    ):
        """
        Greedy chain: each step choose highest-weight unvisited neighbor.
        适合测试“单航带/单序列”。
        """
        if min_covis is None:
            min_covis = self.covisibility_thres

        N = int(g["shape"][0])
        best_walk = np.array([], dtype=np.int64)

        candidates = list(range(N))
        if avoid_set:
            candidates = [x for x in candidates if x not in avoid_set]
        if not candidates:
            return best_walk

        for _ in range(max_retries):
            start = int(self._rng.choice(candidates))
            walk = self._greedy_chain_sampling_csr_once(
                g,
                num_of_samples,
                min_covis=min_covis,
                max_covis=max_covis,
                topk_step=topk_step,
                start=start,
                bidirectional_edge=bidirectional_edge,
                enforce_global_max=True,
            )
            if len(walk) > len(best_walk):
                best_walk = walk
            if len(walk) >= num_of_samples:
                return walk
        return best_walk
    
    # ========================= 
    def _greedy_chain_csr(
        self,
        g,
        length: int,
        start: int = None,
        forbid: set = None,
        min_edge: float = 0.0,
        centers: np.ndarray = None,
        t: np.ndarray = None, 
        s: np.ndarray = None,  
        max_s_dev: float = None, 
        max_turn_deg: float = 75.0, 
        max_dt_jump: float = None,
        max_restarts: int = 4,
        bidirectional_edge: bool = True,
    ):
        if forbid is None:
            forbid = set()
        visited_global = set(forbid)

        N = int(g["shape"][0])
        if N <= 0 or length <= 0:
            return []

        def edge(u: int, v: int) -> float:
            w = float(self._csr_edge(g, u, v))
            if bidirectional_edge:
                w = max(w, float(self._csr_edge(g, v, u)))
            return w
        
        def _estimate_scale_from_start(st: int):
            """自动估计 max_s_dev 和 max_dt_jump，尽量与图结构尺度匹配。"""
            _max_s_dev = max_s_dev
            _max_dt_jump = max_dt_jump

            if centers is None or t is None or s is None:
                return _max_s_dev, _max_dt_jump

            nbrs, w = self._csr_row(g, st)
            if nbrs.size == 0:
                return _max_s_dev, _max_dt_jump

            keep = w >= float(min_edge)
            nbrs = nbrs[keep]
            if nbrs.size == 0:
                return _max_s_dev, _max_dt_jump

            # dt / ds 的稳健尺度
            dt = np.abs(t[nbrs] - float(t[st]))
            ds = np.abs(s[nbrs] - float(s[st]))

            # 用中位数 + MAD 稳健估计
            def robust_tol(x, k=6.0, floor=1e-6):
                x = np.asarray(x, dtype=np.float64)
                med = np.median(x)
                mad = np.median(np.abs(x - med)) + 1e-12
                tol = max(float(med + k * mad), floor)
                return tol

            if _max_dt_jump is None:
                _max_dt_jump = robust_tol(dt, k=6.0, floor=1e-3)
            if _max_s_dev is None:
                # 横向更严格一些，避免跳航带
                _max_s_dev = robust_tol(ds, k=4.0, floor=1e-3)
            return _max_s_dev, _max_dt_jump

        def angle_deg(v1, v2):
            n1 = np.linalg.norm(v1) + 1e-12
            n2 = np.linalg.norm(v2) + 1e-12
            c = float(np.dot(v1, v2) / (n1 * n2))
            c = max(-1.0, min(1.0, c))
            return float(np.degrees(np.arccos(c)))
        
        best_chain = []
        for _ in range(int(max_restarts)):
            visited = set(visited_global)

            if start is None:
                candidates = [i for i in range(N) if i not in visited]
                if not candidates:
                    break
                st = int(self._rng.choice(candidates))
            else:
                st = int(start)
                if st in visited:
                    candidates = [i for i in range(N) if i not in visited]
                    if not candidates:
                        break
                    st = int(self._rng.choice(candidates))

            chain = [st]
            visited.add(st)

            # 自动估计容差
            _max_s_dev, _max_dt_jump = _estimate_scale_from_start(st)

            s0 = float(s[st])
            t_prev = float(t[st])
            dir_sign = None   # +1 或 -1
            prev = None

            while len(chain) < length:
                cur = chain[-1]
                nbrs, w = self._csr_row(g, cur)
                if nbrs.size == 0:
                    break

                cand = []
                for n, ww in zip(nbrs, w):
                    n = int(n)
                    ww = float(ww)

                    if n in visited:
                        continue

                    if ww < float(min_edge) and edge(cur, n) < float(min_edge):
                        continue

                    if prev is not None and n == prev:
                        continue

                    if (centers is not None) and (s is not None) and (_max_s_dev is not None):
                        if abs(float(s[n]) - s0) > float(_max_s_dev):
                            continue

                    if (t is not None) and (_max_dt_jump is not None):
                        if abs(float(t[n]) - float(t[cur])) > float(_max_dt_jump):
                            continue

                    # 几何约束：锁定前进方向（用第一步决定方向）
                    if t is not None:
                        dt = float(t[n]) - float(t[cur])
                        if dir_sign is None:
                            if abs(dt) > 1e-6:
                                dir_sign = 1.0 if dt > 0 else -1.0
                        else:
                            if dir_sign * dt < -1e-6:
                                continue

                    # 几何约束：曲率（转角）受限
                    if (centers is not None) and (prev is not None):
                        v1 = centers[cur] - centers[prev]
                        v2 = centers[n] - centers[cur]
                        if angle_deg(v1, v2) > float(max_turn_deg):
                            continue

                    # 评分：优先大边权，其次航向推进
                    score = edge(cur, n)
                    if t is not None:
                        score += 1e-3 * abs(float(t[n]) - float(t[cur]))
                    cand.append((score, n))

                if not cand:
                    break

                cand.sort(key=lambda x: -x[0])
                nxt = int(cand[0][1])

                prev = cur
                chain.append(nxt)
                visited.add(nxt)

            if len(chain) > len(best_chain):
                best_chain = chain
            if len(best_chain) >= length:
                break

            # 下次重试时，把这次访问过的节点加入全局排除，促使换航带/换起点（但仍满足几何锁带）
            visited_global.update(chain)

        return best_chain

    def _build_strip2_by_pairs(
        self,
        frame_graph,
        strip1_sorted,
        t,
        cross_min: float,
        forbid: set,
        prefer_continuity: bool = True,
        along_min: float = 0.0,
        centers: np.ndarray = None, 
        s: np.ndarray = None,  
        max_s_dev: float = None,  
        max_dt_mismatch: float = None, 
        max_start_candidates: int = 8,
        bidirectional_edge: bool = True,
    ):
        """
        构建 strip2：
        - 先收集所有满足 cross_min 的候选点（来自 strip1 各 anchor 的邻居）
        - 然后从候选集合里选出一条“近似平行 strip1”的连续序列：
        1) strip2 横向坐标 s 稳定（避免跳航带）
        2) strip2 相邻点 edge >= along_min（连续）
        3) strip2 与 strip1 在 t 方向同步推进（与 anchor 的 t 匹配）
        """
        if forbid is None:
            forbid = set()

        g = frame_graph

        def edge(u: int, v: int) -> float:
            w = float(self._csr_edge(g, u, v))
            if bidirectional_edge:
                w = max(w, float(self._csr_edge(g, v, u)))
            return w

        L1 = len(strip1_sorted)
        if L1 == 0:
            return []

        # --- 1) 收集所有 cross_min 候选 ---
        anchor_cands = []   # 每个 anchor 的候选列表 [(n, w_cross), ...]
        cand_set = set()
        for f1 in strip1_sorted:
            nbrs, w = self._csr_row(g, int(f1))
            cands = []
            for n, ww in zip(nbrs, w):
                n = int(n); ww = float(ww)
                if n in forbid:
                    continue
                if ww < float(cross_min) and edge(int(f1), n) < float(cross_min):
                    continue
                cands.append((n, edge(int(f1), n)))
                cand_set.add(n)
            anchor_cands.append(cands)

        if not cand_set:
            return []

        # 自动估计容差（横向偏移、t 匹配窗口）
        if (centers is not None) and (s is not None) and (max_s_dev is None):
            # 用所有候选的 s 分布，估一个较严格的带宽（偏向单航带）
            sc = np.asarray([float(s[i]) for i in cand_set], dtype=np.float64)
            med = np.median(sc)
            mad = np.median(np.abs(sc - med)) + 1e-12
            max_s_dev = max(float(4.0 * mad), 1e-3)

        if (max_dt_mismatch is None):
            # 用 strip1 相邻 dt 的稳健尺度估一个匹配窗
            if (t is not None) and (len(strip1_sorted) > 1):
                dt1 = np.diff(np.asarray([float(t[i]) for i in strip1_sorted], dtype=np.float64))
                dt1 = np.abs(dt1)
                med = float(np.median(dt1))
                mad = float(np.median(np.abs(dt1 - med)) + 1e-12)
                max_dt_mismatch = max(med + 6.0 * mad, 1e-3)
            else:
                max_dt_mismatch = 1e9

        # --- 2) 用多起点尝试构建 strip2（避免早期贪心锁死） ---
        # 起点候选：优先用第一个 anchor 的 top-k cross 邻居；若为空，则从全局候选挑 t 最接近
        start_pool = []
        if anchor_cands[0]:
            start_pool = sorted(anchor_cands[0], key=lambda x: -x[1])[: int(max_start_candidates)]
            start_pool = [int(n) for n, _ in start_pool]
        else:
            # fallback：从全体候选里挑 t 最近的
            a0 = int(strip1_sorted[0])
            target_t = float(t[a0]) if t is not None else 0.0
            tmp = []
            for n in cand_set:
                if n in forbid:
                    continue
                if t is None:
                    tmp.append((0.0, int(n)))
                else:
                    tmp.append((abs(float(t[n]) - target_t), int(n)))
            tmp.sort(key=lambda x: x[0])
            start_pool = [n for _, n in tmp[: int(max_start_candidates)]]

        best_strip2 = []

        for st in start_pool:
            if st in forbid:
                continue

            used = set([st])
            strip2 = [st]
            last = st

            s0 = float(s[st]) if (s is not None) else None

            # 逐 anchor 推进：每一步挑“同时满足 cross_min + 连续 + 平行”的点
            for i in range(1, L1):
                a = int(strip1_sorted[i])
                target_t = float(t[a]) if t is not None else None

                cands = anchor_cands[i]
                if not cands:
                    # 若某个 anchor 没候选，可允许跳过（但这会缩短 strip2）
                    continue

                filtered = []
                for n, w_cross in cands:
                    n = int(n); w_cross = float(w_cross)
                    if n in forbid or n in used:
                        continue
                    if w_cross < float(cross_min):
                        continue

                    if prefer_continuity and float(along_min) > 0.0:
                        if edge(last, n) < float(along_min):
                            continue

                    if (s is not None) and (max_s_dev is not None):
                        if abs(float(s[n]) - s0) > float(max_s_dev):
                            continue

                    if (t is not None) and (target_t is not None):
                        if abs(float(t[n]) - target_t) > float(max_dt_mismatch):
                            continue
                        if float(t[n]) + 1e-6 < float(t[last]):
                            continue

                    score = w_cross
                    if prefer_continuity and float(along_min) > 0.0:
                        score += 0.5 * edge(last, n)
                    if (t is not None) and (target_t is not None):
                        score -= 0.05 * abs(float(t[n]) - target_t)

                    filtered.append((score, n))

                if not filtered:
                    break

                filtered.sort(key=lambda x: -x[0])
                pick = int(filtered[0][1])
                strip2.append(pick)
                used.add(pick)
                last = pick

            if len(strip2) > len(best_strip2):
                best_strip2 = strip2

        for x in best_strip2:
            forbid.add(int(x))
        return best_strip2
    
    
    def _two_strips_sampling(
        self,
        view_graph,
        num_of_samples,
        frame_graph=None,
        view2frame=None,
        frame_centers=None,
        view_centers=None,
        use_frame_graph=False,
        along_min: float = 0.2,
        cross_min: float = 0.1,
        max_tries: int = 12,
        same_extent: bool = True,
        extent_margin: float = 0.15,
        pair_ratio: float = 0.7,
        local_window: int = 80,
        cam_policy="random",
        cam_idxs=None,
        num_views_total=None,
    ):
        """
        Two-strip sampling (fixed photogrammetric mode).

        Goal:
        Sample two "flight strips" that look like photogrammetric acquisition:
        - strip1: a forward-overlap chain (along overlap >= along_min)
        - strip2: for each frame in strip1, pick a cross-strip counterpart (cross overlap >= cross_min),
                    while keeping strip2 itself reasonably continuous along the strip.

        Pseudo-code:
        split N samples into n1 (strip1) and n2 (strip2)
        choose graph level (frame or view)
        compute 1D ordering coordinate t along flight direction (PCA axis)
        repeat max_tries:
            strip1 = greedy_chain(min_edge=along_min)
            strip1_sorted = sort_by_t(strip1)
            strip2 = build_by_pairs(strip1_sorted, cross_min, along_min, local_window)
            strip2_sorted = sort_by_t(strip2)
            trim strip2 to n2 (if longer)
            optionally enforce same_extent wrt strip1
            if paired_ok(strip1_sorted, strip2_sorted, cross_min, pair_ratio):
                return (frames->views if needed) else (view ids)
            keep the best ratio as fallback
        return fallback best
        """
        # -------------------------
        # split samples into two strips
        # -------------------------
        n1 = int(np.ceil(num_of_samples / 2))
        n2 = int(num_of_samples - n1)

        def _pad_to_len(lst, L):
            """Pad by sampling with replacement from itself; or return empty if lst empty."""
            lst = list(lst)
            if len(lst) == 0:
                return np.array([], dtype=np.int64)
            if len(lst) < L:
                extra = self._rng.choice(lst, size=(L - len(lst)), replace=True).tolist()
                lst.extend(extra)
            else:
                lst = lst[:L]
            return np.asarray(lst, dtype=np.int64)

        # -------------------------
        # choose graph: frame-level preferred when available
        # -------------------------
        use_frame = (
            bool(use_frame_graph)
            and (frame_graph is not None)
            and (view2frame is not None)
            and (frame_centers is not None)
        )

        if use_frame:
            g = frame_graph
            centers = np.asarray(frame_centers, dtype=np.float32)
            N = int(g["shape"][0])
            if centers.shape[0] != N:
                use_frame = False

        if not use_frame:
            g = view_graph
            centers = None if view_centers is None else np.asarray(view_centers, dtype=np.float32)
            N = int(g["shape"][0])
            if centers is not None and centers.shape[0] != N:
                centers = None

        if centers is not None:
            C = centers.astype(np.float64)
            C0 = C - C.mean(axis=0, keepdims=True)
            cov = (C0.T @ C0) / max(1, (C0.shape[0] - 1))
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            axis1 = vecs[:, order[0]]; axis1 = axis1 / (np.linalg.norm(axis1) + 1e-12)
            axis2 = vecs[:, order[1]]; axis2 = axis2 / (np.linalg.norm(axis2) + 1e-12)

            t = (centers @ axis1).astype(np.float32)
            s = (centers @ axis2).astype(np.float32)
        else:
            t = np.arange(N, dtype=np.float32)
            s = None

        # -------------------------
        # check if two strips are "paired" enough (cross overlap)
        # -------------------------
        def paired_ok(s1_sorted, s2_sorted):
            """
            Pair by normalized position:
            i in strip1 maps to j in strip2 with same relative position.
            Count how many pairs have edge >= cross_min.
            """
            if len(s1_sorted) == 0 or len(s2_sorted) == 0:
                return False, 0.0
            L1, L2 = len(s1_sorted), len(s2_sorted)
            ok = 0
            for i in range(L1):
                u = 0.0 if L1 == 1 else i / (L1 - 1)
                j = int(round(u * (L2 - 1))) if L2 > 1 else 0
                w = self._csr_edge(g, s1_sorted[i], s2_sorted[j])
                if w >= cross_min:
                    ok += 1
            ratio = ok / max(1, L1)
            return (ratio >= pair_ratio), ratio

        best = None  # (s1_sorted, s2_sorted, ratio)
        for _ in range(max_tries):
            # 1) strip1: greedy along-strip chain (forward overlap >= along_min)
            strip1 = self._greedy_chain_csr(
                g, n1,
                start=None,
                forbid=set(),
                min_edge=along_min,
                centers=centers,
                t=t,
                s=s,
                max_restarts=24,
            )

            if len(strip1) == 0:
                continue
            s1_sorted = self._sort_by_axis(strip1, t)
            forbid = set(s1_sorted)

            # 2) strip2: anchor each strip1 frame and pick best cross neighbor
            strip2 = self._build_strip2_by_pairs(
                frame_graph=g,
                strip1_sorted=s1_sorted,
                t=t,
                s=s,
                cross_min=cross_min,
                forbid=forbid,
                prefer_continuity=True,
                along_min=along_min,
            )
            s2_sorted = self._sort_by_axis(strip2, t)

            # If strip2 is longer than needed, subsample uniformly to length n2
            if len(s2_sorted) > n2:
                if n2 > 1:
                    idxs = np.linspace(0, len(s2_sorted) - 1, n2).round().astype(int).tolist()
                    s2_sorted = [s2_sorted[i] for i in idxs]
                else:
                    s2_sorted = [s2_sorted[len(s2_sorted)//2]]

            # 4) validate pairing quality
            ok, ratio = paired_ok(s1_sorted, s2_sorted)
            if ok:
                best = (s1_sorted, s2_sorted, ratio)
                break
            if best is None or ratio > best[2]:
                best = (s1_sorted, s2_sorted, ratio)

        # -------------------------
        # fallback
        # -------------------------
        if best is None:
            return np.array([], np.int64), np.array([], np.int64)

        s1_sorted, s2_sorted, _ = best
        if use_frame:
            frame2views = self._build_frame2views(view2frame)
            views1 = self._expand_frames_to_views(
                s1_sorted, frame2views, n1, cam_policy=cam_policy,
                num_views_total=num_views_total, cam_idxs=cam_idxs
            )
            views2 = self._expand_frames_to_views(
                s2_sorted, frame2views, n2, cam_policy=cam_policy,
                num_views_total=num_views_total, cam_idxs=cam_idxs
            ) if len(s2_sorted) > 0 else np.array([], np.int64)
            return views1, views2
        else:
            return _pad_to_len(s1_sorted, n1), _pad_to_len(s2_sorted, n2)

    def _build_frame2views(self, view2frame: np.ndarray):
        """view2frame: (N,) int, return list[list[int]] of frame->views"""
        F = int(view2frame.max()) + 1 if view2frame.size > 0 else 0
        frame2views = [[] for _ in range(F)]
        for v, f in enumerate(view2frame.tolist()):
            frame2views[int(f)].append(int(v))
        return frame2views

    def _expand_frames_to_views(self, frame_ids, frame2views, num_views_to_sample, cam_policy="random", num_views_total=None, cam_idxs=None):
        """
        Expands a sequence of frames into views, ensuring each frame selects views of the same camera type.
        
        Args:
            frame_ids (list): List of frame IDs.
            frame2views (list): List of views corresponding to each frame.
            num_views_to_sample (int): Total number of views to sample.
            cam_policy (str): The camera selection policy. Options include 'random', 'same', 'all', 'cycle'.
            num_views_total (int): Total number of views available (if needed for 'same_view_for_all_frames').

        Returns:
            np.ndarray: A numpy array of selected view IDs.
        """
        out = []

        if cam_policy == "same":
            # First, collect all possible views from the given frames
            first_view = self._rng.choice(frame2views[int(frame_ids[0])])
            first_cam_id = cam_idxs[first_view]

            for f in frame_ids:
                vs = frame2views[int(f)]
                if len(vs) == 0:
                    continue
                # Find the view that matches the chosen index if possible
                for v in vs:
                    if cam_idxs[v] == first_cam_id:
                        out.append(int(v))
                        break
                if len(out) >= num_views_to_sample:
                    break
            # If the number of views is less than needed, pad by sampling from existing views
            if len(out) < num_views_to_sample:
                extra = self._rng.choice(out, size=(num_views_to_sample - len(out)), replace=True).tolist()
                out = out + extra

        elif cam_policy == "random":
            for f in frame_ids:
                vs = frame2views[int(f)]
                if len(vs) == 0:
                    continue
                # Randomly pick one view for this frame
                out.append(int(self._rng.choice(vs)))
                if len(out) >= num_views_to_sample:
                    break
        elif cam_policy == "cycle":
            cam_ptr = 0
            for f in frame_ids:
                vs = frame2views[int(f)]
                if len(vs) == 0:
                    continue
                # Cycle through the views
                out.append(int(vs[cam_ptr % len(vs)]))
                cam_ptr += 1
                if len(out) >= num_views_to_sample:
                    break
        elif cam_policy == "all":
            for f in frame_ids:
                vs = frame2views[int(f)]
                out.extend([int(x) for x in vs])
                if len(out) >= num_views_to_sample:
                    out = out[:num_views_to_sample]
                    break
        else:
            raise ValueError(f"Unknown cam_policy: {cam_policy}")

        # If we have fewer views than required, sample with replacement
        if len(out) < num_views_to_sample:
            if len(out) == 0:
                assert num_views_total is not None
                return self._rng.choice(num_views_total, size=num_views_to_sample, replace=True)
            extra = self._rng.choice(out, size=(num_views_to_sample - len(out)), replace=True).tolist()
            out = out + extra
        return np.array(out, dtype=np.int64)

    def _walk_sampling(
        self,
        view_graph,
        num_of_samples,
        frame_graph=None,
        view2frame=None,
        use_frame_graph=False,
        max_retries=4,
        sampling_mode="random_walk",   # supported: "random_walk" / "greedy_chain"
        cam_policy="random",
        use_bidirectional_covis=True,
        cam_idxs=None,
    ):
        # 多镜头：先采 frame，再展开成 view
        if use_frame_graph and (frame_graph is not None) and (view2frame is not None):
            view2frame = np.asarray(view2frame, dtype=np.int64)
            frame2views = self._build_frame2views(view2frame)

            num_frames_to_sample = num_of_samples  # conservative

            if sampling_mode == "greedy_chain":
                frame_ids = self._greedy_chain_sampling_csr(
                    frame_graph,
                    num_frames_to_sample,
                    min_covis=self.covisibility_thres,
                    max_covis=self.covisibility_thres_max,
                    topk_step=self.walk_topk_step,
                    max_retries=max_retries,
                    bidirectional_edge=use_bidirectional_covis,
                )
            else:  # "random_walk"
                frame_ids = self._random_walk_sampling_csr(
                    frame_graph,
                    num_frames_to_sample,
                    max_retries=max_retries,
                    min_covis=self.covisibility_thres,
                    max_covis=self.covisibility_thres_max,
                    restart_prob=self.walk_restart_prob,
                    temperature=self.walk_temperature,
                    topk_step=self.walk_topk_step,
                )

            view_ids = self._expand_frames_to_views(
                frame_ids,
                frame2views,
                num_of_samples,
                cam_policy=cam_policy,
                num_views_total=len(view2frame),
                cam_idxs=cam_idxs,
            )
            return view_ids

        # 单镜头/不使用frame：直接 view graph 采样
        if sampling_mode == "greedy_chain":
            return self._greedy_chain_sampling_csr(
                view_graph,
                num_of_samples,
                min_covis=self.covisibility_thres,
                max_covis=self.covisibility_thres_max,
                topk_step=self.walk_topk_step,
                bidirectional_edge=use_bidirectional_covis,
            )
        else:  # "random_walk"
            return self._random_walk_sampling_csr(
                view_graph,
                num_of_samples,
                max_retries=max_retries,
                min_covis=self.covisibility_thres,
                max_covis=self.covisibility_thres_max,
                restart_prob=self.walk_restart_prob,
                temperature=self.walk_temperature,
                topk_step=self.walk_topk_step,
            )

    def _interleave_two_lists(self, a, b, total_len):
        out = []
        ia = ib = 0
        while len(out) < total_len and (ia < len(a) or ib < len(b)):
            if ia < len(a):
                out.append(a[ia]); ia += 1
                if len(out) >= total_len:
                    break
            if ib < len(b):
                out.append(b[ib]); ib += 1
        return out

    def _sample_view_indices_uav(
        self,
        num_views_to_sample,
        num_views_in_scene,
        view_covis_graph,
        view2frame,
        cam_idxs=None,
        frame_graph=None,
        sampling_mode="random_walk",  # "walk"(==random_walk) / "greedy_chain" / "two_strips"
        frame_centers=None,
        view_centers=None,
        use_frame_graph=False,
        cam_policy="random",
        two_strip_along_min=0.5,
        two_strip_cross_min=0.1,
        two_strip_same_extent=True,
        two_strip_extent_margin=0.15,
        two_strip_pair_ratio=0.7,
        use_bidirectional_covis=True,
    ):
        if num_views_to_sample == num_views_in_scene:
            return self._rng.permutation(num_views_in_scene)
        if num_views_to_sample > num_views_in_scene:
            return self._rng.choice(num_views_in_scene, size=num_views_to_sample, replace=True)

        # frame 路径是否可用（walk 不需要 centers；two_strips 需要 centers）
        use_fg_walk = bool(use_frame_graph) and (frame_graph is not None) and (view2frame is not None)
        use_fg_twostrip = use_fg_walk and (frame_centers is not None)

        # -------------------------
        # 1) random_walk / greedy_chain
        # -------------------------
        if sampling_mode in ["random_walk", "greedy_chain"]:
            view_indices = self._walk_sampling(
                view_covis_graph,
                num_views_to_sample,
                sampling_mode=sampling_mode,
                view2frame=view2frame,
                frame_graph=frame_graph,
                use_frame_graph=use_fg_walk,
                cam_policy=cam_policy,
                use_bidirectional_covis=use_bidirectional_covis,
                cam_idxs=cam_idxs,
            )
            if len(view_indices) < num_views_to_sample and len(view_indices) > 0:
                view_indices = self._rng.choice(view_indices, size=num_views_to_sample, replace=True)
            if len(view_indices) == 0:
                view_indices = self._rng.choice(num_views_in_scene, size=num_views_to_sample, replace=True)
            return view_indices

        # -------------------------
        # 2) two_strips
        # -------------------------
        if sampling_mode == "two_strips":
            views1, views2 = self._two_strips_sampling(
                view_graph=view_covis_graph,
                num_of_samples=num_views_to_sample,
                use_frame_graph=use_fg_twostrip,
                frame_graph=frame_graph,
                view2frame=view2frame,
                frame_centers=frame_centers,
                view_centers=view_centers,
                along_min=two_strip_along_min,
                cross_min=two_strip_cross_min,
                same_extent=two_strip_same_extent,
                extent_margin=two_strip_extent_margin,
                pair_ratio=two_strip_pair_ratio,
                cam_policy=cam_policy, 
                cam_idxs=cam_idxs,  
                num_views_total=num_views_in_scene,
            )

            if self.two_strip_interleave:
                merged = self._interleave_two_lists(views1.tolist(), views2.tolist(), num_views_to_sample)
                view_indices = np.asarray(merged, dtype=np.int64)
            else:
                view_indices = np.concatenate([views1, views2], axis=0)

            if view_indices.size < num_views_to_sample and view_indices.size > 0:
                extra = self._rng.choice(view_indices, size=(num_views_to_sample - view_indices.size), replace=True)
                view_indices = np.concatenate([view_indices, extra], axis=0)
            if view_indices.size == 0:
                view_indices = self._rng.choice(num_views_in_scene, size=num_views_to_sample, replace=True)
            return view_indices

        # unknown mode
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")


    def _getitem_fn(self, idx):
        if isinstance(idx, tuple):
            # The idx is a tuple if specifying the aspect-ratio or/and the number of views
            if isinstance(self.num_views, int):
                idx, ar_idx = idx
            else:
                idx, ar_idx, num_views_to_sample_idx = idx
        else:
            assert len(self._resolutions) == 1
            assert isinstance(self.num_views, int)
            ar_idx = 0

        # Setup the rng
        if self.seed:  # reseed for each _getitem_fn
            # Leads to deterministic sampling where repeating self.seed and self._seed_offset yields the same multi-view set again
            # Scenes will be repeated if size of dataset is artificially increased using "N @" or "N *"
            # When scenes are repeated, self._seed_offset is increased to ensure new multi-view sets
            # This is useful for evaluation if the number of dataset scenes is < N, yet we want unique multi-view sets each iter
            self._rng = np.random.default_rng(seed=self.seed + self._seed_offset + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # Get the views for the given index and check that the number of views is correct
        resolution = self._resolutions[ar_idx]
        if isinstance(self.num_views, int):
            num_views_to_sample = self.num_views
        else:
            num_views_to_sample = self.num_views[num_views_to_sample_idx]

        if isinstance(self.interval, int):
            interval_to_sample = self.interval
        else:
            interval_idx = self._rng.integers(0, len(self.interval))
            interval_to_sample = self.interval[interval_idx]

        views = self._call_get_views_compat(
            idx, num_views_to_sample, resolution,
            interval=interval_to_sample,
            sampling_mode=self.sampling_mode,
            use_frame_graph=self.use_frame_graph,
            cam_policy=self.cam_policy,
            two_strip_along_min=self.two_strip_along_min,
            two_strip_cross_min=self.two_strip_cross_min,
            two_strip_same_extent=self.two_strip_same_extent,
            two_strip_extent_margin=self.two_strip_extent_margin,
            two_strip_pair_ratio=self.two_strip_pair_ratio,
            use_bidirectional_covis=True,
        )

        if isinstance(self.num_views, int):
            assert len(views) == self.num_views
        else:
            assert len(views) in self.num_views

        for v, view in enumerate(views):
            # Store the index and other metadata
            view["idx"] = (idx, ar_idx, v)
            view["is_metric_scale"] = self.is_metric_scale
            view["is_synthetic"] = self.is_synthetic

            # Check the depth, intrinsics, and pose data (also other data if present)
            assert "camera_intrinsics" in view
            assert "camera_pose" in view
            assert np.isfinite(view["camera_pose"]).all(), (
                f"NaN or infinite values in camera pose for view {view_name(view)}"
            )
            assert np.isfinite(view["depthmap"]).all(), (
                f"NaN or infinite values in depthmap for view {view_name(view)}"
            )
            assert "valid_mask" not in view
            assert "pts3d" not in view, (
                f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            )
            if "prior_depth_z" in view:
                assert np.isfinite(view["prior_depth_z"]).all(), (
                    f"NaN or infinite values in prior_depth_z for view {view_name(view)}"
                )
            if "non_ambiguous_mask" in view:
                assert np.isfinite(view["non_ambiguous_mask"]).all(), (
                    f"NaN or infinite values in non_ambiguous_mask for view {view_name(view)}"
                )

            # Encode the image
            width, height = view["img"].size
            view["true_shape"] = np.int32((height, width))
            view["img"] = self.transform(view["img"])
            if "raw_img" in view:
                view["raw_img"] = self.transform(view["raw_img"])
            view["data_norm_type"] = self.data_norm_type

            # Compute the pointmaps, raymap and depth along ray
            (
                pts3d,
                valid_mask,
                ray_origins_world,
                ray_directions_world,
                depth_along_ray,
                ray_directions_cam,
                pts3d_cam,
            ) = get_absolute_pointmaps_and_rays_info(**view)
            view["pts3d"] = pts3d
            view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)
            view["depth_along_ray"] = depth_along_ray
            view["ray_directions_cam"] = ray_directions_cam
            view["pts3d_cam"] = pts3d_cam

            # Compute the prior depth along ray if present
            if "prior_depth_z" in view:
                prior_pts3d, _ = depthmap_to_camera_coordinates(
                    view["prior_depth_z"], view["camera_intrinsics"]
                )
                view["prior_depth_along_ray"] = np.linalg.norm(prior_pts3d, axis=-1)
                view["prior_depth_along_ray"] = view["prior_depth_along_ray"][..., None]
                del view["prior_depth_z"]

            # Convert ambiguous mask dtype to match valid mask dtype
            if "non_ambiguous_mask" in view:
                view["non_ambiguous_mask"] = view["non_ambiguous_mask"].astype(
                    view["valid_mask"].dtype
                )
            else:
                ambiguous_mask = view["depthmap"] < 0
                view["non_ambiguous_mask"] = ~ambiguous_mask
                view["non_ambiguous_mask"] = view["non_ambiguous_mask"].astype(
                    view["valid_mask"].dtype
                )

            # Check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"

            # Check shapes
            assert view["depthmap"].shape == view["img"].shape[1:]
            assert view["depthmap"].shape == view["pts3d"].shape[:2]
            assert view["depthmap"].shape == view["valid_mask"].shape
            assert view["depthmap"].shape == view["depth_along_ray"].shape[:2]
            assert view["depthmap"].shape == view["ray_directions_cam"].shape[:2]
            assert view["depthmap"].shape == view["pts3d_cam"].shape[:2]
            if "prior_depth_along_ray" in view:
                assert view["depthmap"].shape == view["prior_depth_along_ray"].shape[:2]
            if "non_ambiguous_mask" in view:
                assert view["depthmap"].shape == view["non_ambiguous_mask"].shape

            # Expand the last dimennsion of the depthmap
            view["depthmap"] = view["depthmap"][..., None]

            # Append RNG state to the views, this allows to check whether the RNG is in the same state each time
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")

            # Compute and store the quaternions and translation for the camera poses
            # Notation is (x, y, z, w) for quaternions
            # This also ensures that the camera poses have a positive determinant (right-handed coordinate system)
            view["camera_pose_quats"] = (
                Rotation.from_matrix(view["camera_pose"][:3, :3])
                .as_quat()
                .astype(view["camera_pose"].dtype)
            )
            view["camera_pose_trans"] = view["camera_pose"][:3, 3].astype(
                view["camera_pose"].dtype
            )

            # Check the pointmaps, rays, depth along ray, and camera pose quaternions and translation to ensure they are finite
            assert np.isfinite(view["pts3d"]).all(), (
                f"NaN in pts3d for view {view_name(view)}"
            )
            assert np.isfinite(view["valid_mask"]).all(), (
                f"NaN in valid_mask for view {view_name(view)}"
            )
            assert np.isfinite(view["depth_along_ray"]).all(), (
                f"NaN in depth_along_ray for view {view_name(view)}"
            )
            assert np.isfinite(view["ray_directions_cam"]).all(), (
                f"NaN in ray_directions_cam for view {view_name(view)}"
            )
            assert np.isfinite(view["pts3d_cam"]).all(), (
                f"NaN in pts3d_cam for view {view_name(view)}"
            )
            assert np.isfinite(view["camera_pose_quats"]).all(), (
                f"NaN in camera_pose_quats for view {view_name(view)}"
            )
            assert np.isfinite(view["camera_pose_trans"]).all(), (
                f"NaN in camera_pose_trans for view {view_name(view)}"
            )
            if "prior_depth_along_ray" in view:
                assert np.isfinite(view["prior_depth_along_ray"]).all(), (
                    f"NaN in prior_depth_along_ray for view {view_name(view)}"
                )

        return views

    def __getitem__(self, idx):
        if self.max_num_retries == 0:
            return self._getitem_fn(idx)
        
        num_retries = 0
        while num_retries <= self.max_num_retries:
            try:
                return self._getitem_fn(idx)
            except Exception as e:
                scene_idx = idx[0] if isinstance(idx, tuple) else idx
                print(
                    f"Error in {type(self).__name__}.__getitem__ for scene_idx={scene_idx}: {e}"
                )

                if num_retries >= self.max_num_retries:
                    print(
                        f"Max retries ({self.max_num_retries}) reached, raising the exception"
                    )
                    raise e

                # Retry with a different scene index
                num_retries += 1
                if isinstance(idx, tuple):
                    # The scene index is the first element of the tuple
                    idx_list = list(idx)
                    idx_list[0] = np.random.randint(0, len(self))
                    idx = tuple(idx_list)
                else:
                    # The scene index is idx
                    idx = np.random.randint(0, len(self))
                scene_idx = idx[0] if isinstance(idx, tuple) else idx
                print(
                    f"Retrying with scene_idx={scene_idx} ({num_retries} of {self.max_num_retries})"
                )


def is_good_type(v):
    """
    Check if a value has an acceptable data type for processing in the dataset.

    Args:
        v: The value to check.

    Returns:
        tuple: A tuple containing:
            - bool: True if the type is acceptable, False otherwise.
            - str or None: Error message if the type is not acceptable, None otherwise.
    """
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    """
    Generate a string identifier for a view based on its dataset, label, and instance.

    Args:
        view (dict): Dictionary containing view information with 'dataset', 'label', and 'instance' keys.
        batch_index (int, optional): Index to select from batched data. Defaults to None.

    Returns:
        str: A formatted string in the form "dataset/label/instance".
    """

    def sel(x):
        return x[batch_index] if batch_index not in (None, slice(None)) else x

    db = sel(view["dataset"])
    label = sel(view["label"])
    instance = sel(view["instance"])
    return f"{db}/{label}/{instance}"

