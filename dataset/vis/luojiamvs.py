# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
LuoJiaMVS Dataset using WAI format data.
"""

import os
import torch
import cv2
import numpy as np

from dataset.base.base_dataset import BaseDataset
from dataset.wai.core import load_data, load_frame

class LuoJiaMVSWAI(BaseDataset):
    def __init__(
        self,
        *args,
        ROOT,
        dataset_metadata_dir,
        split,
        overfit_num_sets=None,
        sample_specific_scene: bool = False,
        specific_scene_name: str = None,
        **kwargs,
    ):
        """
        Initialize the dataset attributes.
        Args:
            ROOT: Root directory of the dataset.
            dataset_metadata_dir: Path to the dataset metadata directory.
            split: Dataset split (train, val, test).
            overfit_num_sets: If None, use all sets. Else, the dataset will be truncated to this number of sets.
            sample_specific_scene: Whether to sample a specific scene from the dataset.
            specific_scene_name: Name of the specific scene to sample.
        """
        # Initialize the dataset attributes
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.dataset_metadata_dir = dataset_metadata_dir
        self.split = split
        self.overfit_num_sets = overfit_num_sets
        self.sample_specific_scene = sample_specific_scene
        self.specific_scene_name = specific_scene_name
        self._load_data()

        # Define the dataset type flags
        self.is_metric_scale = True
        self.is_synthetic = False

    def _load_data(self):
        "Load the precomputed dataset metadata"
        # Load the dataset metadata corresponding to the split
        split_metadata_path = os.path.join(
            self.dataset_metadata_dir,
            self.split,
            f"luojiamvs_scene_list_{self.split}.npy",
        )
        split_scene_list = np.load(split_metadata_path, allow_pickle=True)

        # Get the list of all scenes
        if not self.sample_specific_scene:
            self.scenes = list(split_scene_list)
        else:
            self.scenes = [self.specific_scene_name]
        self.num_of_scenes = len(self.scenes)

    def _get_views(
        self, 
        sampled_idx, 
        num_views_to_sample, 
        resolution,           
        sampling_mode="random_walk", 
        use_bidirectional_covis=True
    ):
        # Get the scene name of the sampled index
        scene_index = sampled_idx
        scene_name = self.scenes[scene_index]

        # Get the metadata corresponding to the scene
        scene_root = os.path.join(self.ROOT, scene_name)
        
        scene_meta = load_data(
            os.path.join(scene_root, "scene_meta.json"), "scene_meta"
        )
        scene_file_names = list(scene_meta["frame_names"].keys())
        num_views_in_scene = len(scene_file_names)

        # Load view graph for sampling
        g_view = self._load_covis_graph(scene_root, scene_meta)

        # Sample view indices using specified sampling mode
        view_indices = self._sample_view_indices_uav(
            num_views_to_sample=num_views_to_sample,
            num_views_in_scene=num_views_in_scene,
            view_covis_graph=g_view,
            sampling_mode=sampling_mode,
            use_bidirectional_covis=use_bidirectional_covis,
        )

        # Get the views corresponding to the selected view indices
        views = []
        for view_index in view_indices:
            # Load the data corresponding to the view
            view_file_name = scene_file_names[view_index]
            view_data = load_frame(
                scene_root,
                view_file_name,
                modalities=["image", "depth"],
                scene_meta=scene_meta,
            )

            # Convert necessary data to numpy
            raw_image = view_data["image"].permute(1, 2, 0).numpy()  # (H,W,3)
            raw_image = (raw_image * 255).astype(np.uint8)
            depthmap = view_data["depth"].numpy().astype(np.float32)
            intrinsics = view_data["intrinsics"].numpy().astype(np.float32)
            c2w_pose = view_data["extrinsics"].numpy().astype(np.float32)

            # Ensure that the depthmap has all valid values
            depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
            view_data["mask"] = torch.tensor(depthmap > 0.0, device=view_data["depth"].device)

            # Get the non_ambiguous_mask and ensure it matches image resolution
            non_ambiguous_mask = view_data["mask"].numpy().astype(int)
            non_ambiguous_mask = cv2.resize(
                non_ambiguous_mask,
                (raw_image.shape[1], raw_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            # Mask out the GT depth using the non_ambiguous_mask
            depthmap = np.where(non_ambiguous_mask, depthmap, 0)

            # Resize the data to match the desired resolution
            additional_quantities_to_resize = [non_ambiguous_mask]
            image, depthmap, intrinsics, additional_quantities_to_resize = (
                self._crop_resize_if_necessary(
                    image=raw_image,
                    resolution=resolution,
                    depthmap=depthmap,
                    intrinsics=intrinsics,
                    additional_quantities=additional_quantities_to_resize,
                )
            )
            non_ambiguous_mask = additional_quantities_to_resize[0]

            # Append the view dictionary to the list of views
            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=c2w_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    non_ambiguous_mask=non_ambiguous_mask,
                    dataset="WHU-MVS",
                    label=scene_name,
                    instance=os.path.join("images", str(view_file_name)),
                )
            )

        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd", "--root_dir", 
        default="../data/luojiamvs", type=str
    )
    parser.add_argument(
        "-dmd",
        "--dataset_metadata_dir",
        default="../metadata",
        type=str,
    )
    parser.add_argument(
        "-nv",
        "--num_of_views",
        default=10,
        type=int,
    )
    parser.add_argument("--viz", action="store_true", default=False)
    return parser

if __name__ == "__main__":
    # how to use
    # python luojiamvs.py --viz --serve

    import rerun as rr
    from tqdm import tqdm

    from dataset.base.base_dataset import view_name
    from dataset.utils.image import rgb
    from dataset.utils.viz import script_add_rerun_args

    parser = get_parser()
    script_add_rerun_args(
        parser
    )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    dataset = LuoJiaMVSWAI(
        num_views=args.num_of_views,
        split="train",
        ROOT=args.root_dir,
        covisibility_thres=0.1,
        covisibility_thres_max=1.0,
        dataset_metadata_dir=args.dataset_metadata_dir,
        resolution=(518, 252),
        aug_crop=16,
        transform="colorjitter+grayscale+gaublur",
        data_norm_type="dinov2",
        sampling_mode="random_walk",  # "random_walk" or "greedy_chain"
    )
    print(dataset.get_stats())

    if args.viz:
        rr.script_setup(args, "LuoJia-MVS_Dataloader")
        rr.set_time("stable_time", sequence=0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    sampled_indices = np.random.choice(len(dataset), size=1, replace=False)

    for num, idx in enumerate(tqdm(sampled_indices)):
        views = dataset[idx]
        assert len(views) == args.num_of_views
        sample_name = f"{idx}"
        for view_idx in range(args.num_of_views):
            sample_name += f" {view_name(views[view_idx])}"
        print(sample_name)
        for view_idx in range(args.num_of_views):
            image = rgb(
                views[view_idx]["img"], norm_type=views[view_idx]["data_norm_type"]
            )
            depthmap = views[view_idx]["depthmap"]
            pose = views[view_idx]["camera_pose"]
            intrinsics = views[view_idx]["camera_intrinsics"]
            pts3d = views[view_idx]["pts3d"]
            valid_mask = views[view_idx]["valid_mask"]
            if "non_ambiguous_mask" in views[view_idx]:
                non_ambiguous_mask = views[view_idx]["non_ambiguous_mask"]
            else:
                non_ambiguous_mask = None
            if "prior_depth_along_ray" in views[view_idx]:
                prior_depth_along_ray = views[view_idx]["prior_depth_along_ray"]
            else:
                prior_depth_along_ray = None
            if args.viz:
                rr.set_time("stable_time", sequence=num)
                base_name = f"world/view_{view_idx}"
                pts_name = f"world/view_{view_idx}_pointcloud"
                # Log camera info and loaded data
                height, width = image.shape[0], image.shape[1]
                rr.log(
                    base_name,
                    rr.Transform3D(
                        translation=pose[:3, 3],
                        mat3x3=pose[:3, :3],
                    ),
                )
                rr.log(
                    f"{base_name}/pinhole",
                    rr.Pinhole(
                        image_from_camera=intrinsics,
                        height=height,
                        width=width,
                        camera_xyz=rr.ViewCoordinates.RDF,
                    ),
                )
                rr.log(
                    f"{base_name}/pinhole/rgb",
                    rr.Image(image),
                )
                rr.log(
                    f"{base_name}/pinhole/depth",
                    rr.DepthImage(depthmap),
                )
                if prior_depth_along_ray is not None:
                    rr.log(
                        f"prior_depth_along_ray_{view_idx}",
                        rr.DepthImage(prior_depth_along_ray),
                    )
                if non_ambiguous_mask is not None:
                    rr.log(
                        f"{base_name}/pinhole/non_ambiguous_mask",
                        rr.SegmentationImage(non_ambiguous_mask.astype(int)),
                    )
                # Log points in 3D
                filtered_pts = pts3d[valid_mask]
                filtered_pts_col = image[valid_mask]
                rr.log(
                    pts_name,
                    rr.Points3D(
                        positions=filtered_pts.reshape(-1, 3),
                        colors=filtered_pts_col.reshape(-1, 3),
                    ),
                )


                