"""Create train/validation/test scene lists for UAVFF3D datasets.

The split rules are intentionally explicit because several datasets use fixed
scene IDs or acquisition names from the paper. The script can also summarize
per-scene hFOV values from camera files and write split-specific JSON metadata.
"""

import os
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import Dict, List, Tuple, Optional


def whu_whuomvs_split(scenes):
    train_scenes = []
    val_scenes = []
    test_scenes = []

    for scene in scenes:
        # whuomvs 质量太差，还是不用了吧
        if scene.startswith("whuomvs"):
            continue
        if scene.startswith(("whuomvs_area2", "whuomvs_area3")):
            val_scenes.append(scene)
        else:
            train_scenes.append(scene)
    return train_scenes, val_scenes, test_scenes

def uavscenes_split(scenes):
    train_scenes = []
    val_scenes = []
    test_scenes = []
    for scene in scenes:
        if scene in [
            'interval5_AMtown01',
            'interval5_AMtown02',
            'interval5_AMtown03',
            'interval5_AMvalley01',
            'interval5_AMvalley02',
            'interval5_AMvalley03',
            'interval5_HKairport01',
            'interval5_HKairport02',
            'interval5_HKairport03',
            'interval5_HKairport_GNSS01',
            'interval5_HKairport_GNSS02',
            'interval5_HKairport_GNSS03',
            'interval5_HKairport_GNSS_Evening',
            'interval5_HKisland01',
            'interval5_HKisland02',
            'interval5_HKisland03',
            'interval5_HKisland_GNSS01',
            'interval5_HKisland_GNSS02',
            'interval5_HKisland_GNSS03',
            'interval5_HKisland_GNSS_Evening'
        ]:
            train_scenes.append(scene)
    return train_scenes, val_scenes, test_scenes


def ortho_split(scenes):
    test_scenes = []
    train_scenes = []
    val_scenes = []
    for scene in scenes:
        if scene.startswith("test_inPlace") or scene.startswith("test_outPlace"):
            test_scenes.append(scene)
        elif scene.startswith("train") or scene.startswith("val"):
            if scene.startswith("train"):
                train_scenes.append(scene)
            else:
                val_scenes.append(scene)
    return train_scenes, val_scenes, test_scenes


def uavff3d_real_split(scenes):
    train_scenes, val_scenes, test_scenes = [], [], []
    for scene in scenes:
        if scene.startswith(("nanfang_", "yanghaitang_", "xiaoxiang_")):
            test_scenes.append(scene)
            val_scenes.append(scene)
        else:
            train_scenes.append(scene)
    return train_scenes, val_scenes, test_scenes


def uavff3d_syn_small_split(scenes):
    train_scenes, val_scenes, test_scenes = [], [], []
    for scene in scenes:
        train_scenes.append(scene)
    return train_scenes, val_scenes, test_scenes


def uavff3d_syn_large_split(scenes):
    train_scenes, val_scenes, test_scenes = [], [], []
    for scene in scenes:
        train_scenes.append(scene)
    return train_scenes, val_scenes, test_scenes

def uavff3d_fa_split(scenes):
    train_scenes, val_scenes, test_scenes = [], [], []
    for scene in scenes:
        if scene in [
            # school
            "e1b883efa2b8768cfab20347",
            "71040e8faffc08ba7082b029",
            "fa73d296a111a7e3e973f237",
            "23dcba4dffe0c6bf0f59042e",
            "179e2063c562a60e3308d99a",
            "647ac219f9bf5eb6154d0f2b",
            "3b6bb1e3910ef5b714da4f28",
            "1cb8e5e8baf385a3cb2dcf9a",
            # bmvs 59f70ab1e5c5d366af29bf3e
            "7392aec7502366689224419c",
            "768416ab0299c27d86bf292b",
            "a73bdd58a0e011e8e415e625",
            "f6302ec5904cd552d3fda600",
            "a4d4c5752f7a802fd1871d09",
            "5e1ed8ee3c7f5951664de02c",
            "9ef090e43e5036b438022bac",
            "f74808af2ee1b0430f5e1cb2",
            # bmvs 5b271079e0878c3816dacca4
            "c11ff72f5113f4f111d89dbd",
            "9d64efeb3ecfd03c26161c18",
            "20eed7076da120a7d398df66",
            "9777b95bc27e62b2674937ff",
            "2158192f4299118faf68f4fc",
            "b276a6c7098388ff1bbcded5",
            "e3a928e88f9643a03c8a1adc",
            "19ec8ddb25b71a5be6976b93",
            # bmvs 5be3ae47f44e235bdbbc9771
            "6b2399f2b4821c795dcf57ea",
            "667452d4325d8916a88db95a",
            "e63c154a77fb5ca738875320",
            "652584ec9dff985cecebcf3a",
            "078dbc15de74d692b0b30787",
            "1c79ba1167b39cddf16b9c38",
            "64bdf3e8e7a1f57668f00bf8",
            "51e2d0bb51027f115b78f914",	
        ]:
            test_scenes.append(scene)
    return train_scenes, val_scenes, test_scenes


def wilduav_split(scenes):
    return scenes, [], []


def blendedmvs_split(scenes):
    noise_scenes = [
        "5692a4c2adafac1f14201821",
        "5864a935712e2761469111b4",
        "59f87d0bfa6280566fb38c9a",
        "58a44463156b87103d3ed45e",
        "5c2b3ed5e611832e8aed46bf",
        "5bf03590d4392319481971dc",
        "00000000000000000000001a",
        "00000000000000000000000c",
        "000000000000000000000000"
    ]

    val_split_scenes = [
        '000000000000000000000006',
        '000000000000000000000015',
        '000000000000000000000016',
        '00000000000000000000001b',
        '00000000000000000000001d',
        '584bd5587072670e72bffe39',
        '584c9ad27072670e72c060c5',
        '584dfe467072670e72c1665a',
        '584fe07f7072670e72c32e59',
        '585203546789802282f2aaf5',
        '58563650804be1058523da55',
        '5857aa5ab338a62ad5ff4dbe',
        '585834cdb338a62ad5ffab4d',
        '586133c2712e2761468ecfe3',
        '5863915b712e276146909135',
        '5863edf8712e27614690cce0',
        '58647495712e27614690f36d',
        '5864b076712e27614691197e',
        '58660e79712e27614691fe3d',
        '58676c36833dfe3f7b88b7f2',
        '586b4c459d1b5e34c282e66d',
        '586b8f149d1b5e34c283497c',
        '5880675a2366dd5d06e570ca',
        '588084032366dd5d06e59e82',
        '5880e3422366dd5d06e5ff8e',
        '5881fee18ce2c2754d0723f8',
        '588457b8932ba84fbed69942',
        '588a9c5fec4d5a1c088ec350',
        '588aff9d90414422fbe7885a',
        '5890279190414422fbea9734',
        '5899cfa6b76d7a3780a4cb64',
        '58a07ce53d0b45424799fdde',
        '58a1f5d74a4d262a170b65fc',
        '58cf4771d0f5fb221defe6da',
        '58f73e7c9f5b56478738929f',
        '591a467a6109e14d4f09b776',
        '591cf3033162411cf9047f37',
        '59bf97fe7e7b31545da34439',
        '59ecfd02e225f6492d20fcc9',
        '5a4a38dad38c8a075495b5d2',
        '5a77b46b318efe6c6736e68a',
        '5a8315f624b8e938486e0bd8',
        '5aa515e613d42d091d29d300',
        '5acf8ca0f3d8a750097e4b15',
        '5ae2e9c5fe405c5076abc6b2',
        '5b271079e0878c3816dacca4',
        '5b37189a35304b6f75e7583e',
        '5bce7ac9ca24970bce4934b6',
        '5bd43b4ba6b28b1ee86b92dd',
        '5c0d13b795da9479e12e2ee9'
    ]

    test_split_scenes = []
    # test_split_scenes = [
    #     "000000000000000000000005",
    #     "5a2af22b32a1c655cfe46013",
    #     "5a2ba6de32a1c655cfe51b79",
    #     "5a3b9731e24cd76dad1a5f1b",
    #     "5a5a1e48d62c7a12d5d00e47",
    #     "5a6b1c418d100c2f8fdc4411",
    #     "5a6feeb54a7fbc3f874f9db7",
    #     "5a7cb1d6fe5c0d6fb53e64fb",
    #     "5a355c271b63f53d5970f362",
    #     "5a752d42acc41e2423f17674",
    #     "5aa0f478a9efce63548c1cb4",
    #     "000000000000000000000010",
    #     "000000000000000000000017",
    #     "000000000000000000000019",
    #     "58a1d9d14a4d262a170b58fe",
    #     "58a2a09e156b87103d3d668c",
    #     "58a0365e38486e3c984783eb",
    #     "58a160983d0b4542479a7347",
    #     "58a47552156b87103d3f00a4",
    #     "58c6451e4a69c556061894f1",
    #     "58d36897f387231e6c929903",
    #     "58eaf1513353456af3a1682a",
    #     "59a9619a825418241fb88191",
    #     "59da1fb88a126011d0394ae9",
    #     "564a27b26d07883f460d8ab0",
    #     "584a7333fe3cb463906c9fe6",
    #     "584bdadf7072670e72c0005c",
    #     "584c9cc67072670e72c063a1",
    #     "584cea557072670e72c07fb4",
    #     "584e875c7072670e72c1ec94",
    #     "584e05667072670e72c17167",
    #     "585bb25fc49c8507c3ce7812",
    #     "585bbe55c49c8507c3ce81cd",
    #     "585f9661712e2761468dabca",
    #     "586b8f629d1b5e34c28349d6",
    #     "586c4c4d9d1b5e34c28391a1",
    #     "586caab99d1b5e34c283c213",
    #     "586cd0779d1b5e34c28403a7",
    #     "588c203d90414422fbe8319e",
    #     "589af2c97dc3d323d55691e8",
    #     "000000000000000000000002",
    #     "00000000000000000000000e",
    #     "590f91851225725be9e25d4e",
    #     "5889e344ec4d5a1c088e59be",
    #     "5898b31cc9dccc22987b82ec",
    #     "00000000000000000000000a",
    #     "5862388b712e2761468f84aa",
    #     "58897f62c02346100f4b8ee6",
    #     "58669c02712e27614692851a",
    #     "58598db2b338a62ad500bc38",
    # ]

    train_scenes, val_scenes, test_scenes = [], [], []

    for scene in scenes:
        if scene in noise_scenes:
            continue
        if scene in val_split_scenes:
            val_scenes.append(scene)
        elif scene in test_split_scenes:
            test_scenes.append(scene)
        else:
            train_scenes.append(scene)
    return train_scenes, val_scenes, test_scenes


def usegeo_split(scenes):
    return [], scenes, scenes

def urbanscene3d_split(scenes):
    return [], scenes, scenes

def enrich_split(scenes):
    return [], scenes, scenes

def save_scene_lists(scene_list, output_path):
    """Save the list of scene names as both numpy array and txt."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    scene_array = np.array(scene_list, dtype=object)
    np.save(output_path, scene_array)

    txt_path = output_path.replace(".npy", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for scene in scene_list:
            f.write(scene + "\n")

    print(f"Saved {len(scene_list)} scenes to:")
    print(f"  {output_path}")
    print(f"  {txt_path}")


def _read_file_head_tail(cam_txt_path: str, head_bytes: int = 512, tail_bytes: int = 256) -> Tuple[str, str]:
    """Read only the head and tail of a camera txt for faster parsing."""
    file_size = os.path.getsize(cam_txt_path)
    with open(cam_txt_path, "rb") as f:
        head = f.read(min(head_bytes, file_size))
        if file_size > tail_bytes:
            f.seek(-tail_bytes, os.SEEK_END)
            tail = f.read(tail_bytes)
        else:
            f.seek(0)
            tail = f.read()
    return head.decode("utf-8", errors="ignore"), tail.decode("utf-8", errors="ignore")


def _parse_fx_from_head(head_text: str, cam_txt_path: str) -> float:
    lines = [line.strip() for line in head_text.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if line.lower().startswith("intrinsic"):
            if i + 1 >= len(lines):
                raise ValueError(f"Intrinsic row1 missing in: {cam_txt_path}")
            row1 = lines[i + 1].split()
            if len(row1) < 1:
                raise ValueError(f"Intrinsic row1 format error in: {cam_txt_path}")
            return float(row1[0])
    raise ValueError(f"fx not found in: {cam_txt_path}")


def _parse_size_from_tail(tail_text: str, cam_txt_path: str) -> Tuple[int, int]:
    lines = [line.strip() for line in tail_text.splitlines() if line.strip()]

    # Fast path: the size line is usually the last non-empty line, e.g. "720 1024 31"
    if lines:
        last_tokens = lines[-1].split()
        if len(last_tokens) >= 2:
            try:
                h = int(round(float(last_tokens[0])))
                w = int(round(float(last_tokens[1])))
                return w, h
            except ValueError:
                pass

    for i, line in enumerate(lines):
        if line.lower().startswith("h w hfov"):
            if i + 1 >= len(lines):
                raise ValueError(f"Image size row missing in: {cam_txt_path}")
            vals = lines[i + 1].split()
            if len(vals) < 2:
                raise ValueError(f"Image size row format error in: {cam_txt_path}")
            h = int(round(float(vals[0])))
            w = int(round(float(vals[1])))
            return w, h

    raise ValueError(f"Image size row not found in: {cam_txt_path}")


def compute_hfov_from_txt_fast(cam_txt_path: str) -> int:
    """Fast path: read only a small head/tail chunk and recompute rounded integer hfov."""
    head_text, tail_text = _read_file_head_tail(cam_txt_path)
    fx = _parse_fx_from_head(head_text, cam_txt_path)
    width, _ = _parse_size_from_tail(tail_text, cam_txt_path)
    hfov = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    return int(round(hfov))


def compute_hfov_from_txt_fallback(cam_txt_path: str) -> int:
    """Robust fallback: read the full file and recompute rounded integer hfov."""
    with open(cam_txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    fx = None
    width = None
    cx = None

    for i, line in enumerate(lines):
        lower_line = line.lower()
        if lower_line.startswith("intrinsic"):
            if i + 1 >= len(lines):
                raise ValueError(f"Intrinsic matrix missing rows in: {cam_txt_path}")
            row1 = [float(x) for x in lines[i + 1].split()]
            if len(row1) < 3:
                raise ValueError(f"Intrinsic row1 format error in: {cam_txt_path}")
            fx = float(row1[0])
            cx = float(row1[2])
        elif lower_line.startswith("h w hfov"):
            if i + 1 >= len(lines):
                raise ValueError(f"Image size row missing in: {cam_txt_path}")
            size_vals = [float(x) for x in lines[i + 1].split()]
            if len(size_vals) < 2:
                raise ValueError(f"Image size row format error in: {cam_txt_path}")
            width = int(round(size_vals[1]))

    if fx is None:
        raise ValueError(f"fx not found in: {cam_txt_path}")
    if width is None:
        if cx is None:
            raise ValueError(f"width/cx not found in: {cam_txt_path}")
        width = int(round(cx * 2.0))

    hfov = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    return int(round(hfov))


def compute_hfov_from_txt(cam_txt_path: str) -> int:
    try:
        return compute_hfov_from_txt_fast(cam_txt_path)
    except Exception:
        return compute_hfov_from_txt_fallback(cam_txt_path)


def _compute_hfov_for_file(txt_path: str) -> Tuple[str, Optional[int], Optional[str]]:
    txt_name = os.path.basename(txt_path)
    try:
        return txt_name, compute_hfov_from_txt(txt_path), None
    except Exception as e:
        return txt_name, None, str(e)


def collect_scene_hfovs(scene_dir: str, file_workers: int = 0) -> Tuple[List[int], Dict[str, str]]:
    """
    Collect unique integer hfov values from all txt files under scene_dir/cams.

    Args:
        scene_dir: scene directory
        file_workers: optional per-scene parallel workers for txt parsing.
            0 or 1 means sequential.
    Returns:
        hfovs: sorted unique hfov ints
        errors: {filename: error_message}
    """
    cams_dir = os.path.join(scene_dir, "cams")
    hfov_values = []
    errors = {}

    if not os.path.isdir(cams_dir):
        errors["cams"] = f"Missing cams directory: {cams_dir}"
        return [], errors

    txt_paths = sorted([
        entry.path for entry in os.scandir(cams_dir)
        if entry.is_file() and entry.name.lower().endswith(".txt")
    ])

    if len(txt_paths) == 0:
        errors["cams"] = f"No txt files found in: {cams_dir}"
        return [], errors

    if file_workers and file_workers > 1:
        with ThreadPoolExecutor(max_workers=file_workers) as executor:
            futures = [executor.submit(_compute_hfov_for_file, txt_path) for txt_path in txt_paths]
            for future in as_completed(futures):
                txt_name, hfov_int, err = future.result()
                if err is None:
                    hfov_values.append(hfov_int)
                else:
                    errors[txt_name] = err
    else:
        for txt_path in txt_paths:
            txt_name, hfov_int, err = _compute_hfov_for_file(txt_path)
            if err is None:
                hfov_values.append(hfov_int)
            else:
                errors[txt_name] = err

    hfov_values = sorted(set(hfov_values))
    return hfov_values, errors


def build_split_hfov_json(
    scenes_path: str,
    split_scene_list: List[str],
    verbose: bool = True,
    scene_workers: int = 8,
    file_workers: int = 0,
) -> Dict[str, Dict[str, List[int]]]:
    """
    Build split hfov json in the following format:
    {
      "scene_001": {"hfovs": [58]},
      "scene_002": {"hfovs": [42, 61]}
    }

    Speed-up strategy:
      1. Parallelize across scenes (recommended).
      2. Use fast txt parsing from head/tail chunks.
      3. Optional per-scene file-level threads (usually unnecessary).
    """
    split_hfov_info = {}
    missing_or_error_scenes = 0

    if not split_scene_list:
        return split_hfov_info

    # Scene-level parallelism is usually the best trade-off because each scene is independent.
    if scene_workers and scene_workers > 1:
        with ThreadPoolExecutor(max_workers=scene_workers) as executor:
            future_to_scene = {
                executor.submit(collect_scene_hfovs, os.path.join(scenes_path, scene), file_workers): scene
                for scene in split_scene_list
            }
            for future in as_completed(future_to_scene):
                scene = future_to_scene[future]
                hfovs, errors = future.result()
                split_hfov_info[scene] = {"hfovs": hfovs}
                if errors:
                    missing_or_error_scenes += 1
                    if verbose:
                        print(f"[Warning] Scene '{scene}' has {len(errors)} parsing issue(s).")
                        for key, msg in errors.items():
                            print(f"    - {key}: {msg}")
    else:
        for scene in split_scene_list:
            scene_dir = os.path.join(scenes_path, scene)
            hfovs, errors = collect_scene_hfovs(scene_dir, file_workers=file_workers)
            split_hfov_info[scene] = {"hfovs": hfovs}
            if errors:
                missing_or_error_scenes += 1
                if verbose:
                    print(f"[Warning] Scene '{scene}' has {len(errors)} parsing issue(s).")
                    for key, msg in errors.items():
                        print(f"    - {key}: {msg}")

    split_hfov_info = dict(sorted(split_hfov_info.items(), key=lambda kv: kv[0]))

    if verbose:
        print(
            f"Built hfov json for {len(split_scene_list)} scenes, "
            f"scenes with warnings: {missing_or_error_scenes}"
        )

    return split_hfov_info


def save_hfov_json(hfov_info: Dict[str, Dict[str, List[int]]], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hfov_info, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Saved hfov json: {output_path}")


def save_split_outputs(
    scenes_path: str,
    metadata: str,
    dataset: str,
    split_name: str,
    scene_list: List[str],
    scene_suffix: str = "scene_list",
    hfov_suffix: str = "scene_hfov",
    scene_workers: int = 8,
    file_workers: int = 0,
    skip_existing_json: bool = False,
):
    if len(scene_list) == 0:
        return

    split_dir = os.path.join(metadata, split_name)
    os.makedirs(split_dir, exist_ok=True)

    scene_output_path = os.path.join(
        split_dir, f"{dataset}_{scene_suffix}_{split_name}.npy"
    )
    save_scene_lists(scene_list, scene_output_path)

    hfov_output_path = os.path.join(
        split_dir, f"{dataset}_{hfov_suffix}_{split_name}.json"
    )
    if skip_existing_json and os.path.isfile(hfov_output_path):
        print(f"Skip existing hfov json: {hfov_output_path}")
        return

    hfov_info = build_split_hfov_json(
        scenes_path,
        scene_list,
        scene_workers=scene_workers,
        file_workers=file_workers,
    )
    save_hfov_json(hfov_info, hfov_output_path)


def process_dataset(root, metadata, dataset, scene_workers=8, file_workers=0, skip_existing_json=False):
    """Process a single dataset."""
    scenes_path = os.path.join(root, dataset)

    if not os.path.isdir(scenes_path):
        print(f"Warning: Dataset path does not exist: {scenes_path}")
        return

    scenes = [
        s for s in os.listdir(scenes_path)
        if os.path.isdir(os.path.join(scenes_path, s))
    ]
    scenes = sorted(scenes)

    if dataset == "whu_whuomvs":
        train, val, test = whu_whuomvs_split(scenes)
    elif dataset == "uavscenes":
        train, val, test = uavscenes_split(scenes)
    elif dataset == "ortholoc":
        train, val, test = ortho_split(scenes)
    elif dataset == "UAVFF3D-Real":
        train, val, test = uavff3d_real_split(scenes)
    elif dataset == "UAVFF3D-Syn-L":
        train, val, test = uavff3d_syn_large_split(scenes)
    elif dataset == "UAVFF3D-Syn-S":
        train, val, test = uavff3d_syn_small_split(scenes)
    elif dataset == "UAVFF3D-FA":
        train, val, test = uavff3d_fa_split(scenes)
    elif dataset == "wilduav":
        train, val, test = wilduav_split(scenes)
    elif dataset == "blendedmvs":
        train, val, test = blendedmvs_split(scenes)
    elif dataset == "usegeo":
        train, val, test = usegeo_split(scenes)
    elif dataset == "urbanscene3d":
        train, val, test = urbanscene3d_split(scenes)
    elif dataset == "enrich":
        train, val, test = enrich_split(scenes)
    else:
        print(f"Warning: Unknown dataset '{dataset}'. Skipping.")
        return

    save_split_outputs(
        scenes_path=scenes_path,
        metadata=metadata,
        dataset=dataset,
        split_name="train",
        scene_list=train,
        scene_suffix="scene_list",
        hfov_suffix="scene_hfov",
        scene_workers=scene_workers,
        file_workers=file_workers,
        skip_existing_json=skip_existing_json,
    )
    save_split_outputs(
        scenes_path=scenes_path,
        metadata=metadata,
        dataset=dataset,
        split_name="val",
        scene_list=val,
        scene_suffix="scene_list",
        hfov_suffix="scene_hfov",
        scene_workers=scene_workers,
        file_workers=file_workers,
        skip_existing_json=skip_existing_json,
    )
    save_split_outputs(
        scenes_path=scenes_path,
        metadata=metadata,
        dataset=dataset,
        split_name="test",
        scene_list=test,
        scene_suffix="scene_list",
        hfov_suffix="scene_hfov",
        scene_workers=scene_workers,
        file_workers=file_workers,
        skip_existing_json=skip_existing_json,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument(
        "--dataset",
        nargs='+',
        default=["UAVFF3D-Real", "UAVFF3D-Syn-L", "UAVFF3D-Syn-S", "UAVFF3D-FA", 
                 "uavscenes", "blendedmvs", "usegeo", "whu_whuomvs", "urbanscene3d", "enrich"]
    )
    parser.add_argument(
        "--scene_workers",
        type=int,
        default=min(16, max(1, (os.cpu_count() or 8))),
        help="Number of worker threads used across scenes. This is the main acceleration switch.",
    )
    parser.add_argument(
        "--file_workers",
        type=int,
        default=0,
        help="Optional per-scene worker threads for camera txt files. Usually keep 0.",
    )
    parser.add_argument(
        "--skip_existing_json",
        action="store_true",
        help="Skip hfov json generation if the target json already exists.",
    )
    cfg = parser.parse_args()

    datasets = [cfg.dataset] if isinstance(cfg.dataset, str) else cfg.dataset

    for dataset in datasets:
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {dataset}")
        print(f"{'=' * 80}")
        process_dataset(
            cfg.root,
            cfg.metadata,
            dataset,
            scene_workers=cfg.scene_workers,
            file_workers=cfg.file_workers,
            skip_existing_json=cfg.skip_existing_json,
        )


if __name__ == "__main__":
    main()
