import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


def opk_to_rotation_matrix(omega_deg: float, phi_deg: float, kappa_deg: float) -> np.ndarray:
    """
    ContextCapture / BlocksExchange 官方 OPK 公式
    返回:
        R_cw: world -> camera
    这里默认 CameraOrientation = XRightYDown
    （你的 DJI 导出里没有该字段时，按规范默认值处理）
    """
    o = np.deg2rad(omega_deg)
    p = np.deg2rad(phi_deg)
    k = np.deg2rad(kappa_deg)

    co, so = np.cos(o), np.sin(o)
    cp, sp = np.cos(p), np.sin(p)
    ck, sk = np.cos(k), np.sin(k)

    R = np.array([
        [ cp * ck,                 co * sk + so * sp * ck,   so * sk - co * sp * ck],
        [-cp * sk,                 co * ck - so * sp * sk,   so * ck + co * sp * sk],
        [ sp,                     -so * cp,                  co * cp]
    ], dtype=np.float64)
    return R


def build_intrinsic_matrix(
    focal_length_pixels: float,
    cx: float,
    cy: float,
    aspect_ratio: float = 1.0,
    skew: float = 0.0,
) -> np.ndarray:
    fx = float(focal_length_pixels)
    fy = float(focal_length_pixels) * float(aspect_ratio)
    s = float(skew)

    K = np.array([
        [fx, s,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)
    return K


def normalize_image_path(path: str) -> str:
    return path.replace("\\", "/")


def parse_srs_origin(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"SRSOrigin 格式错误: {text}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def parse_model_metadata(metadata_xml_path: str) -> dict:
    tree = ET.parse(metadata_xml_path)
    root = tree.getroot()

    srs = root.findtext("SRS", default="").strip()
    srs_origin_text = root.findtext("SRSOrigin", default="").strip()

    if not srs_origin_text:
        raise ValueError(f"metadata.xml 中未找到 SRSOrigin: {metadata_xml_path}")

    srs_origin = parse_srs_origin(srs_origin_text)

    return {
        "srs": srs,
        "srs_origin": srs_origin,
    }


def make_extrinsic_from_center(R_cw: np.ndarray, camera_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    已知:
        R_cw: world -> camera 旋转
        camera_center: 相机中心在 world 坐标系中的坐标
    """
    C = camera_center.reshape(3, 1)
    t_cw = -R_cw @ C

    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = t_cw[:, 0]

    T_wc = np.linalg.inv(T_cw)
    return T_cw, T_wc


def parse_rotation(photo, camera_orientation: str = "XRightYDown") -> np.ndarray:
    """
    先尝试读旋转矩阵 M_ij；
    若没有，再读 Omega/Phi/Kappa。
    目前你的 DJI 导出没有 CameraOrientation，按默认 XRightYDown 处理。
    """
    rot = photo.find("Pose/Rotation")
    if rot is None:
        raise ValueError("缺少 Pose/Rotation")

    if camera_orientation != "XRightYDown":
        raise NotImplementedError(
            f"当前代码只按 XRightYDown 处理，实际为: {camera_orientation}"
        )

    # 优先读取矩阵
    m00 = rot.findtext("M_00")
    if m00 is not None:
        R_cw = np.array([
            [float(rot.findtext("M_00")), float(rot.findtext("M_01")), float(rot.findtext("M_02"))],
            [float(rot.findtext("M_10")), float(rot.findtext("M_11")), float(rot.findtext("M_12"))],
            [float(rot.findtext("M_20")), float(rot.findtext("M_21")), float(rot.findtext("M_22"))],
        ], dtype=np.float64)
        return R_cw

    omega = float(rot.findtext("Omega", default="0"))
    phi   = float(rot.findtext("Phi", default="0"))
    kappa = float(rot.findtext("Kappa", default="0"))
    return opk_to_rotation_matrix(omega, phi, kappa)


def parse_dji_blocks_exchange_with_obj_local(
    blocks_xml_path: str,
    metadata_xml_path: str | None = None,
) -> dict:
    obj_meta = None
    srs_origin = None
    if metadata_xml_path is not None:
        obj_meta = parse_model_metadata(metadata_xml_path)
        srs_origin = obj_meta["srs_origin"]
        print(f"srs origin: {srs_origin}")

    tree = ET.parse(blocks_xml_path)
    root = tree.getroot()

    srs_elem = root.find(".//SpatialReferenceSystems/SRS")
    blocks_srs_name = srs_elem.findtext("Name", default="") if srs_elem is not None else ""
    blocks_srs_definition = srs_elem.findtext("Definition", default="") if srs_elem is not None else ""

    results = {}

    photogroups = root.findall(".//Photogroup")
    for pg in photogroups:
        photogroup_name = pg.findtext("Name", default="")
        camera_orientation = pg.findtext("CameraOrientation", default="XRightYDown").strip()

        width = int(pg.findtext("ImageDimensions/Width", default="0"))
        height = int(pg.findtext("ImageDimensions/Height", default="0"))
        camera_model_type = pg.findtext("CameraModelType", default="")

        focal_length_pixels = float(pg.findtext("FocalLengthPixels", default="0"))
        cx = float(pg.findtext("PrincipalPoint/x", default="0"))
        cy = float(pg.findtext("PrincipalPoint/y", default="0"))
        aspect_ratio = float(pg.findtext("AspectRatio", default="1"))
        skew = float(pg.findtext("Skew", default="0"))

        distortion = {
            "K1": float(pg.findtext("Distortion/K1", default="0")),
            "K2": float(pg.findtext("Distortion/K2", default="0")),
            "K3": float(pg.findtext("Distortion/K3", default="0")),
            "P1": float(pg.findtext("Distortion/P1", default="0")),
            "P2": float(pg.findtext("Distortion/P2", default="0")),
        }

        K = build_intrinsic_matrix(
            focal_length_pixels=focal_length_pixels,
            cx=cx,
            cy=cy,
            aspect_ratio=aspect_ratio,
            skew=skew,
        )

        photos = pg.findall("Photo")
        for photo in photos:
            image_path_raw = photo.findtext("ImagePath", default="")
            image_path = normalize_image_path(image_path_raw)
            image_name = os.path.basename(image_path)

            omega = float(photo.findtext("Pose/Rotation/Omega", default="0"))
            phi = float(photo.findtext("Pose/Rotation/Phi", default="0"))
            kappa = float(photo.findtext("Pose/Rotation/Kappa", default="0"))

            x_global = float(photo.findtext("Pose/Center/x", default="0"))
            y_global = float(photo.findtext("Pose/Center/y", default="0"))
            z_global = float(photo.findtext("Pose/Center/z", default="0"))

            camera_center_global = np.array([x_global, y_global, z_global], dtype=np.float64)

            if srs_origin is not None:
                camera_center_local = camera_center_global - srs_origin
            else:
                camera_center_local = camera_center_global.copy()

            R_cw = parse_rotation(photo, camera_orientation)

            T_cw_global, T_wc_global = make_extrinsic_from_center(R_cw, camera_center_global)
            T_cw_local, T_wc_local = make_extrinsic_from_center(R_cw, camera_center_local)

            results[image_name] = {
                "image_name": image_name,
                "image_path": image_path,
                "image_size": [width, height],
                "intrinsic_matrix": K.tolist(),

                "camera_center": camera_center_local.tolist(),
                "extrinsic_matrix_world_to_camera": T_cw_local.tolist(),
                "extrinsic_matrix_camera_to_world": T_wc_local.tolist(),

                "camera_center_global": camera_center_global.tolist(),
                "camera_center_local": camera_center_local.tolist(),
                "extrinsic_matrix_world_to_camera_global": T_cw_global.tolist(),
                "extrinsic_matrix_camera_to_world_global": T_wc_global.tolist(),
                "extrinsic_matrix_world_to_camera_local": T_cw_local.tolist(),
                "extrinsic_matrix_camera_to_world_local": T_wc_local.tolist(),
                "rotation_matrix_world_to_camera": R_cw.tolist(),
                "distortion": distortion,
                "photogroup_name": photogroup_name,
                "camera_model_type": camera_model_type,
                "camera_orientation": camera_orientation,
                "blocks_exchange_srs_name": blocks_srs_name,
                "blocks_exchange_srs_definition": blocks_srs_definition,
                "obj_metadata_srs": obj_meta["srs"] if obj_meta is not None else None,
                "obj_srs_origin": srs_origin.tolist() if srs_origin is not None else None,
                "opk_degree": {
                    "omega": omega,
                    "phi": phi,
                    "kappa": kappa,
                }
            }

    return results


def calculate_fov(fy, h):
    return 2.0 * np.arctan(float(h) / (2.0 * float(fy))) * 180.0 / np.pi


def write_cam_file(cam_path: Path, extrinsic, fx, fy, cx, cy, h, w):
    fov = calculate_fov(fy, h)
    with open(cam_path, "w", encoding="utf-8") as f:
        f.write("extrinsic opencv(x Right, y Down, z Forward) world2camera\n")
        for i in range(4):
            r = extrinsic[i]
            f.write(f"{r[0]:.12f} {r[1]:.12f} {r[2]:.12f} {r[3]:.12f}\n")
        f.write("\n")

        f.write("intrinsic: fx fy cx cy (pixel)\n")
        f.write(f"{fx:.12f} 0.000000000000 {cx:.12f}\n")
        f.write(f"0.000000000000 {fy:.12f} {cy:.12f}\n")
        f.write("0.000000000000 0.000000000000 1.000000000000\n")
        f.write("\n")

        f.write("h w fov\n")
        f.write(f"{h} {w} {fov:.12f}\n")


def save_all_cam_txt(data: dict, result_dir: str):
    result_dir = Path(result_dir)
    cams_dir = result_dir / "cams"
    cams_dir.mkdir(parents=True, exist_ok=True)

    for image_name, item in data.items():
        w, h = item["image_size"]  # 原数据里是 [W, H]
        K = np.asarray(item["intrinsic_matrix"], dtype=np.float64)
        extrinsic = np.asarray(item["extrinsic_matrix_world_to_camera"], dtype=np.float64)

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        txt_name = Path(image_name).stem + ".txt"
        cam_path = cams_dir / txt_name

        write_cam_file(
            cam_path=cam_path,
            extrinsic=extrinsic,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            h=h,
            w=w,
        )

    print(f"Saved {len(data)} camera txt files to: {cams_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse DJI Terra BlocksExchange XML and save camera parameters as txt files"
    )
    parser.add_argument(
        "--blocks_xml_path",
        type=str,
        default="BlocksExchangeUndistortAT_WithoutTiePoints.xml",
        help="Path to BlocksExchange XML"
    )
    parser.add_argument(
        "--metadata_xml_path",
        type=str,
        default="metadata.xml",
        help="Path to OBJ metadata.xml"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Result directory. A subfolder named cams will be created inside."
    )
    args = parser.parse_args()

    data = parse_dji_blocks_exchange_with_obj_local(
        blocks_xml_path=args.blocks_xml_path,
        metadata_xml_path=args.metadata_xml_path,
    )

    print(f"Parsed {len(data)} images.")

    if len(data) > 0:
        first_key = next(iter(data))
        print(f"Example image: {first_key}")

    save_all_cam_txt(data, args.result_dir)


if __name__ == "__main__":
    main()

"""
python djiterra2wai.py --blocks_xml_path nanfang/AT/BlocksExchangeUndistortAT_WithoutTiePoints.xml \
    --metadata_xml_path nanfang/models/pc/0/terra_obj/metadata.xml \
    --result_dir nanfang

python djiterra2wai.py --blocks_xml_path yanghaitang/AT/BlocksExchangeUndistortAT_WithoutTiePoints.xml \
    --metadata_xml_path yanghaitang/models/pc/0/terra_obj/metadata.xml \
    --result_dir yanghaitang

python djiterra2wai.py --blocks_xml_path xiaoxiang/AT/BlocksExchangeUndistortAT_WithoutTiePoints.xml \
    --metadata_xml_path xiaoxiang/models/pc/0/terra_obj/metadata.xml \
    --result_dir xiaoxiang
"""
