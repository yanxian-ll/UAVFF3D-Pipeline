from __future__ import annotations

import argparse
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "kvuong2711/aerialmegadepth"
ALLOW_PATTERNS = ("**.zip", "aerial_megadepth_all.npz")
DEFAULT_MAX_WORKERS = 8
ZIP_DIR_NAME = "aerialmegadepth_zip"
EXTRACT_DIR_NAME = "aerialmegadepth"


def download_archives(zip_dir: Path, max_workers: int):
    """Download dataset archives into ``zip_dir``."""
    zip_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID} archives to {zip_dir}...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(zip_dir),
        max_workers=max_workers,
        allow_patterns=list(ALLOW_PATTERNS),
    )
    print("Download complete!")


def extract_zip_archives(
    target_dir: Path,
    output_dir: Path,
    n_workers: int = DEFAULT_MAX_WORKERS,
    remove_zip_after_extract: bool = False,
):
    """
    Extract all zip archives under ``target_dir`` into ``output_dir``.

    Args:
        target_dir: Directory containing zip archives.
        output_dir: Directory to extract all contents into.
        n_workers: Number of parallel extraction workers.
        remove_zip_after_extract: Whether to delete zip files after successful extraction.
    """
    target_dir = Path(target_dir)
    output_dir = Path(output_dir)

    if not target_dir.exists():
        raise FileNotFoundError(f"Zip directory does not exist: {target_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(target_dir.rglob("*.zip"))
    if len(zip_files) == 0:
        raise FileNotFoundError(f"No zip archives found under: {target_dir}")

    print(f"Found {len(zip_files)} zip archives in {target_dir}")
    print(f"Extracting to {output_dir} with {n_workers} workers...")

    def _extract_one(zip_path: Path):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                bad_file = zf.testzip()
                if bad_file is not None:
                    raise zipfile.BadZipFile(
                        f"Corrupted member '{bad_file}' found in {zip_path}"
                    )
                zf.extractall(output_dir)

            if remove_zip_after_extract:
                zip_path.unlink()

            return zip_path.name, None
        except Exception as e:
            return zip_path.name, e

    errors = []

    if n_workers <= 1:
        for zip_path in zip_files:
            name, err = _extract_one(zip_path)
            if err is None:
                print(f"[OK] Extracted: {name}")
            else:
                print(f"[FAIL] {name}: {err}")
                errors.append((name, err))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_extract_one, z): z for z in zip_files}
            for future in as_completed(futures):
                name, err = future.result()
                if err is None:
                    print(f"[OK] Extracted: {name}")
                else:
                    print(f"[FAIL] {name}: {err}")
                    errors.append((name, err))

    if errors:
        err_msg = "\n".join([f"{name}: {err}" for name, err in errors])
        raise RuntimeError(f"Some zip archives failed to extract:\n{err_msg}")

    print("All zip archives extracted successfully.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the Aerial MegaDepth dataset and optionally extract archives.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Base directory for downloaded archives and extracted data.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Number of parallel workers used by the Hugging Face downloader.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    zip_dir = target_dir / ZIP_DIR_NAME
    extract_dir = target_dir / EXTRACT_DIR_NAME

    # 1. Download zip files from huggingface
    download_archives(zip_dir, max_workers=args.max_workers)

    # 2. Extract zip files
    extract_zip_archives(
        target_dir=zip_dir,
        output_dir=extract_dir,
        n_workers=args.max_workers,
    )

    # 3. Move the aerial_megadepth_all.npz to the extract_dir
    shutil.move(
        zip_dir / "aerial_megadepth_all.npz",
        extract_dir / "aerial_megadepth_all.npz",
    )

    print("All tasks completed successfully.")


if __name__ == "__main__":
    main()
    