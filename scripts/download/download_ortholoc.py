#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast downloader for OrthoLoC: unpacked/d/* assets

Remote layout:
  <BASE_URL>/unpacked/d/{train,val,test_inPlace,test_outPlace}/{point_maps,queries,cameras}/

File types:
  cameras    -> *.json
  point_maps -> *.ply
  queries    -> *.jpg / *.jpeg

Features:
  - connection reuse via requests.Session (keep-alive)
  - concurrent downloads via ThreadPoolExecutor
  - resumable downloads via .part + HTTP Range (if supported)
  - retries for transient errors (429/5xx)

Deps:
  pip install requests beautifulsoup4 tqdm

Examples:
  # download all splits to ./data
  python download_ortholoc_unpacked_d.py --out_dir ./data

  # download only train, with 32 workers
  python download_ortholoc_unpacked_d.py --split train --out_dir ./data --workers 32

  # custom base url
  python download_ortholoc_unpacked_d.py --base_url https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/OrthoLoC/
"""

from __future__ import annotations

import argparse
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


DATASET_URL = "https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/OrthoLoC/"

SPLITS = ("train", "val", "test_inPlace", "test_outPlace")
SPLITS = ("train", "val")

# subfolder -> regex for files to download
SUBFOLDERS = {
    "cameras": r"^.*\.json$",
    "point_maps": r"^.*\.ply$",
    "queries": r"^.*\.(jpg|jpeg)$",
}


def default_cache_dir() -> str:
    # Keep same convention as earlier: env overrides; else user cache.
    env = os.environ.get("ORTHOLOC_CACHE_DIR")
    if env:
        return env
    return str(Path.home() / ".cache" / "ortholoc")


def make_session() -> requests.Session:
    """
    Build a Session with a larger connection pool + retries.
    This often narrows the gap vs browser download performance.
    """
    s = requests.Session()

    retries = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)

    # Some hosts throttle/behave differently for default python UA.
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )
    return s


def get_file_links(session: requests.Session, url: str, pattern: Optional[str] = None) -> list[str]:
    """
    Parse Apache directory listing, return file URLs filtered by regex on href.
    """
    r = session.get(url, timeout=60)
    if r.status_code != 200:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    hrefs = [a.get("href") for a in soup.find_all("a") if a.get("href")]

    # Filter out parent dirs and subdirs (we only want files)
    # Apache listings often include "../" and folder links ending with "/"
    hrefs = [h for h in hrefs if h not in ("../",) and not h.endswith("/")]

    if pattern is None:
        return [urljoin(url.rstrip("/") + "/", h) for h in hrefs]

    reg = re.compile(pattern)
    return [urljoin(url.rstrip("/") + "/", h) for h in hrefs if reg.match(h)]


def head_content_length(session: requests.Session, url: str) -> Optional[int]:
    """
    Try to get Content-Length. Not all servers provide it.
    """
    try:
        r = session.head(url, timeout=30, allow_redirects=True)
        if r.status_code >= 400:
            return None
        cl = r.headers.get("Content-Length")
        return int(cl) if cl is not None else None
    except Exception:
        return None


def download_file(
    session: requests.Session,
    url: str,
    save_path: str,
    chunk_size: int = 8 * 1024 * 1024,
    verify_size: bool = True,
    lock: Optional[threading.Lock] = None,
) -> Optional[str]:
    """
    Download one file with resume support (.part + HTTP Range).

    Behavior:
      - if save_path exists:
          - if verify_size and Content-Length known and matches -> skip
          - else skip (best effort)
      - else if save_path.part exists:
          - try Range resume from current .part size
          - if server ignores Range -> restart from 0
      - write to .part then atomic replace into final
    """
    save_path = str(save_path)
    tmp_path = save_path + ".part"

    # If already downloaded, optionally validate size
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        if verify_size:
            remote_size = head_content_length(session, url)
            if remote_size is not None and os.path.getsize(save_path) != remote_size:
                # size mismatch -> re-download
                try:
                    os.remove(save_path)
                except Exception:
                    return None
            else:
                return save_path
        else:
            return save_path

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    resume_from = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else {}

    try:
        with session.get(url, stream=True, timeout=120, headers=headers) as r:
            # If we requested Range but server ignored it, restart
            if resume_from > 0 and r.status_code == 200:
                resume_from = 0
                headers = {}
                # restart clean
                r.close()
                with session.get(url, stream=True, timeout=120) as r2:
                    if r2.status_code != 200:
                        return None
                    with open(tmp_path, "wb") as f:
                        for chunk in r2.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
            else:
                if r.status_code not in (200, 206):
                    return None
                mode = "ab" if (resume_from > 0 and r.status_code == 206) else "wb"
                with open(tmp_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

        # Optional size check after download
        if verify_size:
            remote_size = head_content_length(session, url)
            if remote_size is not None and os.path.getsize(tmp_path) != remote_size:
                # incomplete -> keep .part for future resume
                return None

        os.replace(tmp_path, save_path)
        return save_path

    except Exception:
        return None


def download_dir(
    session: requests.Session,
    remote_dir_url: str,
    local_dir: str,
    pattern: str,
    workers: int = 16,
    chunk_size: int = 8 * 1024 * 1024,
    verify_size: bool = True,
) -> bool:
    """
    Download all matching files from a remote Apache listing directory using concurrency.
    Returns True if directory exists and at least one file is present (downloaded or already exists).
    """
    os.makedirs(local_dir, exist_ok=True)

    links = get_file_links(session, remote_dir_url, pattern=pattern)
    if not links:
        return False

    # If some files already exist, we still schedule them; download_file will skip quickly
    lock = threading.Lock()
    futures = []
    ok_any = False

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for file_url in links:
            fname = os.path.basename(file_url)
            out_path = os.path.join(local_dir, fname)
            futures.append(
                ex.submit(
                    download_file,
                    session,
                    file_url,
                    out_path,
                    chunk_size,
                    verify_size,
                    lock,
                )
            )

        pbar = tqdm(as_completed(futures), total=len(futures), desc=f"{os.path.basename(local_dir)}", unit="file")
        for fu in pbar:
            res = fu.result()
            if res is not None:
                ok_any = True

    return ok_any


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default='../raw_data/ortholoc', help="Local output root.")
    ap.add_argument("--base_url", type=str, default=DATASET_URL, help="Dataset root URL.")
    ap.add_argument("--split", type=str, default="all", choices=("all",) + SPLITS, help="Split to download.")
    ap.add_argument("--workers", type=int, default=16, help="Concurrent workers per folder (8-32 recommended).")
    ap.add_argument("--chunk_mb", type=int, default=8, help="Chunk size in MB for streaming download.")
    ap.add_argument("--no_verify_size", action="store_true", help="Skip Content-Length size verification.")
    args = ap.parse_args()

    session = make_session()

    base_url = args.base_url.rstrip("/") + "/"
    out_root = args.out_dir
    workers = max(1, int(args.workers))
    chunk_size = max(1, int(args.chunk_mb)) * 1024 * 1024
    verify_size = not args.no_verify_size

    splits = SPLITS if args.split == "all" else (args.split,)

    for sp in splits:
        for sub, regex in SUBFOLDERS.items():
            remote_dir = urljoin(base_url, f"unpacked/{sp}/{sub}/")
            local_dir = os.path.join(out_root, "unpacked", sp, sub)

            ok = download_dir(
                session=session,
                remote_dir_url=remote_dir,
                local_dir=local_dir,
                pattern=regex,
                workers=workers,
                chunk_size=chunk_size,
                verify_size=verify_size,
            )

            if ok:
                print(f"[OK] {remote_dir} -> {local_dir}")
            else:
                print(f"[WARN] empty or inaccessible: {remote_dir}")


if __name__ == "__main__":
    main()
