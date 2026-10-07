#!/usr/bin/env python3
"""Download the LIBERO demonstration datasets used for BC training.

The datasets are not redistributed with this repository. They are published by
the LIBERO authors under CC BY 4.0:

  HuggingFace  https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
  Project page https://libero-project.github.io/datasets.html

The HuggingFace hub is the reproducible source. The original Box links used by
LIBERO's own downloader expire, so they are kept only as a fallback.

Suite -> HuggingFace directory and size (2025-05 revision):

  spatial  -> libero_spatial   10 files   6.24 GB
  object   -> libero_object    10 files   7.44 GB
  goal     -> libero_goal      10 files   6.37 GB
  scene10  -> libero_10        10 files  13.73 GB
  scene90  -> libero_90        90 files  66.66 GB

Usage:
    python scripts/download_datasets.py --suite spatial
    python scripts/download_datasets.py --suite all --out-dir /data/libero_datasets
    python scripts/download_datasets.py --suite scene90 --verify-only
"""
from __future__ import annotations

import argparse
import os
import sys

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"
# Pinned so a re-download is byte-for-byte the revision this repo was tested on.
HF_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"

SUITE_TO_DIR = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "scene10": "libero_10",
    "scene90": "libero_90",
}

BOX_FALLBACK = {
    "libero_spatial": "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip",
    "libero_object": "https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip",
    "libero_goal": "https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip",
    "libero_10": "https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip",
}

EXPECTED_FILES = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}

DEFAULT_OUT = os.environ.get(
    "LIBERO_DATASETS",
    os.path.join(os.environ.get("LIBERO_BASIL_PATH", "/workspace/libero_basil"),
                 "libero", "datasets"),
)


def verify(out_dir: str, dirs) -> bool:
    ok = True
    for d in dirs:
        path = os.path.join(out_dir, d)
        n = len([f for f in os.listdir(path) if f.endswith(".hdf5")]) if os.path.isdir(path) else 0
        want = EXPECTED_FILES[d]
        status = "ok" if n == want else "MISSING"
        if n != want:
            ok = False
        print(f"  {d:<16} {n:>3}/{want:<3} {status}")
    return ok


def download_hf(out_dir: str, dirs, revision: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    for d in dirs:
        print(f"[datasets] {d} <- {HF_REPO_ID}/{d} @ {revision[:12]}")
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision=revision,
            local_dir=out_dir,
            allow_patterns=f"{d}/*",
            local_dir_use_symlinks=False,
        )


def download_box(out_dir: str, dirs) -> None:
    import urllib.request
    import zipfile

    for d in dirs:
        url = BOX_FALLBACK.get(d)
        if url is None:
            print(f"[datasets] {d}: no Box fallback, use HuggingFace")
            continue
        dest = os.path.join(out_dir, f"{d}.zip")
        print(f"[datasets] {d} <- {url}")
        urllib.request.urlretrieve(url, dest)
        with zipfile.ZipFile(dest) as z:
            z.extractall(out_dir)
        os.unlink(dest)


def main() -> int:
    p = argparse.ArgumentParser(description="Download LIBERO demo datasets")
    p.add_argument("--suite", default="spatial",
                   choices=["spatial", "object", "goal", "scene10", "scene90", "all"])
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--revision", default=HF_REVISION)
    p.add_argument("--source", choices=["huggingface", "box"], default="huggingface")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    if args.suite == "all":
        dirs = list(SUITE_TO_DIR.values())
    else:
        dirs = [SUITE_TO_DIR[args.suite]]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[datasets] out-dir: {args.out_dir}")

    if not args.verify_only:
        if args.source == "huggingface":
            download_hf(args.out_dir, dirs, args.revision)
        else:
            download_box(args.out_dir, dirs)

    print("[datasets] verify")
    ok = verify(args.out_dir, dirs)
    print("[datasets] " + ("OK" if ok else "INCOMPLETE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
