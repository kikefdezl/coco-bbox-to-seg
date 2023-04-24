from __future__ import annotations

import json
from pathlib import Path

import requests
from tqdm import tqdm

from .settings import MODEL, MODELS, PACKAGE_ROOT


def download_ckpt():
    print("Downloading model checkpoint... ")
    r = requests.get(
        MODELS[MODEL]["DOWNLOAD_LINK"], allow_redirects=True, stream=True
    )
    total = int(r.headers.get("content-length", 0))
    fname = (Path(PACKAGE_ROOT) / MODELS[MODEL]["CKPT_PATH"]).as_posix()
    with open(fname, "wb") as file, tqdm(
        desc=fname, total=total, unit="iB", unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in r.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def load_json(filepath: str | Path) -> dict:
    """Returns the contents of a JSON file"""
    with open(filepath, "r") as jf:
        data = json.load(jf)
    return data


def save_json(data: dict, filepath: str | Path, pretty: bool = True):
    """Saves dictionary to a JSON file"""
    indent = 4 if pretty else None
    with open(filepath, "w") as jf:
        json.dump(data, jf, indent=indent)
