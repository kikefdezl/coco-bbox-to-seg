from __future__ import annotations

import json
from pathlib import Path

import requests

from .settings import MODEL, MODELS


def download_ckpt():
    print("Downloading model checkpoint... ", end="")
    r = requests.get(MODELS[MODEL]["DOWNLOAD_LINK"], allow_redirects=True)
    open(MODELS[MODEL]["CKPT_PATH"], "wb").write(r.content)
    print("done.")


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
