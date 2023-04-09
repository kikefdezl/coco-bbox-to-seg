"""
Evaluate the performance of SAM auto-segmentation in comparison to the real
segmentations in the COCO dataset.
"""
from __future__ import annotations

import copy
from argparse import ArgumentParser
from pathlib import Path

from shapely.geometry import Polygon
from tqdm import tqdm

from bbox_to_seg.coco import CocoDataset
from bbox_to_seg.inference import coco_bbox_to_coco_seg
from bbox_to_seg.utils import load_json

TMP_COCO_FILENAME = "tmp_coco.json"
OUTPUT_TXT_FILENAME = "avg_iou.txt"


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("coco_json", help="Path to the COCO JSON file.")
    parser.add_argument("images", help="Path to the folder with the images.")
    return parser.parse_args()


def coco_to_polygon(coco_points: list[int | float]):
    """Convert COCO format points to shapely Polygon"""
    coords = [
        (coco_points[i], coco_points[i + 1]) for i in range(0, len(coco_points), 2)
    ]
    return Polygon(coords)


def evaluate_sam_on_coco_dataset(coco_path: str | Path, images_path: str | Path):
    original_data = load_json(coco_path)
    # no support for crowd annotations
    original_data["annotations"] = [
        a for a in original_data["annotations"] if not a["iscrowd"]
    ]
    original_dataset = CocoDataset(**original_data)
    inferred_dataset = copy.deepcopy(original_dataset)

    print("Inferring on COCO dataset...")
    inferred_dataset = coco_bbox_to_coco_seg(inferred_dataset, images_path)

    ious = []
    for org_annot, inf_annot in tqdm(
        list(zip(original_dataset.annotations, inferred_dataset.annotations)),
        "Evaluating annotations...",
    ):
        assert org_annot.bbox == inf_annot.bbox
        org_polygon = coco_to_polygon(org_annot.segmentation)
        inf_polygon = coco_to_polygon(inf_annot.segmentation)

        intersection = org_polygon.intersection(inf_polygon).area
        union = org_polygon.union(inf_polygon).area
        ious.append(intersection / union)

    msg = f"Average IoU = {sum(ious) / len(ious)}"
    print(msg)
    with open(OUTPUT_TXT_FILENAME, "w") as outfile:
        outfile.write(msg)
        print(f"Saved result at: {Path(OUTPUT_TXT_FILENAME).resolve().as_posix()}")


if __name__ == "__main__":
    args = parse_arguments()
    evaluate_sam_on_coco_dataset(args.coco_json, args.images)
