from argparse import ArgumentParser
from pathlib import Path

from bbox_to_seg.inference import coco_bbox_to_coco_seg
from bbox_to_seg.utils import load_json, save_json
from bbox_to_seg.coco import CocoDataset
from bbox_to_seg.settings import DEFAULT_OUTPUT_FILENAME


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("coco_json", help="Path to the COCO JSON file.")
    parser.add_argument("images", help="Path to the folder with the images.")
    parser.add_argument("output", help="Output path.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    data = load_json(args.coco_json)
    coco_dataset = CocoDataset(**data)

    output_filename = Path(args.output)
    if output_filename.suffix.lower() != ".json":
        output_filename = output_filename / DEFAULT_OUTPUT_FILENAME

    inferred_data = coco_bbox_to_coco_seg(args.coco_json, args.images)

    save_json(inferred_data.dict(), output_filename)
    print(f"Saved auto-segmented JSON at: {output_filename.as_posix()}")
