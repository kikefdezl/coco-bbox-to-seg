from argparse import ArgumentParser

from bbox_to_seg.inference import coco_bbox_to_coco_seg


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("coco_json", help="Path to the COCO JSON file.")
    parser.add_argument("images", help="Path to the folder with the images.")
    parser.add_argument("output", help="Output path.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    coco_bbox_to_coco_seg(args.coco_json, args.images, args.output)
