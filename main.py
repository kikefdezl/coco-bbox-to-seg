from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from segment_anything import build_sam, SamPredictor
from skimage import measure
from skimage.io import imread
from tqdm import tqdm

import settings
from coco import CocoDataset
from utils import load_json, save_json

CKPT_PATH = Path(settings.CKPT_PATH)
IMAGE_EXTS = settings.IMAGE_EXTS
DEFAULT_OUTPUT_FILENAME = settings.DEFAULT_OUTPUT_FILENAME


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("coco_json", help="Path to the COCO JSON file.")
    parser.add_argument("images", help="Path to the folder with the images.")
    parser.add_argument("output", help="Output path.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = load_json(args.coco_json)
    coco_dataset = CocoDataset(**data)

    output_filename = Path(args.output)
    if output_filename.suffix.lower() != ".json":
        output_filename = output_filename / DEFAULT_OUTPUT_FILENAME

    predictor = SamPredictor(build_sam(checkpoint=CKPT_PATH.as_posix()))

    image_dir = Path(args.images)

    for coco_image in tqdm(coco_dataset.images, "Images", position=0,
                           leave=False):
        image_filepath = image_dir / coco_image.file_name
        annotations = [a for a in coco_dataset.annotations if
                       a.image_id == coco_image.id]
        image = imread(image_filepath.as_posix())
        predictor.set_image(image, image_format="BGR")
        for annotation in tqdm(annotations, "Annotations", position=1,
                               leave=False):
            bbox = np.array([
                annotation.bbox[0],
                annotation.bbox[1],
                annotation.bbox[0] + annotation.bbox[2],
                annotation.bbox[1] + annotation.bbox[3]
            ])
            mask = predictor.predict(box=bbox, multimask_output=False)[0][0]
            contours = measure.find_contours(mask.T, 0.5)
            if not contours:
                continue
            contour = contours[0]
            annotation.segmentation = [contour.flatten().tolist()]

    save_json(coco_dataset.dict(), output_filename)
    print(f"Saved auto-segmented JSON at: {output_filename.as_posix()}")


if __name__ == "__main__":
    main()
