from pathlib import Path
from argparse import ArgumentParser

from segment_anything import build_sam, SamPredictor
import numpy as np
from skimage import measure
import cv2
from tqdm import tqdm

from coco import CocoDataset
from utils import load_json, save_json

CKPT_PATH = Path("ckpt/sam_vit_h_4b8939.pth")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif"}

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("coco_json")
    parser.add_argument("images")
    parser.add_argument("output")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    data = load_json(args.coco_json)
    coco_dataset = CocoDataset(**data)

    output_filename = Path(args.output)
    assert output_filename.suffix.lower() == ".json"

    predictor = SamPredictor(build_sam(checkpoint=CKPT_PATH.as_posix()))

    image_dir = Path(args.images)

    for coco_image in tqdm(coco_dataset.images, "Images", position=0, leave=False):
        image_filepath = image_dir / coco_image.file_name
        annotations = [a for a in coco_dataset.annotations if a.image_id == coco_image.id]
        image = cv2.imread(image_filepath.as_posix())
        predictor.set_image(image)
        for annotation in tqdm(annotations, "Annotations", position=1, leave=False):
            bbox = np.array(annotation.bbox)
            mask = predictor.predict(box=bbox, multimask_output=False)[0][0]
            contours = measure.find_contours(mask, 0.5)
            if not contours:
                continue
            contour = contours[0]
            annotation.segmentation = [contour.flatten().tolist()]
            x=0

    save_json(coco_dataset.dict(), output_filename)
    print(f"Saved auto-segmented JSON at: {output_filename.as_posix()}")


if __name__ == "__main__":
    main()