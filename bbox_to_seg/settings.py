IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif"}
DEFAULT_OUTPUT_FILENAME = "segmented.json"
MODEL = "VIT_H"
MODELS = {
    "VIT_H": {
        "CKPT_PATH": "ckpt/sam_vit_h_4b8939.pth",
        "DOWNLOAD_LINK": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }
}
