from __future__ import annotations

from pydantic import BaseModel
from pydantic.typing import List, Union


class CocoCategory(BaseModel):
    id: int
    name: str
    supercategory: str


class CocoAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    bbox: List[Union[int, float]]
    segmentation: List[List[Union[int, float]]]
    area: float
    score: Union[float, None]
    iscrowd: bool


class CocoImage(BaseModel):
    id: int
    file_name: str
    width: int
    height: int


class CocoDataset(BaseModel):
    images: List[CocoImage]
    annotations: List[CocoAnnotation]
    categories: List[CocoCategory]
    licenses: list = []
    info: dict = []
