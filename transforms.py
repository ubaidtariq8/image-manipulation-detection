from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 384) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, interpolation=1),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.08,
                contrast_limit=0.08,
                p=0.3,
            ),
            A.ImageCompression(quality_range=(85, 100), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(transpose_mask=True),
        ]
    )


def get_eval_transforms(image_size: int = 384) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, interpolation=1),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=0,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(transpose_mask=True),
        ]
    )


def build_transforms(config: dict[str, Any], split: str) -> A.Compose:
    image_size = int(config.get("data", config).get("image_size", 384))
    if split == "train":
        return get_train_transforms(image_size)
    return get_eval_transforms(image_size)
