from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


NONE_MASK = "None"


class ManipulationDataset(Dataset):
    def __init__(
        self,
        split_json: str | Path,
        transform: Any | None = None,
    ) -> None:
        self.split_json = Path(split_json)
        self.transform = transform
        with self.split_json.open("r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image_path = Path(sample["img_path"])
        mask_path = sample["gt_mask"]

        image = _read_rgb(image_path)
        label = 0.0 if mask_path == NONE_MASK else 1.0
        if mask_path == NONE_MASK:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        else:
            mask = _read_binary_mask(Path(mask_path))

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image_tensor = transformed["image"].float()
            mask_tensor = transformed["mask"].float()
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask).float()

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        elif mask_tensor.ndim == 3 and mask_tensor.shape[0] != 1:
            mask_tensor = mask_tensor[:1]

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor([label], dtype=torch.float32),
            "img_path": str(image_path),
            "gt_mask": str(mask_path),
        }


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.float32)
