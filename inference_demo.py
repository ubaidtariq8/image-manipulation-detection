from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from model import build_model
from transforms import IMAGENET_MEAN, IMAGENET_STD


DEFAULT_CONFIG_PATH = Path("configs/unet_mit_b1_casia2.yaml")
DEFAULT_CHECKPOINT_PATH = Path("weights/best_weights.pt")
DEFAULT_IMAGE_SIZE = 384


def run(
    image_source: str,
    threshold: float = 0.5,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
) -> dict[str, float]:
    if not image_source:
        raise ValueError("Set IMAGE_SOURCE to a local image path or image URL first.")

    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    config = _load_config(config_path)
    image_size = int(config.get("data", {}).get("image_size", DEFAULT_IMAGE_SIZE))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(_extract_state_dict(checkpoint))
    model.eval()

    original = _load_rgb_image(image_source)
    tensor, meta = _preprocess(original, image_size)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        output = model(tensor)
        class_probability = torch.sigmoid(output["class_logits"]).item()
        mask_probability = torch.sigmoid(output["mask_logits"])[0, 0].float().cpu().numpy()

    mask_probability = _restore_mask(mask_probability, meta)
    binary_mask = mask_probability >= threshold
    overlay = _overlay_mask(original, binary_mask)
    classification = "manipulated" if class_probability >= threshold else "authentic"

    print(f"Classification: {classification}")
    print(f"Manipulation probability: {class_probability:.4f}")
    print(f"Threshold: {threshold:.2f}")

    _display_results(original, binary_mask, overlay)
    return {
        "classification_probability": float(class_probability),
        "threshold": float(threshold),
        "predicted_label": int(class_probability >= threshold),
        "mask_positive_ratio": float(binary_mask.mean()),
    }


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _load_rgb_image(source: str) -> np.ndarray:
    if source.startswith(("http://", "https://")):
        with urlopen(source) as response:
            image = Image.open(BytesIO(response.read())).convert("RGB")
    else:
        image = Image.open(source).convert("RGB")
    return np.array(image)


def _preprocess(image: np.ndarray, image_size: int) -> tuple[torch.Tensor, dict[str, int]]:
    height, width = image.shape[:2]
    scale = image_size / max(height, width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_top = (image_size - resized_height) // 2
    pad_bottom = image_size - resized_height - pad_top
    pad_left = (image_size - resized_width) // 2
    pad_right = image_size - resized_width - pad_left
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    normalized = padded.astype(np.float32) / 255.0
    normalized = (normalized - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
        IMAGENET_STD,
        dtype=np.float32,
    )
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
    return tensor, {
        "height": height,
        "width": width,
        "resized_height": resized_height,
        "resized_width": resized_width,
        "pad_top": pad_top,
        "pad_left": pad_left,
    }


def _restore_mask(mask: np.ndarray, meta: dict[str, int]) -> np.ndarray:
    top = meta["pad_top"]
    left = meta["pad_left"]
    resized_height = meta["resized_height"]
    resized_width = meta["resized_width"]
    cropped = mask[top : top + resized_height, left : left + resized_width]
    return cv2.resize(
        cropped,
        (meta["width"], meta["height"]),
        interpolation=cv2.INTER_LINEAR,
    )


def _overlay_mask(image: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image.copy()
    highlight = np.zeros_like(image)
    highlight[..., 0] = 255
    overlay[binary_mask] = (
        (1.0 - alpha) * image[binary_mask] + alpha * highlight[binary_mask]
    ).astype(np.uint8)
    return overlay


def _display_results(
    image: np.ndarray,
    binary_mask: np.ndarray,
    overlay: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Source Image")
    axes[1].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Binary Mask")
    axes[2].imshow(overlay)
    axes[2].set_title("Highlighted Manipulated Regions")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.show()
