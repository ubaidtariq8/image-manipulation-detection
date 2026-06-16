from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch import nn


class ManipulationUnet(nn.Module):
    """U-Net segmentation model with an auxiliary binary classification head."""

    def __init__(
        self,
        encoder_name: str = "mit_b1",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        aux_pooling: str = "avg",
        aux_dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        aux_params = {
            "pooling": aux_pooling,
            "dropout": aux_dropout,
            "activation": None,
            "classes": 1,
        }
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            aux_params=aux_params,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.model(x)
        if isinstance(output, tuple):
            mask_logits, class_logits = output
        else:
            mask_logits = output
            class_logits = torch.empty(
                (x.shape[0], 1), device=x.device, dtype=mask_logits.dtype
            )
        return {"mask_logits": mask_logits, "class_logits": class_logits}


def build_model(config: dict[str, Any]) -> ManipulationUnet:
    model_config = config.get("model", config)
    return ManipulationUnet(
        encoder_name=model_config.get("encoder_name", "mit_b1"),
        encoder_weights=model_config.get("encoder_weights", "imagenet"),
        in_channels=model_config.get("in_channels", 3),
        classes=model_config.get("classes", 1),
        aux_pooling=model_config.get("aux_pooling", "avg"),
        aux_dropout=model_config.get("aux_dropout", 0.2),
    )
