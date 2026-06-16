from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from segmentation_models_pytorch.losses import BINARY_MODE, DiceLoss, FocalLoss
from torch import nn


@dataclass
class LossBreakdown:
    total: torch.Tensor
    segmentation: torch.Tensor
    classification: torch.Tensor
    dice: torch.Tensor
    focal: torch.Tensor


class ManipulationLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        classification_weight: float = 0.25,
        focal_alpha: float | None = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.classification_weight = classification_weight
        self.dice = DiceLoss(mode=BINARY_MODE, from_logits=True)
        self.focal = FocalLoss(
            mode=BINARY_MODE,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
        self.classification = nn.BCEWithLogitsLoss()

    def forward(
        self,
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
    ) -> LossBreakdown:
        dice_loss = self.dice(mask_logits, masks)
        focal_loss = self.focal(mask_logits, masks)
        segmentation_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        classification_loss = self.classification(class_logits, labels)
        total = segmentation_loss + self.classification_weight * classification_loss
        return LossBreakdown(
            total=total,
            segmentation=segmentation_loss.detach(),
            classification=classification_loss.detach(),
            dice=dice_loss.detach(),
            focal=focal_loss.detach(),
        )


def build_loss(config: dict[str, Any]) -> ManipulationLoss:
    loss_config = config.get("loss", config)
    return ManipulationLoss(
        dice_weight=loss_config.get("dice_weight", 1.0),
        focal_weight=loss_config.get("focal_weight", 1.0),
        classification_weight=loss_config.get("classification_weight", 0.25),
        focal_alpha=loss_config.get("focal_alpha", 0.25),
        focal_gamma=loss_config.get("focal_gamma", 2.0),
    )
