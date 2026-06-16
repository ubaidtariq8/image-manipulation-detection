from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ManipulationDataset
from loss import build_loss
from model import build_model
from transforms import build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SMP U-Net on CASIA2.")
    parser.add_argument("--config", type=Path, default=Path("configs/unet_mit_b1_casia2.yaml"))
    parser.add_argument("--log-file", type=Path, default=Path("logs.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    logger = _setup_logger(args.log_file)
    _set_seed(int(config["training"].get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting training")
    logger.info("Config: %s", args.config)
    logger.info("Device: %s", device)
    model = build_model(config).to(device)
    criterion = build_loss(config)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["optimizer"].get("lr", 3e-4)),
        weight_decay=float(config["optimizer"].get("weight_decay", 1e-4)),
    )

    train_loader = _build_loader(config, "train")
    val_loader = _build_loader(config, "val")
    logger.info("Train batches: %d", len(train_loader))
    logger.info("Val batches: %d", len(val_loader))

    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    logger.info("AMP enabled: %s", amp_enabled)

    weights_dir = Path(config["training"].get("weights_dir", "weights"))
    weights_dir.mkdir(parents=True, exist_ok=True)
    early_stopping_config = config.get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping_config.get("enabled", True))
    patience = int(early_stopping_config.get("patience", 5))
    min_delta = float(early_stopping_config.get("min_delta", 0.0))
    best_val_dice = float("-inf")
    best_val_accuracy = float("-inf")
    epochs_without_improvement = 0
    epochs = int(config["training"].get("epochs", 100))
    scheduler = _build_scheduler(optimizer, config, epochs)
    logger.info(
        "Optimizer: AdamW lr=%s weight_decay=%s",
        config["optimizer"].get("lr", 3e-4),
        config["optimizer"].get("weight_decay", 1e-4),
    )
    scheduler_config = config.get("scheduler", {})
    logger.info(
        "Scheduler: warmup_epochs=%s cosine_min_lr=%s total_epochs=%s",
        scheduler_config.get("warmup_epochs", 1),
        scheduler_config.get("min_lr", 1e-6),
        epochs,
    )
    logger.info(
        "Early stopping: enabled=%s patience=%d min_delta=%s monitors=val_dice,val_cls_acc",
        early_stopping_enabled,
        patience,
        min_delta,
    )
    logger.info("Weights dir: %s", weights_dir)

    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %d/%d started lr=%.8f", epoch, epochs, lr)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            logger=logger,
        )
        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            logger=logger,
        )
        logger.info(
            (
                "Epoch %d/%d finished train_loss=%.4f train_cls_acc=%.4f "
                "val_loss=%.4f val_dice=%.4f val_cls_acc=%.4f val_pixel_acc=%.4f"
            ),
            epoch,
            epochs,
            train_metrics["loss"],
            train_metrics["class_accuracy"],
            val_metrics["loss"],
            val_metrics["dice_score"],
            val_metrics["class_accuracy"],
            val_metrics["pixel_accuracy"],
        )
        scheduler.step()

        dice_improved = val_metrics["dice_score"] > best_val_dice + min_delta
        accuracy_improved = val_metrics["class_accuracy"] > best_val_accuracy + min_delta
        if dice_improved:
            best_val_dice = val_metrics["dice_score"]
        if accuracy_improved:
            best_val_accuracy = val_metrics["class_accuracy"]
        improved = dice_improved or accuracy_improved

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_val_dice": best_val_dice,
            "best_val_accuracy": best_val_accuracy,
        }
        torch.save(checkpoint, weights_dir / "latest.pt")
        logger.info("Saved latest checkpoint: %s", weights_dir / "latest.pt")
        if improved:
            epochs_without_improvement = 0
            torch.save(checkpoint, weights_dir / "best.pt")
            logger.info(
                "Saved best checkpoint: %s dice_improved=%s accuracy_improved=%s best_dice=%.4f best_cls_acc=%.4f",
                weights_dir / "best.pt",
                dice_improved,
                accuracy_improved,
                best_val_dice,
                best_val_accuracy,
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                "No val Dice/class-accuracy improvement for %d/%d epochs.",
                epochs_without_improvement,
                patience,
            )

        if early_stopping_enabled and epochs_without_improvement >= patience:
            logger.info(
                "Early stopping triggered at epoch %d. Best val_dice=%.4f best_val_cls_acc=%.4f",
                epoch,
                best_val_dice,
                best_val_accuracy,
            )
            break


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    logger: logging.Logger,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_class_accuracy = 0.0
    total_pixel_accuracy = 0.0
    total_samples = 0
    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        images, masks, labels = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            output = model(images)
            losses = criterion(output["mask_logits"], output["class_logits"], masks, labels)
        scaler.scale(losses.total).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += losses.total.detach().item() * batch_size
        total_class_accuracy += _classification_accuracy(output["class_logits"], labels).item() * batch_size
        total_pixel_accuracy += _pixel_accuracy(output["mask_logits"], masks).item() * batch_size
        total_samples += batch_size
        progress.set_postfix(
            loss=total_loss / max(total_samples, 1),
            cls_acc=total_class_accuracy / max(total_samples, 1),
        )
    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "class_accuracy": total_class_accuracy / max(total_samples, 1),
        "pixel_accuracy": total_pixel_accuracy / max(total_samples, 1),
    }
    logger.info("Train epoch metrics: %s", metrics)
    return metrics


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool,
    logger: logging.Logger,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_class_accuracy = 0.0
    total_pixel_accuracy = 0.0
    total_samples = 0
    for batch in tqdm(loader, desc="val", leave=False):
        images, masks, labels = _batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            output = model(images)
            losses = criterion(output["mask_logits"], output["class_logits"], masks, labels)
        batch_size = images.size(0)
        total_loss += losses.total.detach().item() * batch_size
        total_dice += _dice_score(output["mask_logits"], masks).item() * batch_size
        total_class_accuracy += _classification_accuracy(output["class_logits"], labels).item() * batch_size
        total_pixel_accuracy += _pixel_accuracy(output["mask_logits"], masks).item() * batch_size
        total_samples += batch_size
    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "dice_score": total_dice / max(total_samples, 1),
        "class_accuracy": total_class_accuracy / max(total_samples, 1),
        "pixel_accuracy": total_pixel_accuracy / max(total_samples, 1),
    }
    logger.info("Val epoch metrics: %s", metrics)
    return metrics


def _build_loader(config: dict[str, Any], split: str) -> DataLoader:
    data_config = config["data"]
    split_path = Path(data_config["splits_dir"]) / f"{split}.json"
    dataset = ManipulationDataset(split_path, transform=build_transforms(config, split))
    workers = int(data_config.get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 32)),
        shuffle=split == "train",
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=split == "train",
    )


def _batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return images, masks, labels


def _dice_score(mask_logits: torch.Tensor, masks: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    preds = (torch.sigmoid(mask_logits) > 0.5).float()
    dims = (1, 2, 3)
    intersection = (preds * masks).sum(dim=dims)
    union = preds.sum(dim=dims) + masks.sum(dim=dims)
    return ((2.0 * intersection + eps) / (union + eps)).mean()


def _classification_accuracy(class_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = (torch.sigmoid(class_logits) > 0.5).float()
    return (preds == labels).float().mean()


def _pixel_accuracy(mask_logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    preds = (torch.sigmoid(mask_logits) > 0.5).float()
    return (preds == masks).float().mean()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler_config = config.get("scheduler", {})
    warmup_epochs = int(scheduler_config.get("warmup_epochs", 1))
    min_lr = float(scheduler_config.get("min_lr", 1e-6))
    base_lr = float(config["optimizer"].get("lr", 3e-4))
    eta_min = min_lr
    cosine_epochs = max(1, epochs - warmup_epochs)

    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=eta_min)

    warmup = LinearLR(
        optimizer,
        start_factor=float(scheduler_config.get("warmup_start_factor", 0.1)),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=eta_min)
    # Ensure the optimizer starts at the warmup LR before the first batch.
    for group in optimizer.param_groups:
        group["lr"] = base_lr * float(scheduler_config.get("warmup_start_factor", 0.1))
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file.parent.mkdir(parents=True, exist_ok=True) if log_file.parent != Path(".") else None
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


if __name__ == "__main__":
    main()
