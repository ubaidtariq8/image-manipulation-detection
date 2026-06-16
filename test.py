from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ManipulationDataset
from model import build_model
from transforms import build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split.")
    parser.add_argument("--config", type=Path, default=Path("configs/unet_mit_b1_casia2.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/best_weights.pt"))
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--class-threshold", type=float, default=0.5)
    parser.add_argument("--log-file", type=Path, default=Path("logs.txt"))
    parser.add_argument("--output-json", type=Path, default=Path("weights/test_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    logger = _setup_logger(args.log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Starting evaluation")
    logger.info("Config: %s", args.config)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Split: %s", args.split)
    logger.info("Device: %s", device)

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(
        "Loaded checkpoint epoch: %s",
        checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown",
    )

    loader = _build_loader(config, args.split)
    logger.info("Samples: %d", len(loader.dataset))
    logger.info("Batches: %d", len(loader))

    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    metrics = evaluate(
        model=model,
        loader=loader,
        device=device,
        amp_enabled=amp_enabled,
        mask_threshold=args.mask_threshold,
        class_threshold=args.class_threshold,
    )

    logger.info("Evaluation metrics:")
    for name, value in metrics.items():
        logger.info("  %s: %s", name, value)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    logger.info("Saved metrics JSON: %s", args.output_json)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    mask_threshold: float,
    class_threshold: float,
) -> dict[str, float | int | None]:
    seg_counts = _empty_counts()
    seg_counts_forged = _empty_counts()
    cls_counts = _empty_counts()
    image_dice_scores = []
    image_iou_scores = []
    forged_image_dice_scores = []
    forged_image_iou_scores = []
    class_probs = []
    class_targets = []
    mask_probs = []
    mask_targets = []

    for batch in tqdm(loader, desc="test", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            output = model(images)

        batch_mask_probs = torch.sigmoid(output["mask_logits"]).float()
        batch_class_probs = torch.sigmoid(output["class_logits"]).float()
        batch_mask_preds = batch_mask_probs > mask_threshold
        batch_class_preds = batch_class_probs > class_threshold
        batch_mask_targets = masks > 0.5
        batch_class_targets = labels > 0.5

        _update_counts(seg_counts, batch_mask_preds, batch_mask_targets)
        _update_counts(cls_counts, batch_class_preds, batch_class_targets)

        forged_selector = batch_class_targets.view(-1)
        if forged_selector.any():
            _update_counts(
                seg_counts_forged,
                batch_mask_preds[forged_selector],
                batch_mask_targets[forged_selector],
            )

        per_image = _per_image_segmentation_scores(batch_mask_preds, batch_mask_targets)
        image_dice_scores.extend(per_image["dice"])
        image_iou_scores.extend(per_image["iou"])
        if forged_selector.any():
            forged_scores = _per_image_segmentation_scores(
                batch_mask_preds[forged_selector],
                batch_mask_targets[forged_selector],
            )
            forged_image_dice_scores.extend(forged_scores["dice"])
            forged_image_iou_scores.extend(forged_scores["iou"])

        class_probs.append(batch_class_probs.detach().cpu().flatten())
        class_targets.append(batch_class_targets.detach().cpu().flatten())
        mask_probs.append(batch_mask_probs.detach().cpu().flatten())
        mask_targets.append(batch_mask_targets.detach().cpu().flatten())

    class_probs_tensor = torch.cat(class_probs).numpy()
    class_targets_tensor = torch.cat(class_targets).numpy().astype("int64")
    mask_probs_tensor = torch.cat(mask_probs).numpy()
    mask_targets_tensor = torch.cat(mask_targets).numpy().astype("int64")

    metrics: dict[str, float | int | None] = {}
    metrics.update(_prefixed_metrics("classification", cls_counts))
    metrics.update(_prefixed_metrics("segmentation_global", seg_counts))
    metrics.update(_prefixed_metrics("segmentation_forged_only", seg_counts_forged))
    metrics["segmentation_mean_image_dice"] = _mean(image_dice_scores)
    metrics["segmentation_mean_image_iou"] = _mean(image_iou_scores)
    metrics["segmentation_forged_mean_image_dice"] = _mean(forged_image_dice_scores)
    metrics["segmentation_forged_mean_image_iou"] = _mean(forged_image_iou_scores)
    metrics["classification_roc_auc"] = _safe_auc(class_targets_tensor, class_probs_tensor)
    metrics["classification_average_precision"] = _safe_average_precision(
        class_targets_tensor,
        class_probs_tensor,
    )
    metrics["segmentation_pixel_roc_auc"] = _safe_auc(mask_targets_tensor, mask_probs_tensor)
    metrics["segmentation_pixel_average_precision"] = _safe_average_precision(
        mask_targets_tensor,
        mask_probs_tensor,
    )
    return metrics


def _build_loader(config: dict[str, Any], split: str) -> DataLoader:
    data_config = config["data"]
    split_path = Path(data_config["splits_dir"]) / f"{split}.json"
    dataset = ManipulationDataset(split_path, transform=build_transforms(config, split))
    workers = int(data_config.get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 32)),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}


def _update_counts(
    counts: dict[str, int],
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    preds = preds.bool()
    targets = targets.bool()
    counts["tp"] += int((preds & targets).sum().item())
    counts["tn"] += int((~preds & ~targets).sum().item())
    counts["fp"] += int((preds & ~targets).sum().item())
    counts["fn"] += int((~preds & targets).sum().item())


def _prefixed_metrics(prefix: str, counts: dict[str, int]) -> dict[str, float | int]:
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    balanced_accuracy = (recall + specificity) / 2.0
    mcc = _mcc(tp, tn, fp, fn)
    return {
        f"{prefix}_tp": tp,
        f"{prefix}_tn": tn,
        f"{prefix}_fp": fp,
        f"{prefix}_fn": fn,
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_balanced_accuracy": balanced_accuracy,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_specificity": specificity,
        f"{prefix}_f1": f1,
        f"{prefix}_iou": iou,
        f"{prefix}_mcc": mcc,
    }


def _per_image_segmentation_scores(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
) -> dict[str, list[float]]:
    dims = (1, 2, 3)
    preds = preds.float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=dims)
    pred_sum = preds.sum(dim=dims)
    target_sum = targets.sum(dim=dims)
    union_for_dice = pred_sum + target_sum
    union_for_iou = pred_sum + target_sum - intersection
    dice = ((2.0 * intersection + eps) / (union_for_dice + eps)).detach().cpu().tolist()
    iou = ((intersection + eps) / (union_for_iou + eps)).detach().cpu().tolist()
    return {"dice": dice, "iou": iou}


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return _safe_div((tp * tn) - (fp * fn), denominator)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_auc(targets: Any, scores: Any) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score

        if len(set(targets.tolist())) < 2:
            return None
        return float(roc_auc_score(targets, scores))
    except ImportError:
        return None


def _safe_average_precision(targets: Any, scores: Any) -> float | None:
    try:
        from sklearn.metrics import average_precision_score

        if len(set(targets.tolist())) < 2:
            return None
        return float(average_precision_score(targets, scores))
    except ImportError:
        return None


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file.parent != Path("."):
        log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


if __name__ == "__main__":
    main()
