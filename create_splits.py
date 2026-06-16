from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
NONE_MASK = "None"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CASIA2 train/val/test JSON splits.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/CASIA2"))
    parser.add_argument("--out-dir", type=Path, default=Path("splits/casia2"))
    parser.add_argument("--log-file", type=Path, default=Path("logs.txt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--drop-missing-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop tampered images that cannot be paired with a mask.",
    )
    parser.add_argument(
        "--check-mask-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip tampered image/mask pairs whose width and height do not match.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = _setup_logger(args.log_file)
    logger.info("Creating CASIA2 splits")
    logger.info("Dataset root: %s", args.dataset_root)
    logger.info("Output dir: %s", args.out_dir)
    logger.info("Ratios: train=%.3f val=%.3f test=%.3f", args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio)
    logger.info("Seed: %d", args.seed)
    authentic = _build_authentic_samples(args.dataset_root)
    tampered, dropped_missing, dropped_size = _build_tampered_samples(
        args.dataset_root,
        drop_missing_masks=args.drop_missing_masks,
        check_mask_size=args.check_mask_size,
    )
    logger.info("Authentic samples: %d", len(authentic))
    logger.info("Tampered samples with valid masks: %d", len(tampered))

    train = []
    val = []
    test = []
    for samples in (authentic, tampered):
        part_train, part_val, part_test = _split_samples(
            samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        train.extend(part_train)
        val.extend(part_val)
        test.extend(part_test)

    rng = random.Random(args.seed)
    for split in (train, val, test):
        rng.shuffle(split)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "train.json", train)
    _write_json(args.out_dir / "val.json", val)
    _write_json(args.out_dir / "test.json", test)

    logger.info("Wrote %d train, %d val, %d test samples.", len(train), len(val), len(test))
    if dropped_missing:
        logger.warning("Dropped %d tampered images with missing masks:", len(dropped_missing))
        for path in dropped_missing:
            logger.warning("  %s", path)
    if dropped_size:
        logger.warning(
            "Dropped %d tampered image/mask pairs with size mismatch:",
            len(dropped_size),
        )
        for image, mask, image_size, mask_size in dropped_size:
            logger.warning("  %s (%s) != %s (%s)", image, image_size, mask, mask_size)


def _build_authentic_samples(dataset_root: Path) -> list[dict[str, str]]:
    au_dir = dataset_root / "Au"
    return [
        {"img_path": _as_posix(path), "gt_mask": NONE_MASK}
        for path in _list_images(au_dir)
    ]


def _build_tampered_samples(
    dataset_root: Path,
    drop_missing_masks: bool,
    check_mask_size: bool,
) -> tuple[list[dict[str, str]], list[Path], list[tuple[Path, Path, tuple[int, int], tuple[int, int]]]]:
    tp_dir = dataset_root / "Tp"
    gt_dir = dataset_root / "CASIA 2 Groundtruth"
    masks = _list_images(gt_dir)
    masks_by_stem = {mask.stem.removesuffix("_gt"): mask for mask in masks}
    masks_by_suffix: dict[str, list[Path]] = {}
    for mask in masks:
        suffix = mask.stem.removesuffix("_gt").rsplit("_", 1)[-1]
        masks_by_suffix.setdefault(suffix, []).append(mask)

    samples = []
    dropped_missing = []
    dropped_size = []
    for image in _list_images(tp_dir):
        mask = masks_by_stem.get(image.stem)
        if mask is None:
            candidates = masks_by_suffix.get(image.stem.rsplit("_", 1)[-1], [])
            if check_mask_size:
                candidates = [
                    candidate
                    for candidate in candidates
                    if _same_image_size(image, candidate)
                ]
            if len(candidates) == 1:
                mask = candidates[0]

        if mask is None:
            if drop_missing_masks:
                dropped_missing.append(image)
                continue
            samples.append({"img_path": _as_posix(image), "gt_mask": NONE_MASK})
            continue

        if check_mask_size:
            image_size = _image_size(image)
            mask_size = _image_size(mask)
            if image_size != mask_size:
                dropped_size.append((image, mask, image_size, mask_size))
                continue

        samples.append({"img_path": _as_posix(image), "gt_mask": _as_posix(mask)})
    return samples, dropped_missing, dropped_size


def _split_samples(
    samples: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def _list_images(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected directory: {path}")
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def _write_json(path: Path, samples: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
        f.write("\n")


def _same_image_size(image_path: Path, mask_path: Path) -> bool:
    return _image_size(image_path) == _image_size(mask_path)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _as_posix(path: Path) -> str:
    return path.as_posix()


def _setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("create_splits")
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
