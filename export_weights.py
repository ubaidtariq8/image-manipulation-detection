from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a compact model-only weights file from a training checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("weights/best_weights.pt"))
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help="Save model weights in fp16 for a smaller inference checkpoint or fp32 for full precision.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    if args.precision == "fp16":
        state_dict = {
            key: value.half() if torch.is_floating_point(value) else value
            for key, value in state_dict.items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.output)
    input_mb = args.checkpoint.stat().st_size / (1024 * 1024)
    output_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Loaded: {args.checkpoint} ({input_mb:.2f} MB)")
    print(f"Saved:  {args.output} ({output_mb:.2f} MB, {args.precision})")


if __name__ == "__main__":
    main()
