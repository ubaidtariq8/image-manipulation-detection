# Image Manipulation Detection and Localization

Binary image manipulation detection and localization using a U-Net segmentation model from `segmentation_models_pytorch`.

The first training stage uses CASIA2 with:

- SMP `Unet`
- `mit_b1` encoder with ImageNet weights
- 384 x 384 inputs
- 1-channel binary mask logits
- auxiliary binary classification logits
- Dice + Focal segmentation loss and BCE-with-logits classification loss
- PyTorch AMP with `torch.amp.autocast("cuda")` and `torch.amp.GradScaler("cuda")`
- 1 warmup epoch followed by cosine learning-rate decay

## Project Structure

```text
.
|-- configs/
|   `-- unet_mit_b1_casia2.yaml
|-- create_splits.py
|-- dataset.py
|-- export_weights.py
|-- fine_tune.py
|-- inference_demo.ipynb
|-- inference_demo.py
|-- loss.py
|-- model.py
|-- test.py
|-- train.py
`-- transforms.py
```

Expected CASIA2 layout:

```text
datasets/
`-- CASIA2/
    |-- Au/
    |-- Tp/
    `-- CASIA 2 Groundtruth/
```

`fine_tune.py` is intentionally a placeholder for the future FantasyID fine-tuning stage.

## Setup

Create and activate a Python environment, then install dependencies:

```powershell
python -m pip install -r requirements.txt
```

For CUDA training on Windows, install the PyTorch build that matches your NVIDIA driver from the official PyTorch instructions if your current environment does not already have CUDA-enabled PyTorch.

## Create Splits

```powershell
python create_splits.py --dataset-root datasets/CASIA2 --out-dir splits/casia2
```

This writes:

```text
splits/casia2/train.json
splits/casia2/val.json
splits/casia2/test.json
```

Each item is:

```json
{
  "img_path": "datasets/CASIA2/Tp/example.jpg",
  "gt_mask": "datasets/CASIA2/CASIA 2 Groundtruth/example_gt.png"
}
```

Authentic images use `"gt_mask": "None"` and get an all-zero mask in the dataset loader.
Tampered images are skipped if no mask can be found or if the image and mask dimensions do not match.

## Train

```powershell
python train.py --config configs/unet_mit_b1_casia2.yaml
```

Default training uses 100 epochs, 1 warmup epoch, cosine LR decay, and early stopping with patience 5. Early stopping resets when either validation classification accuracy or validation Dice improves.

Epoch summaries include loss, Dice, classification accuracy, and mask pixel accuracy.

Checkpoints are written to:

```text
weights/latest.pt
weights/best.pt
```

Export the model-only checkpoint for inference and GitHub:

```powershell
python export_weights.py --checkpoint weights/best.pt --output weights/best_weights.pt
```

By default, this exports FP16 model-only weights. The full training checkpoint `weights/best.pt` includes training state and stays ignored; `weights/best_weights.pt` is the compact inference checkpoint to commit.

Console output from `create_splits.py` and `train.py` is also appended to:

```text
logs.txt
```

## Current CASIA2 Results

The current checkpoint was trained on the generated CASIA2 80/10/10 split with `mit_b1`, 384 x 384 inputs, batch size 32, AMP/fp16, AdamW at `1e-4`, 1 warmup epoch, cosine LR decay, and early stopping enabled. The best checkpoint selected from validation was epoch 13.

Held-out test split size: 1,261 images.

Headline test metrics from `weights/best.pt`:

| Task | Metric | Score |
| --- | --- | ---: |
| Classification | Accuracy | 0.8914 |
| Classification | Balanced accuracy | 0.8884 |
| Classification | F1 | 0.8669 |
| Classification | ROC-AUC | 0.9696 |
| Localization | Pixel F1/Dice | 0.7519 |
| Localization | Pixel IoU | 0.6024 |
| Localization | Pixel ROC-AUC | 0.9562 |
| Localization | Pixel average precision | 0.7692 |

Note: pixel accuracy is high at 0.9888 but is less informative for manipulation localization because background pixels dominate the masks.

## Test

After training, evaluate the best checkpoint on the held-out test split:

```powershell
python test.py --config configs/unet_mit_b1_casia2.yaml --checkpoint weights/best_weights.pt
```

This reports paper-style classification metrics and segmentation metrics, including accuracy, balanced accuracy, precision, recall, specificity, F1/Dice, IoU, MCC, ROC-AUC, and average precision where available. Metrics are appended to `logs.txt` and saved to:

```text
weights/test_metrics.json
```

## Inference Demo

Open [inference_demo.ipynb](inference_demo.ipynb), set the first cell, and run both cells:

```python
IMAGE_SOURCE = ""
THRESHOLD = 0.5
```

`IMAGE_SOURCE` can be a local image path or an image URL. The notebook imports only `run` from [inference_demo.py](inference_demo.py), loads `weights/best_weights.pt`, prints the classification probability, and displays the source image, predicted binary mask, and highlighted manipulated regions.
