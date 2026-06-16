# Image Manipulation Detection And Localization

A PyTorch project for binary image manipulation detection and pixel-level localization on CASIA v2.0.

The model is a `segmentation_models_pytorch` U-Net with a MiT-B1 encoder, trained to output:

- a 1-channel manipulation mask logit map
- an auxiliary image-level manipulation classification logit

## Highlights

- Dataset: CASIA v2.0
- Backbone: `mit_b1`
- Architecture: SMP `Unet`
- Input size: 384 x 384
- Encoder weights: ImageNet
- Segmentation loss: Dice + Focal
- Classification loss: BCE with logits
- Training: AMP/fp16, AdamW, 1 warmup epoch, cosine LR decay, early stopping
- Evaluation: classification metrics and pixel-level localization metrics
- Inference demo: notebook with image/URL input, predicted mask, and highlighted manipulation overlay

## Project Structure

```text
.
|-- configs/
|   `-- unet_mit_b1_casia2.yaml
|-- create_splits.py
|-- dataset.py
|-- export_weights.py
|-- inference_demo.ipynb
|-- inference_demo.py
|-- loss.py
|-- model.py
|-- test.py
|-- train.py
`-- transforms.py
```

## Dataset Layout

Place CASIA v2.0 under `datasets/CASIA2`:

```text
datasets/
`-- CASIA2/
    |-- Au/
    |-- Tp/
    `-- CASIA 2 Groundtruth/
```

Authentic images from `Au/` use all-zero masks. Tampered images from `Tp/` are paired with masks from `CASIA 2 Groundtruth/`. The split script drops tampered images if no usable mask is found or if the image and mask dimensions do not match.

## Setup

```powershell
python -m pip install -r requirements.txt
```

For GPU training, use a CUDA-enabled PyTorch build that matches your NVIDIA driver.

## Create Splits

```powershell
python create_splits.py --dataset-root datasets/CASIA2 --out-dir splits/casia2
```

This creates:

```text
splits/casia2/train.json
splits/casia2/val.json
splits/casia2/test.json
```

Each split item has this shape:

```json
{
  "img_path": "datasets/CASIA2/Tp/example.jpg",
  "gt_mask": "datasets/CASIA2/CASIA 2 Groundtruth/example_gt.png"
}
```

Authentic samples use:

```json
{
  "img_path": "datasets/CASIA2/Au/example.jpg",
  "gt_mask": "None"
}
```

## Train

```powershell
python train.py --config configs/unet_mit_b1_casia2.yaml
```

Default training settings:

- epochs: 100
- batch size: 32
- workers: 4
- learning rate: `1e-4`
- warmup: 1 epoch
- scheduler: cosine decay
- early stopping patience: 5
- checkpoints: `weights/latest.pt` and `weights/best.pt`

Logs are appended to:

```text
logs.txt
```

## Training Hardware

The current checkpoint was trained locally on:

- CPU: AMD Ryzen 5 7600X
- RAM: 32 GB DDR5 6000 MHz
- GPU: NVIDIA RTX 5070 12 GB

## Export Compact Weights

The full `weights/best.pt` checkpoint includes optimizer, scheduler, config, and metrics. Export model-only FP16 weights for inference and GitHub:

```powershell
python export_weights.py --checkpoint weights/best.pt --output weights/best_weights.pt
```

The compact checkpoint is saved as:

```text
weights/best_weights.pt
```

## Test

Evaluate the held-out CASIA v2 test split:

```powershell
python test.py --config configs/unet_mit_b1_casia2.yaml --checkpoint weights/best_weights.pt
```

`test.py` reports classification and localization metrics including accuracy, balanced accuracy, precision, recall, specificity, F1/Dice, IoU, MCC, ROC-AUC, and average precision where available.

Metrics are saved to:

```text
weights/test_metrics.json
```

## Current CASIA v2 Results

The current checkpoint was trained on the generated CASIA v2 80/10/10 split. The best checkpoint was selected from validation.

Held-out test split size: 1,261 images.

Headline test metrics:

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

Pixel accuracy was 0.9888, but it is less informative for localization because background pixels dominate manipulation masks.

## Inference Demo

Open [inference_demo.ipynb](inference_demo.ipynb), set the first cell, and run both cells:

```python
IMAGE_SOURCE = ""
THRESHOLD = 0.5
```

`IMAGE_SOURCE` can be a local image path or an image URL. The notebook loads `weights/best_weights.pt`, prints the classification output, and displays:

- source image
- predicted binary mask
- highlighted manipulated regions
