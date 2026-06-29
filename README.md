# SAR2EO-Research

SAR-to-EO image translation for the [GalaxEye AI Research Internship](https://galaxeye.space) technical assignment. Given a single-channel Sentinel-1 SAR patch, the model generates a corresponding Sentinel-2 RGB optical image using a conditional GAN pipeline (Pix2Pix-style) with an EfficientNet-B0 encoder, CBAM attention, residual decoder, and PatchGAN discriminator.

**Repository:** [github.com/Karthikpasupuleti11/SAR2EO-Research](https://github.com/Karthikpasupuleti11/SAR2EO-Research)

---

## Approach

| Component | Choice |
|-----------|--------|
| Generator | EfficientNet-B0 encoder (ImageNet-pretrained, 1-channel SAR) → skip connections → CBAM → residual decoder → RGB (Tanh) |
| Discriminator | 70×70 PatchGAN |
| Final loss (E4) | GAN + 100×L1 + 10×perceptual (VGG) + 5×(1−SSIM) |
| Split strategy | Tile-aware grouped split per terrain (no spatial leakage) |
| Training | Kaggle T4 GPU, AMP, 50 epochs per ablation experiment |

Ablation progression: **E1** (L1 only) → **E2** (+ GAN) → **E3** (+ CBAM) → **E4** (+ perceptual + SSIM, final model).

---

## Requirements

- **Python:** 3.10+ (tested on 3.10 locally, 3.12 on Kaggle)
- **GPU:** ≤16 GB VRAM for training/inference (Kaggle T4 / Colab)
- **Dependencies:** pinned in [`requirements.txt`](requirements.txt)

Key packages: `torch`, `torchvision`, `timm`, `lpips`, `torchmetrics`, `torch-fidelity`, `albumentations`, `pandas`, `pyyaml`.

---

## Environment Setup

```bash
git clone https://github.com/Karthikpasupuleti11/SAR2EO-Research.git
cd SAR2EO-Research

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install torch-fidelity   # required for FID metric in evaluate.py
```

---

## Dataset

### Source

We use the permitted Kaggle dataset:

**[Sentinel-1 & 2 Image Pairs Segregated by Terrain](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain)**

- 16,000 paired 256×256 PNG patches across 4 terrains: `agri`, `barrenland`, `grassland`, `urban`
- Pre-paired and co-registered SAR (Sentinel-1) + optical (Sentinel-2 RGB)

### Subset & split

We use the full **v_2** release (all 16,000 pairs). Splits are **tile-grouped** so patches from the same spatial tile never appear in more than one split:

| Split | Images | Tiles | % |
|-------|--------|-------|---|
| Train | 13,137 | 19 | 82.1% |
| Val   | 1,213  | 7  | 7.6%  |
| Test  | 1,650  | 6  | 10.3% |

Split manifests: [`splits/train.json`](splits/train.json), [`splits/val.json`](splits/val.json), [`splits/test.json`](splits/test.json).

### Expected directory layout

```
datasets/sentinel12/v_2/
├── agri/
│   ├── s1/   # SAR patches (*.png)
│   └── s2/   # optical patches (*.png)
├── barrenland/
├── grassland/
└── urban/
```

### Kaggle notebooks (reproducibility)

| Step | Notebook |
|------|----------|
| Data setup & metadata | [sar2eo-dataspliting](https://www.kaggle.com/code/pasupuletikarthik11/sar2eo-dataspliting) |
| Model training (E1–E4) | [sar2eo](https://www.kaggle.com/code/pasupuletikarthik11/sar2eo) |
| Full pipeline | [sar2eo (edit)](https://www.kaggle.com/code/pasupuletikarthik11/sar2eo/edit) |

### Local / Kaggle data pipeline

```bash
# 1. Link Kaggle input to local path (on Kaggle)
python scripts/setup_kaggle_data.py \
  --src /kaggle/input/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain/v_2

# 2. Build metadata
python utils/build_metadata.py --dataset-root datasets/sentinel12/v_2

# 3. Create tile-aware splits
python data/split.py
```

---

## Training

Each ablation experiment is trained independently for **50 epochs** with batch size 8 and AMP enabled.

```bash
# Smoke test (5 batches)
python train.py --config configs/config.yaml --max-batches 5

# Ablation experiments
python train.py --config configs/config.yaml --epochs 50                              # E1: L1 only
python train.py --config configs/experiments/e2_gan.yaml --epochs 50                    # E2: + GAN
python train.py --config configs/experiments/e3_cbam.yaml --epochs 50                   # E3: + CBAM
python train.py --config configs/experiments/e4_full.yaml --epochs 50                   # E4: full model
```

Checkpoints and per-epoch train loss JSON are saved under `Results/<experiment>_results/` (e.g. `E4.pt`, `E4_history.json`).

Generate loss curves:

```bash
python scripts/plot_losses.py
# → outputs/loss_curves/
```

> **Note:** Training logs per-epoch **train** loss only. Held-out **validation and test metrics** (PSNR, SSIM, LPIPS, FID) are computed post-training via `evaluate.py`.

---

## Inference

Conforms to the GalaxEye I/O contract:

- **Input:** directory of single-channel SAR PNGs, 256×256, `[0, 255]`
- **Output:** directory of RGB PNGs, same filenames
- **No internet** required at inference time (weights loaded locally)

```bash
python infer.py \
  --input_dir path/to/sar_pngs \
  --output_dir path/to/eo_output \
  --weights path/to/E4.pt
```

Example (local):

```bash
python infer.py \
  --input_dir outputs/infer_test_in \
  --output_dir outputs/infer_test_out \
  --weights Results/E4_results/E4.pt
```

Download final weights: **[E4.pt on Google Drive](https://drive.google.com/file/d/1BOs6j_3g6BBtKY9IyfNr5dlkFXZ5yLs_/view?usp=sharing)**

Direct download (for scripts):

```bash
gdown 1BOs6j_3g6BBtKY9IyfNr5dlkFXZ5yLs_ -O E4.pt
```

---

## Evaluation

Compute PSNR, SSIM, LPIPS, and FID on val/test splits:

```bash
# Single checkpoint
python evaluate.py --weights Results/E4_results/E4.pt --splits val test

# All ablation checkpoints (E1–E4)
python scripts/evaluate_all.py --results-root .
```

Outputs:

- `outputs/evaluation/<Exp>_metrics.json`
- `outputs/evaluation/ablation_summary.json`
- `outputs/evaluation/<Exp>/qualitative/` — SAR | ground-truth | generated triplets

---

## Results

Final model: **E4** (EfficientNet-B0 + CBAM + GAN + L1 + perceptual + SSIM).

### Test split (primary comparison)

| Exp | Config | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|-----|--------|--------|--------|---------|-------|
| E1 | L1 only | 14.13 | 0.179 | 0.752 | **179.9** |
| E2 | + GAN | 14.03 | 0.174 | 0.760 | 184.2 |
| E3 | + CBAM | **14.27** | **0.180** | **0.751** | 191.7 |
| **E4** | **+ perceptual + SSIM** | 13.88 | 0.166 | 0.860 | 291.1 |

### Validation split

| Exp | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|-----|--------|--------|---------|-------|
| E1 | 13.68 | 0.179 | 0.776 | 226.1 |
| E2 | 13.81 | 0.185 | 0.772 | 234.0 |
| E3 | 13.48 | 0.174 | 0.776 | 232.6 |
| **E4** | 13.79 | 0.181 | 0.851 | 341.0 |

E4 does not win on all pixel/distribution metrics — it optimizes a richer perceptual objective. Qualitative triplets in `outputs/evaluation/E4/qualitative/` show sharper, more realistic structure than the L1-only baseline. See the technical report for the pixel-vs-perceptual discussion.

Training loss curves: `outputs/loss_curves/` and `Results/*/E*_history.json`.

---

## Model Weights

| Model | File | Link |
|-------|------|------|
| **E4 (final submission)** | `E4.pt` | [Google Drive](https://drive.google.com/file/d/1BOs6j_3g6BBtKY9IyfNr5dlkFXZ5yLs_/view?usp=sharing) |

Ablation checkpoints (E1–E3) are available locally under `Results/`; not hosted publicly due to size. Reproduce by running the training commands above.

---

## Project Structure

```
SAR2EO-Research/
├── configs/              # Base + experiment YAML configs
├── data/                 # Dataset, transforms, split script
├── models/               # Encoder, decoder, CBAM, generator, discriminator
├── losses/               # Generator / discriminator losses
├── metrics/              # PSNR, SSIM, LPIPS, FID
├── scripts/              # Kaggle setup, plot_losses, evaluate_all
├── splits/               # Train/val/test JSON manifests
├── utils/                # Config, checkpoint, I/O helpers
├── Results/              # Checkpoints, history JSON, sample images
├── train.py
├── evaluate.py
└── infer.py
```

---

## Citations & References

### Dataset

- Requiemonk. *Sentinel-1 & 2 Image Pairs Segregated by Terrain.* Kaggle.  
  https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain

### Methods & codebases

- Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). *Image-to-Image Translation with Conditional Adversarial Networks.* CVPR. (Pix2Pix)
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). *CBAM: Convolutional Block Attention Module.* ECCV.
- Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR. (LPIPS)
- Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* NeurIPS. (FID)

### Libraries

- PyTorch, torchvision, timm, lpips, torchmetrics, albumentations

---

## License

Code in this repository is submitted as part of the GalaxEye technical assignment. Dataset usage follows the Kaggle source terms. Pretrained ImageNet weights (EfficientNet-B0, VGG for perceptual loss) are used as permitted by the assignment.
