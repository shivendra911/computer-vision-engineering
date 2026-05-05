# 🧠 Computer Vision Journey — 30-Week Mastery Plan

> **A complete, production-grade roadmap to mastering Computer Vision from mathematical foundations to deployed real-world systems.**
> Built by [Shivendra Pratap](https://github.com/shivendra911) · Maintained collaboratively · Started May 2026

---

## Table of Contents

- [What This Is](#what-this-is)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [VS Code Setup](#vs-code-setup)
- [Git & Version Control Workflow](#git--version-control-workflow)
- [Experiment Tracking with W&B](#experiment-tracking-with-wb)
- [Data Version Control with DVC](#data-version-control-with-dvc)
- [The 30-Week Roadmap](#the-30-week-roadmap)
- [Project Ladder](#project-ladder)
- [Tech Stack Reference](#tech-stack-reference)
- [Code Style & Conventions](#code-style--conventions)
- [Commit Message Guide](#commit-message-guide)
- [Key Papers to Read](#key-papers-to-read)
- [Resource Library](#resource-library)
- [Troubleshooting](#troubleshooting)
- [Quick Reference Cheatsheet](#quick-reference-cheatsheet)

---

## What This Is

This repository is the single source of truth for a structured, 30-week deep dive into Computer Vision — every concept, every implementation, every project, every experiment. It is designed so that:

- **Nothing is skipped.** We go from writing a convolution in NumPy to deploying a TensorRT-optimized inference API.
- **Every concept is implemented before being used from a library.** You don't understand conv until you write it. Libraries are acceleration, not understanding.
- **Every week ships a real project.** Not tutorials. Not colab notebooks. Real pipelines.
- **Everything is tracked.** Code in Git. Data in DVC. Experiments in W&B. No lost work.

The mental model: CV is a stack. Each layer only makes sense if the one below it is solid.

```
Foundation → Classical CV → Deep Learning → Data Mastery → Advanced → Production
```

---

## Repository Structure

```
computer-vision-journey/
│
├── phase0_math/                    ← Weeks 1–3: Linear algebra, calculus, signal processing
│   ├── 01_convolution_numpy.py
│   ├── 02_svd_image_compression.py
│   ├── 03_pca_faces.py
│   └── 04_fft_frequency_filters.py
│
├── phase1_classical/               ← Weeks 4–9: Classical CV from scratch
│   ├── 01_edge_detection/
│   ├── 02_sift_feature_matching/
│   ├── 03_camera_calibration/
│   ├── 04_panorama_stitcher/
│   ├── 05_optical_flow_tracker/
│   └── 06_lane_detector/
│
├── phase2_deeplearning/            ← Weeks 10–18: Deep learning for vision
│   ├── 01_mlp_numpy/
│   ├── 02_cnn_numpy_then_pytorch/
│   ├── 03_resnet_implementation/
│   ├── 04_vit_implementation/
│   ├── 05_detection_yolo/
│   ├── 06_segmentation_unet/
│   └── 07_retinal_multilabel/
│
├── phase3_data/                    ← Weeks 19–21: Data engineering, SSL, fairness
│   ├── 01_annotation_pipeline/
│   ├── 02_ssl_simclr/
│   └── 03_fairness_audit/
│
├── phase4_advanced/                ← Weeks 22–26: Generative, 3D, multimodal
│   ├── 01_vae_gan/
│   ├── 02_diffusion_models/
│   ├── 03_video_understanding/
│   ├── 04_pointcloud_3d/
│   ├── 05_clip_multimodal/
│   └── 06_uncertainty_quantification/
│
├── phase5_production/              ← Weeks 27–30: Deployment, MLOps, final system
│   ├── 01_model_optimization/
│   ├── 02_fastapi_server/
│   ├── 03_mlops_pipeline/
│   └── 04_capstone_retinal_system/
│
├── notebooks/                      ← Exploration and experimentation (not production code)
├── experiments/                    ← Training configs, logs, results
├── data/                           ← Datasets (gitignored, tracked with DVC)
├── models/                         ← Saved weights (gitignored, tracked with DVC)
├── utils/                          ← Shared helper functions used across phases
│   ├── metrics.py
│   ├── visualize.py
│   ├── data_utils.py
│   └── training.py
├── configs/                        ← YAML training configs (Hydra)
│   └── train.yaml
├── .gitignore
├── .dvcignore
├── environment.yml                 ← Conda environment (reproduce with one command)
├── requirements.txt                ← pip fallback
└── README.md                       ← You are here
```

**Rule:** Notebooks go in `notebooks/` for exploration. Once something works, it becomes a `.py` script in the correct phase folder.

---

## Environment Setup

### Prerequisites

- Python 3.11+
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (not Anaconda)
- Git 2.39+
- NVIDIA GPU recommended (GTX 1060+ / RTX series)

### Step 1 — Check your system

```bash
python --version          # need 3.10+
nvidia-smi                # check GPU + CUDA version
nvcc --version            # check CUDA toolkit version
git --version             # need 2.39+
```

### Step 2 — Install Miniconda

```bash
# Linux / WSL2
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# macOS (Apple Silicon)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# Restart terminal, then verify:
conda --version
```

### Step 3 — Clone this repo and create the environment

```bash
git clone git@github.com:shivendra911/computer-vision-journey.git
cd computer-vision-journey

# Create environment from the saved file (one command, fully reproducible)
conda env create -f environment.yml
conda activate cv
```

If `environment.yml` is not yet available, create from scratch:

```bash
conda create -n cv python=3.11 -y
conda activate cv
```

### Step 4 — Install PyTorch

**Go to [pytorch.org](https://pytorch.org/get-started/locally/) and use the selector for your OS + CUDA version.**

```bash
# Example — CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (no GPU)
pip install torch torchvision torchaudio

# Verify GPU is accessible
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Step 5 — Install all dependencies

```bash
pip install \
  opencv-python \
  numpy \
  matplotlib \
  Pillow \
  scikit-image \
  scikit-learn \
  scipy \
  albumentations \
  timm \
  segmentation-models-pytorch \
  einops \
  kornia \
  monai \
  pydicom \
  SimpleITK \
  wandb \
  hydra-core \
  omegaconf \
  dvc \
  tqdm \
  rich \
  jupyterlab \
  ipywidgets \
  black \
  flake8 \
  isort \
  ipykernel

# Register environment as Jupyter kernel
python -m ipykernel install --user --name cv --display-name "CV (Python 3.11)"
```

### Step 6 — Save the environment

```bash
# Always update this after installing new packages
conda env export > environment.yml
pip freeze > requirements.txt
git add environment.yml requirements.txt
git commit -m "env: update dependencies"
```

### Reproducing on a new machine

```bash
# Full conda environment (recommended)
conda env create -f environment.yml
conda activate cv

# Or pip only
pip install -r requirements.txt
```

---

## VS Code Setup

### Required Extensions

Install all of these. Open VS Code → Extensions (`Ctrl+Shift+X`) → search each name.

| Extension | Publisher | Purpose |
|---|---|---|
| Python | Microsoft | Core Python support |
| Pylance | Microsoft | Type checking, fast autocomplete |
| Jupyter | Microsoft | Run notebooks inside VS Code |
| GitLens | Eric Amodio | Git blame, history, diff in editor |
| Git Graph | mhutchie | Visual branch and commit graph |
| GitHub Copilot | GitHub | AI autocomplete (free student access) |
| autoDocstring | NilsJPWerner | Auto-generate function docstrings |
| indent-rainbow | oderwat | Color-coded indentation levels |
| Error Lens | Alexander | Errors shown inline in code |
| Thunder Client | Rangav | Test REST APIs without Postman |
| Docker | Microsoft | Manage containers from VS Code |
| Remote - SSH | Microsoft | Connect to remote GPU servers |
| Remote - WSL | Microsoft | Windows Subsystem for Linux |
| Better Comments | Aaron Bond | Color-coded comment types |
| Material Icon Theme | PKief | File type icons in explorer |
| One Dark Pro | binaryify | Dark theme for long sessions |

### settings.json

Open with `Ctrl+Shift+P` → `Preferences: Open User Settings (JSON)`:

```json
{
  "python.defaultInterpreterPath": "~/miniconda3/envs/cv/bin/python",
  "editor.formatOnSave": true,
  "editor.rulers": [88],
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  },
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.pylintEnabled": false,
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "files.autoSave": "afterDelay",
  "files.autoSaveDelay": 1000,
  "editor.fontSize": 14,
  "editor.lineHeight": 1.6,
  "editor.minimap.enabled": false,
  "terminal.integrated.defaultProfile.linux": "bash",
  "git.autofetch": true,
  "editor.bracketPairColorization.enabled": true,
  "editor.guides.bracketPairs": true
}
```

### Useful Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+Shift+P` | Command palette (your best friend) |
| `Ctrl+`` ` | Toggle terminal |
| `Ctrl+Shift+`` ` | New terminal |
| `Ctrl+/` | Toggle line comment |
| `Alt+Shift+F` | Format document |
| `F12` | Go to definition |
| `Shift+F12` | Find all references |
| `Ctrl+D` | Select next occurrence |
| `Ctrl+Shift+L` | Select all occurrences |
| `Alt+↑/↓` | Move line up/down |
| `Ctrl+Shift+K` | Delete line |

---

## Git & Version Control Workflow

### First-time Git Setup

```bash
git config --global user.name "Your Name"
git config --global user.email "you@email.com"
git config --global core.editor "code --wait"
git config --global init.defaultBranch main
git config --global pull.rebase false
```

### SSH Key for GitHub (do once, never enter password again)

```bash
# Generate key
ssh-keygen -t ed25519 -C "you@email.com"
# Press Enter for all prompts

# Copy public key
cat ~/.ssh/id_ed25519.pub
# Go to github.com → Settings → SSH Keys → New SSH Key → paste

# Test connection
ssh -T git@github.com
# Expected: "Hi username! You've successfully authenticated"
```

### Daily Workflow — Follow This Every Day

```bash
# 1. Start of day — always pull first
git pull

# 2. Create a branch for what you're building
git checkout -b week1/numpy-convolution

# 3. Write code, experiment, iterate

# 4. Stage specific files (never use git add . blindly)
git add phase0_math/01_convolution_numpy.py

# 5. Commit with a meaningful message
git commit -m "feat: implement 2D convolution from scratch using NumPy"

# 6. Push branch to GitHub
git push origin week1/numpy-convolution

# 7. When the week's work is complete, merge to main
git checkout main
git merge week1/numpy-convolution
git push origin main

# 8. Delete the branch (keep things clean)
git branch -d week1/numpy-convolution
git push origin --delete week1/numpy-convolution
```

### Branch Naming Convention

```
week{N}/{short-description}

Examples:
  week1/numpy-convolution
  week5/panorama-stitcher
  week12/resnet-retinal-finetune
  week18/multilabel-retinal-classifier
  fix/unet-dice-loss-bug
  exp/simclr-pretrain-test
```

### Useful Git Commands

```bash
# Status and history
git status                          # what's changed right now
git log --oneline --graph --all     # visual branch history
git diff                            # see exact line changes
git diff HEAD~1 HEAD                # compare with last commit

# Undoing things
git reset --soft HEAD~1             # undo last commit, keep changes staged
git reset --mixed HEAD~1            # undo last commit, unstage changes
git checkout -- filename.py         # discard all changes to a file
git restore filename.py             # same as above (newer syntax)
git stash                           # temporarily shelve all changes
git stash pop                       # restore shelved changes
git stash list                      # see all stashes

# Working with remotes
git fetch origin                    # download changes without merging
git pull origin main                # fetch + merge main
git push --force-with-lease         # safer force push

# Inspecting
git show HEAD                       # see last commit details
git blame filename.py               # who wrote each line
git grep "function_name"            # search across all files
```

### The .gitignore

Key things that are ignored — never commit these:

```gitignore
# Large files — use DVC for these
data/
models/
*.pth
*.pt
*.onnx
*.h5
*.pkl

# Jupyter
.ipynb_checkpoints/

# Python
__pycache__/
*.pyc
*.pyo

# Secrets
.env
*.env
secrets.yaml
wandb/

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log
runs/
```

**Rule: If a file is large, binary, or secret — it does not go in Git. Period.**

---

## Experiment Tracking with W&B

Every training run is logged automatically. Never lose track of what hyperparameters produced what result.

### Setup

```bash
pip install wandb

# Authenticate (get API key from wandb.ai/authorize)
wandb login
```

### Standard Usage in Training Scripts

```python
import wandb

# Initialize at the top of your training script
wandb.init(
    project="cv-journey",
    name="week12-resnet50-retinal-v2",
    tags=["resnet50", "retinal", "finetune"],
    config={
        "architecture": "resnet50",
        "dataset": "ODIR-5K",
        "img_size": 224,
        "epochs": 50,
        "lr": 1e-4,
        "batch_size": 32,
        "loss": "focal",
        "augmentation": "heavy",
    }
)

# Log metrics inside your training loop
for epoch in range(cfg.epochs):
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/acc": train_acc,
        "val/loss": val_loss,
        "val/auroc": val_auroc,
        "val/dice": val_dice,
        "lr": optimizer.param_groups[0]["lr"],
    })

# Log sample predictions with images
wandb.log({
    "predictions": [
        wandb.Image(img, caption=f"GT: {gt} | Pred: {pred}")
        for img, gt, pred in zip(sample_images, gts, preds)
    ]
})

# Save model checkpoint as artifact
artifact = wandb.Artifact("model", type="model")
artifact.add_file("models/best_model.pth")
wandb.log_artifact(artifact)

wandb.finish()
```

### Hyperparameter Sweeps

```python
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/auroc", "goal": "maximize"},
    "parameters": {
        "lr":         {"min": 1e-5, "max": 1e-3},
        "batch_size": {"values": [16, 32, 64]},
        "dropout":    {"values": [0.1, 0.3, 0.5]},
    }
}

sweep_id = wandb.sweep(sweep_config, project="cv-journey")
wandb.agent(sweep_id, function=train, count=20)
```

---

## Data Version Control with DVC

Git tracks code. DVC tracks data and model weights. Together, every experiment is fully reproducible.

### Setup (already initialized in this repo)

```bash
pip install dvc
dvc init
git add .dvc .dvcignore
git commit -m "feat: initialize DVC"
```

### Tracking a Dataset

```bash
# Track a dataset directory
dvc add data/retinal_dataset/

# This creates a small pointer file — commit that, not the data
git add data/retinal_dataset.dvc .gitignore
git commit -m "data: track retinal dataset with DVC"
```

### Setting up Remote Storage

```bash
# Google Drive (free, easy setup)
dvc remote add -d gdrive gdrive://your-google-drive-folder-id

# AWS S3 (production grade)
dvc remote add -d s3remote s3://your-bucket/cv-data

# Save remote config to git
git add .dvc/config
git commit -m "feat: configure DVC remote storage"
```

### Daily Data Workflow

```bash
dvc push                  # upload tracked data to remote
dvc pull                  # download data on a new machine
dvc status                # check what's out of sync
dvc pull data/retinal_dataset.dvc    # pull a specific file only
```

---

## The 30-Week Roadmap

### Phase 0 — Mathematical Bedrock (Weeks 1–3)

| Week | Focus | Key Build |
|---|---|---|
| 1 | Linear algebra, calculus, matrix ops | 2D convolution engine in pure NumPy |
| 2 | Probability, statistics, Fourier transform | Frequency-domain image filter toolkit |
| 3 | OpenCV entry, color spaces, morphology | Medical image preprocessing pipeline |

**Milestone:** You can implement any image operation without a library.

---

### Phase 1 — Classical Computer Vision (Weeks 4–9)

| Week | Focus | Key Build |
|---|---|---|
| 4 | Canny, Harris, SIFT from scratch | Feature matching engine |
| 5 | Pinhole model, RANSAC, homography | Panorama stitcher |
| 6 | Stereo vision, epipolar geometry, optical flow | Depth estimator + real-time tracker |
| 7 | HOG, SVM, sliding window detection | Pedestrian detector on real video |
| 8 | Watershed, GrabCut, contour analysis | Classical retinal optic disc segmentation |
| 9 | Capstone | Lane detector + document scanner |

**Milestone:** You can build a CV system with zero deep learning.

---

### Phase 2 — Deep Learning for Vision (Weeks 10–18)

| Week | Focus | Key Build |
|---|---|---|
| 10 | MLP from scratch in NumPy | MNIST classifier — no frameworks |
| 11 | CNN forward + backward pass, PyTorch entry | CNN from scratch → 85%+ CIFAR-10 |
| 12 | ResNet, EfficientNet, transfer learning | ResNet-50 fine-tuned on retinal data |
| 13 | Self-attention, ViT, Swin Transformer | ViT vs CNN comparative study |
| 14 | Anchor boxes, Faster R-CNN, YOLO, DETR | Custom object detector — full pipeline |
| 15 | U-Net, DeepLab, Mask R-CNN, SAM | Retinal vessel segmentation |
| 16 | Loss functions, LR schedulers, AMP, augmentation | Ablation study — what actually helps |
| 17 | mAP, Dice, Grad-CAM, error analysis | Model evaluation dashboard |
| 18 | Capstone | Multi-label retinal disease classifier |

**Milestone:** You can build, train, evaluate, and debug any standard CV model.

---

### Phase 3 — Data Mastery (Weeks 19–21)

| Week | Focus | Key Build |
|---|---|---|
| 19 | Annotation, DICOM, data loaders, DVC | SAM-assisted annotation pipeline |
| 20 | SimCLR, DINO, MAE, pseudo-labeling | SSL pre-training on unlabeled fundus images |
| 21 | Fairness, domain adaptation, robustness | Fairness audit report |

**Milestone:** You understand data better than most people understand models.

---

### Phase 4 — Advanced Topics (Weeks 22–26)

| Week | Focus | Key Build |
|---|---|---|
| 22 | VAE, GAN, CycleGAN, Diffusion, Stable Diffusion | Synthetic retinal image generator |
| 23 | Two-stream, 3D conv, video transformers, ByteTrack | Real-time multi-object tracker |
| 24 | Point clouds, PointNet, NeRF, monocular depth | 3D scene reconstruction with NeRF |
| 25 | CLIP, Grounding DINO, LLaVA, vision-language | Medical image search + report generator |
| 26 | MC Dropout, Deep Ensembles, OOD detection | Uncertainty-aware triage system |

**Milestone:** You can work on frontier CV problems.

---

### Phase 5 — Production & Deployment (Weeks 27–30)

| Week | Focus | Key Build |
|---|---|---|
| 27 | ONNX, INT8 quantization, TensorRT, distillation | 5x faster, 4x smaller optimized model |
| 28 | FastAPI, Docker, Triton, load testing | Production inference API |
| 29 | W&B, CI/CD, drift detection, retraining loop | Full MLOps pipeline |
| 30 | Capstone | Complete retinal AI system — clinical grade |

**Milestone:** You can ship a CV model to production.

---

## Project Ladder

Build these in order. Each one forces real understanding.

| # | Project | Concepts Practiced |
|---|---|---|
| 1 | NumPy convolution engine | Filters, kernels, signal processing |
| 2 | PCA face compressor | Linear algebra, dimensionality reduction |
| 3 | Frequency-domain filter | FFT, low/high/band-pass |
| 4 | Lane detector (classical only) | Canny, Hough, perspective warp |
| 5 | SIFT feature matcher | Scale space, descriptor, ratio test |
| 6 | Panorama stitcher | RANSAC, homography, image warping |
| 7 | Real-time optical flow tracker | Lucas-Kanade, sparse tracking |
| 8 | HOG+SVM pedestrian detector | HOG, SVM, NMS, sliding window |
| 9 | Document scanner | Contour detection, perspective correction |
| 10 | MLP from scratch | Backprop, chain rule, no autograd |
| 11 | CNN from scratch → PyTorch | Forward/backward conv, batch norm |
| 12 | Retinal disease classifier | Transfer learning, fine-tuning |
| 13 | Custom YOLO object detector | Anchors, multi-scale, mAP evaluation |
| 14 | Retinal vessel segmenter | U-Net, Dice loss, medical metrics |
| 15 | SAM annotation pipeline | Foundation models, prompt engineering |
| 16 | SSL pre-training + fine-tune | SimCLR, contrastive learning |
| 17 | Synthetic data generator | Diffusion models, FID evaluation |
| 18 | Real-time multi-object tracker | Detection + ByteTrack + video pipeline |
| 19 | NeRF 3D reconstruction | Volume rendering, implicit representation |
| 20 | CLIP-powered image search | Vision-language, embedding similarity |
| 21 | Uncertainty-aware triage system | MC Dropout, epistemic uncertainty |
| 22 | Production retinal AI system | Full pipeline: diagnose + explain + serve |

---

## Tech Stack Reference

### Core Libraries

```python
import torch                          # deep learning
import torch.nn as nn
import torchvision

import cv2                            # classical CV
import numpy as np

from PIL import Image                 # image I/O
import skimage                        # scientific image processing

import timm                           # 700+ pretrained models
import segmentation_models_pytorch    # segmentation architectures (SMP)

import albumentations as A            # augmentation pipelines

import monai                          # medical imaging DL
import pydicom                        # DICOM file handling
```

### Training & Experiment Utilities

```python
import wandb                          # experiment tracking
import hydra                          # config management
from omegaconf import DictConfig
from tqdm import tqdm                 # progress bars
from rich import print                # pretty terminal output
```

### Evaluation

```python
from torchmetrics import AUROC, Dice, JaccardIndex
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from pytorch_grad_cam import GradCAM  # explainability
```

### Deployment

```python
import fastapi                        # REST API server
import uvicorn
import onnx                           # model export
import onnxruntime                    # CPU inference
# TensorRT → CLI tool for GPU inference
# Docker → containerization
# NVIDIA Triton → multi-model serving
```

### Config Pattern (Hydra)

```yaml
# configs/train.yaml
data:
  root: data/retinal/
  img_size: 224
  batch_size: 32
  num_workers: 4

model:
  name: resnet50
  pretrained: true
  num_classes: 8

training:
  epochs: 50
  lr: 1e-4
  weight_decay: 1e-5
  scheduler: cosine

logging:
  project: cv-journey
  entity: shivendra911
```

```python
# train.py
@hydra.main(config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print(cfg.model.name)     # resnet50

# Override anything from CLI:
# python train.py training.lr=0.001 model.name=efficientnet_b4
```

---

## Code Style & Conventions

### Formatting Tools

```bash
black .        # auto-format all Python files (run before every commit)
flake8 .       # check for style violations
isort .        # sort imports into stdlib / third-party / local groups
```

### File Structure for Every Script

```python
"""
Short description of what this module does and what it implements.

Usage:
    python phase2_deeplearning/06_segmentation_unet/train.py

Author: Shivendra Pratap
Week: 15
"""

# === Standard library ===
import os
from pathlib import Path

# === Third-party ===
import torch
import torch.nn as nn
import numpy as np
import cv2

# === Local ===
from utils.metrics import dice_coefficient
from utils.visualize import plot_predictions


# === Constants ===
IMG_SIZE = 224
NUM_CLASSES = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    pass


if __name__ == "__main__":
    main()
```

### Docstring Style (Google format)

```python
def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute Intersection over Union between predicted and ground truth masks.

    Args:
        pred_mask: Predicted binary mask, shape (H, W), values in [0, 1].
        gt_mask:   Ground truth binary mask, shape (H, W), values in {0, 1}.
        threshold: Threshold to binarize pred_mask. Defaults to 0.5.

    Returns:
        IoU score as float in [0, 1].

    Raises:
        ValueError: If pred_mask and gt_mask have different shapes.

    Example:
        >>> iou = compute_iou(pred, gt, threshold=0.5)
        >>> print(f"IoU: {iou:.4f}")
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: {pred_mask.shape} vs {gt_mask.shape}")
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    intersection = (pred_binary & gt_mask).sum()
    union = (pred_binary | gt_mask).sum()
    return intersection / (union + 1e-8)
```

### Comment Conventions (Better Comments extension)

```python
# Standard comment — explain the why, not the what

# ! CRITICAL: this must run before the DataLoader is initialized

# ? Why RoI Align instead of RoI Pooling?
# ? Because RoI Pooling has a quantization error causing ~1px misalignment.

# TODO: replace with a calibrated threshold per disease class

# * NOTE: model expects input normalized to ImageNet mean/std
```

---

## Commit Message Guide

```
<type>: <short description in present tense, lowercase>
```

### Types

| Type | When to use |
|---|---|
| `feat` | New implementation or feature |
| `fix` | Bug fix |
| `exp` | Experiment — training run, ablation, test |
| `docs` | README, comments, docstrings |
| `refactor` | Code restructuring, no behavior change |
| `data` | Dataset changes, DVC updates |
| `model` | Architecture changes |
| `eval` | Evaluation results, metric reports |
| `env` | Environment, requirements, dependencies |
| `chore` | Misc — gitignore, configs, cleanup |

### Good Examples

```bash
git commit -m "feat: implement U-Net with skip connections from scratch"
git commit -m "fix: correct dice loss for multi-class — was missing softmax"
git commit -m "exp: train resnet50 on ODIR-5K — val AUROC 0.87 baseline"
git commit -m "data: add ODIR-5K dataset pointer via DVC"
git commit -m "eval: add per-class AUROC and ECE to retinal classifier"
git commit -m "refactor: extract training loop into utils/training.py"
git commit -m "env: add monai and pydicom for medical imaging support"
```

### Never Do This

```bash
git commit -m "update"
git commit -m "fix bug"
git commit -m "changes"
git commit -m "asdfgh"
```

---

## Key Papers to Read

Read in this order. Papers marked 🔥 are absolutely foundational — read them in full.

### Classical CV

| Paper | Year | Why |
|---|---|---|
| 🔥 Lowe — Distinctive Image Features from Scale-Invariant Keypoints (SIFT) | 2004 | Foundation of feature detection |
| Dalal & Triggs — Histograms of Oriented Gradients for Human Detection | 2005 | HOG, still used in production |
| Viola & Jones — Rapid Object Detection using a Boosted Cascade | 2001 | Real-time face detection architecture |

### Architectures

| Paper | Year | Why |
|---|---|---|
| 🔥 He et al. — Deep Residual Learning for Image Recognition (ResNet) | 2015 | Skip connections solve vanishing gradient |
| 🔥 Dosovitskiy et al. — An Image is Worth 16x16 Words (ViT) | 2020 | Transformers applied to vision |
| Liu et al. — Swin Transformer | 2021 | Hierarchical ViT — best for dense prediction |
| Tan & Le — EfficientNet: Rethinking Model Scaling | 2019 | Compound scaling, NAS |
| Ronneberger et al. — U-Net | 2015 | Gold standard medical image segmentation |

### Detection

| Paper | Year | Why |
|---|---|---|
| 🔥 Ren et al. — Faster R-CNN | 2015 | Region Proposal Network |
| Lin et al. — Focal Loss for Dense Object Detection (RetinaNet) | 2017 | Solved class imbalance in detection |
| Carion et al. — End-to-End Object Detection with Transformers (DETR) | 2020 | Detection without NMS |

### Generative & Self-Supervised

| Paper | Year | Why |
|---|---|---|
| Chen et al. — A Simple Framework for Contrastive Learning (SimCLR) | 2020 | Foundation of SSL |
| 🔥 He et al. — Masked Autoencoders Are Scalable Vision Learners (MAE) | 2021 | Modern SSL pre-training |
| Ho et al. — Denoising Diffusion Probabilistic Models (DDPM) | 2020 | Foundation of diffusion |

### Multimodal & Foundation Models

| Paper | Year | Why |
|---|---|---|
| 🔥 Radford et al. — Learning Transferable Visual Models from Natural Language (CLIP) | 2021 | Vision-language alignment |
| Kirillov et al. — Segment Anything (SAM) | 2023 | Foundation model for segmentation |
| Liu et al. — Visual Instruction Tuning (LLaVA) | 2023 | Vision-language instruction following |

---

## Resource Library

### Textbooks

| Book | Use It For |
|---|---|
| Szeliski — Computer Vision: Algorithms and Applications *(free PDF)* | Classical CV reference bible |
| Hartley & Zisserman — Multiple View Geometry | Geometric CV, stereo, SfM |
| Goodfellow et al. — Deep Learning *(deeplearningbook.org, free)* | DL foundations |
| Nielsen — Neural Networks and Deep Learning *(neuralnetworksanddeeplearning.com, free)* | Backprop from scratch |

### Courses

| Course | Platform | Use It For |
|---|---|---|
| CS231n — CNNs for Visual Recognition | Stanford / YouTube | Best CV course ever made |
| CS231a — Computer Vision: 3D Reconstruction | Stanford | Geometric CV |
| fast.ai Practical Deep Learning | fast.ai | Pragmatic DL for vision |
| Karpathy — Neural Networks: Zero to Hero | YouTube | Building everything from scratch |

### Libraries

| Library | Docs | Purpose |
|---|---|---|
| PyTorch | pytorch.org/docs | Everything deep learning |
| OpenCV | docs.opencv.org | Classical CV operations |
| Albumentations | albumentations.ai/docs | Augmentation pipelines |
| timm | huggingface.co/docs/timm | 700+ pretrained models |
| MONAI | docs.monai.io | Medical imaging DL |
| Ultralytics | docs.ultralytics.com | YOLOv8 training and inference |
| W&B | docs.wandb.ai | Experiment tracking |
| DVC | dvc.org/doc | Data version control |

### Datasets

| Dataset | Task | Size | Source |
|---|---|---|---|
| CIFAR-10 | Classification | 60K images | torchvision |
| ImageNet-1K | Classification | 1.28M images | image-net.org |
| COCO 2017 | Detection + Seg | 330K images | cocodataset.org |
| ODIR-5K | Retinal disease | 5K fundus | kaggle.com |
| IDRiD | DR grading + seg | 516 images | idrid.grand-challenge.org |
| DRIVE | Retinal vessels | 40 images | drive.grand-challenge.org |
| LiTS | 3D liver seg | 201 CT scans | lits-challenge.com |
| UCF-101 | Action recognition | 13K videos | crcv.ucf.edu |

---

## Troubleshooting

### CUDA / GPU Issues

```bash
# Verify CUDA is accessible
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version matches PyTorch build
python -c "import torch; print(torch.version.cuda)"
nvcc --version

# Free GPU memory in notebook
import torch; torch.cuda.empty_cache()

# Check GPU memory usage
nvidia-smi
python -c "import torch; print(torch.cuda.memory_summary())"
```

### Common Training Issues

| Symptom | Likely Cause | Fix |
|---|---|---|
| Loss is NaN | Exploding gradients or LR too high | Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` |
| Loss not decreasing | LR too low, or data pipeline broken | Print a single batch and verify shapes and values |
| Overfitting from epoch 1 | Too little data or too large model | Add augmentation, reduce capacity |
| Val loss > Train from epoch 1 | Data leakage | Verify patient-level splits |
| GPU not being used | Model/data not on device | Add `.to(DEVICE)` to model and every batch |
| Slow training | num_workers=0 or data on HDD | Set `num_workers=4, pin_memory=True` |

### Environment Issues

```bash
# Wrong Python or packages not found
conda activate cv
which python          # must point to miniconda path

# Jupyter kernel not showing
python -m ipykernel install --user --name cv --display-name "CV (Python 3.11)"

# Package conflict — recreate environment
conda env remove -n cv
conda env create -f environment.yml
```

### Git Issues

```bash
# Accidentally committed a large file
git rm --cached models/large_model.pth
echo "models/" >> .gitignore
git commit -m "chore: remove large file from tracking"

# Merge conflict
git status                      # identify conflicted files
# Edit the file, resolve <<<<<<< HEAD markers manually
git add resolved_file.py
git commit -m "fix: resolve merge conflict in training utils"

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Pushed wrong thing — revert safely
git revert HEAD
git push
```

---

## Quick Reference Cheatsheet

```bash
# ─── Environment ───────────────────────────────────────────────────
conda activate cv                        # activate environment
conda deactivate                         # exit environment
conda env export > environment.yml       # save full environment
conda env create -f environment.yml      # recreate on new machine
pip freeze > requirements.txt            # save pip packages only

# ─── Jupyter ───────────────────────────────────────────────────────
jupyter lab                              # start JupyterLab
jupyter notebook                         # start classic notebook

# ─── Git ───────────────────────────────────────────────────────────
git status                               # what changed
git log --oneline --graph --all          # visual history
git checkout -b week1/feature-name       # create + switch branch
git add filename.py                      # stage specific file
git commit -m "type: message"            # commit with convention
git push origin branch-name             # push branch to GitHub
git pull                                 # pull latest from remote
git stash                                # shelve changes temporarily
git stash pop                            # restore shelved changes
git reset --soft HEAD~1                  # undo last commit, keep changes

# ─── DVC ───────────────────────────────────────────────────────────
dvc add data/dataset/                    # start tracking a dataset
dvc push                                 # upload data to remote
dvc pull                                 # download data from remote
dvc status                               # check sync status

# ─── W&B ───────────────────────────────────────────────────────────
wandb login                              # authenticate
wandb offline                            # disable for no-internet sessions
wandb sync                               # sync offline runs later

# ─── Code Quality ──────────────────────────────────────────────────
black .                                  # auto-format all Python
flake8 .                                 # check style violations
isort .                                  # sort imports

# ─── PyTorch Snippets ──────────────────────────────────────────────
torch.cuda.is_available()                # check GPU
torch.cuda.empty_cache()                 # free GPU memory
torch.cuda.memory_summary()              # detailed memory report

# ─── Docker ────────────────────────────────────────────────────────
docker build -t cv-api .                 # build image
docker run -p 8000:8000 cv-api           # run container
docker ps                                # list running containers
docker stop container_id                 # stop a container
docker logs container_id                 # view container logs
```

---

## Contributing to This Repo

If working on this together:

1. **Never push directly to `main`.** Always branch → commit → push → merge.
2. **Pull before you start every session.** Prevents merge conflicts.
3. **One branch per week or feature.** Keep branches small and focused.
4. **Leave `# ?` comments** on anything confusing. Future us will thank present us.
5. **Update `environment.yml`** any time you install a new package.
6. **Run `black .` before every commit.** No unformatted code in main.
7. **Update W&B** with results before closing a training run.

---

*"Every concept you learn — implement from scratch first. Then use the library."*

*Built with PyTorch · Tracked with W&B · Versioned with DVC · Deployed with FastAPI + Docker*
