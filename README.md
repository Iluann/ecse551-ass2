# ECSE 551 Assignment 2 (RIVAL10 image classification)

```
data/
  fall-2026-ecse-551-task-1/   Kaggle dataset (git-ignored, see "Data" below)
  split.csv                    shared 70/15/15 train/val/test split (stratified, seed 0); ALL models use it
notebooks/
  cnn.ipynb                    Task 1(a): data pipeline, first CNN (baseline), ablation, final CNN
  vit.ipynb                    Task 1(b): ViT (copy of Vit.ipynb from branch vit2, data path updated)
  task2.ipynb                  Task 2: frozen DINOv2 / SigLIP2 / ConvNeXt features, linear probes, ensemble
outputs/
  weights/                     final weights (cnn.pth, vit.pth, ...)
  preds/                       predictions on our own test split, for the CNN-vs-ViT comparison (part f)
  submissions/                 Kaggle CSVs, format id,label_name (00042.png -> 42); nothing submitted yet
  features/                    Task 2 feature cache (git-ignored)
report/main.tex                IEEE report (compile on Overleaf), figures in report/figures/
```

Notebooks run from `notebooks/`, so every path in them starts with `../`. The final submission is merged into one notebook at the end.

## Setup (Mac, Apple GPU)

```bash
uv venv .venv && uv pip install --python .venv torch numpy pandas pillow matplotlib scikit-learn ipykernel seaborn transformers torchvision
```
Pick `.venv` as the notebook kernel. (`requirements.txt` is the Linux/CUDA environment.)

## Data

Download the competition zip (Kaggle → Data → Download All) and unzip it into `data/fall-2026-ecse-551-task-1/` (git-ignored):

```
fall-2026-ecse-551-task-1/
  train.csv  test.csv  sample_submission.csv
  archive/RIVAL10_Kaggle/train/ordinary/            images (224×224) + *_merged_mask.JPEG object masks
  archive/RIVAL10_Kaggle/test/test_images|test_masks/
```

Label encoding (shared by all notebooks): truck 0, frog 1, cat 2, plane 3, bird 4, car 5, equine 6, ship 7, deer 8, dog 9.
