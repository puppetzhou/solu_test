# SurfSol

Overview: SurfSol is a PyTorch-based project for surface and solvation-related predictions. It includes data processing, model definitions, training utilities (including cross-validation), and evaluation scripts.

## Repository Structure

- `surf_dataset.py`: dataset and data loader implementations
- `surf_model.py`: model architecture definitions
- `surf_train.py`: single-run training script
- `surf_train_cv.py`: cross-validation training script (multi-fold)
- `surf_test.py`: single-model evaluation script
- `surf_test_cv.py`: cross-validation evaluation script
- `checkpoints/`: saved model weights and checkpoints
- `data/`: example CSV data files (e.g. `sufsol_train2.csv`, `sufsol_test2.csv`)
- `scripts/`: helper scripts for preprocessing and feature extraction

## Quick Start

1. Requirements

`environment.yml` 

2. Prepare Data

Place your training and test CSV files in the `data/` directory. Example filenames used in this project:

- `data/sufsol_train2.csv`
- `data/sufsol_test2.csv`

3. Training Examples

Single-run training (example):

```bash
python surf_train.py --data data/sufsol_train2.csv --save-dir checkpoints/SurfSol_Ablation_Allmodalities/
```

Cross-validation training (example, 5 folds):

```bash
python surf_train_cv.py --data data/sufsol_train2.csv --folds 5 --checkpoints_dir checkpoints/checkpoints_cv/
```

4. Testing / Evaluation Examples

Single-model test (example):

```bash
python surf_test.py --model checkpoints/SurfSol_Ablation_Allmodalities/best_SurfSol_model.pth --data data/sufsol_test2.csv
```

Cross-validation evaluation (example):

```bash
python surf_test_cv.py --checkpoints_dir checkpoints/checkpoints_cv/ --data data/sufsol_test2.csv
```

5. Example Checkpoint Paths

The repository contains example saved weights for folds and single models:

- `checkpoints/checkpoints_cv`
- `checkpoints/SurfSol_Ablation_Allmodalities/best_SurfSol_model.pth`

6. Helper Scripts

The `scripts/` directory contains preprocessing and feature extraction utilities:

- `extract_esm_features.py`: extract ESM sequence features
- `surface_build_with_pos.py`: surface construction utilities
- `01-masif.py`: example surface-related workflow

7. Customization and Extension

- Modify training hyperparameters or model architecture in `surf_train.py` and `surf_model.py`.
- Add support for new dataset formats by extending `surf_dataset.py`.

8. License & Support

If you need help or want to report issues, please open an issue in the repository or contact the maintainer.

---