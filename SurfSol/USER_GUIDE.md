# SurfSol ARM64 Usage Guide

## Quick Run

```bash
cd /home/xuyzh/solu_test/SurfSol
./run_surfsol_prediction.sh
```

Default inputs:
- CSV: `data/sufsol_test2.csv`
- checkpoint: `checkpoints/SurfSol_Ablation_Allmodalities/best_SurfSol_model.pth`
- output: `outputs/sufsol_test2_predictions.csv`
- ESM feature cache: `outputs/esm_features_sufsol_test2.pkl`

## Common Parameters

```bash
./run_surfsol_prediction.sh \
  --input-csv data/sufsol_test2.csv \
  --checkpoint checkpoints/SurfSol_Ablation_Allmodalities/best_SurfSol_model.pth \
  --output-csv outputs/my_predictions.csv \
  --esm-features outputs/my_esm_features.pkl \
  --esm-batch-size 8 \
  --batch-size 64 \
  --device cuda
```

Parameter notes:
- `--input-csv`: must contain `sequence` and either `gene` or `uniprot`.
- `--checkpoint`: SurfSol `.pth` checkpoint.
- `--output-csv`: prediction output path.
- `--esm-features`: ESM feature cache path. If the file is missing or incomplete, missing features are extracted and saved.
- `--esm-batch-size`: ESM embedding batch size. Reduce this if GPU memory is insufficient.
- `--batch-size`: lightweight prediction batch size after ESM features exist.
- `--max-length`: sequence truncation length for ESM2, default `1022`.
- `--limit`: optional row limit for smoke tests.
- `--device`: `cuda` or `cpu`.

## Environment

Use the deployed environment:

```bash
conda activate SurfSol_aarch64
python run_surfsol_prediction.py --help
```

The shell wrapper uses `SurfSol_aarch64` by default. To use a different conda env:

```bash
SURFSOL_ENV=my_env ./run_surfsol_prediction.sh --input-csv data/sufsol_test2.csv
```

## Runtime Panel

At startup the runner prints a panel showing input CSV, checkpoint, output path, ESM cache path, model weight path, device, and inference mode. The inference mode is `ESM inference; surface/structure vectors are zero-filled`.

## Output Format

The output CSV contains:
- `gene` or `uniprot`: copied from the input identifier column.
- `predicted_solubility`: model prediction.
- `true_solubility`: included when the input CSV has `solubility`.

If labels are present, RMSE and R2 are printed at the end.
