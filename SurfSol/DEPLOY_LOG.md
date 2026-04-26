# SurfSol Deployment Log

Deployment date: 2026-04-26

Host:
- Machine: DGX Spark, `aarch64`
- GPU visible to PyTorch: NVIDIA GB10
- Working conda environment: `SurfSol_aarch64`
- Python: `3.13.12`
- PyTorch: `2.11.0+cu130`, CUDA runtime `13.0`, `torch.cuda.is_available() == True`

## Environment

The original `environment.yml` is an old, heavily pinned Linux x86_64/CUDA environment and is not usable on this aarch64 host. I created the working environment with:

```bash
conda create -y -n SurfSol_aarch64 --clone base
conda run -n SurfSol_aarch64 python -m pip install fair-esm scikit-learn
conda run -n SurfSol_aarch64 python -m pip install e3nn mdtraj plyfile torch-geometric
```

Key installed packages:
- `torch 2.11.0+cu130`
- `fair-esm 2.0.0`
- `e3nn 0.6.0`
- `mdtraj 1.11.1.post1`
- `torch-geometric 2.7.0`
- `plyfile 1.1.3`
- `scikit-learn 1.8.0`
- `pandas 2.3.3`
- `numpy 2.4.4`
- `tqdm 4.67.3`

`environment.aarch64.yml` records a minimal aarch64-oriented environment specification. On this host, the important part is using the available NVIDIA aarch64 PyTorch build (`2.11.0+cu130`).

## Code Changes

Added:
- `run_surfsol_prediction.py`: ARM64-compatible inference runner.
- `run_surfsol_prediction.sh`: shell wrapper using conda environment `SurfSol_aarch64`.
- `environment.aarch64.yml`: minimal environment notes/spec.
- `DEPLOY_LOG.md`: this deployment record.
- `USER_GUIDE.md`: usage guide.

No original source files were modified.

## Dependency Workaround

The repository does not include the raw feature directories expected by `surf_test.py`:
- `raw_data/surfgraph_with_pos/intra_surf1`
- `raw_data/esm_features/esm_features/esm_features_test.pkl`
- `raw_data/pdb_test`

The original model code also imports packages that remain hard to resolve on this aarch64 host and are unnecessary without the raw feature files:
- `torch_scatter`
- `torch_cluster`

To make prediction runnable with the available repository artifacts, `run_surfsol_prediction.py` loads the checkpoint-compatible ESM branch, attention fusion module, MLP, and regression head. Surface and structure modality vectors are zero-filled placeholders because the corresponding `.ply` and `.pdb` feature files are absent. This preserves checkpoint tensor compatibility and provides a reproducible inference path from the supplied CSV sequences.

Later follow-up:
- `e3nn`, `mdtraj`, `plyfile`, and `torch-geometric` installed successfully on aarch64.
- `torch_scatter` and `torch_cluster` still required source compilation on this host; compilation was started and then stopped after deciding that full three-modal reproduction should be done on an x86_64 CUDA machine where PyG wheels are available.

## ESM Dependency

Installed PLM dependency:
- Package: `fair-esm==2.0.0`
- Model used: `esm2_t30_150M_UR50D`
- Embedding dimension: `640`
- Default download paths:
  - `/home/xuyzh/.cache/torch/hub/checkpoints/esm2_t30_150M_UR50D.pt`
  - `/home/xuyzh/.cache/torch/hub/checkpoints/esm2_t30_150M_UR50D-contact-regression.pt`

The above ESM weight files were removed during cache cleanup after validation. They will be downloaded again automatically on the next run if missing.

The script also caches extracted per-sequence features as pickle files, for example:
- `outputs/esm_features_sufsol_test2.pkl`

## Validation Run

Command run:

```bash
./run_surfsol_prediction.sh \
  --input-csv data/sufsol_test2.csv \
  --output-csv outputs/sufsol_test2_predictions.csv \
  --esm-features outputs/esm_features_sufsol_test2.pkl \
  --esm-batch-size 8 \
  --batch-size 64
```

Result:
- Input rows: `655`
- Output file: `outputs/sufsol_test2_predictions.csv`
- ESM feature cache: `outputs/esm_features_sufsol_test2.pkl`
- Metrics printed because `solubility` labels exist in the CSV:
  - RMSE: `0.2883`
  - R2: `0.1990`
