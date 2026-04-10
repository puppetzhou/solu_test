# solu_test

This repository collects local copies of 11 protein solubility prediction tools for benchmark preparation and reproducible comparison. The original nested Git histories from upstream tool repositories have been removed so the collection can be managed as a single benchmark project.

## Included tools

| Directory | Tool | Brief description |
| --- | --- | --- |
| `PLM_Sol/` | PLM_Sol | Protein solubility prediction based on ProtT5 protein language model embeddings. |
| `DeepSoluE/` | DeepSoluE | LSTM-based protein solubility predictor using physicochemical and amino-acid representation features. |
| `FGNNSol/` | FGNNSol | Graph neural network solubility predictor using structure-derived graph features and protein language model features. |
| `ProtSolM/` | ProtSolM | Multimodal solubility predictor combining sequence, structure, and feature information. |
| `Pro4S/` | Pro4S | Multimodal qualitative and quantitative protein solubility predictor using language, structure, and surface descriptors. |
| `GraphSol/` | GraphSol | Graph convolutional network predictor using sequence-derived graph/contact-map features. |
| `netsolp-1.0.ALL/` | NetSolP | Solubility and usability prediction package using ONNX model files. |
| `SurfSol/` | SurfSol | PyTorch-based surface and solvation-related prediction workflow. |
| `MMSol/` | MMSol | Multimodal solubility predictor with sequence, structure, and function modalities plus noise-resistant learning. |
| `MTPSol/` | MTPSol | Multimodal twin protein solubility prediction architecture based on pretrained sequence and structure models. |
| `ProG-SOL/` | ProG-SOL | Protein solubility predictor using protein embeddings and dual-GraphSAGE convolutional networks. |

Each tool directory keeps its own upstream README, environment files, scripts, and license files when available. Use those local README files for tool-specific installation and execution details.

## Repository notes

- This project is intended as a benchmark workspace, not a unified Python package.
- Large archives, local intermediate benchmark data, macOS resource-fork files, and files above GitHub's normal 100 MB limit are intentionally ignored.
- In particular, `Pro4S/checkpoints/*.ckpt` and `netsolp-1.0.ALL/models/*.onnx` are not tracked in Git because Git LFS is not installed in this workspace. Re-download those files from the tool authors' documented release channels when running the corresponding predictors.
- Local data preparation folders such as `407/` and `408recollection/` are kept outside Git by default.

## Benchmark workflow

1. Prepare a shared input FASTA/PDB dataset for the predictors being compared.
2. Follow each tool's README to create its required environment and download any external model weights or databases.
3. Run each predictor in its own directory and collect outputs into a consistent tabular format.
4. Evaluate all predictions with the same train/test split and metrics.

