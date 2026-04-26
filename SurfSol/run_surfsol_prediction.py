#!/usr/bin/env python3
"""ARM64-compatible SurfSol prediction runner.

This runner uses the repository checkpoint but avoids unavailable surface/PDB
dependencies during inference. ESM embeddings are computed from sequence, while
surface and structure modality vectors are zero-filled placeholders because the
original raw feature files are not included in this repository checkout.
"""

import argparse
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm


class _CheckpointConfig:
    """Stub class used only for unpickling SurfSolConfig from checkpoints."""


_CheckpointConfig.__module__ = "surf_model"


def _install_checkpoint_stub():
    module = types.ModuleType("surf_model")
    module.SurfSolConfig = _CheckpointConfig
    sys.modules["surf_model"] = module


class MultiModalAttentionFusion(nn.Module):
    def __init__(self, feature_dims, output_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.projectors = nn.ModuleList([nn.Linear(dim, output_dim) for dim in feature_dims])
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        projected = [projector(feat) for projector, feat in zip(self.projectors, features)]
        stacked = torch.stack(projected, dim=1)
        attended, _ = self.attention(stacked, stacked, stacked)
        return self.dropout(self.layer_norm(attended.mean(dim=1)))


class CompatibleSurfSolModel(nn.Module):
    """Checkpoint-compatible ESM inference model with zero-filled missing modalities."""

    def __init__(self, config):
        super().__init__()
        self.surface_dim = int(getattr(config, "surface_emb_dim", 8))
        self.esm_dim = int(getattr(config, "esm_dim", 640))
        self.esm_emb_dim = int(getattr(config, "esm_emb_dim", 4))
        self.structure_dim = int(getattr(config, "structure_emb_dim", 8))
        self.hidden_dim = int(getattr(config, "hidden_dim", 16))
        attention_heads = int(getattr(config, "attention_heads", 4))
        dropout = float(getattr(config, "dropout", 0.0))

        self.esm_encoder = nn.Linear(self.esm_dim, self.esm_emb_dim)
        self.fusion_module = MultiModalAttentionFusion(
            [self.surface_dim, self.esm_emb_dim, self.structure_dim],
            self.hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.regression_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, esm_features):
        batch_size = esm_features.shape[0]
        surface = torch.zeros(batch_size, self.surface_dim, device=esm_features.device)
        structure = torch.zeros(batch_size, self.structure_dim, device=esm_features.device)
        esm_repr = self.esm_encoder(esm_features)
        fused = self.fusion_module([surface, esm_repr, structure])
        hidden = self.fusion_mlp(fused)
        return self.regression_head(hidden).squeeze(-1)


def load_checkpoint(checkpoint_path, device):
    _install_checkpoint_stub()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", _CheckpointConfig())
    model = CompatibleSurfSolModel(config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    unexpected = [key for key in unexpected if not key.startswith(("surface_", "structure_"))]
    if unexpected:
        print(f"[WARN] Ignored unexpected checkpoint keys: {len(unexpected)}")
    if missing:
        print(f"[WARN] Missing model keys: {missing}")
    model.eval()
    return model, config, checkpoint


def read_feature_cache(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return {str(key): np.asarray(value, dtype=np.float32) for key, value in data.items()}


def write_feature_cache(path, features):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(features, handle)


def extract_esm_features(rows, device, batch_size, max_length):
    import esm

    print("[ESM] Loading esm2_t30_150M_UR50D")
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    features = {}
    for start in tqdm(range(0, len(rows), batch_size), desc="Extracting ESM", ncols=100):
        batch_rows = rows[start : start + batch_size]
        batch = [(name, seq[:max_length]) for name, seq in batch_rows]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            result = model(tokens, repr_layers=[30], return_contacts=False)
        reps = result["representations"][30]
        for i, (name, seq) in enumerate(batch):
            features[name] = reps[i, 1 : len(seq) + 1].mean(0).detach().cpu().numpy()
    return features


def print_panel(args, device, model_cache_path):
    line = "=" * 78
    print(line)
    print("SurfSol ARM64 Prediction Runner")
    print(line)
    print(f"Input CSV      : {args.input_csv}")
    print(f"Checkpoint     : {args.checkpoint}")
    print(f"Output CSV     : {args.output_csv}")
    print(f"Feature cache  : {args.esm_features}")
    print(f"ESM model      : esm2_t30_150M_UR50D (fair-esm 2.0.0)")
    print(f"ESM weight dir : {model_cache_path}")
    print(f"Device         : {device}")
    print("Mode           : ESM inference; surface/structure vectors are zero-filled")
    print(line)


def main():
    parser = argparse.ArgumentParser(description="Run SurfSol prediction on ARM64/DGX Spark.")
    parser.add_argument("--input-csv", default="data/sufsol_test2.csv", help="CSV with gene/uniprot and sequence columns.")
    parser.add_argument("--checkpoint", default="checkpoints/SurfSol_Ablation_Allmodalities/best_SurfSol_model.pth")
    parser.add_argument("--output-csv", default="outputs/sufsol_test2_predictions.csv")
    parser.add_argument("--esm-features", default="outputs/esm_features_sufsol_test2.pkl")
    parser.add_argument("--batch-size", type=int, default=32, help="Prediction batch size.")
    parser.add_argument("--esm-batch-size", type=int, default=8, help="ESM embedding batch size.")
    parser.add_argument("--max-length", type=int, default=1022, help="Maximum amino-acid sequence length for ESM2.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_cache_path = Path(torch.hub.get_dir()) / "checkpoints" / "esm2_t30_150M_UR50D.pt"
    print_panel(args, device, model_cache_path)

    df = pd.read_csv(args.input_csv)
    if args.limit:
        df = df.head(args.limit).copy()
    id_col = "gene" if "gene" in df.columns else "uniprot"
    if id_col not in df.columns or "sequence" not in df.columns:
        raise ValueError("Input CSV must contain sequence and either gene or uniprot column.")

    df[id_col] = df[id_col].astype(str)
    df["sequence"] = df["sequence"].astype(str).str.strip()
    df = df[df["sequence"].str.len() > 0].reset_index(drop=True)

    features = read_feature_cache(args.esm_features)
    missing_rows = [
        (row[id_col], row["sequence"])
        for _, row in df.iterrows()
        if row[id_col] not in features
    ]
    if missing_rows:
        print(f"[ESM] Missing cached features for {len(missing_rows)} sequences; extracting now.")
        features.update(extract_esm_features(missing_rows, device, args.esm_batch_size, args.max_length))
        write_feature_cache(args.esm_features, features)
        print(f"[ESM] Saved feature cache: {args.esm_features}")
    else:
        print(f"[ESM] Loaded all features from cache: {args.esm_features}")

    model, config, checkpoint = load_checkpoint(args.checkpoint, device)
    print(f"[MODEL] Checkpoint epoch: {checkpoint.get('epoch', 'NA')}")
    print(
        "[MODEL] Config: "
        f"surface={getattr(config, 'use_surface', True)}, "
        f"esm_dim={getattr(config, 'esm_dim', 640)}, "
        f"hidden_dim={getattr(config, 'hidden_dim', 16)}, "
        f"fusion={getattr(config, 'fusion_type', 'attention')}"
    )

    predictions = []
    names = df[id_col].tolist()
    with torch.no_grad():
        for start in tqdm(range(0, len(names), args.batch_size), desc="Predicting", ncols=100):
            batch_names = names[start : start + args.batch_size]
            batch_features = np.stack([features[name] for name in batch_names])
            tensor = torch.tensor(batch_features, dtype=torch.float32, device=device)
            pred = model(tensor).detach().cpu().numpy()
            predictions.extend(pred.tolist())

    output = pd.DataFrame({id_col: names, "predicted_solubility": predictions})
    if "solubility" in df.columns:
        output["true_solubility"] = df["solubility"].to_numpy()
        try:
            from sklearn.metrics import mean_squared_error, r2_score

            rmse = mean_squared_error(output["true_solubility"], output["predicted_solubility"]) ** 0.5
            r2 = r2_score(output["true_solubility"], output["predicted_solubility"])
            print(f"[METRIC] RMSE={rmse:.4f} R2={r2:.4f}")
        except Exception as exc:
            print(f"[WARN] Metric calculation skipped: {exc}")

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"[DONE] Wrote predictions for {len(output)} sequences to {args.output_csv}")


if __name__ == "__main__":
    main()
