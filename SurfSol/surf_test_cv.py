import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
from surf_model import SurfSolProteinModel, SurfSolConfig
from surf_dataset import SurfSolDataset, surf_collate_fn
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats


def print_logo():
    """Print logo ASCII art"""
    try:
        with open('raw_data/logo.txt', 'r', encoding='utf-8') as f:
            logo = f.read()
        print(logo)
    except FileNotFoundError:
        print("🌊 SURFSOL - Cross-Validation Testing 🌊")
        print("=" * 60)
    except Exception as e:
        print(f"Failed to read logo: {e}")
        print("🌊 SURFSOL - Cross-Validation Testing 🌊")
        print("=" * 60)


def load_cv_checkpoints(checkpoint_dir, device, n_folds=5):
    """Load all cross-validation model checkpoints"""
    models = []
    configs = []
    fold_metrics = []
    
    print(f"\n📂 Loading cross-validation models from {checkpoint_dir}")
    
    for fold_idx in range(1, n_folds + 1):
        checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold_idx}", f"best_fold_{fold_idx}_model.pth")
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Warning: Checkpoint not found for fold {fold_idx}: {checkpoint_path}")
            continue
        
        print(f"   Loading fold {fold_idx} model...")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load configuration
            config = checkpoint.get('config', SurfSolConfig())
            
            # Create model
            model = SurfSolProteinModel(config).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Get validation metrics from training
            val_metrics = checkpoint.get('val_metrics', {})
            val_loss = checkpoint.get('val_loss', float('inf'))
            
            models.append(model)
            configs.append(config)
            fold_metrics.append({
                'fold': fold_idx,
                'val_loss': val_loss,
                'val_metrics': val_metrics
            })
            
            print(f"      ✅ Fold {fold_idx} loaded | Val Loss: {val_loss:.4f} | Val R²: {val_metrics.get('regression_r2', 0):.4f}")
            
        except Exception as e:
            print(f"      ❌ Failed to load fold {fold_idx}: {e}")
            continue
    
    if len(models) == 0:
        raise ValueError(f"No valid checkpoints found in {checkpoint_dir}")
    
    print(f"\n✅ Successfully loaded {len(models)}/{n_folds} models")
    
    return models, configs, fold_metrics


def predict_with_model(model, dataloader, device):
    """Make predictions with a single model"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            hetero_data = batch['hetero_data'].to(device)
            
            # Forward pass
            outputs = model(hetero_data)
            
            # Get predictions
            if 'regression' in outputs:
                pred = outputs['regression'].squeeze().cpu().numpy()
            elif 'classification' in outputs:
                pred = torch.sigmoid(outputs['classification']).squeeze().cpu().numpy()
            else:
                # Fallback: use any available output
                if len(outputs) > 0:
                    first_key = list(outputs.keys())[0]
                    pred = outputs[first_key].squeeze().cpu().numpy()
                else:
                    continue
            
            # Ensure pred is array
            if pred.ndim == 0:
                pred = np.array([pred])
            
            all_predictions.extend(pred)
    
    return np.array(all_predictions)


def ensemble_predict(models, dataloader, device, aggregation='mean'):
    """Make ensemble predictions using all models"""
    all_predictions = []
    protein_names = []
    
    # Collect predictions from all models
    fold_predictions = []
    
    for fold_idx, model in enumerate(models, 1):
        print(f"   Predicting with fold {fold_idx} model...")
        preds = predict_with_model(model, dataloader, device)
        fold_predictions.append(preds)
    
    # Get protein names from dataloader
    for batch in dataloader:
        protein_names.extend(batch['protein_names'])
    
    # Aggregate predictions
    fold_predictions = np.array(fold_predictions)  # Shape: (n_folds, n_samples)
    
    if aggregation == 'mean':
        ensemble_preds = np.mean(fold_predictions, axis=0)
    elif aggregation == 'median':
        ensemble_preds = np.median(fold_predictions, axis=0)
    elif aggregation == 'weighted_mean':
        # Weight by inverse validation loss (better models get higher weight)
        # This would require validation losses, which we can get from fold_metrics
        weights = np.ones(len(models))  # Default: equal weights
        ensemble_preds = np.average(fold_predictions, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    return ensemble_preds, fold_predictions, protein_names


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Pearson correlation
    pearson_r, pearson_p = scipy.stats.pearsonr(y_true, y_pred)
    pearson_r2 = pearson_r ** 2
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_r2': pearson_r2,
        'pearson_p': pearson_p,
        'mae': mae
    }


def main():
    """Main cross-validation testing function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Test SurfSol cross-validation models on test dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default paths
  python surf_test_cv.py --checkpoint_dir checkpoints_cv
  
  # With custom test file and output
  python surf_test_cv.py --checkpoint_dir checkpoints_cv \\
                         --test_csv S.cerevisiae_test.csv \\
                         --output_csv cv_predictions.csv
  
  # With custom number of folds
  python surf_test_cv.py --checkpoint_dir checkpoints_cv \\
                         --n_folds 5 \\
                         --aggregation median
        """
    )
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='checkpoints',
                       help='Directory containing cross-validation checkpoints')
    parser.add_argument('--test_csv', type=str, 
                       default='raw_data/ATA.csv',
                       help='Path to test CSV file')
    parser.add_argument('--output_csv', type=str,
                       default='ATA_predictions_ALL.csv',
                       help='Path to save predictions')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for inference')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--aggregation', type=str, default='mean',
                       choices=['mean', 'median', 'weighted_mean'],
                       help='Method to aggregate predictions from multiple folds')
    args = parser.parse_args()
    
    # Check if checkpoint directory exists
    if not os.path.exists(args.checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {args.checkpoint_dir}")
        print("   Please provide a valid checkpoint directory using --checkpoint_dir")
        return
    
    # Check if test CSV exists
    if not os.path.exists(args.test_csv):
        print(f"❌ Test CSV file not found: {args.test_csv}")
        print("   Please provide a valid test CSV path using --test_csv")
        return
    
    print_logo()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load all CV models
    models, configs, fold_metrics = load_cv_checkpoints(
        args.checkpoint_dir, device, args.n_folds
    )
    
    if len(models) == 0:
        print("❌ No models loaded. Exiting.")
        return
    
    # Use the first config (all should be the same)
    config = configs[0]
    
    print(f"\n📊 Model configuration:")
    print(f"   Training mode: {config.training_mode}")
    print(f"   Use surface: {config.use_surface}")
    print(f"   Use ESM: {config.use_esm}")
    print(f"   Use structure: {config.use_structure}")
    print(f"   Fusion type: {config.fusion_type}")
    print(f"   Number of models: {len(models)}")
    print(f"   Aggregation method: {args.aggregation}")
    
    # Load test data
    print(f"\n📂 Loading test data from {args.test_csv}")
    
    # Read test CSV
    test_data = pd.read_csv(args.test_csv)
    print(f"   Found {len(test_data)} test samples")
    
    # Check if CSV has the required columns
    required_columns = ['uniprot', 'sequence']
    missing_columns = [col for col in required_columns if col not in test_data.columns]
    
    if missing_columns:
        # Check if 'gene' exists as alternative
        if 'gene' not in test_data.columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print(f"   CSV must have either 'uniprot' or 'gene' column")
            return
        elif 'uniprot' not in test_data.columns:
            print("   Note: Using 'gene' column instead of 'uniprot'")
    
    # Data paths
    surface_path = 'raw_data/surfgraph_with_pos/intra_surf1'
    esm_features_path = 'raw_data/esm_features/esm_features/esm_features_ATA.pkl'
    pdb_path = 'raw_data/pdb'
    cache_dir = 'preprocessed_data'
    
    # Load ESM features
    esm_features = {}
    if esm_features_path and os.path.exists(esm_features_path):
        try:
            with open(esm_features_path, 'rb') as f:
                esm_features = pickle.load(f)
            print(f"   Loaded ESM features for {len(esm_features)} proteins")
        except Exception as e:
            print(f"   ⚠️  Failed to load ESM features: {e}")
    
    # Note: We need to rename 'uniprot' to 'gene' for the dataset to work properly
    if 'gene' not in test_data.columns and 'uniprot' in test_data.columns:
        test_data['gene'] = test_data['uniprot']
        print(f"   Mapped 'uniprot' to 'gene' column")
    elif 'gene' not in test_data.columns:
        print("❌ CSV must have either 'gene' or 'uniprot' column")
        return
    
    # Also add 'solubility' column if not present (for evaluation if available)
    has_labels = 'solubility' in test_data.columns
    if not has_labels:
        test_data['solubility'] = 0.0  # Placeholder, won't be used
        print(f"   Note: No 'solubility' column found, using placeholder for predictions only")
    
    # Create dataset (disable resampling for prediction)
    test_config = config.__class__()
    # Copy all attributes from training config
    for attr_name in dir(config):
        if not attr_name.startswith('_'):
            setattr(test_config, attr_name, getattr(config, attr_name))
    test_config.use_resampling = False
    
    # Create test dataset
    test_dataset = SurfSolDataset(
        test_data, 
        surface_path, 
        esm_features, 
        pdb_path, 
        cache_dir, 
        test_config
    )
    
    print(f"   Valid test samples: {len(test_dataset)}/{len(test_data)}")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        collate_fn=surf_collate_fn,
        num_workers=2
    )
    
    # Make ensemble predictions
    print(f"\n🔮 Making ensemble predictions with {len(models)} models...")
    ensemble_preds, fold_predictions, protein_names = ensemble_predict(
        models, test_loader, device, aggregation=args.aggregation
    )
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'uniprot': protein_names,
        'ensemble_prediction': ensemble_preds
    })
    
    # Add individual fold predictions
    for fold_idx in range(len(models)):
        results_df[f'fold_{fold_idx+1}_prediction'] = fold_predictions[fold_idx]
    
    # Add true labels if available
    if has_labels:
        results_df['true_solubility'] = results_df['uniprot'].map(
            dict(zip(test_data['uniprot'], test_data['solubility']))
        )
        
        # Calculate metrics for ensemble
        print(f"\n📈 Ensemble Model Performance:")
        ensemble_metrics = calculate_metrics(
            results_df['true_solubility'], 
            results_df['ensemble_prediction']
        )
        
        print(f"   RMSE: {ensemble_metrics['rmse']:.4f}")
        print(f"   R²: {ensemble_metrics['r2']:.4f}")
        print(f"   MSE: {ensemble_metrics['mse']:.4f}")
        print(f"   MAE: {ensemble_metrics['mae']:.4f}")
        print(f"   Pearson R: {ensemble_metrics['pearson_r']:.4f}")
        print(f"   Pearson R²: {ensemble_metrics['pearson_r2']:.4f}")
        print(f"   Pearson p-value: {ensemble_metrics['pearson_p']:.6f}")
        
        # Calculate metrics for each fold
        print(f"\n📊 Individual Fold Performance:")
        print(f"{'Fold':<6} {'RMSE':<10} {'R²':<10} {'MSE':<10} {'MAE':<10} {'Pearson R':<12}")
        print("-" * 70)
        
        fold_metrics_list = []
        for fold_idx in range(len(models)):
            fold_pred_col = f'fold_{fold_idx+1}_prediction'
            fold_metrics = calculate_metrics(
                results_df['true_solubility'],
                results_df[fold_pred_col]
            )
            fold_metrics_list.append(fold_metrics)
            
            print(f"{fold_idx+1:<6} {fold_metrics['rmse']:<10.4f} {fold_metrics['r2']:<10.4f} "
                  f"{fold_metrics['mse']:<10.4f} {fold_metrics['mae']:<10.4f} "
                  f"{fold_metrics['pearson_r']:<12.4f}")
        
        # Calculate statistics across folds
        print(f"\n📈 Cross-Fold Statistics:")
        fold_r2s = [m['r2'] for m in fold_metrics_list]
        fold_rmses = [m['rmse'] for m in fold_metrics_list]
        fold_pearson_rs = [m['pearson_r'] for m in fold_metrics_list]
        
        print(f"   R²: {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")
        print(f"   RMSE: {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")
        print(f"   Pearson R: {np.mean(fold_pearson_rs):.4f} ± {np.std(fold_pearson_rs):.4f}")
        
        # Compare ensemble vs individual folds
        print(f"\n🔄 Ensemble vs Individual Folds:")
        print(f"   Ensemble R²: {ensemble_metrics['r2']:.4f}")
        print(f"   Best Fold R²: {max(fold_r2s):.4f} (Fold {fold_r2s.index(max(fold_r2s))+1})")
        print(f"   Mean Fold R²: {np.mean(fold_r2s):.4f}")
        print(f"   Improvement: {ensemble_metrics['r2'] - np.mean(fold_r2s):.4f}")
    
    # Save predictions
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n💾 Predictions saved to {args.output_csv}")
    
    # Print sample predictions
    print(f"\n📝 Sample predictions:")
    display_cols = ['uniprot', 'ensemble_prediction']
    if has_labels:
        display_cols.append('true_solubility')
    if len(models) <= 3:  # Only show individual folds if not too many
        for i in range(len(models)):
            display_cols.append(f'fold_{i+1}_prediction')
    
    print(results_df[display_cols].head(10).to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    main()

