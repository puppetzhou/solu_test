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


def print_logo():
    """Print logo ASCII art"""
    try:
        with open('test_log/logo.txt', 'r', encoding='utf-8') as f:
            logo = f.read()
        print(logo)
    except FileNotFoundError:
        print("SURFSOL - Protein Solubility Prediction Model")
        print("=" * 60)
    except Exception as e:
        print(f"Failed to read logo: {e}")
        print("SURFSOL - Protein Solubility Prediction Model")
        print("=" * 60)


def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load configuration
    config = checkpoint.get('config', SurfSolConfig())
    
    # Create model
    model = SurfSolProteinModel(config).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(" Model loaded successfully")
    
    return model, config


def predict(model, dataloader, device):
    """Make predictions on test dataset"""
    model.eval()
    
    all_predictions = []
    all_protein_names = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", ncols=100):
            hetero_data = batch['hetero_data'].to(device)
            protein_names = batch['protein_names']
            
            # Forward pass
            outputs = model(hetero_data)
            
            # Get predictions
            if 'regression' in outputs:
                pred = outputs['regression'].squeeze().cpu().numpy()
                
                # Ensure pred is array
                if pred.ndim == 0:
                    pred = np.array([pred])
                
                all_predictions.extend(pred)
            elif 'classification' in outputs:
                # For classification output, convert to probability
                pred = torch.sigmoid(outputs['classification']).squeeze().cpu().numpy()
                
                if pred.ndim == 0:
                    pred = np.array([pred])
                
                all_predictions.extend(pred)
            else:
                # Fallback: use any available output
                if len(outputs) > 0:
                    first_key = list(outputs.keys())[0]
                    pred = outputs[first_key].squeeze().cpu().numpy()
                    if pred.ndim == 0:
                        pred = np.array([pred])
                    all_predictions.extend(pred)
            
            all_protein_names.extend(protein_names)
    
    return all_predictions, all_protein_names


def main():
    """Main testing function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Test SurfSol protein solubility prediction model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default paths
  python surf_test.py --checkpoint checkpoints/best_SurfSol_model.pth
  
  # With custom test file and output
  python surf_test.py --checkpoint checkpoints/best_SurfSol_model.pth \\
                      --test_csv S.cerevisiae_test.csv \\
                      --output_csv my_predictions.csv
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file (.pth)')
    parser.add_argument('--test_csv', type=str, 
                       default='raw_data/S.cerevisiae_test.csv',
                       help='Path to test CSV file')
    parser.add_argument('--output_csv', type=str,
                       default='predictions.csv',
                       help='Path to save predictions')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for inference')
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        print("   Please provide a valid checkpoint path using --checkpoint")
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
    
    # Load checkpoint
    model, config = load_checkpoint(args.checkpoint, device)
    
    print(f"\n📊 Model configuration:")
    print(f"   Training mode: {config.training_mode}")
    print(f"   Use surface: {config.use_surface}")
    print(f"   Use ESM: {config.use_esm}")
    print(f"   Use structure: {config.use_structure}")
    print(f"   Fusion type: {config.fusion_type}")
    
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
    
    # Data paths (adjust these to match your environment)
    surface_path = 'raw_data/surfgraph_with_pos/intra_surf1'
    esm_features_path = 'raw_data/esm_features/esm_features/esm_features_test.pkl'
    pdb_path = 'raw_data/pdb_test'
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
    if 'solubility' not in test_data.columns:
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
    
    # Make predictions
    print(f"\n🔮 Making predictions...")
    predictions, protein_names = predict(model, test_loader, device)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'uniprot': protein_names,
        'predicted_solubility': predictions
    })
    
    # Add true labels if available (for evaluation)
    if 'solubility' in test_data.columns:
        results_df['true_solubility'] = results_df['uniprot'].map(
            dict(zip(test_data['uniprot'], test_data['solubility']))
        )
        
        # Calculate metrics if true labels are available
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(results_df['true_solubility'], results_df['predicted_solubility'])
        r2 = r2_score(results_df['true_solubility'], results_df['predicted_solubility'])
        rmse = np.sqrt(mse)
        
        print(f"\n📈 Prediction metrics:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²: {r2:.4f}")
        print(f"   MSE: {mse:.4f}")
    
    # Save predictions
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n Predictions saved to {args.output_csv}")
    
    # Print sample predictions
    print(f"\n📝 Sample predictions:")
    print(results_df.head(10).to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    main()

