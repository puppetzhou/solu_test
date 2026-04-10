import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
import os
import shutil
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
import json
from surf_model import SurfSolProteinModel, SurfSolConfig
from surf_dataset import SurfSolDataset, surf_collate_fn


def print_logo():
    """Print logo ASCII art"""
    try:
        with open('raw_data/logo.txt', 'r', encoding='utf-8') as f:
            logo = f.read()
        print(logo)
    except FileNotFoundError:
        print("🌊 SURFSOL - Protein Solubility Prediction Model 🌊")
        print("=" * 60)
    except Exception as e:
        print(f"Failed to read logo: {e}")
        print("🌊 SURFSOL - Protein Solubility Prediction Model 🌊")
        print("=" * 60)


def load_cv_data(train_csv_path, surface_path, esm_features_path=None, pdb_path=None, cache_dir="data_cache", config=None):
    """Load training data for cross-validation
    
    Note: Cross-validation should only use training data to avoid data leakage.
    Test data should be kept separate for final model evaluation.
    """
    import pandas as pd
    
    # Load only training data for cross-validation
    train_data = pd.read_csv(train_csv_path)
    
    print(f"Loading training data for cross-validation: {len(train_data)} records")
    print(f"  - Training data: {len(train_data)} records from {train_csv_path}")
    print(f"  - Note: Test data is kept separate for final evaluation")
    
    # Load ESM features
    esm_features = {}
    if esm_features_path and os.path.exists(esm_features_path):
        try:
            with open(esm_features_path, 'rb') as f:
                esm_features = pickle.load(f)
            print(f"Loading ESM features: {len(esm_features)} proteins")
        except Exception as e:
            print(f"Failed to load ESM features: {e}")
    
    # Create combined dataset (disable resampling for CV)
    cv_config = config.__class__() if config else None
    if cv_config and config:
        # Copy config but disable resampling for CV
        for attr_name in dir(config):
            if not attr_name.startswith('_'):
                setattr(cv_config, attr_name, getattr(config, attr_name))
        cv_config.use_resampling = False
    
    cv_dataset = SurfSolDataset(train_data, surface_path, esm_features, pdb_path, cache_dir, cv_config)
    
    return cv_dataset


def create_cv_folds(dataset, n_splits=5, random_state=42, stratify=True):
    """Create K-fold splits for cross-validation"""
    
    # Get all valid indices and corresponding solubility values
    # Note: dataset.__getitem__ returns dict with 'regression_target', not 'solubility'
    # So we access dataset.valid_data directly which contains 'solubility' key
    valid_indices = list(range(len(dataset)))
    solubility_values = [dataset.valid_data[i]['solubility'] for i in valid_indices]
    
    if stratify:
        # Stratified split based on solubility bins
        solubility_array = np.array(solubility_values)
        
        # Create stratification bins based on solubility quartiles
        quartiles = np.percentile(solubility_array, [25, 50, 75])
        strat_labels = np.digitize(solubility_array, quartiles)
        
        print(f"Stratified CV based on solubility quartiles:")
        print(f"  - Q1 (≤{quartiles[0]:.2f}): {np.sum(strat_labels==0)} samples")
        print(f"  - Q2 ({quartiles[0]:.2f}-{quartiles[1]:.2f}): {np.sum(strat_labels==1)} samples")
        print(f"  - Q3 ({quartiles[1]:.2f}-{quartiles[2]:.2f}): {np.sum(strat_labels==2)} samples")
        print(f"  - Q4 (>{quartiles[2]:.2f}): {np.sum(strat_labels==3)} samples")
        
        # Use StratifiedKFold
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_splits = list(kf.split(valid_indices, strat_labels))
    else:
        # Regular K-fold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_splits = list(kf.split(valid_indices))
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        train_indices = [valid_indices[i] for i in train_idx]
        val_indices = [valid_indices[i] for i in val_idx]
        
        # Calculate fold statistics
        train_sol = [solubility_values[i] for i in train_indices]
        val_sol = [solubility_values[i] for i in val_indices]
        
        folds.append({
            'fold': fold_idx + 1,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'train_sol_mean': np.mean(train_sol),
            'val_sol_mean': np.mean(val_sol),
            'train_sol_std': np.std(train_sol),
            'val_sol_std': np.std(val_sol)
        })
        
        print(f"Fold {fold_idx + 1}: Train({len(train_indices)}) | Val({len(val_indices)}) | "
              f"Sol_mean(Train:{np.mean(train_sol):.3f}, Val:{np.mean(val_sol):.3f})")
    
    return folds


class SurfSolCrossValidator:
    """SurfSol Cross-Validator for K-fold training"""
    
    def __init__(self, config, base_log_dir="log/SurfSol_CV", base_checkpoint_dir="checkpoints_cv", loss_function="mse"):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_log_dir = base_log_dir
        self.base_checkpoint_dir = base_checkpoint_dir
        self.loss_function = loss_function
        
        print(f"Using device: {self.device}")
        
        # Create base directories
        os.makedirs(self.base_log_dir, exist_ok=True)
        os.makedirs(self.base_checkpoint_dir, exist_ok=True)
        
        # CV results storage
        self.fold_results = []
        
    def _create_trainer_for_fold(self, fold_idx):
        """Create a trainer instance for a specific fold"""
        
        # Create fold-specific directories
        fold_log_dir = os.path.join(self.base_log_dir, f"fold_{fold_idx}")
        fold_checkpoint_dir = os.path.join(self.base_checkpoint_dir, f"fold_{fold_idx}")
        
        os.makedirs(fold_log_dir, exist_ok=True)
        os.makedirs(fold_checkpoint_dir, exist_ok=True)
        
        # Create TensorBoard writer
        writer = SummaryWriter(fold_log_dir)
        
        # Initialize model
        model = SurfSolProteinModel(self.config).to(self.device)
        
        # Loss function
        if self.loss_function == "mse":
            regression_criterion = nn.MSELoss()
        elif self.loss_function == "smoothl1":
            regression_criterion = nn.SmoothL1Loss()
        elif self.loss_function == "huber":
            regression_criterion = nn.HuberLoss()
        elif self.loss_function == "mae":
            regression_criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
        
        classification_criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.config.patience//2, factor=0.5)
        
        return {
            'model': model,
            'regression_criterion': regression_criterion,
            'classification_criterion': classification_criterion,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'writer': writer,
            'checkpoint_dir': fold_checkpoint_dir,
            'global_step': 0
        }
    
    def pearson_correlation_loss(self, y_pred, y_true):
        """Calculate Pearson correlation coefficient squared as a differentiable loss"""
        if y_pred.numel() == 0 or y_true.numel() == 0:
            return torch.tensor(0.0, device=y_pred.device)
        
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        pred_mean = torch.mean(y_pred)
        true_mean = torch.mean(y_true)
        
        pred_centered = y_pred - pred_mean
        true_centered = y_true - true_mean
        
        numerator = torch.sum(pred_centered * true_centered)
        pred_std = torch.sqrt(torch.sum(pred_centered ** 2))
        true_std = torch.sqrt(torch.sum(true_centered ** 2))
        
        denominator = pred_std * true_std
        if denominator < 1e-8:
            return torch.tensor(0.0, device=y_pred.device)
        
        correlation = numerator / denominator
        return correlation ** 2
    
    def train_fold(self, fold_info, train_loader, val_loader, num_epochs):
        """Train a single fold"""
        fold_idx = fold_info['fold']
        print(f"\n{'='*60}")
        print(f"🔄 Training Fold {fold_idx}/{len(self.fold_results) + 1}")
        print(f"   Train samples: {fold_info['train_size']}")
        print(f"   Val samples: {fold_info['val_size']}")
        print(f"{'='*60}")
        
        # Create trainer components for this fold
        trainer = self._create_trainer_for_fold(fold_idx)
        
        best_val_loss = float('inf')
        best_val_r2 = float('-inf')  # Track best R² separately
        patience_counter = 0
        fold_history = []
        
        # Create epoch progress bar
        epoch_pbar = tqdm(range(num_epochs), desc=f"Fold {fold_idx} Training", 
                         ncols=150, position=0, leave=True,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        
        # Initialize progress bar
        epoch_pbar.set_postfix({'Status': 'Initializing...'})
        epoch_pbar.refresh()
        
        print(f"  📦 Preparing data loaders...")
        
        for epoch in epoch_pbar:
            # Update progress bar status
            epoch_pbar.set_postfix({'Status': f'Epoch {epoch+1}/{num_epochs}...'})
            epoch_pbar.refresh()
            
            # Training
            train_metrics = self._train_epoch(trainer, train_loader, epoch)
            
            # Validation
            val_metrics = self._evaluate(trainer, val_loader, epoch, 'Val')
            
            # Learning rate scheduling
            trainer['scheduler'].step(val_metrics['loss'])
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'ValLoss': f'{val_metrics["loss"]:.4f}',
                'ValR²': f'{val_metrics["regression_r2"]:.3f}',
                'BestR²': f'{best_val_r2:.3f}',
                'Patience': f'{patience_counter}/{self.config.patience}'
            })
            
            # Record fold history
            fold_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_r2': val_metrics['regression_r2'],
                'lr': trainer['optimizer'].param_groups[0]['lr']
            })
            
            # Early stopping and model saving based on R² (better metric for regression)
            if val_metrics['regression_r2'] > best_val_r2:
                best_val_r2 = val_metrics['regression_r2']
                patience_counter = 0
                
                # Save best model based on R²
                checkpoint_path = os.path.join(trainer['checkpoint_dir'], f'best_fold_{fold_idx}_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer['model'].state_dict(),
                    'optimizer_state_dict': trainer['optimizer'].state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_r2': val_metrics['regression_r2'],
                    'val_metrics': val_metrics,
                    'config': self.config,
                    'fold_info': fold_info
                }, checkpoint_path)
                
                best_val_metrics = val_metrics.copy()
                best_val_loss = val_metrics['loss']  # Also track best loss for reference
                tqdm.write(f"💾 Best model saved (R²={best_val_r2:.4f}, Loss={val_metrics['loss']:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    tqdm.write(f"🛑 Early stopping at epoch {epoch+1}")
                    break
        
        # Close writer
        trainer['writer'].close()
        
        # Record fold results
        fold_result = {
            'fold': fold_idx,
            'best_val_loss': best_val_loss,
            'best_val_r2': best_val_r2,
            'best_val_metrics': best_val_metrics,
            'training_history': fold_history,
            'fold_info': fold_info
        }
        
        self.fold_results.append(fold_result)
        
        print(f"✅ Fold {fold_idx} completed | Best Val R²: {best_val_r2:.4f} | Best Val Loss: {best_val_loss:.4f}")
        
        return fold_result
    
    def _train_epoch(self, trainer, dataloader, epoch):
        """Train one epoch for a fold"""
        trainer['model'].train()
        total_loss = 0
        regression_losses = []
        
        # Add debug output for first batch
        if epoch == 0:
            print("  ⏳ Loading first batch (this may take a moment)...")
        
        for batch_idx, batch in enumerate(dataloader):
            # Debug output for first batch
            if epoch == 0 and batch_idx == 0:
                print("  ✅ First batch loaded, starting training...")
            hetero_data = batch['hetero_data'].to(self.device)
            regression_targets = batch['regression_targets'].to(self.device)
            
            trainer['optimizer'].zero_grad()
            
            # Forward pass
            outputs = trainer['model'](hetero_data)
            
            # Regression loss
            regression_loss = trainer['regression_criterion'](
                outputs['regression'].squeeze(), 
                regression_targets.squeeze()
            )
            
            # Add Pearson correlation loss if enabled
            if self.config.use_pearson_loss:
                pearson_r2 = self.pearson_correlation_loss(
                    outputs['regression'].squeeze(), 
                    regression_targets.squeeze()
                )
                regression_loss = regression_loss - self.config.pearson_weight * pearson_r2
            
            # Backward pass
            regression_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer['model'].parameters(), max_norm=1.0)
            trainer['optimizer'].step()
            
            total_loss += regression_loss.item()
            regression_losses.append(regression_loss.item())
            
            # Log to TensorBoard
            trainer['writer'].add_scalar('Train/Batch_Loss', regression_loss.item(), trainer['global_step'])
            trainer['writer'].add_scalar('Train/Learning_Rate', trainer['optimizer'].param_groups[0]['lr'], trainer['global_step'])
            trainer['global_step'] += 1
        
        avg_loss = total_loss / len(dataloader)
        trainer['writer'].add_scalar('Train/Epoch_Loss', avg_loss, epoch)
        
        return {'loss': avg_loss}
    
    def _evaluate(self, trainer, dataloader, epoch, split_name):
        """Evaluate model for a fold"""
        trainer['model'].eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                hetero_data = batch['hetero_data'].to(self.device)
                regression_targets = batch['regression_targets'].to(self.device)
                
                outputs = trainer['model'](hetero_data)
                
                regression_loss = trainer['regression_criterion'](
                    outputs['regression'].squeeze(), 
                    regression_targets.squeeze()
                )
                
                # Add Pearson correlation loss if enabled
                if self.config.use_pearson_loss:
                    pearson_r2 = self.pearson_correlation_loss(
                        outputs['regression'].squeeze(), 
                        regression_targets.squeeze()
                    )
                    regression_loss = regression_loss - self.config.pearson_weight * pearson_r2
                
                total_loss += regression_loss.item()
                
                # Collect predictions
                pred = outputs['regression'].squeeze().cpu().numpy()
                target = regression_targets.squeeze().cpu().numpy()
                
                if pred.ndim == 0:
                    pred = np.array([pred])
                if target.ndim == 0:
                    target = np.array([target])
                    
                all_preds.extend(pred)
                all_targets.extend(target)
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        mse = mean_squared_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        rmse = np.sqrt(mse)
        
        # Calculate Pearson correlation
        pearson_r2_eval = 0.0
        if self.config.use_pearson_loss:
            import scipy.stats
            pearson_r, _ = scipy.stats.pearsonr(all_targets, all_preds)
            pearson_r2_eval = pearson_r ** 2
        
        # Log to TensorBoard
        trainer['writer'].add_scalar(f'{split_name}/Loss', avg_loss, epoch)
        trainer['writer'].add_scalar(f'{split_name}/MSE', mse, epoch)
        trainer['writer'].add_scalar(f'{split_name}/RMSE', rmse, epoch)
        trainer['writer'].add_scalar(f'{split_name}/R2', r2, epoch)
        if self.config.use_pearson_loss:
            trainer['writer'].add_scalar(f'{split_name}/Pearson_R2', pearson_r2_eval, epoch)
        
        return {
            'loss': avg_loss,
            'regression_mse': mse,
            'regression_rmse': rmse,
            'regression_r2': r2,
            'pearson_r2': pearson_r2_eval
        }
    
    def run_cross_validation(self, dataset, folds, num_epochs, batch_size=10):
        """Run complete cross-validation"""
        print(f"\n🚀 Starting {len(folds)}-Fold Cross-Validation")
        print(f"Total dataset size: {len(dataset)}")
        print(f"Epochs per fold: {num_epochs}")
        print(f"Batch size: {batch_size}")
        
        for fold_info in folds:
            # Create data loaders for this fold
            train_indices = fold_info['train_indices']
            val_indices = fold_info['val_indices']
            
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, 
                                    shuffle=True, collate_fn=surf_collate_fn, num_workers=2)
            val_loader = DataLoader(val_subset, batch_size=batch_size, 
                                   shuffle=False, collate_fn=surf_collate_fn, num_workers=2)
            
            # Test data loading (this helps identify if data loading is the bottleneck)
            print(f"  🔍 Testing data loader (loading first batch)...")
            try:
                test_batch = next(iter(train_loader))
                print(f"  ✅ Data loader working! Batch keys: {test_batch.keys()}")
            except Exception as e:
                print(f"  ❌ Error loading data: {e}")
                raise
            
            # Train this fold
            fold_result = self.train_fold(fold_info, train_loader, val_loader, num_epochs)
        
        # Aggregate and report results
        return self._aggregate_results()
    
    def _aggregate_results(self):
        """Aggregate results across all folds"""
        print(f"\n{'='*80}")
        print("📊 CROSS-VALIDATION RESULTS SUMMARY")
        print(f"{'='*80}")
        
        # Extract metrics from all folds
        val_losses = [fold['best_val_loss'] for fold in self.fold_results]
        val_r2s = [fold['best_val_metrics']['regression_r2'] for fold in self.fold_results]
        val_mses = [fold['best_val_metrics']['regression_mse'] for fold in self.fold_results]
        val_rmses = [fold['best_val_metrics']['regression_rmse'] for fold in self.fold_results]
        
        # Calculate aggregated statistics
        aggregated_results = {
            'n_folds': len(self.fold_results),
            'val_loss_mean': np.mean(val_losses),
            'val_loss_std': np.std(val_losses),
            'val_r2_mean': np.mean(val_r2s),
            'val_r2_std': np.std(val_r2s),
            'val_mse_mean': np.mean(val_mses),
            'val_mse_std': np.std(val_mses),
            'val_rmse_mean': np.mean(val_rmses),
            'val_rmse_std': np.std(val_rmses),
            'fold_results': self.fold_results
        }
        
        # Print individual fold results
        print(f"Individual Fold Results:")
        print(f"{'Fold':<6} {'Val Loss':<10} {'Val R²':<10} {'Val MSE':<10} {'Val RMSE':<10}")
        print("-" * 50)
        for i, fold in enumerate(self.fold_results):
            metrics = fold['best_val_metrics']
            print(f"{fold['fold']:<6} {fold['best_val_loss']:<10.4f} {metrics['regression_r2']:<10.4f} "
                  f"{metrics['regression_mse']:<10.4f} {metrics['regression_rmse']:<10.4f}")
        
        print(f"\nAggregated Results:")
        print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'95% CI':<20}")
        print("-" * 55)
        
        # Calculate 95% confidence intervals
        def confidence_interval(values, confidence=0.95):
            import scipy.stats as stats
            n = len(values)
            mean = np.mean(values)
            std = np.std(values)
            h = std * stats.t.ppf((1 + confidence) / 2, n - 1) / np.sqrt(n)
            return mean - h, mean + h
        
        metrics_to_report = [
            ('Val Loss', val_losses),
            ('Val R²', val_r2s),
            ('Val MSE', val_mses),
            ('Val RMSE', val_rmses)
        ]
        
        for metric_name, values in metrics_to_report:
            mean_val = np.mean(values)
            std_val = np.std(values)
            ci_low, ci_high = confidence_interval(values)
            print(f"{metric_name:<15} {mean_val:<10.4f} {std_val:<10.4f} [{ci_low:.4f}, {ci_high:.4f}]")
        
        # Save detailed results
        results_path = os.path.join(self.base_log_dir, 'cv_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {}
            for key, value in aggregated_results.items():
                if key == 'fold_results':
                    json_results[key] = value  # Keep as is for now
                else:
                    json_results[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
            
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_path}")
        print(f"📁 Individual fold models saved in: {self.base_checkpoint_dir}/")
        print(f"📊 TensorBoard logs saved in: {self.base_log_dir}/")
        
        return aggregated_results


def main():
    """Main cross-validation function"""
    # ================================================================
    # Cross-validation and model configuration - freely adjustable
    # ================================================================
    
    # Cross-validation parameters
    n_folds = 5
    stratified = True  # Whether to use stratified K-fold
    random_state = 42
    
    # Basic training parameters
    learning_rate = 1e-4
    weight_decay = 1e-4
    patience = 20  # Reduced for CV to avoid overfitting
    batch_size = 10
    num_epochs = 200  # Reduced for CV efficiency
    
    # Loss function configuration
    loss_function = "mse"  # "mse", "smoothl1", "huber", "mae"
    
    # Pearson correlation loss configuration
    use_pearson_loss = False
    pearson_weight = 1
    
    # Training mode configuration
    training_mode = "regression"  # Focus on regression for CV
    
    # Modal switches - control which modalities to use
    use_surface = True
    use_esm = True
    use_structure = True
    
    # Fusion method configuration
    fusion_type = "attention"  # "concat" or "attention"
    attention_heads = 4
    
    # Model architecture parameters
    surface_emb_dim = 8
    surface_edge_dim = 3
    num_conv_layers = 3
    esm_dim = 640
    esm_emb_dim = 4
    structure_hidden_channels = 150
    structure_num_filters = 150
    structure_num_interactions = 2
    structure_num_gaussians = 300
    structure_cutoff = 15.0
    structure_max_num_neighbors = 150
    structure_emb_dim = 8
    hidden_dim = 16
    dropout = 0
    
    # Data paths
    # Note: Only use training data for cross-validation
    # Test data should be kept separate for final evaluation via surf_test_cv.py
    train_csv_path = 'raw_data/sufsol_train2.csv'
    surface_path = 'raw_data/surfgraph_with_pos/intra_surf1'
    esm_features_path = 'raw_data/esm_features/esm_features/esm_features2.pkl'
    pdb_path = 'raw_data/pdb'

    # Cache settings
    cache_dir = 'preprocessed_data'
    
    # Output paths configuration
    base_log_dir = "log/SurfSol_CV"
    base_checkpoint_dir = "checkpoints_cv"
    
    # ================================================================
    # Create configuration object
    # ================================================================
    config = SurfSolConfig()
    
    # Training parameters
    config.learning_rate = learning_rate
    config.weight_decay = weight_decay
    config.patience = patience
    config.batch_size = batch_size
    
    # Pearson correlation loss parameters
    config.use_pearson_loss = use_pearson_loss
    config.pearson_weight = pearson_weight
    
    # Training mode configuration
    config.training_mode = training_mode
    config.use_classification_head = False  # Disable for regression CV
    
    # Disable resampling for CV (we want fair comparison across folds)
    config.use_resampling = False
    
    # Modal switches
    config.use_surface = use_surface
    config.use_esm = use_esm
    config.use_structure = use_structure
    
    # Fusion method configuration
    config.fusion_type = fusion_type
    config.attention_heads = attention_heads
    
    # Network parameters
    config.surface_emb_dim = surface_emb_dim
    config.surface_edge_dim = surface_edge_dim
    config.num_conv_layers = num_conv_layers
    config.esm_dim = esm_dim
    config.esm_emb_dim = esm_emb_dim
    config.structure_hidden_channels = structure_hidden_channels
    config.structure_num_filters = structure_num_filters
    config.structure_num_interactions = structure_num_interactions
    config.structure_num_gaussians = structure_num_gaussians
    config.structure_cutoff = structure_cutoff
    config.structure_max_num_neighbors = structure_max_num_neighbors
    config.structure_emb_dim = structure_emb_dim
    config.hidden_dim = hidden_dim
    config.dropout = dropout
    
    # Print logo and info
    print_logo()
    
    print("🚀 Starting SurfSol Cross-Validation Training")
    print(f"Modal configuration: Surface({use_surface}) + ESM({use_esm}) + 3D Structure({use_structure})")
    print(f"Cross-validation: {n_folds}-fold {'stratified' if stratified else 'standard'}")
    print(f"Fusion method: {fusion_type.upper()}" + (f" (heads: {attention_heads})" if fusion_type == "attention" else ""))
    print(f"Loss function: {loss_function.upper()}")
    
    # Load training data for cross-validation
    # Note: Cross-validation only uses training data to avoid data leakage
    # Test data is kept separate and should be evaluated using surf_test_cv.py
    print("\n📊 Loading training data for cross-validation...")
    dataset = load_cv_data(
        train_csv_path, surface_path, esm_features_path, pdb_path, cache_dir, config
    )
    
    print(f"📈 Total dataset size: {len(dataset)}")
    
    # Print dataset statistics
    labels = [item['solubility'] for item in dataset.valid_data]
    labels = np.array(labels, dtype=np.float32)
    print(f"Solubility statistics: mean={labels.mean():.3f}, std={labels.std():.3f}, min={labels.min():.3f}, max={labels.max():.3f}")
    
    # Create cross-validation folds
    print(f"\n🔄 Creating {n_folds}-fold cross-validation splits...")
    folds = create_cv_folds(dataset, n_splits=n_folds, random_state=random_state, stratify=stratified)
    
    # Initialize cross-validator
    cv = SurfSolCrossValidator(config, base_log_dir, base_checkpoint_dir, loss_function)
    
    print(f"\n🏗️ Model parameters: {sum(p.numel() for p in SurfSolProteinModel(config).parameters()):,}")
    
    # Run cross-validation
    print("🏋️ Starting cross-validation training...")
    
    cv_results = cv.run_cross_validation(dataset, folds, num_epochs, batch_size)
    
    print("\n✅ Cross-validation completed!")
    print(f"📊 Final CV R² Score: {cv_results['val_r2_mean']:.4f} ± {cv_results['val_r2_std']:.4f}")
    print(f"📁 Results saved to: {base_log_dir}/")
    
    return cv_results


if __name__ == "__main__":
    main()
