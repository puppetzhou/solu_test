import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import os
import shutil
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from surf_model import SurfSolProteinModel, SurfSolConfig
from surf_dataset import load_train_test_data, surf_collate_fn


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


class SurfSolTrainer:
    """SurfSol-based trainer"""
    def __init__(self, config, tensorboard_log_dir=None, checkpoint_dir=None, loss_function="mse"):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create TensorBoard log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = tensorboard_log_dir or "log/root/tf-logs/SurfSol_model"
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Backup source code files to log directory
        self._backup_source_code(timestamp)
        
        # Create checkpoint directory
        self.checkpoint_dir = checkpoint_dir or "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize model
        self.model = SurfSolProteinModel(config).to(self.device)
        
        # Loss function - dynamically selected
        self.loss_function_name = loss_function
        if loss_function == "mse":
            self.regression_criterion = nn.MSELoss()
        elif loss_function == "smoothl1":
            self.regression_criterion = nn.SmoothL1Loss()
        elif loss_function == "huber":
            self.regression_criterion = nn.HuberLoss()
        elif loss_function == "mae":
            self.regression_criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        # Classification loss for mix_train mode
        self.classification_criterion = nn.BCEWithLogitsLoss()
        
        # 优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=config.patience//2, factor=0.5)
        
        # Record global steps
        self.global_step = 0
        
    def pearson_correlation_loss(self, y_pred, y_true):
        """Calculate Pearson correlation coefficient squared as a differentiable loss"""
        # Ensure we have valid data
        if y_pred.numel() == 0 or y_true.numel() == 0:
            return torch.tensor(0.0, device=y_pred.device)
        
        # Flatten tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # Calculate means
        pred_mean = torch.mean(y_pred)
        true_mean = torch.mean(y_true)
        
        # Calculate centered values
        pred_centered = y_pred - pred_mean
        true_centered = y_true - true_mean
        
        # Calculate correlation coefficient
        numerator = torch.sum(pred_centered * true_centered)
        pred_std = torch.sqrt(torch.sum(pred_centered ** 2))
        true_std = torch.sqrt(torch.sum(true_centered ** 2))
        
        # Avoid division by zero
        denominator = pred_std * true_std
        if denominator < 1e-8:
            return torch.tensor(0.0, device=y_pred.device)
        
        correlation = numerator / denominator
        
        # Return r² (squared correlation)
        return correlation ** 2
        
    def _backup_source_code(self, timestamp):
        """Backup source code files to log directory"""
        # Create code backup directory
        code_backup_dir = os.path.join(self.log_dir, "source_code")
        os.makedirs(code_backup_dir, exist_ok=True)
        
        # List of source files to backup
        source_files = [
            "surf_dataset.py",
            "surf_model.py", 
            "surf_train.py"
        ]
        
        for file_name in source_files:
            if os.path.exists(file_name):
                # Add timestamp to backup filename
                backup_name = f"{timestamp}_{file_name}"
                backup_path = os.path.join(code_backup_dir, backup_name)
                shutil.copy2(file_name, backup_path)
        
    def train_epoch(self, dataloader, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        regression_losses = []
        classification_losses = []
        
        # 创建详细的进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", 
                   leave=False, ncols=120, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        
        for batch_idx, batch in enumerate(pbar):
            hetero_data = batch['hetero_data'].to(self.device)
            regression_targets = batch['regression_targets'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(hetero_data)
            
            # Calculate losses based on training mode
            classification_loss = torch.tensor(0.0, device=self.device)
            regression_loss = torch.tensor(0.0, device=self.device)
            
            if self.config.training_mode in ["classification", "regression+classification"] and 'classification' in outputs:
                # Classification task
                classification_targets = batch['classification_targets'].to(self.device)
                classification_loss = self.classification_criterion(
                    outputs['classification'].squeeze(), 
                    classification_targets.squeeze()
                )
                classification_losses.append(classification_loss.item())
            
            if self.config.training_mode in ["regression", "regression+classification"] and 'regression' in outputs:
                # Regression task
                regression_loss = self.regression_criterion(
                    outputs['regression'].squeeze(), 
                    regression_targets.squeeze()
                )
                
                # Add Pearson correlation loss if enabled
                if self.config.use_pearson_loss:
                    pearson_r2 = self.pearson_correlation_loss(
                        outputs['regression'].squeeze(), 
                        regression_targets.squeeze()
                    )
                    # Subtract Pearson loss to encourage higher correlation
                    regression_loss = regression_loss - self.config.pearson_weight * pearson_r2
            
            # Combine losses based on training mode
            if self.config.training_mode == "regression+classification":
                total_batch_loss = (self.config.classification_weight * classification_loss + 
                                  self.config.regression_weight * regression_loss)
            elif self.config.training_mode == "classification":
                total_batch_loss = self.config.classification_weight * classification_loss
            else:  # regression
                total_batch_loss = regression_loss
            
            # 反向传播
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            regression_losses.append(regression_loss.item())
            
            # 记录到TensorBoard
            self.writer.add_scalar('Train/Batch_Total_Loss', total_batch_loss.item(), self.global_step)
            self.writer.add_scalar('Train/Batch_Regression_Loss', regression_loss.item(), self.global_step)
            if self.config.training_mode in ["classification", "regression+classification"] and 'classification' in outputs:
                self.writer.add_scalar('Train/Batch_Classification_Loss', classification_loss.item(), self.global_step)
            if self.config.use_pearson_loss and self.config.training_mode in ["regression", "regression+classification"]:
                # Calculate and log current batch Pearson correlation
                current_pearson_r2 = self.pearson_correlation_loss(
                    outputs['regression'].squeeze(), 
                    regression_targets.squeeze()
                )
                self.writer.add_scalar('Train/Batch_Pearson_R2', current_pearson_r2.item(), self.global_step)
            self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            self.global_step += 1
            
            # 更新进度条
            if self.config.training_mode in ["classification", "regression+classification"] and classification_losses:
                if self.config.training_mode == "regression+classification":
                    pbar.set_postfix({
                        'RegLoss': f'{regression_loss.item():.4f}',
                        'ClsLoss': f'{classification_loss.item():.4f}',
                        'TotalLoss': f'{total_batch_loss.item():.4f}',
                        'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                    })
                else:  # classification only
                    pbar.set_postfix({
                        'ClsLoss': f'{classification_loss.item():.4f}',
                        'TotalLoss': f'{total_batch_loss.item():.4f}',
                        'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                    })
            else:  # regression only
                pbar.set_postfix({
                    'RegLoss': f'{regression_loss.item():.4f}',
                    'TotalLoss': f'{total_batch_loss.item():.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        avg_loss = total_loss / len(dataloader)
        avg_reg_loss = np.mean(regression_losses)
        avg_cls_loss = np.mean(classification_losses) if classification_losses else 0.0
        
        # 记录epoch级别的指标
        self.writer.add_scalar('Train/Epoch_Total_Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/Epoch_Regression_Loss', avg_reg_loss, epoch)
        if self.config.training_mode == "mix_train" and classification_losses:
            self.writer.add_scalar('Train/Epoch_Classification_Loss', avg_cls_loss, epoch)
        
        return avg_loss, avg_reg_loss, avg_cls_loss
    
    def evaluate(self, dataloader, epoch, split_name):
        """Evaluate model with support for mixed training"""
        self.model.eval()
        total_loss = 0
        all_regression_preds = []
        all_regression_targets = []
        all_classification_preds = []
        all_classification_targets = []
        
        # 创建评估进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [{split_name}]", 
                   leave=False, ncols=120,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        
        with torch.no_grad():
            for batch in pbar:
                hetero_data = batch['hetero_data'].to(self.device)
                regression_targets = batch['regression_targets'].to(self.device)
                
                outputs = self.model(hetero_data)
                
                # Calculate losses based on training mode
                classification_loss = torch.tensor(0.0, device=self.device)
                regression_loss = torch.tensor(0.0, device=self.device)
                
                if self.config.training_mode in ["classification", "regression+classification"] and 'classification' in outputs:
                    # Classification task
                    classification_targets = batch['classification_targets'].to(self.device)
                    classification_loss = self.classification_criterion(
                        outputs['classification'].squeeze(), 
                        classification_targets.squeeze()
                    )
                    
                    # 收集分类预测结果
                    cls_pred = torch.sigmoid(outputs['classification']).squeeze().cpu().numpy()
                    cls_target = classification_targets.squeeze().cpu().numpy()
                    
                    if cls_pred.ndim == 0:
                        cls_pred = np.array([cls_pred])
                    if cls_target.ndim == 0:
                        cls_target = np.array([cls_target])
                        
                    all_classification_preds.extend(cls_pred)
                    all_classification_targets.extend(cls_target)
                
                if self.config.training_mode in ["regression", "regression+classification"] and 'regression' in outputs:
                    # Regression task
                    regression_loss = self.regression_criterion(
                        outputs['regression'].squeeze(), 
                        regression_targets.squeeze()
                    )
                    
                    # Add Pearson correlation loss if enabled
                    if self.config.use_pearson_loss:
                        pearson_r2 = self.pearson_correlation_loss(
                            outputs['regression'].squeeze(), 
                            regression_targets.squeeze()
                        )
                        # Subtract Pearson loss to encourage higher correlation
                        regression_loss = regression_loss - self.config.pearson_weight * pearson_r2
                    
                    # 收集回归预测结果
                    reg_pred = outputs['regression'].squeeze().cpu().numpy()
                    reg_target = regression_targets.squeeze().cpu().numpy()
                    
                    # 确保是数组形式
                    if reg_pred.ndim == 0:
                        reg_pred = np.array([reg_pred])
                    if reg_target.ndim == 0:
                        reg_target = np.array([reg_target])
                        
                    all_regression_preds.extend(reg_pred)
                    all_regression_targets.extend(reg_target)
                
                # Combine losses based on training mode
                if self.config.training_mode == "regression+classification":
                    total_batch_loss = (self.config.classification_weight * classification_loss + 
                                      self.config.regression_weight * regression_loss)
                elif self.config.training_mode == "classification":
                    total_batch_loss = self.config.classification_weight * classification_loss
                else:  # regression
                    total_batch_loss = regression_loss
                
                total_loss += total_batch_loss.item()
                
                # 更新进度条
                if self.config.training_mode in ["classification", "regression+classification"] and all_classification_targets:
                    if self.config.training_mode == "regression+classification":
                        pbar.set_postfix({
                            'RegLoss': f'{regression_loss.item():.4f}',
                            'ClsLoss': f'{classification_loss.item():.4f}',
                            'TotalLoss': f'{total_batch_loss.item():.4f}'
                        })
                    else:  # classification only
                        pbar.set_postfix({
                            'ClsLoss': f'{classification_loss.item():.4f}',
                            'TotalLoss': f'{total_batch_loss.item():.4f}'
                        })
                else:  # regression only
                    pbar.set_postfix({
                        'RegLoss': f'{regression_loss.item():.4f}',
                        'TotalLoss': f'{total_batch_loss.item():.4f}'
                    })
        
        # 计算指标
        avg_loss = total_loss / len(dataloader)
        
        # 回归指标
        reg_mse = 0.0
        reg_r2 = 0.0
        reg_rmse = 0.0
        if all_regression_targets:
            reg_mse = mean_squared_error(all_regression_targets, all_regression_preds)
            reg_r2 = r2_score(all_regression_targets, all_regression_preds)
            reg_rmse = np.sqrt(reg_mse)
        
        # 分类指标（如果是包含分类的训练模式）
        cls_acc = 0.0
        cls_auc = 0.0
        if self.config.training_mode in ["classification", "regression+classification"] and all_classification_targets:
            # 二分类准确率
            cls_pred_binary = (np.array(all_classification_preds) > 0.5).astype(int)
            cls_acc = accuracy_score(all_classification_targets, cls_pred_binary)
            # AUC
            try:
                cls_auc = roc_auc_score(all_classification_targets, all_classification_preds)
            except:
                cls_auc = 0.0
        
        # 记录到TensorBoard
        self.writer.add_scalar(f'{split_name}/Loss', avg_loss, epoch)
        if all_regression_targets:
            self.writer.add_scalar(f'{split_name}/Regression_MSE', reg_mse, epoch)
            self.writer.add_scalar(f'{split_name}/Regression_RMSE', reg_rmse, epoch)
            self.writer.add_scalar(f'{split_name}/Regression_R2', reg_r2, epoch)
            
            # Calculate and log Pearson correlation for the entire evaluation set
            if self.config.use_pearson_loss:
                import scipy.stats
                pearson_r, _ = scipy.stats.pearsonr(all_regression_targets, all_regression_preds)
                pearson_r2_eval = pearson_r ** 2
                self.writer.add_scalar(f'{split_name}/Pearson_R2', pearson_r2_eval, epoch)
        
        if self.config.training_mode in ["classification", "regression+classification"] and all_classification_targets:
            self.writer.add_scalar(f'{split_name}/Classification_Accuracy', cls_acc, epoch)
            self.writer.add_scalar(f'{split_name}/Classification_AUC', cls_auc, epoch)
        
        # 添加预测vs实际值的散点图
        import matplotlib.pyplot as plt
        
        if self.config.training_mode == "mix_train" and all_classification_targets:
            # 创建2个子图：回归 + 分类
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            # 创建1个子图：仅回归
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        
        # 回归散点图
        ax1.scatter(all_regression_targets, all_regression_preds, alpha=0.6)
        ax1.plot([min(all_regression_targets), max(all_regression_targets)], 
               [min(all_regression_targets), max(all_regression_targets)], 'r--')
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predictions')
        ax1.set_title(f'{split_name} Regression: R²={reg_r2:.3f}')
        
        # 分类图（如果存在）
        if self.config.training_mode == "mix_train" and all_classification_targets:
            # ROC曲线或分类准确率可视化
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(all_classification_targets, cls_pred_binary)
            im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax2.set_title(f'{split_name} Classification: Acc={cls_acc:.3f}, AUC={cls_auc:.3f}')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('True')
            
            # 添加文本注释
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax2.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
        
        self.writer.add_figure(f'{split_name}/Predictions', fig, epoch)
        plt.close(fig)
        
        # 计算Pearson相关系数（用于返回结果）
        pearson_r2_eval = 0.0
        if all_regression_targets and self.config.use_pearson_loss:
            import scipy.stats
            pearson_r, _ = scipy.stats.pearsonr(all_regression_targets, all_regression_preds)
            pearson_r2_eval = pearson_r ** 2
        
        return {
            'loss': avg_loss,
            'regression_mse': reg_mse,
            'regression_rmse': reg_rmse,
            'regression_r2': reg_r2,
            'pearson_r2': pearson_r2_eval,
            'classification_accuracy': cls_acc,
            'classification_auc': cls_auc
        }
    
    def train(self, train_loader, test_loader, num_epochs):
        """Complete training process - train on training set, evaluate on test set"""
        best_test_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs")
        
        # 创建epoch级别的进度条
        epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", 
                         ncols=150, position=0, leave=True,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        
        for epoch in epoch_pbar:
            # 训练
            train_loss, train_reg_loss, train_cls_loss = self.train_epoch(train_loader, epoch)
            
            # 测试集评估
            test_metrics = self.evaluate(test_loader, epoch, 'Test')
            
            # 学习率调度
            self.scheduler.step(test_metrics['loss'])
            
            # 更新epoch进度条
            if self.config.training_mode == "mix_train":
                epoch_pbar.set_postfix({
                    'TestLoss': f'{test_metrics["loss"]:.4f}',
                    'TestR²': f'{test_metrics["regression_r2"]:.3f}',
                    'TestAcc': f'{test_metrics["classification_accuracy"]:.3f}',
                    'Patience': f'{patience_counter}/{self.config.patience}'
                })
            else:
                epoch_pbar.set_postfix({
                    'TestLoss': f'{test_metrics["loss"]:.4f}',
                    'TestR²': f'{test_metrics["regression_r2"]:.3f}',
                    'Patience': f'{patience_counter}/{self.config.patience}'
                })
            
            # 每个epoch结束后打印详细信息
            if self.config.training_mode == "mix_train":
                tqdm.write(f"\nEpoch {epoch+1:3d}: "
                          f"Train Loss={train_loss:.4f} (Reg={train_reg_loss:.4f}, Cls={train_cls_loss:.4f}) | "
                          f"Test Loss={test_metrics['loss']:.4f} R²={test_metrics['regression_r2']:.3f} Acc={test_metrics['classification_accuracy']:.3f}")
            else:
                tqdm.write(f"\nEpoch {epoch+1:3d}: "
                          f"Train Loss={train_loss:.4f} | "
                          f"Test Loss={test_metrics['loss']:.4f} R²={test_metrics['regression_r2']:.3f}")
            
            # 早停
            if test_metrics['loss'] < best_test_loss:
                best_test_loss = test_metrics['loss']
                patience_counter = 0
                # 保存最佳模型到checkpoints目录
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_SurfSol_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'test_loss': test_metrics['loss'],
                    'test_metrics': test_metrics,
                    'config': self.config
                }, checkpoint_path)
                tqdm.write("💾 Best model saved")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    tqdm.write(f"🛑 Early stopping at epoch {epoch+1}")
                    break
        
        # 加载最佳模型进行最终评估
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_SurfSol_model.pth')
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("\n🏆 Training completed, best model loaded")
        
        # 最终测试集评估
        final_test_metrics = self.evaluate(test_loader, epoch, 'Final_Test')
        print(f"\n📊 Final test results:")
        print(f"   - Loss: {final_test_metrics['loss']:.4f}")
        print(f"   - Regression MSE: {final_test_metrics['regression_mse']:.4f}, RMSE: {final_test_metrics['regression_rmse']:.4f}, R²: {final_test_metrics['regression_r2']:.4f}")
        if self.config.training_mode == "mix_train":
            print(f"   - Classification Acc: {final_test_metrics['classification_accuracy']:.4f}, AUC: {final_test_metrics['classification_auc']:.4f}")
        
        # 关闭TensorBoard writer
        self.writer.close()
        
        return final_test_metrics


def main():
    """Main training function"""
    # ================================================================
    # Model hyperparameter configuration - freely adjustable
    # ================================================================
    
    # Basic training parameters
    learning_rate = 1e-4
    weight_decay = 1e-4
    patience = 1000
    batch_size = 10
    num_epochs = 1000
    
    # Loss function configuration
    loss_function = "mse"  # "mse", "smoothl1", "huber", "mae"
    # - "mse": Mean Squared Error (L2 loss)
    # - "smoothl1": Smooth L1 Loss (less sensitive to outliers)
    # - "huber": Huber Loss (robust to outliers)
    # - "mae": Mean Absolute Error (L1 loss)
    
    # Pearson correlation loss configuration
    use_pearson_loss = False  # Whether to add Pearson correlation loss
    pearson_weight = 1     # λ: Weight for Pearson correlation loss (loss = primary_loss - λ * pearson_r²)
    
    # NEW: Training mode configuration
    training_mode = "regression"  # "regression", "classification", or "regression+classification"
    # - "regression": Only regression task (original mode)
    # - "classification": Only binary classification task (solubility >= threshold vs < threshold)
    # - "regression+classification": Both tasks with mixed loss (recommended for imbalanced data)
    
    # Mixed training parameters (used when training_mode includes classification)
    use_classification_head = True  # Whether to use classification head
    classification_weight = 1.0  # α: Classification loss weight
    regression_weight = 1.0  # β: Regression loss weight (only used in "regression+classification" mode)
    classification_threshold = 1.0  # Threshold for binary classification (1.0 means solubility == 1.0 vs < 1.0)
    
    # Resampling configuration (independent of training_mode)
    use_resampling = True  # Whether to apply resampling strategy
    oversample_range = (0.3, 0.8)  # Range for oversampling 
    oversample_factor = 1  # Oversampling multiplier
    undersample_target = 1.0  # Target value for undersampling
    undersample_factor = 1  # Undersampling ratio
    
    # Modal switches - control which modalities to use
    use_surface = True      # Whether to use surface information
    use_esm = True          # Whether to use ESM embeddings
    use_structure = True    # Whether to use protein 3D structure
    
    # Fusion method configuration
    fusion_type = "attention"  # "concat" or "attention"
    attention_heads = 4  # Number of attention heads (only for attention fusion)
    
    # Surface network parameters
    surface_emb_dim = 8
    surface_edge_dim = 3
    num_conv_layers = 3
    
    # ESM feature parameters
    esm_dim = 640           # ESM2-150M feature dimension
    esm_emb_dim = 4
    
    # 3D structure network parameters
    structure_hidden_channels = 150
    structure_num_filters = 150
    structure_num_interactions = 2
    structure_num_gaussians = 300
    structure_cutoff = 15.0
    structure_max_num_neighbors = 150
    structure_emb_dim = 8
    
    # Fusion layer parameters
    hidden_dim = 16
    dropout = 0
    
    # Data paths
    train_csv_path = 'raw_data/sufsol_train2.csv'
    test_csv_path = 'raw_data/sufsol_test2.csv'
    surface_path = 'raw_data/surfgraph/intra_surf1'
    esm_features_path = 'raw_data/esm_features/esm_features/esm_features2.pkl'
    pdb_path = 'raw_data/pdb'
    
    # Cache settings
    cache_dir = 'preprocessed_data'  # Directory to store preprocessed data cache
    
    # Output paths configuration
    tensorboard_log_dir = "./log/SurfSol2"  # TensorBoard log directory
    checkpoint_dir = "checkpoints_ablation"  # Model checkpoint directory
    
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
    
    # Mixed training configuration
    config.training_mode = training_mode
    config.use_classification_head = use_classification_head
    config.classification_weight = classification_weight
    config.regression_weight = regression_weight
    config.classification_threshold = classification_threshold
    
    # Resampling configuration  
    config.use_resampling = use_resampling
    config.oversample_range = oversample_range
    config.oversample_factor = oversample_factor
    config.undersample_target = undersample_target
    config.undersample_factor = undersample_factor
    
    # Modal switches
    config.use_surface = use_surface
    config.use_esm = use_esm
    config.use_structure = use_structure
    
    # Fusion method configuration
    config.fusion_type = fusion_type
    config.attention_heads = attention_heads
    
    # Surface network parameters
    config.surface_emb_dim = surface_emb_dim
    config.surface_edge_dim = surface_edge_dim
    config.num_conv_layers = num_conv_layers
    
    # ESM parameters
    config.esm_dim = esm_dim
    config.esm_emb_dim = esm_emb_dim
    
    # 3D structure parameters
    config.structure_hidden_channels = structure_hidden_channels
    config.structure_num_filters = structure_num_filters
    config.structure_num_interactions = structure_num_interactions
    config.structure_num_gaussians = structure_num_gaussians
    config.structure_cutoff = structure_cutoff
    config.structure_max_num_neighbors = structure_max_num_neighbors
    config.structure_emb_dim = structure_emb_dim
    
    # Fusion layer parameters
    config.hidden_dim = hidden_dim
    config.dropout = dropout
    
    # Print logo
    print_logo()
    
    print("🚀 Starting trimodal SurfSol training")
    # print("=" * 60)
    # print(f"Training mode: {training_mode.upper()}")
    # if training_mode == "mix_train":
    #     print(f"  - Mixed loss: α={classification_weight} (classification) + β={regression_weight} (regression)")
    #     print(f"  - Resampling: {use_resampling} (oversample {oversample_range}, undersample {undersample_target})")
    print(f"Modal configuration: Surface({use_surface}) + ESM({use_esm}) + 3D Structure({use_structure})")
    # print(f"Fusion method: {fusion_type.upper()}" + (f" (heads: {attention_heads})" if fusion_type == "attention" else ""))
    # print(f"Loss function: {loss_function.upper()}")
    # print(f"Data split method: {split_method.upper()}")
    # print(f"Cache directory: {cache_dir}")
    # print(f"TensorBoard logs: {tensorboard_log_dir}")
    # print(f"Checkpoints: {checkpoint_dir}")
    # print("=" * 60)
    
    # Load data with config for resampling
    print("📊 Loading training and test data...")
    train_dataset, test_dataset = load_train_test_data(
        train_csv_path, test_csv_path, surface_path, esm_features_path, pdb_path, cache_dir, config
    )
    
    print(f"📈 Dataset sizes:")
    print(f"   Training set: {len(train_dataset)}")
    print(f"   Test set: {len(test_dataset)}")
    
    # Print solubility distribution statistics
    def print_dataset_stats(name, dataset):
        labels = [item['solubility'] for item in dataset.valid_data]
        labels = np.array(labels, dtype=np.float32)
        print(f"{name}: mean={labels.mean():.3f}, std={labels.std():.3f}")
    
    print_dataset_stats("Train", train_dataset)
    print_dataset_stats("Test", test_dataset)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, collate_fn=surf_collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, collate_fn=surf_collate_fn, num_workers=2)
    
    # Trainer
    trainer = SurfSolTrainer(config, tensorboard_log_dir, checkpoint_dir, loss_function)
    
    print(f"\n🏗️ Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    # 开始训练
    print("🏋️ Starting training...")
    
    final_metrics = trainer.train(train_loader, test_loader, num_epochs)
    
    print("\n✅ Training completed!")
    print(f"📁 Model saved to: checkpoints/best_SurfSol_model.pth")
    print(f"📊 TensorBoard logs: {trainer.log_dir}")
    
    return final_metrics


if __name__ == "__main__":
    main()
