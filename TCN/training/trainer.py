import logging
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
import numpy as np
from typing import Dict, Tuple, Optional

from TCN.models.tcn_model import AutoregressiveTCNModel
from TCN.models.baseline_model import GradientBoostingBaseline
from TCN.datasets.base_dataset import AutoregressiveResidualsDataset
from TCN.datasets.cached_dataset import CachedScheduledSamplingDataset
from TCN.datasets.data_utils import prepare_data_for_training
from .cached_sampling import CachedScheduledSampling
from .metrics import evaluate_autoregressive, evaluate_teacher_forcing, compare_evaluation_methods
from TCN.utils.model_utils import save_models

logger = logging.getLogger(__name__)


class CachedScheduledSamplingTCNTrainer:
    """
    Trainer that incorporates cached scheduled sampling for performance and compatibility.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
        
        self.sampling_strategy = config.get('sampling_strategy', 'linear')
        self.noise_factor = config.get('noise_factor', 0.15)
        self.cache_update_frequency = config.get('cache_update_frequency', 5)
        self.max_cache_size = config.get('max_cache_size', 100000)
        
        try:
            import torch._inductor.config as inductor_config
            if hasattr(inductor_config, 'triton'):
                inductor_config.triton.cudagraphs = False
            if hasattr(inductor_config, 'fx_graph_cache'):
                inductor_config.fx_graph_cache = False
            if hasattr(inductor_config, 'cuda_graph_pooling'):
                inductor_config.cuda_graph_pooling = False
            
            import os
            os.environ['TORCH_COMPILE_DISABLE_CUDAGRAPHS'] = '1'
            
            logger.info("CUDA graphs disabled for inductor")
        except Exception as e:
            logger.warning(f"Could not configure inductor settings: {e}")
        
        self.prediction_cache = CachedScheduledSampling(
            cache_update_frequency=self.cache_update_frequency,
            max_cache_size=self.max_cache_size
        )
        
        self._set_seed(config.get('seed', 42))
    
    def _set_seed(self, seed: int = 42):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed}")
    
    def train_model(self) -> Dict[str, float]:
        """Train the autoregressive TCN model with cached scheduled sampling"""
        logger.info("Starting cached scheduled sampling autoregressive TCN training...")
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Prepare data
        data = prepare_data_for_training(self.config)
        
        # Train baseline model
        gb_model = self._train_baseline(data)
        
        # Get residuals
        train_residuals, y_train_pred_gb = gb_model.get_residuals(
            data['X_train_static'], data['y_train']
        )
        val_residuals, y_val_pred_gb = gb_model.get_residuals(
            data['X_val_static'], data['y_val']
        )
        test_residuals, y_test_pred_gb = gb_model.get_residuals(
            data['X_test_static'], data['y_test']
        )
        
        # Evaluate baseline
        baseline_metrics = self._evaluate_baseline(gb_model, data)
        
        # Create and train TCN model
        tcn_model = self._create_tcn_model(data)
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = self._create_datasets(
            data, train_residuals, val_residuals, test_residuals
        )
        
        # Train TCN
        best_model_state = self._train_tcn(
            tcn_model, gb_model, train_dataset, val_dataset, data
        )
        
        # Load best model
        tcn_model.load_state_dict(best_model_state)
        
        # Final evaluation
        final_metrics = self._final_evaluation(
            tcn_model, gb_model, val_dataset, test_dataset, data
        )
        
        # Save models
        save_models(gb_model.model, tcn_model, self.config)
        
        # Cleanup
        self._cleanup()
        
        return final_metrics
    
    def _train_baseline(self, data: Dict) -> GradientBoostingBaseline:
        """Train GradientBoosting baseline model"""
        logger.info("Training GradientBoosting baseline...")
        
        gb_model = GradientBoostingBaseline(self.config)
        gb_model.fit(data['X_train_static'], data['y_train'])
        
        return gb_model
    
    def _evaluate_baseline(self, gb_model: GradientBoostingBaseline, data: Dict) -> Dict[str, float]:
        """Evaluate baseline model performance on training, validation, and test datasets"""
        logger.info("Evaluating GradientBoosting baseline on train, validation, and test sets...")
        
        all_metrics = {}
        

        train_closed_mask = None
        if 'closed' in data['train_df'].columns:
            train_closed_mask = data['train_df']['closed'].values > 0
        
        train_baseline_metrics = gb_model.evaluate(
            data['X_train_static'], 
            data['y_train'],
            exclude_closed=True,
            closed_mask=train_closed_mask
        )

        train_metrics = {f"baseline_train_{k}": v for k, v in train_baseline_metrics.items()}
        all_metrics.update(train_metrics)
        
        val_closed_mask = None
        if 'closed' in data['val_df'].columns:
            val_closed_mask = data['val_df']['closed'].values > 0
        
        val_baseline_metrics = gb_model.evaluate(
            data['X_val_static'], 
            data['y_val'],
            exclude_closed=True,
            closed_mask=val_closed_mask
        )
        
        val_metrics = {f"baseline_val_{k}": v for k, v in val_baseline_metrics.items()}
        all_metrics.update(val_metrics)
        
        test_closed_mask = None
        if 'closed' in data['test_df'].columns:
            test_closed_mask = data['test_df']['closed'].values > 0
        
        test_baseline_metrics = gb_model.evaluate(
            data['X_test_static'], 
            data['y_test'],
            exclude_closed=True,
            closed_mask=test_closed_mask
        )
        
        test_metrics = {f"baseline_test_{k}": v for k, v in test_baseline_metrics.items()}
        all_metrics.update(test_metrics)
        
        if wandb.run:
            wandb.log(all_metrics)
        
        logger.info(f"Validation baseline metrics: {val_metrics}")
        logger.info(f"Test baseline metrics: {test_metrics}")
        
        return all_metrics
    
    def _create_tcn_model(self, data: Dict) -> AutoregressiveTCNModel:
        """Create and configure TCN model"""
        # Update config with data-specific parameters
        self.config['static_features_size'] = len(data['static_feature_cols'])
        
        # Create model
        model = AutoregressiveTCNModel.from_config(self.config)
        model = model.to(self.device)
        
        # FIXED: More conservative torch.compile configuration
        if self.config.get('use_torch_compile', True) and hasattr(torch, 'compile'):
            logger.info("Enabling torch.compile with CUDA graph workarounds")
            try:
                # Use a more conservative compile mode to avoid CUDA graph issues
                model = torch.compile(model, mode='default', dynamic=True)
            except Exception as e:
                logger.warning(f"torch.compile failed, continuing without compilation: {e}")
        
        return model
    
    def _create_datasets(self, data: Dict, train_residuals: np.ndarray,
                        val_residuals: np.ndarray, test_residuals: np.ndarray) -> Tuple:
        """Create train, validation, and test datasets"""
        seq_length = self.config['seq_length']
        opening_hour = self.config.get('opening_hour', 11)
        closing_hour = self.config.get('closing_hour', 17)
        total_epochs = self.config['epochs']
        
        train_dataset = CachedScheduledSamplingDataset(
            data['X_train_static'], train_residuals, data['y_train'], 
            seq_length, data['train_df']['timestamp'], 
            opening_hour, closing_hour,
            current_epoch=0, total_epochs=total_epochs,
            sampling_strategy=self.sampling_strategy, 
            noise_factor=self.noise_factor,
            prediction_cache=self.prediction_cache
        )
        
        val_dataset = AutoregressiveResidualsDataset(
            data['X_val_static'], val_residuals, data['y_val'], 
            seq_length, data['val_df']['timestamp'], 
            opening_hour, closing_hour
        )
        
        test_dataset = AutoregressiveResidualsDataset(
            data['X_test_static'], test_residuals, data['y_test'], 
            seq_length, data['test_df']['timestamp'], 
            opening_hour, closing_hour
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def _train_tcn(self, model: AutoregressiveTCNModel, gb_model: GradientBoostingBaseline,
                train_dataset: CachedScheduledSamplingDataset,
                val_dataset: AutoregressiveResidualsDataset,
                data: Dict) -> Dict:
        """Train the TCN model"""
        batch_size = self.config['batch_size']
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,
            T_mult=1,
            eta_min=self.config['learning_rate'] * 0.2
        ) 

        # Mixed precision training
        scaler = None
        if self.config.get('use_mixed_precision', True) and torch.cuda.is_available():
            scaler = torch.amp.GradScaler()
            logger.info("Mixed precision training enabled")
        
        # Training loop
        best_val_ar_mae = float('inf')
        patience_counter = 0
        patience = self.config.get('patience', 40)
        best_model_state = None
        
        for epoch in range(self.config['epochs']):
            train_dataset.update_epoch(epoch)

            if self.prediction_cache.should_update_cache(epoch):
                train_dataset.static_feature_cols = data['static_feature_cols']
                self.prediction_cache.update_cache(
                    model, gb_model.model, train_dataset, self.device
                )

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=torch.cuda.is_available()
            )

            train_loss = self._train_epoch(
                model, train_loader, criterion, optimizer, scaler
            )

            scheduler.step()  

            val_loss, val_metrics = self._validate(
                model, gb_model, val_dataset, batch_size, criterion, scaler, data
            )

            autoregressive_metrics = evaluate_autoregressive(
                model, gb_model.model, val_dataset, batch_size, scaler, 
                data['val_df'], data['val_indices'], self.device, self.prediction_cache
            )

            cache_size = len(self.prediction_cache.prediction_cache)
            
            # FIXED: Consolidated metrics logging with consistent step
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics,
                "learning_rate": scheduler.get_last_lr()[0],
                "teacher_forcing_prob": train_dataset.teacher_forcing_prob,
                "cache_size": cache_size,
                **{f"val_ar_{k.replace('test_', '')}": v for k, v in autoregressive_metrics.items()}
            }
            
            # FIXED: Use consistent step numbering (epoch + 1 to avoid step 0 conflict)
            if wandb.run:
                logger.info("Transfering metrics to wandb")
                wandb.log(epoch_metrics, step=epoch + 1)  
            
            # Logging
            logger.info(
                f'Epoch {epoch+1}/{self.config["epochs"]} - '
                f'Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, '
                f'Val MAE: {val_metrics["val_mae"]:.4f}, Val sMAPE: {val_metrics["val_smape"]:.2f}%, '
                f'Autoregressive MAE: {autoregressive_metrics.get("test_mae", 0):.4f}, '
                f'TF prob: {train_dataset.teacher_forcing_prob:.3f}, LR: {scheduler.get_last_lr()[0]:.6f}'
            )

            # Early stopping logic
            if epoch >= 10:
                if autoregressive_metrics.get('test_mae', 0) < best_val_ar_mae:
                    best_val_ar_mae = autoregressive_metrics.get('test_mae', 0)
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                    logger.info(f"New best model saved at epoch {epoch+1}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

        return best_model_state

    
    def _train_epoch(self, model, train_loader, criterion, optimizer, scaler):
        """Train for one epoch"""
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)

        return train_loss / train_samples

    
    def _validate(self, model, gb_model, val_dataset, batch_size, criterion, scaler, data):
        """Validate the model"""
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True if torch.cuda.is_available() else False
        )
        
        model.eval()
        val_loss = 0.0
        val_samples = 0
        val_residual_preds = []
        val_residual_targets = []
        val_indices_list = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                if scaler is not None:
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)
                
                val_residual_preds.extend(outputs.cpu().numpy().flatten())
                val_residual_targets.extend(targets.cpu().numpy().flatten())
                
                # Track indices
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + len(targets), len(val_dataset))
                val_indices_list.extend(range(batch_start, batch_end))
        
        val_loss /= val_samples
        
        # Calculate validation metrics
        val_metrics = self._calculate_val_metrics(
            val_residual_preds, val_indices_list, val_dataset, 
            gb_model, data['val_df'], data['y_val']
        )
        
        return val_loss, val_metrics
    
    def _calculate_val_metrics(self, residual_preds, indices_list, dataset, 
                              gb_model, val_df, y_val):
        """Calculate validation metrics"""
        from .metrics import evaluate_predictions
        
        residual_preds = np.array(residual_preds)
        
        # Get corresponding validation data indices
        dataset_indices = [dataset.valid_indices[i] for i in indices_list]
        
        # Get baseline predictions
        val_static_features = val_df.iloc[dataset_indices][
            [col for col in val_df.columns if col not in ['wait_time', 'timestamp']]
        ].values
        val_gb_preds = gb_model.predict(val_static_features)
        val_actual_wait_times = y_val[dataset_indices]
        
        # Combined predictions
        val_combined_preds = val_gb_preds + residual_preds
        
        # Create evaluation dataframe
        eval_df = val_df.iloc[dataset_indices].reset_index(drop=True)
        eval_df['combined_pred'] = val_combined_preds
        eval_df['actual_wait_time'] = val_actual_wait_times
        
        # Evaluate
        metrics = evaluate_predictions(eval_df, 'combined_pred', 'actual_wait_time')
        
        return {f"val_{k}": v for k, v in metrics.items()}
    
    def _log_epoch_metrics(self, epoch, train_loss, val_loss, val_metrics, 
                          teacher_forcing_prob, learning_rate):
        """Log metrics for the epoch"""
        cache_size = len(self.prediction_cache.prediction_cache)
        
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
            "learning_rate": learning_rate,
            "teacher_forcing_prob": teacher_forcing_prob,
            "cache_size": cache_size
        }
        
        if wandb.run:
            wandb.log(metrics)
        
        logger.info(
            f'Epoch {epoch+1}/{self.config["epochs"]} - '
            f'Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, '
            f'Val MAE: {val_metrics["val_mae"]:.4f}, Val sMAPE: {val_metrics["val_smape"]:.2f}%, '
            f'TF prob: {teacher_forcing_prob:.3f}, LR: {learning_rate:.6f}'
        )
    
    def _final_evaluation(self, model, gb_model, val_dataset, test_dataset, data):
        """Perform final dual evaluation"""
        logger.info("Performing dual evaluation (autoregressive + teacher forcing)...")
        
        batch_size = self.config['batch_size']
        scaler = torch.amp.GradScaler() if self.config.get('use_mixed_precision', True) and torch.cuda.is_available() else None
        
        # Autoregressive evaluation
        val_autoregressive_metrics = evaluate_autoregressive(
            model, gb_model.model, val_dataset, batch_size, scaler, 
            data['val_df'], data['val_indices'], self.device, self.prediction_cache
        )
        
        test_autoregressive_metrics = evaluate_autoregressive(
            model, gb_model.model, test_dataset, batch_size, scaler, 
            data['test_df'], data['test_indices'], self.device, self.prediction_cache
        )
        
        # Teacher forcing evaluation
        test_teacher_forcing_metrics = evaluate_teacher_forcing(
            model, gb_model.model, test_dataset, batch_size, scaler,
            data['test_df'], data['test_indices'], self.device
        )
        
        # Performance gap analysis
        performance_gap = compare_evaluation_methods(
            test_autoregressive_metrics, test_teacher_forcing_metrics
        )
        
        # Combine all metrics
        final_metrics = {
            **test_autoregressive_metrics,
            **test_teacher_forcing_metrics,
            **{f"gap_{k}": v for k, v in performance_gap.items()}
        }

        final_step = self.config['epochs'] + 10
        if wandb.run:
            wandb.log(final_metrics, step=final_step)
        
        logger.info(f"Final dual evaluation metrics: {final_metrics}")
        
        return final_metrics
    
    def _cleanup(self):
        """Clean up resources"""
        self.prediction_cache.clear_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()