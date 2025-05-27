import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     prefix: str = "") -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        prefix: Prefix for metric names
        
    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate sMAPE
    non_zero_mask = y_true > 0.1
    if np.sum(non_zero_mask) > 0:
        y_true_nz = y_true[non_zero_mask]
        y_pred_nz = y_pred[non_zero_mask]
        numerator = np.abs(y_true_nz - y_pred_nz)
        denominator = np.abs(y_true_nz) + np.abs(y_pred_nz)
        smape = np.mean(numerator / denominator) * 100
    else:
        smape = 0.0
    
    metrics = {
        f"{prefix}mae": mae,
        f"{prefix}rmse": rmse,
        f"{prefix}r2": r2,
        f"{prefix}smape": smape,
        f"{prefix}n_samples": len(y_true)
    }
    
    return metrics


def evaluate_predictions(df: pd.DataFrame, pred_col: str, true_col: str = 'wait_time',
                        exclude_closed: bool = True) -> Dict[str, float]:
    """
    Evaluate predictions in a DataFrame.
    
    Args:
        df: DataFrame with predictions and ground truth
        pred_col: Name of prediction column
        true_col: Name of ground truth column
        exclude_closed: Whether to exclude closed rides
        
    Returns:
        Dictionary of metrics
    """
    eval_df = df.copy()
    
    # Filter out closed rides if requested
    if exclude_closed and 'closed' in eval_df.columns:
        logger.info(f"Excluding {eval_df['closed'].sum()} closed ride data points")
        eval_df = eval_df[eval_df['closed'] == 0]
    
    if len(eval_df) == 0:
        logger.warning("No data points for evaluation!")
        return {
            "mae": float('inf'),
            "rmse": float('inf'),
            "r2": -float('inf'),
            "smape": float('inf'),
            "n_samples": 0
        }
    
    y_true = eval_df[true_col].values
    y_pred = eval_df[pred_col].values
    
    return calculate_metrics(y_true, y_pred)


def evaluate_autoregressive(model, gb_model, dataset, batch_size: int, 
                          scaler: Optional[object], test_df: pd.DataFrame,
                          test_indices: np.ndarray, device: torch.device,
                          prediction_cache: Optional[object] = None) -> Dict[str, float]:
    """
    Evaluate model performance using autoregressive prediction (real-world conditions).
    
    Args:
        model: TCN model
        gb_model: Baseline GradientBoosting model
        dataset: Test dataset
        batch_size: Batch size for evaluation
        scaler: GradScaler for mixed precision (optional)
        test_df: Test DataFrame
        test_indices: Test indices
        device: Torch device
        prediction_cache: CachedScheduledSampling instance (optional)
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Evaluating with autoregressive prediction (model uses own predictions)...")
    
    from TCN.datasets.cached_dataset import CachedScheduledSamplingDataset
    
    model.eval()
    autoregressive_preds = []
    actual_wait_times = []
    test_indices_used = []
    
    # Create a copy of test dataset with high scheduled sampling for autoregressive evaluation
    autoregressive_dataset = CachedScheduledSamplingDataset(
        dataset.X_static, 
        dataset.residuals, 
        dataset.wait_times, 
        dataset.seq_length,
        dataset.timestamps, 
        dataset.opening_hour, 
        dataset.closing_hour,
        current_epoch=99,  # High epoch = mostly scheduled sampling (5% teacher forcing)
        total_epochs=100,
        sampling_strategy="linear",
        noise_factor=0.15,
        prediction_cache=prediction_cache
    )
    
    # Process samples in smaller batches to avoid memory issues
    autoregressive_batch_size = min(32, batch_size)
    
    with torch.no_grad():
        for start_idx in range(0, len(autoregressive_dataset), autoregressive_batch_size):
            end_idx = min(start_idx + autoregressive_batch_size, len(autoregressive_dataset))
            
            batch_preds = []
            batch_actuals = []
            batch_indices = []
            
            for idx in range(start_idx, end_idx):
                try:
                    # Get autoregressive sequence
                    inputs, targets = autoregressive_dataset[idx]
                    
                    # Model prediction
                    inputs = inputs.unsqueeze(0).to(device)
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            residual_pred = model(inputs).cpu().numpy().flatten()[0]
                    else:
                        residual_pred = model(inputs).cpu().numpy().flatten()[0]
                    
                    # Get corresponding dataset index and static features
                    dataset_idx = autoregressive_dataset.valid_indices[idx]
                    static_features = autoregressive_dataset.X_static[dataset_idx]
                    
                    # Get baseline prediction
                    gb_pred = gb_model.predict(static_features.reshape(1, -1))[0]
                    
                    # Combined prediction
                    combined_pred = gb_pred + residual_pred
                    actual_wait_time = autoregressive_dataset.wait_times[dataset_idx]
                    
                    batch_preds.append(combined_pred)
                    batch_actuals.append(actual_wait_time)
                    batch_indices.append(dataset_idx)
                    
                except Exception as e:
                    logger.debug(f"Autoregressive evaluation failed for sample {idx}: {e}")
                    continue
            
            # Accumulate results
            autoregressive_preds.extend(batch_preds)
            actual_wait_times.extend(batch_actuals)
            test_indices_used.extend(batch_indices)
            
            # Log progress
            if len(autoregressive_preds) % 1000 == 0:
                logger.info(f"Processed {len(autoregressive_preds)} autoregressive samples...")
    
    # Convert to arrays
    autoregressive_preds = np.array(autoregressive_preds)
    actual_wait_times = np.array(actual_wait_times)
    
    logger.info(f"Generated {len(autoregressive_preds)} autoregressive predictions")
    
    # Create evaluation dataframe
    eval_df = test_df.iloc[test_indices_used].reset_index(drop=True)
    eval_df['autoregressive_pred'] = autoregressive_preds
    eval_df['actual_wait_time'] = actual_wait_times
    
    # Evaluate
    metrics = evaluate_predictions(eval_df, 'autoregressive_pred', 'actual_wait_time')
    
    # Add prefix to distinguish from teacher forcing
    return {f"test_{k}": v for k, v in metrics.items()}


def evaluate_teacher_forcing(model, gb_model, dataset, batch_size: int,
                           scaler: Optional[object], test_df: pd.DataFrame,
                           test_indices: np.ndarray, device: torch.device) -> Dict[str, float]:
    """
    Evaluate model performance using teacher forcing (ideal conditions).
    
    Args:
        model: TCN model
        gb_model: Baseline GradientBoosting model
        dataset: Test dataset
        batch_size: Batch size for evaluation
        scaler: GradScaler for mixed precision (optional)
        test_df: Test DataFrame
        test_indices: Test indices
        device: Torch device
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Evaluating with teacher forcing (model uses ground truth)...")
    
    test_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )
    
    model.eval()
    test_residual_preds = []
    test_indices_list = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device, non_blocking=True)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            
            test_residual_preds.extend(outputs.cpu().numpy().flatten())
            
            # Track indices
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + len(targets), len(dataset))
            test_indices_list.extend(range(batch_start, batch_end))
    
    # Calculate combined predictions
    test_residual_preds = np.array(test_residual_preds)
    test_dataset_indices = [dataset.valid_indices[i] for i in test_indices_list]
    
    # Get GB baseline predictions for these indices
    test_static_features = test_df.iloc[test_dataset_indices][
        [col for col in test_df.columns if col not in ['wait_time', 'timestamp']]
    ].values
    test_gb_preds = gb_model.predict(test_static_features)
    test_actual_wait_times = test_df.iloc[test_dataset_indices]['wait_time'].values
    
    # Combined predictions = GB baseline + TCN residuals
    test_combined_preds = test_gb_preds + test_residual_preds
    
    # Evaluation dataframe
    eval_df = test_df.iloc[test_dataset_indices].reset_index(drop=True)
    eval_df['teacher_forcing_pred'] = test_combined_preds
    eval_df['actual_wait_time'] = test_actual_wait_times
    
    # Evaluate
    metrics = evaluate_predictions(eval_df, 'teacher_forcing_pred', 'actual_wait_time')
    
    # Add suffix to distinguish from autoregressive
    return {f"test_{k}_tf": v for k, v in metrics.items()}


def compare_evaluation_methods(autoregressive_metrics: Dict[str, float],
                              teacher_forcing_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Compare autoregressive and teacher forcing evaluation results.
    
    Args:
        autoregressive_metrics: Metrics from autoregressive evaluation
        teacher_forcing_metrics: Metrics from teacher forcing evaluation
        
    Returns:
        Dictionary with performance gaps
    """
    performance_gap = {
        "mae_gap": autoregressive_metrics["test_mae"] - teacher_forcing_metrics["test_mae_tf"],
        "rmse_gap": autoregressive_metrics["test_rmse"] - teacher_forcing_metrics["test_rmse_tf"],
        "r2_gap": teacher_forcing_metrics["test_r2_tf"] - autoregressive_metrics["test_r2"],
        "smape_gap": autoregressive_metrics["test_smape"] - teacher_forcing_metrics["test_smape_tf"]
    }
    
    # Calculate percentage gaps
    if teacher_forcing_metrics["test_mae_tf"] > 0:
        performance_gap["mae_gap_pct"] = (performance_gap["mae_gap"] / teacher_forcing_metrics["test_mae_tf"]) * 100
    
    if teacher_forcing_metrics["test_rmse_tf"] > 0:
        performance_gap["rmse_gap_pct"] = (performance_gap["rmse_gap"] / teacher_forcing_metrics["test_rmse_tf"]) * 100
    
    # Log comparison
    logger.info("=== PERFORMANCE COMPARISON ===")
    logger.info(f"Real-world (Autoregressive) MAE: {autoregressive_metrics['test_mae']:.4f}")
    logger.info(f"Ideal conditions (Teacher Forcing) MAE: {teacher_forcing_metrics['test_mae_tf']:.4f}")
    logger.info(f"Performance gap: {performance_gap['mae_gap']:.4f} ({performance_gap.get('mae_gap_pct', 0):.1f}%)")
    
    return performance_gap