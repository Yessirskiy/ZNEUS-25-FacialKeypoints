import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute evaluation metrics for keypoint predictions.
    
    Args:
        predictions: Model predictions (batch_size, 30)
        targets: Ground truth keypoints (batch_size, 30)
        
    Returns:
        Dictionary with MSE, MAE, and RMSE metrics
    """
    mse = nn.functional.mse_loss(predictions, targets).item()
    mae = nn.functional.l1_loss(predictions, targets).item()
    rmse = np.sqrt(mse)
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
    }


def get_wandb_run_name(
    model_name: str,
    epochs: int,
    lr: float,
    optimizer: str,
    loss_fn: str,
    custom_name: Optional[str] = None,
) -> str:
    """
    Generate a descriptive WandB run name from hyperparameters.
    
    Args:
        model_name: Name of the model architecture
        epochs: Number of training epochs
        lr: Learning rate
        optimizer: Optimizer name
        loss_fn: Loss function name
        custom_name: Optional custom prefix for the run name
        
    Returns:
        Formatted run name string
    """
    if custom_name:
        return f"{custom_name}_{model_name}_e{epochs}_lr{lr}_{optimizer}_{loss_fn}"
    return f"{model_name}_e{epochs}_lr{lr}_{optimizer}_{loss_fn}"


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    save_path: str = "training_history.png",
) -> None:
    """
    Plot training and validation loss/metrics over epochs.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_metrics: Dictionary of metric name -> list of training values
        val_metrics: Dictionary of metric name -> list of validation values
        save_path: Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create subplots: loss + metrics
    num_metrics = len(train_metrics)
    fig, axes = plt.subplots(1, num_metrics + 1, figsize=(6 * (num_metrics + 1), 5))
    
    if num_metrics == 0:
        axes = [axes]
    
    # Plot loss
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot each metric
    for idx, metric_name in enumerate(train_metrics.keys()):
        ax = axes[idx + 1]
        ax.plot(epochs, train_metrics[metric_name], 'b-', label=f'Training {metric_name.upper()}', linewidth=2)
        ax.plot(epochs, val_metrics[metric_name], 'r-', label=f'Validation {metric_name.upper()}', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name.upper(), fontsize=12)
        ax.set_title(f'Training and Validation {metric_name.upper()}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {save_path}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    train_losses: List[float],
    val_losses: List[float],
    save_path: str,
    model_name: str,
    hyperparameters: Dict,
) -> None:
    """
    Save model checkpoint with metadata.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state
        epoch: Current epoch number
        best_val_loss: Best validation loss achieved
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the checkpoint
        model_name: Name of the model architecture
        hyperparameters: Dictionary of hyperparameters used
    """
    checkpoint = {
        'epoch': epoch,
        'model_name': model_name,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'hyperparameters': hyperparameters,
    }
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved to: {save_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = 'cpu',
) -> Tuple[nn.Module, Dict]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load the model on
        
    Returns:
        Tuple of (loaded model, checkpoint metadata)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    metadata = {
        'epoch': checkpoint.get('epoch'),
        'best_val_loss': checkpoint.get('best_val_loss'),
        'hyperparameters': checkpoint.get('hyperparameters', {}),
    }
    
    return model, metadata


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True
            
            return False
