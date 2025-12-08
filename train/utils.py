import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


def _to_kpts(x: torch.Tensor) -> torch.Tensor:
    """
    Force tensor into shape (N, K, 2)
    Works for both (N, 30) and (N, 15, 2)
    """
    if x.dim() == 2:
        return x.view(x.size(0), 15, 2)
    return x


def nme(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = _to_kpts(preds)
    targets = _to_kpts(targets)

    d = math.sqrt(96**2 + 96**2)
    errs = torch.norm(preds - targets, dim=-1)
    per_img = errs.mean(dim=1) / d
    return per_img.mean().item()


def pck(preds: torch.Tensor, targets: torch.Tensor, alpha: float = 0.05) -> float:
    preds = _to_kpts(preds)
    targets = _to_kpts(targets)

    d = math.sqrt(96**2 + 96**2)
    thresh = alpha * d

    errs = torch.norm(preds - targets, dim=-1)
    correct = (errs <= thresh).float()
    return correct.mean().item()


def compute_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute evaluation metrics for keypoint predictions.

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
        "nme": nme(predictions, targets),
        "pck": pck(predictions, targets),
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
    Generate a WandB run name from hyperparameters.
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
    epochs = range(1, len(train_losses) + 1)

    # Create subplots: loss + metrics
    num_metrics = len(train_metrics)
    fig, axes = plt.subplots(1, num_metrics + 1, figsize=(6 * (num_metrics + 1), 5))

    if num_metrics == 0:
        axes = [axes]

    # Plot loss
    axes[0].plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    axes[0].plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot each metric
    for idx, metric_name in enumerate(train_metrics.keys()):
        ax = axes[idx + 1]
        ax.plot(
            epochs,
            train_metrics[metric_name],
            "b-",
            label=f"Training {metric_name.upper()}",
            linewidth=2,
        )
        ax.plot(
            epochs,
            val_metrics[metric_name],
            "r-",
            label=f"Validation {metric_name.upper()}",
            linewidth=2,
        )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(metric_name.upper(), fontsize=12)
        ax.set_title(
            f"Training and Validation {metric_name.upper()}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
    """
    checkpoint = {
        "epoch": epoch,
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "hyperparameters": hyperparameters,
    }
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved to: {save_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict]:
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metadata = {
        "epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "hyperparameters": checkpoint.get("hyperparameters", {}),
    }

    return model, metadata


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """

    def __init__(
        self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as improvement
            verbose: Print messages
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
                    print(
                        f"Early stopping triggered after {self.counter} epochs without improvement"
                    )
                return True

            return False


def visualize_predictions(
    image: np.ndarray,
    predicted_keypoints: np.ndarray,
    ground_truth_keypoints: Optional[np.ndarray] = None,
    title: str = "Facial Keypoint Predictions",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize predicted facial keypoints on an image.

    Args:
        image: Image array (H, W, C) with values in [0, 1] or [0, 255]
        predicted_keypoints: Array of shape (30,) containing predicted keypoints [x1, y1, x2, y2, ...]
        ground_truth_keypoints: Optional array of shape (30,) containing ground truth keypoints
        title: Title for the plot
        save_path: Optional path to save the visualization
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Normalize image to [0, 1] if needed
    if image.max() > 1.0:
        image = image / 255.0

    ax.imshow(image, cmap="gray" if len(image.shape) == 2 else None)

    # Reshape keypoints from (30,) to (15, 2)
    pred_kpts = predicted_keypoints.reshape(-1, 2)

    # Plot predicted keypoints
    ax.scatter(
        pred_kpts[:, 0],
        pred_kpts[:, 1],
        c="red",
        s=100,
        marker="x",
        linewidths=3,
        label="Predicted",
        alpha=0.8,
    )

    # Plot ground truth keypoints if provided
    if ground_truth_keypoints is not None:
        gt_kpts = ground_truth_keypoints.reshape(-1, 2)
        ax.scatter(
            gt_kpts[:, 0],
            gt_kpts[:, 1],
            c="lime",
            s=50,
            marker="o",
            alpha=0.7,
            label="Ground Truth",
        )

        # Draw lines connecting predicted to ground truth
        for i in range(len(pred_kpts)):
            ax.plot(
                [pred_kpts[i, 0], gt_kpts[i, 0]],
                [pred_kpts[i, 1], gt_kpts[i, 1]],
                "yellow",
                alpha=0.3,
                linewidth=1,
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_batch_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor,
    ground_truth: Optional[torch.Tensor] = None,
    num_samples: int = 8,
    save_path: Optional[str] = None,
    denormalize: bool = True,
) -> None:
    """
    Visualize a batch of predictions in a grid.

    Args:
        images: Batch of images (B, C, H, W)
        predictions: Batch of predicted keypoints (B, 30)
        ground_truth: Optional batch of ground truth keypoints (B, 30)
        num_samples: Number of samples to visualize
        save_path: Optional path to save the visualization
        denormalize: Whether to denormalize images (assuming ImageNet normalization)
    """
    num_samples = min(num_samples, len(images))
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    # ImageNet normalization constants
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx in range(num_samples):
        ax = axes[idx]

        # Get image and convert to numpy (C, H, W) -> (H, W, C)
        img = images[idx].cpu().numpy().transpose(1, 2, 0)

        # Denormalize if needed
        if denormalize:
            img = img * std + mean
            img = np.clip(img, 0, 1)

        # Plot image
        ax.imshow(img)

        # Get predicted keypoints
        pred_kpts = predictions[idx].cpu().numpy().reshape(-1, 2)

        # Plot predicted keypoints
        ax.scatter(
            pred_kpts[:, 0],
            pred_kpts[:, 1],
            c="red",
            s=50,
            marker="x",
            linewidths=2,
            label="Predicted",
            alpha=0.8,
        )

        # Plot ground truth if provided
        if ground_truth is not None:
            gt_kpts = ground_truth[idx].cpu().numpy().reshape(-1, 2)
            ax.scatter(
                gt_kpts[:, 0],
                gt_kpts[:, 1],
                c="lime",
                s=30,
                marker="o",
                alpha=0.7,
                label="Ground Truth",
            )

        ax.set_title(f"Sample {idx + 1}", fontsize=10)
        ax.axis("off")

        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    # Hide empty subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Batch visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()
