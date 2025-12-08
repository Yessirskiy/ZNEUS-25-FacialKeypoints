import os
import pandas as pd
from PIL import Image
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from tqdm import tqdm
import wandb
from dotenv import load_dotenv

load_dotenv()

from . import MODELS
from .utils import (
    compute_metrics,
    get_wandb_run_name,
    plot_training_history,
    save_checkpoint,
    EarlyStopping,
    visualize_batch_predictions,
)

# CONFIG
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "YOUR_WANDB_API_KEY")
WANDB_PROJECT = "facial-keypoints"


class FacialKeypointDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        training_dir: str,
        augmented_dir: str,
        img_size: int = 96,
    ) -> None:
        self.df = pd.read_csv(dataset_path)
        self.training_dir = training_dir
        self.augmented_dir = augmented_dir

        self.transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ]
        )

        self.kcols = [c for c in self.df.columns if c != "ImageId"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if "aug" in row.ImageId:
            img_path = os.path.join(self.augmented_dir, row["ImageId"] + ".png")
        else:
            img_path = os.path.join(self.training_dir, row["ImageId"] + ".png")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        keypoints = row[self.kcols].values.astype("float32")
        keypoints = torch.tensor(keypoints)

        return img, keypoints


def get_optimizer(optimizer_name: str, model_params, lr: float):
    optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
    }

    name = optimizer_name.lower()
    if name not in optimizers:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. Choose from {list(optimizers.keys())}"
        )

    opt_class = optimizers[name]
    weight_decay = 1e-4

    if name == "sgd":
        return opt_class(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adamw":
        return opt_class(model_params, lr=lr, weight_decay=weight_decay)
    return opt_class(model_params, lr=lr, weight_decay=weight_decay)


def get_loss_function(loss_name: str):
    losses = {
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "smoothl1": nn.SmoothL1Loss,
    }
    if loss_name.lower() not in losses:
        raise ValueError(
            f"Unknown loss function: {loss_name}. Choose from {list(losses.keys())}"
        )
    return losses[loss_name.lower()]()


def train(
    model_name: str,
    dataset_path: str,
    training_dir: str,
    augmented_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    optimizer_name: str = "adam",
    loss_fn_name: str = "mse",
    val_split: float = 0.2,
    early_stop_patience: Optional[int] = 10,
    use_wandb: bool = False,
    plot_results: bool = True,
    auto_visualize: bool = True,
):
    """
    Train a facial keypoint detection model.

    Args:
        model_name: Name of the model architecture to use
        dataset_path: Path to the CSV file with training data
        training_dir: Directory containing original training images
        augmented_dir: Directory containing augmented training images
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        optimizer_name: Optimizer to use ('adam', 'sgd', 'adamw')
        loss_fn_name: Loss function to use ('mse', 'mae', 'smoothl1')
        val_split: Fraction of data to use for validation
        early_stop_patience: Patience for early stopping (None to disable)
        plot_results: Whether to generate training plots
        auto_visualize: Whether to automatically show prediction visualizations after training
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("=" * 70)
    print("FACIAL KEYPOINT DETECTION - TRAINING")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Dataset: {dataset_path}")
    print(f"Hyperparameters:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Learning Rate: {lr}")
    print(f"   - Optimizer: {optimizer_name}")
    print(f"   - Loss Function: {loss_fn_name}")
    print(f"   - Validation Split: {val_split}")
    print(
        f"   - Early Stopping: {'Enabled (patience=' + str(early_stop_patience) + ')' if early_stop_patience else 'Disabled'}"
    )
    print("=" * 70)

    run_name = get_wandb_run_name(model_name, epochs, lr, optimizer_name, loss_fn_name)

    if use_wandb and WANDB_API_KEY != "YOUR_WANDB_API_KEY":
        wandb.login(key=WANDB_API_KEY)

        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config={
                "model": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "optimizer": optimizer_name,
                "loss_function": loss_fn_name,
                "val_split": val_split,
                "early_stop_patience": early_stop_patience,
            },
        )

    save_model_path = f"models/{run_name}_best.pth"
    os.makedirs("models", exist_ok=True)

    # Load and Split Dataset
    full_dataset = FacialKeypointDataset(dataset_path, training_dir, augmented_dir)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"Dataset split: {train_size} training, {val_size} validation samples")
    print("=" * 70)

    # Model Setup
    model = MODELS[model_name]()
    model.to(device)

    optimizer = get_optimizer(optimizer_name, model.parameters(), lr)
    loss_fn = get_loss_function(loss_fn_name)

    # Early Stopping Setup
    early_stopping = (
        EarlyStopping(patience=early_stop_patience, verbose=True)
        if early_stop_patience
        else None
    )

    # Training Loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_metrics_history = {"mae": [], "rmse": []}
    val_metrics_history = {"mae": [], "rmse": []}

    hyperparameters = {
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "optimizer": optimizer_name,
        "loss_fn": loss_fn_name,
    }

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_preds_all = []
        train_targets_all = []

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", disable=False
        )
        for imgs, keypoints in train_pbar:
            imgs, keypoints = imgs.to(device), keypoints.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, keypoints)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds_all.append(preds.detach().cpu())
            train_targets_all.append(keypoints.detach().cpu())

            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Compute training metrics
        train_preds_all = torch.cat(train_preds_all)
        train_targets_all = torch.cat(train_targets_all)
        train_metrics = compute_metrics(train_preds_all, train_targets_all)
        train_metrics_history["mae"].append(train_metrics["mae"])
        train_metrics_history["rmse"].append(train_metrics["rmse"])

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds_all = []
        val_targets_all = []

        with torch.no_grad():
            val_pbar = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", disable=False
            )
            for imgs, keypoints in val_pbar:
                imgs, keypoints = imgs.to(device), keypoints.to(device)

                preds = model(imgs)
                loss = loss_fn(preds, keypoints)

                val_loss += loss.item()
                val_preds_all.append(preds.cpu())
                val_targets_all.append(keypoints.cpu())

                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Compute validation metrics
        val_preds_all = torch.cat(val_preds_all)
        val_targets_all = torch.cat(val_targets_all)
        val_metrics = compute_metrics(val_preds_all, val_targets_all)
        val_metrics_history["mae"].append(val_metrics["mae"])
        val_metrics_history["rmse"].append(val_metrics["rmse"])

        # Logging
        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(
            f"   Train MAE: {train_metrics['mae']:.4f} | Val MAE: {val_metrics['mae']:.4f}"
        )
        print(
            f"   Train RMSE: {train_metrics['rmse']:.4f} | Val RMSE: {val_metrics['rmse']:.4f}"
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_mae": train_metrics["mae"],
                    "val_mae": val_metrics["mae"],
                    "train_rmse": train_metrics["rmse"],
                    "val_rmse": val_metrics["rmse"],
                    "val_nme": val_metrics["nme"],
                    "val_pck": val_metrics["pck"],
                    "learning_rate": lr,
                }
            )

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_val_loss,
                train_losses,
                val_losses,
                save_model_path,
                model_name,
                hyperparameters,
            )
            print(f"New best validation loss: {best_val_loss:.4f}")

        # Early Stopping Check
        if early_stopping and early_stopping(avg_val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

        print("-" * 70)

    # Final Results
    print("=" * 70)
    print("TRAINING COMPLETE")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_model_path}")
    print("=" * 70)

    # Plot Results
    if plot_results:
        plot_path = save_model_path.replace(".pth", "_history.png")
        plot_training_history(
            train_losses,
            val_losses,
            train_metrics_history,
            val_metrics_history,
            save_path=plot_path,
        )

    # Visualize Predictions
    if auto_visualize:
        print("\n" + "=" * 70)
        print("GENERATING PREDICTION VISUALIZATIONS")
        print("=" * 70)

        model.eval()
        with torch.no_grad():
            # Get a batch from validation set
            val_iter = iter(val_loader)
            sample_imgs, sample_targets = next(val_iter)
            sample_imgs = sample_imgs.to(device)

            # Generate predictions
            sample_preds = model(sample_imgs)

            # Create visualization
            viz_path = save_model_path.replace(".pth", "_predictions.png")
            visualize_batch_predictions(
                images=sample_imgs,
                predictions=sample_preds,
                ground_truth=sample_targets,
                num_samples=min(8, len(sample_imgs)),
                save_path=viz_path,
                denormalize=False,  # Images are already in [0, 1] range
            )
            print(f"Predictions visualization saved to: {viz_path}")
            print("=" * 70)

    if use_wandb:
        wandb.finish()

    return {
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_epoch": len(train_losses),
    }
