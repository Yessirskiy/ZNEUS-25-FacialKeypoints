import os
import hashlib
import argparse
from train import MODELS, train


# Replace with dropped train_final.csv hash
def validateDataset(dataset: str) -> bool:
    REQ_HASH = "e729effacc3ed8c912ebb73e33adb3be"
    with open(dataset, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash == REQ_HASH


def main():
    parser = argparse.ArgumentParser(
        description="Train facial keypoint detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        "-m", "--model", 
        default="BasicCNN", 
        choices=list(MODELS.keys()),
        help="Model architecture to use"
    )
    parser.add_argument(
        "-d", "--dataset",
        default="train_final.csv",
        help="Path to training CSV file"
    )
    parser.add_argument(
        "-td", "--training_dir",
        default="training",
        help="Directory containing original training images"
    )
    parser.add_argument(
        "-ad", "--augmented_dir",
        default="augmented",
        help="Directory containing augmented training images"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "-bs", "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "-lr", "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "-opt", "--optimizer",
        choices=["adam", "sgd", "adamw"],
        default="adam",
        help="Optimizer to use"
    )
    parser.add_argument(
        "-loss", "--loss_function",
        choices=["mse", "mae", "smoothl1"],
        default="mse",
        help="Loss function to use"
    )
    parser.add_argument(
        "-vs", "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio (0.0 to 1.0)"
    )
    parser.add_argument(
        "-es", "--early_stop",
        type=int,
        default=10,
        help="Early stopping patience (epochs). Set to 0 to disable."
    )
    
    # Output arguments
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable training plot generation"
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.model not in MODELS:
        raise ValueError(
            f"Unknown model {args.model}. Available: {list(MODELS.keys())}"
        )
    if not os.path.exists(args.dataset):
        raise ValueError(f"No dataset {args.dataset} found.")
    if not validateDataset(args.dataset):
        raise ValueError(f"Dataset {args.dataset} does not match original hash.")
    if not os.path.exists(args.training_dir):
        raise ValueError(f"No directory {args.training_dir} with images found.")
    if not os.path.exists(args.augmented_dir):
        raise ValueError(f"No directory {args.augmented_dir} with images found.")
    
    print("\n" + "=" * 70)
    print("CONFIGURATION VALIDATED")
    print("=" * 70)
    print(f"Dataset: {args.dataset} (hash verified)")
    print(f"Training images: {args.training_dir}")
    print(f"Augmented images: {args.augmented_dir}")
    print(f"Model: {args.model}")
    print("=" * 70 + "\n")
    
    train(
        model_name=args.model,
        dataset_path=args.dataset,
        training_dir=args.training_dir,
        augmented_dir=args.augmented_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        optimizer_name=args.optimizer,
        loss_fn_name=args.loss_function,
        val_split=args.val_split,
        early_stop_patience=args.early_stop if args.early_stop > 0 else None,
        plot_results=not args.no_plot,
    )


if __name__ == "__main__":
    main()

