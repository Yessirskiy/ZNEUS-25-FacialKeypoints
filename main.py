import os
import hashlib
import argparse
from train import MODELS, train


def validateDataset(dataset: str) -> bool:
    REQ_HASH = "60e5ef3b5eea20f1e94718a84ddf7614"
    with open(dataset, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash == REQ_HASH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="BasicCNN", help="Model name (default: BasicCNN)"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="train_final.csv",
        help="Path to training CSV (default: train_final.csv)",
    )
    parser.add_argument(
        "-td",
        "--training_dir",
        default="training",
        help="Folder containing original training images (default: training)",
    )
    parser.add_argument(
        "-ag",
        "--augmented_dir",
        default="augmented",
        help="Folder containing augmented training images (default: augmented)",
    )
    args = parser.parse_args()

    if args.model not in MODELS:
        raise ValueError(
            f"Unknown model {args.model}. Available: {list(MODELS.keys())}"
        )
    if not os.path.exists(args.dataset):
        raise ValueError(f"No dataset {args.datset} found.")
    if not validateDataset(args.dataset):
        raise ValueError(f"Dataset {args.datset} does not match original.")
    if not os.path.exists(args.training_dir):
        raise ValueError(f"No directory {args.training_dir} with images found.")
    if not os.path.exists(args.augmented_dir):
        raise ValueError(f"No directory {args.augmented_dir} with images found.")

    train(args.model, args.dataset, args.training_dir, args.augmented_dir)


if __name__ == "__main__":
    main()
