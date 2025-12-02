import os
import pandas as pd
from PIL import Image
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import wandb


class TestDataset(Dataset):
    """Dataset for test images without labels."""
    
    def __init__(
        self,
        test_dir: str,
        img_size: int = 96,
    ) -> None:
        # Get all image files in test directory
        self.test_dir = test_dir
        self.image_files = sorted(
            [f for f in os.listdir(test_dir) if f.endswith('.png')],
            key=lambda x: int(x.replace('.png', ''))
        )
        
        self.transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        # Extract image ID from filename (e.g., "1.png" -> 1)
        image_id = int(img_file.replace('.png', ''))
        
        return img, image_id


def visualize_test_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor,
    num_samples: int = 8,
    save_path: Optional[str] = None,
    log_to_wandb: bool = True,
) -> None:
    """
    Visualize test predictions with keypoints overlaid on images.
    
    Args:
        images: Tensor of shape (batch_size, 3, H, W)
        predictions: Tensor of shape (batch_size, 30) with keypoint coordinates
        num_samples: Number of samples to visualize
        save_path: Path to save visualization
        log_to_wandb: Whether to log to wandb
    """
    num_samples = min(num_samples, len(images))
    
    # Reshape predictions to (num_samples, 15, 2) for 15 keypoints with (x, y)
    keypoints = predictions[:num_samples].reshape(num_samples, 15, 2).cpu().numpy()
    
    # Convert images to numpy
    imgs_np = images[:num_samples].permute(0, 2, 3, 1).cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # Display image
        img = imgs_np[idx]
        ax.imshow(img)
        
        # Plot keypoints
        kpts = keypoints[idx]
        ax.scatter(kpts[:, 0], kpts[:, 1], c='red', s=20, marker='x', linewidths=2)
        
        # Optionally connect keypoints for face structure
        # Eyes
        if len(kpts) >= 4:
            # Left eye
            ax.plot([kpts[2, 0], kpts[3, 0]], [kpts[2, 1], kpts[3, 1]], 'yellow', linewidth=1)
            # Right eye  
            ax.plot([kpts[4, 0], kpts[5, 0]], [kpts[4, 1], kpts[5, 1]], 'yellow', linewidth=1)
        
        ax.axis('off')
        ax.set_title(f'Test Image {idx + 1}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Test predictions visualization saved to: {save_path}")
    
    if log_to_wandb and wandb.run is not None:
        wandb.log({"test_predictions": wandb.Image(fig)})
        print("Test predictions logged to WandB")
    
    plt.close(fig)



def generate_submission(
    model,
    test_dir: str,
    lookup_table_path: str,
    output_path: str,
    device: str = "cuda",
    batch_size: int = 32,
    img_size: int = 96,
) -> pd.DataFrame:
    """
    Generate predictions on test set and create submission file.
    
    Args:
        model: Trained model
        test_dir: Directory containing test images
        lookup_table_path: Path to IdLookupTable.csv
        output_path: Path to save submission CSV
        device: Device to run inference on
        batch_size: Batch size for inference
        img_size: Image size
        
    Returns:
        DataFrame with predictions
    """
    print("==" * 35)
    print("GENERATING TEST SET PREDICTIONS")
    print("==" * 35)
    
    # Load lookup table
    lookup = pd.read_csv(lookup_table_path)
    print(f"Loaded lookup table: {len(lookup)} rows")
    
    # Create test dataset and loader
    test_dataset = TestDataset(test_dir, img_size=img_size)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"Test dataset: {len(test_dataset)} images")
    
    # Feature names corresponding to model output (30 values)
    feature_names = [
        'left_eye_center_x', 'left_eye_center_y',
        'right_eye_center_x', 'right_eye_center_y',
        'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
        'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
        'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
        'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
        'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
        'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
        'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
        'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
        'nose_tip_x', 'nose_tip_y',
        'mouth_left_corner_x', 'mouth_left_corner_y',
        'mouth_right_corner_x', 'mouth_right_corner_y',
        'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
        'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y',
    ]
    
    # Run inference
    model.eval()
    predictions_dict = {}  # {image_id: {feature_name: value}}
    all_images = []  # Store images for visualization
    all_predictions = []  # Store predictions for visualization
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Generating predictions")
        for imgs, image_ids in pbar:
            imgs = imgs.to(device)
            preds = model(imgs)  # Shape: (batch_size, 30)
            
            # Store first batch for visualization
            if len(all_images) == 0:
                all_images.append(imgs.cpu())
                all_predictions.append(preds.cpu())
            
            # Store predictions
            for i, image_id in enumerate(image_ids.numpy()):
                image_id = int(image_id)
                predictions_dict[image_id] = {}
                
                for j, feature_name in enumerate(feature_names):
                    predictions_dict[image_id][feature_name] = preds[i, j].cpu().item()
    
    print(f"Generated predictions for {len(predictions_dict)} images")
    
    # Create submission DataFrame
    submission_data = []
    
    for _, row in tqdm(lookup.iterrows(), total=len(lookup), desc="Building submission"):
        row_id = row['RowId']
        image_id = row['ImageId']
        feature_name = row['FeatureName']
        
        # Get prediction for this image and feature
        if image_id in predictions_dict and feature_name in predictions_dict[image_id]:
            location = predictions_dict[image_id][feature_name]
        else:
            # If prediction not available, use 0 (shouldn't happen with full model)
            location = 0.0
            
        submission_data.append({
            'RowId': row_id,
            'Location': location
        })
    
    submission_df = pd.DataFrame(submission_data)
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    print("==" * 35)
    
    # Print sample predictions
    print("\nSample predictions (first 10 rows):")
    print(submission_df.head(10))
    print("==" * 35)
    
    # Visualize test predictions
    if len(all_images) > 0:
        print("\nGenerating test visualization...")
        viz_path = output_path.replace("_submission.csv", "_test_viz.png")
        visualize_test_predictions(
            images=all_images[0],
            predictions=all_predictions[0],
            num_samples=min(8, len(all_images[0])),
            save_path=viz_path,
            log_to_wandb=True
        )
    
    return submission_df
