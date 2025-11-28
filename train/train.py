import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from . import MODELS


class Dataset(Dataset):
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


# TODO: add new parameters: epochs, optimizer, loss fn (one), metrics (multiple), early_stop, wandb
# TODO: add validation of the dataset (all images present, each is 96x96)
# TODO: add training dataset split (training, validation)
# TODO: add test dataset run at the end
# TODO: add pretty prints, preferrably `verbose` parameter
# TODO: add "save model" parameter
# TODO: add graphical representation of loss and metric


def train(model_name: str, dataset_path: str, trainding_dir: str, augmented_dir: str):
    model = MODELS[model_name]()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset = Dataset(dataset_path, trainding_dir, augmented_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(100):
        total_loss = 0
        for imgs, keypoints in loader:
            imgs, keypoints = imgs.to(device), keypoints.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, keypoints)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Avg Loss: {total_loss/len(loader):.4f}")
