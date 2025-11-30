import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 12 * 12, 512), nn.ReLU(), nn.Linear(512, 30)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DeeperCNN(nn.Module):
    """
    Enhanced CNN with 5 convolutional layers, batch normalization, 
    dropout, and residual-like skip connections.
    """

    def __init__(self, dropout_rate=0.3):
        super().__init__()

        # Convolutional blocks with batch normalization
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 96 -> 48
            nn.Dropout2d(dropout_rate),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Dropout2d(dropout_rate),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24 -> 12
            nn.Dropout2d(dropout_rate),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12 -> 6
            nn.Dropout2d(dropout_rate),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 6 -> 3
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 30),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class WiderCNN(nn.Module):
    """
    Wider architecture with progressive channel expansion (64-128-256-512)
    and global average pooling for spatial invariance.
    """

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 96x96 -> 48x48
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            # Block 2: 48x48 -> 24x24
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            # Block 3: 24x24 -> 12x12
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            # Block 4: 12x12 -> 6x6
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 30),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class PyramidCNN(nn.Module):
    """
    Multi-scale feature extraction with parallel convolutional branches
    at different kernel sizes (1x1, 3x3, 5x5), then concatenation.
    """

    def __init__(self, dropout_rate=0.3):
        super().__init__()

        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 96 -> 48
        )

        # Pyramid block 1
        self.pyramid1_1x1 = nn.Conv2d(64, 32, kernel_size=1)
        self.pyramid1_3x3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.pyramid1_5x5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=5, padding=2)
        )
        self.pyramid1_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(64, 32, kernel_size=1)
        )

        self.pyramid1_bn = nn.Sequential(
            nn.BatchNorm2d(128),  # 32*4 = 128
            nn.ReLU(),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Dropout2d(dropout_rate),
        )

        # Pyramid block 2
        self.pyramid2_1x1 = nn.Conv2d(128, 64, kernel_size=1)
        self.pyramid2_3x3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.pyramid2_5x5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=5, padding=2)
        )

        self.pyramid2_bn = nn.Sequential(
            nn.BatchNorm2d(192),  # 64*3 = 192
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24 -> 12
            nn.Dropout2d(dropout_rate),
        )

        # Final convolution layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12 -> 6
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 30),
        )

    def forward(self, x):
        x = self.initial(x)

        # Pyramid block 1
        p1_1 = self.pyramid1_1x1(x)
        p1_3 = self.pyramid1_3x3(x)
        p1_5 = self.pyramid1_5x5(x)
        p1_p = self.pyramid1_pool(x)
        x = torch.cat([p1_1, p1_3, p1_5, p1_p], dim=1)
        x = self.pyramid1_bn(x)

        # Pyramid block 2
        p2_1 = self.pyramid2_1x1(x)
        p2_3 = self.pyramid2_3x3(x)
        p2_5 = self.pyramid2_5x5(x)
        x = torch.cat([p2_1, p2_3, p2_5], dim=1)
        x = self.pyramid2_bn(x)

        # Final convolution and FC
        x = self.final_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


MODELS = {
    "BasicCNN": BasicCNN,
    "DeeperCNN": DeeperCNN,
    "WiderCNN": WiderCNN,
    "PyramidCNN": PyramidCNN,
}
