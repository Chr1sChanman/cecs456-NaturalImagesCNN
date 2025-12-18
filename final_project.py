# PyTorch checking as I'm running a 5070ti and needs sm_120 support
# I had to install a dual boot Ubuntu to utilize my GPU
import torch
print("Torch version:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Arch list:", torch.cuda.get_arch_list())

import os
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensures reproducible results across runs for project criteria
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA params to ensure deterministic algorithms
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Hyperparams and training settings
@dataclass
class TrainConfig:
    # Data settings
    data_dir: str = "./archive/data/natural_images"
    val_split: float = 0.2
    img_size: int = 128

    # Training hyperparams
    batch_size: int = 64
    num_epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Model architecture
    num_classes: int = 8

    # System settings
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Logging and checkpoints
    log_interval: int = 50
    min_val_acc: float = 0.80
    checkpoint_path: str = "best_natural_images_cnn.pth"

class NaturalImagesCnn(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()

        # Feature extraction w/4 convolutional blocks w/progressive channel expansion
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Classification head w/dropout regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_dataloaders(cfg: TrainConfig):

    # ImageNet Normalization (Per-channel mean & std)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    
    # Training transforms, aggressive augmentation to prevent overfitting
    train_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    # Validation transforms, no augmentation for deterministic evaluation
    val_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    # Creation of two seperate ImageFolder instances to avoid data leakage
    full_dataset_train = datasets.ImageFolder(root=cfg.data_dir, transform=train_tfms)
    full_dataset_val = datasets.ImageFolder(root=cfg.data_dir, transform=val_tfms)

    # Split size calculation
    val_len = int(len(full_dataset_train) * cfg.val_split)
    train_len = len(full_dataset_train) - val_len

    # Seed before split for reproducibility
    set_seed(cfg.seed)
    # Getting indices from random_split
    train_ds_temp, val_ds_temp = random_split(full_dataset_train, [train_len, val_len])
    
    # Applying same indices to both dataset objects
    from torch.utils.data import Subset
    train_ds = Subset(full_dataset_train, train_ds_temp.indices)
    val_ds = Subset(full_dataset_val, val_ds_temp.indices)

    # Enabling pinned memory for transter CPU to GPU transfer if CUDA is available
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_mem,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_mem,
    )

    return train_loader, val_loader

def train_one_epoch(model, loader, optimizer, device, epoch, log_interval=50):
    # Enable dropout and BatchNorm training mode
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)

        # Training step (zero gradients, forward pass, backprop, update)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Function to print progress every log_interval batch
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            print(f"Epoch {epoch} Step {batch_idx+1}: Loss = {avg_loss:.4f}")
            running_loss = 0.0

# Disabling gradient computation for inference
@torch.no_grad()
def evaluate(model, loader, device):
    # Disable dropout for inference, use BatchNorm running statistics
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)

        loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)

        # Get predicted class (idx of max logit)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total

from torch.optim.lr_scheduler import ReduceLROnPlateau

cfg = TrainConfig()
set_seed(cfg.seed)

# Model init and move to GPU if avail
model = NaturalImagesCnn(num_classes=cfg.num_classes).to(cfg.device)
train_loader, val_loader = get_dataloaders(cfg)

# AdamW gradient descent w/decoupled weight decay
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=cfg.lr,
                              weight_decay=cfg.weight_decay)

# Adaptive LR scheduler, reduce LR by 50% if val acc doesn't improve for 2 epochs
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

best_acc = 0.0

for epoch in range(1, cfg.num_epochs + 1):
    train_one_epoch(model, train_loader, optimizer, cfg.device, epoch, cfg.log_interval)
    val_loss, val_acc = evaluate(model, val_loader, cfg.device)

    print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}")
    
    # Scheduler implementation
    scheduler.step(val_acc)

    # Checkpoint saves if best model by current epoch
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cfg": cfg.__dict__,
                "val_acc": val_acc,
            },
            "best_natural_images_cnn.pth",
        )
        print(f"New best model saved! Acc = {val_acc:.4f}")

# Load best checkpoint
checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
model.load_state_dict(checkpoint["model_state_dict"])  # Correct key
print(f"Loaded model with validation accuracy: {checkpoint['val_acc']:.4f}")