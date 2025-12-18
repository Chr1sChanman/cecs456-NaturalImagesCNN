import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "tests" / "debug_training.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_project import TrainConfig, NaturalImagesCnn, get_dataloaders, set_seed
import torch
from collections import Counter

cfg = TrainConfig()
set_seed(cfg.seed)

train_loader, val_loader = get_dataloaders(cfg)

# Printing basic dataset stats
print("=== Dataset Info ===")
print(f"Training samples: {len(train_loader)}")
print(f"Validation samples: {len(val_loader)}")
print(f"Number of classes: {cfg.num_classes}")
print(f"Batch size: {cfg.batch_size}")

# Extraing labels from subset datasets
train_labels = [train_loader.dataset.dataset.targets[i] for i in train_loader.dataset.indices]  # type: ignore
val_labels = [val_loader.dataset.dataset.targets[i] for i in val_loader.dataset.indices]  # type: ignore

# Checking for class imbalance
print("\n=== Class Distribution ===")
print("Train:", Counter(train_labels))
print("Val:", Counter(val_labels))

# Calculating majority class frequency
print(f"\nMost common class frequency (train): {max(Counter(train_labels).values()) / len(train_labels):.2%}")
print(f"Most common class frequency (val): {max(Counter(val_labels).values()) / len(val_labels):.2%}")

# Testing forward pass w/real data
print("\n=== Model Test ===")
model = NaturalImagesCnn(cfg.num_classes)
images, labels = next(iter(train_loader))
print(f"Input shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Label range: {labels.min().item()} to {labels.max().item()}")

# Checking model output shape and values
outputs = model(images)
print(f"Output shape: {outputs.shape}")
print(f"Output sample (first 3):\n{outputs[:3]}")

# Checking class prediction diversity
preds = outputs.argmax(dim=1)
print(f"\nPredictions for first batch: {preds.tolist()}")
print(f"Unique predictions: {preds.unique().tolist()}")
print(f"Prediction distribution: {Counter(preds.tolist())}")