import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "tests" / "test_tiny_overfit.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torchvision import transforms

from final_project import TrainConfig, NaturalImagesCnn, get_dataloaders, set_seed

def test_can_overfit_tiny_subset():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    # Get the dataset w/o augmentations for overfitting test
    train_loader, val_loader = get_dataloaders(cfg)
    
    # Use validation dataset which has no augmentations
    full_ds = val_loader.dataset

    # take a tiny subset (e.g., 32 samples)
    tiny_indices = list(range(32))
    tiny_ds = Subset(full_ds, tiny_indices)
    tiny_loader = DataLoader(
        tiny_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    model = NaturalImagesCnn(cfg.num_classes)
    criterion = nn.CrossEntropyLoss()
    # Lower learning rate for better convergence on the last few samples
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    for epoch in range(100):  # More epochs to ensure convergence
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for images, targets in tiny_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        avg_loss = running_loss / total
        acc = correct / total

        # optional debug print if you want
        # print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.4f}")

        if acc > 0.99:
            break

    # After up to 100 epochs we should basically memorize 32 images
    assert acc > 0.99, f"Did not overfit tiny set. Final loss={avg_loss:.4f}, acc={acc:.4f}"