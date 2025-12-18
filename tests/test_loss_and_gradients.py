import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "tests" / "test_loss_and_gradients.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_project import TrainConfig, NaturalImagesCnn, get_dataloaders
import torch
import torch.nn as nn

# Verifies optimizer can reduce loss over multiple steps
def test_multi_step_reduces_loss_on_real_batch():
    cfg = TrainConfig()
    model = NaturalImagesCnn(cfg.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Getting one batch
    train_loader, _ = get_dataloaders(cfg)
    images, labels = next(iter(train_loader))

    # Training for 10 steps
    losses = []
    for step in range(10):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Test check of loss decreasing by at least 10%
    assert losses[-1] < losses[0] * 0.9

# Verifying that backprop computes non-zero gradients
def test_gradients_non_zero():
    cfg = TrainConfig()
    model = NaturalImagesCnn(cfg.num_classes)
    criterion = nn.CrossEntropyLoss()

    # Creating random inputs
    x = torch.randn(8, 3, cfg.img_size, cfg.img_size)
    y = torch.randint(0, cfg.num_classes, (8,))

    # Forward pass + backprop
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    # Sum gradient magnitudes across all params
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm().item()

    # Test check that at least some gradients are non-zero
    assert total_norm > 0
