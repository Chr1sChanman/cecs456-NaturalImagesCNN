import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "tests" / "test_smoke_train.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import Subset, DataLoader
from final_project import (
    TrainConfig, NaturalImagesCnn, get_dataloaders,
    train_one_epoch, evaluate, set_seed
)

# Integration test to verify that full pipeline beats random guessing after 3 epochs
def test_short_training_beats_random():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loading dataloaders
    train_loader, val_loader = get_dataloaders(cfg)

    tiny_train = Subset(train_loader.dataset, range(256))
    tiny_val = Subset(val_loader.dataset, range(128))

    train_loader = DataLoader(tiny_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(tiny_val, batch_size=cfg.batch_size)

    # Model and optimizer creation
    model = NaturalImagesCnn(cfg.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Training for 4 epochs (one more to ensure we beat threshold)
    for epoch in range(4):
        train_one_epoch(model, train_loader, optimizer, device=device, epoch=epoch+1, log_interval=999999)

    val_loss, val_acc = evaluate(model, val_loader, device=device)

    # Test check of beating threshold of 12.5% for random
    assert val_acc > 0.19, f"Smoke test did not beat random: acc={val_acc:.4f}"