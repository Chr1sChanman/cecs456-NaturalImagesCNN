import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "tests" / "test_training_loop.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_project import TrainConfig, NaturalImagesCnn, train_one_epoch, get_dataloaders
from torch.utils.data import TensorDataset, DataLoader
import types
import torch

from final_project import (
    TrainConfig,
    NaturalImagesCnn,
    get_dataloaders,
    train_one_epoch,
    evaluate,
)

# Verifies that the function train_one_epoch() updates model weights
def test_train_one_epoch_changes_weights():
    cfg = TrainConfig()
    train_loader, _ = get_dataloaders(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NaturalImagesCnn(cfg.num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Snapshot initial weights
    initial_params = [p.detach().cpu().clone() for p in model.parameters()]

    # Running one training epoch
    train_one_epoch(model, train_loader, optimizer,
                    device=device, epoch=1, log_interval=999999)

    # Checking for any param changes
    changed = False
    for p0, p1 in zip(initial_params, model.parameters()):
        if not torch.allclose(p0, p1.detach().cpu()):
            changed = True
            break

    # Test check if at least one param changed
    assert changed, "Model parameters did not change after one training epoch"

# Verifies the evaluate() function correctly computes accuracy
def test_evaluate_accuracy_computation():
    cfg = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NaturalImagesCnn(cfg.num_classes).to(device)

    # Creating dummy data where all labels are 0
    x = torch.randn(10, 3, cfg.img_size, cfg.img_size)
    y = torch.zeros(10, dtype=torch.long)

    # Monkeypatch model to always predict class 0 w/high confidence
    def forward_stub(self, inp):
        B = inp.size(0)
        out = torch.zeros(B, cfg.num_classes, device=inp.device)
        out[:, 0] = 10.0
        return out

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=5)

    model.forward = types.MethodType(forward_stub, model)
    loss, acc = evaluate(model, loader, device=device)

    # Test check to see that w/perfect predictions, accuracy must be 1.0
    assert acc == 1.0
