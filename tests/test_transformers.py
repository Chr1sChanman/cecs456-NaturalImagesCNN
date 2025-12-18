import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "tests" / "test_transformers.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_project import TrainConfig, get_dataloaders
from torch.utils.data import Subset

# Ensures train and validation use seperate images (prevent data leakage)
def test_different_transforms_applied():
    cfg = TrainConfig()
    train_loader, val_loader = get_dataloaders(cfg)
    
    # Both loaders have Subset datasets
    train_subset = train_loader.dataset
    val_subset = val_loader.dataset
    
    assert isinstance(train_subset, Subset)
    assert isinstance(val_subset, Subset)
    
    # Access the underlying ImageFolder dataset
    train_base = train_subset.dataset  # type: ignore
    val_base = val_subset.dataset  # type: ignore
    
    train_transform = train_base.transform  #type: ignore
    val_transform = val_base.transform  #type: ignore
    
    print("\n=== Transform Diagnostic ===")
    print(f"Train transform: {train_transform}")
    print(f"Val transform: {val_transform}")
    print(f"Same object? {train_transform is val_transform}")
    
    # They should NOT be the same object
    assert train_transform is not val_transform, \
        "ERROR: Train and val are using the SAME transform object!"