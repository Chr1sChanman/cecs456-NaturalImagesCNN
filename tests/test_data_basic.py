import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "tests" / "test_data_basic.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_project import TrainConfig, get_dataloaders

# Validates data loader returns correctly shaped tensors w/valid labels
def test_dataloader_shapes_and_classes():
    cfg = TrainConfig()
    train_loader, val_loader = get_dataloaders(cfg)

    # Test check to ensure datasets are not empty
    assert len(train_loader.dataset) > 0    # type: ignore
    assert len(val_loader.dataset) > 0      # type: ignore

    # First batch
    images, labels = next(iter(train_loader))
    
    # Test checks for 4D shape
    assert images.ndim == 4
    assert images.shape[1] in (1, 3)  # grayscale or RGB
    assert labels.ndim == 1

    # Test checks for valid class indices
    assert labels.min() >= 0
    assert labels.max() < cfg.num_classes

# Verifies normalization process produces reasonable pixel value ranges
def test_image_value_range():
    cfg = TrainConfig()
    train_loader, _ = get_dataloaders(cfg)
    images, _ = next(iter(train_loader))

    # Checks rough range after normalization
    assert images.min() > -5
    assert images.max() < 5
