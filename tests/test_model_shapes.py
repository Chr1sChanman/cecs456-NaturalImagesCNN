import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "tests" / "test_model_shapes.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_project import TrainConfig, NaturalImagesCnn
import torch

# Validates model ouputs correct shape and contains no NaN values
def test_model_output_shape():
    cfg = TrainConfig()
    model = NaturalImagesCnn(cfg.num_classes)

    # Create dummy batch
    x = torch.randn(4, 3, cfg.img_size, cfg.img_size)
    y = model(x)

    # Test checks for expected shape and numerical instability
    assert y.shape == (4, cfg.num_classes)
    assert not torch.isnan(y).any()