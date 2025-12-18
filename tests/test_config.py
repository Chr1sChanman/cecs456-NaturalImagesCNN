import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "tests" / "test_config.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_project import TrainConfig, set_seed
import torch, numpy as np, random

# Verifies that set_seed() produces deterministics values across runs
def test_deterministic_seed():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    # Generate random values from 3 different libraries
    a1 = torch.randn(3, 3)
    b1 = np.random.rand(3)
    c1 = random.random()

    # Reset seed and generate again for comparison
    set_seed(cfg.seed)
    a2 = torch.randn(3, 3)
    b2 = np.random.rand(3)
    c2 = random.random()

    # Test checks to see if produced results are identical
    assert torch.allclose(a1, a2)
    assert (b1 == b2).all()
    assert c1 == c2