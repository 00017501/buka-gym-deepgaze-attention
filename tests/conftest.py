"""Shared fixtures: a small synthetic density and a matching post spec."""

from pathlib import Path

import numpy as np
import pytest

from src.buka_attention.schemas import PostSpec
from src.buka_attention.stats import Density

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def density() -> Density:
    """Return a 40 x 40 density with half the mass in the top-left 10 x 10 block."""
    grid = np.full((40, 40), 0.5 / (40 * 40 - 100))
    grid[:10, :10] = 0.5 / 100
    return grid


@pytest.fixture
def spec() -> PostSpec:
    """Return a post with a logo on the hot block and a CTA elsewhere; no offer AOI."""
    return PostSpec.model_validate(
        {
            "image": "synthetic.png",
            "label": "Synthetic",
            "aois": {
                "logo": {"type": "logo", "boxes": [[0, 0, 10, 10]]},
                "cta": {"type": "cta", "boxes": [[20, 20, 30, 30]]},
            },
        }
    )
