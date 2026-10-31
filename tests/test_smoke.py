"""Installation check: DeepGaze IIE yields a valid density whose peak lands on an obvious salient object."""

import numpy as np
import pytest

from src.buka_attention.model import DeepGazeRunner
from tests.conftest import REPO_ROOT

CENTERBIAS_PATH = REPO_ROOT / "centerbias_mit1003.npy"


@pytest.mark.slow
@pytest.mark.skipif(not CENTERBIAS_PATH.exists(), reason="run ./setup.sh to download the centre-bias file")
def test_density_is_normalised_and_peaks_on_salient_square() -> None:
    # Grey background with one bright square off-centre, so a centre-only prior would fail.
    image = np.full((768, 1024, 3), 120, dtype=np.uint8)
    image[200:350, 650:800] = [240, 60, 60]

    density = DeepGazeRunner(CENTERBIAS_PATH, "cpu").predict_density(image)

    assert density.shape == image.shape[:2]
    assert density.sum() == pytest.approx(1.0, abs=1e-2)
    peak_y, peak_x = np.unravel_index(density.argmax(), density.shape)
    assert 150 <= peak_y <= 400
    assert 600 <= peak_x <= 850
