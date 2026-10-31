"""DeepGaze IIE inference: centre-bias prior and fixation density per image."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from PIL import Image

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def load_centerbias(path: Path | None, height: int, width: int) -> npt.NDArray[np.float32]:
    """Resize the MIT1003 centre-bias template to the image and normalise it to a log-density.

    Falls back to a uniform prior when the template file is missing.
    """
    if path is not None and path.exists():
        centerbias = np.load(path).astype(np.float32)
        centerbias = np.array(Image.fromarray(centerbias, mode="F").resize((width, height), Image.NEAREST))
    else:
        logger.warning("centerbias_missing path=%s fallback=uniform", path)
        centerbias = np.zeros((height, width), np.float32)
    peak = centerbias.max()
    # Subtract logsumexp so exp(centerbias) sums to 1.
    return centerbias - (peak + np.log(np.exp(centerbias - peak).sum()))


class DeepGazeRunner:
    """Run DeepGaze IIE, loading the pretrained model once on first use."""

    def __init__(self, centerbias_path: Path | None, device: str) -> None:
        self._centerbias_path = centerbias_path
        self._device = device
        self._model: torch.nn.Module | None = None

    def predict_density(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
        """Return the fixation density for an `H x W x 3` RGB image; it sums to 1."""
        import torch

        height, width = image.shape[:2]
        centerbias = load_centerbias(self._centerbias_path, height, width)
        with torch.no_grad():
            image_tensor = torch.tensor(image.transpose(2, 0, 1)[None].astype(np.float32)).to(self._device)
            centerbias_tensor = torch.tensor(centerbias[None]).to(self._device)
            log_density = self._get_model()(image_tensor, centerbias_tensor)[0, 0].cpu().numpy().astype(np.float64)
        density = np.exp(log_density)
        return density / density.sum()

    def predict_density_from_file(self, image_path: Path) -> npt.NDArray[np.float64]:
        """Return the fixation density for the image stored at `image_path`."""
        return self.predict_density(np.array(Image.open(image_path).convert("RGB")))

    def _get_model(self) -> "torch.nn.Module":
        """Return the cached model, loading the pretrained weights on the first call."""
        if self._model is None:
            import deepgaze_pytorch

            logger.info("model_loading name=DeepGazeIIE device=%s", self._device)
            self._model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(self._device).eval()
        return self._model


def pick_device() -> str:
    """Return `cuda` when a GPU is available, otherwise `cpu`."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
