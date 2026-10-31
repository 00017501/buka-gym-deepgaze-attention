"""DeepGaze IIE smoke test: run on a synthetic image, confirm a log-density map comes out."""
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom
from scipy.special import logsumexp

import deepgaze_pytorch

HERE = Path(__file__).resolve().parent.parent  # repo root
DEVICE = "cpu"

# synthetic test image: grey background, bright square off-centre
image = np.full((768, 1024, 3), 120, dtype=np.uint8)
image[200:350, 650:800] = [240, 60, 60]

model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE).eval()

centerbias_template = np.load(HERE / "centerbias_mit1003.npy", allow_pickle=False)
centerbias = zoom(
    centerbias_template,
    (image.shape[0] / centerbias_template.shape[0], image.shape[1] / centerbias_template.shape[1]),
    order=0,
    mode="nearest",
)
centerbias -= logsumexp(centerbias)

image_tensor = torch.tensor(np.array([image.transpose(2, 0, 1)])).float().to(DEVICE)
centerbias_tensor = torch.tensor(np.array([centerbias])).float().to(DEVICE)

with torch.no_grad():
    log_density = model(image_tensor, centerbias_tensor)

ld = log_density.cpu().numpy()[0, 0]
prob_sum = float(np.exp(ld).sum())
peak_y, peak_x = np.unravel_index(ld.argmax(), ld.shape)
print("log-density shape:", ld.shape)
print("sum(exp(log-density)) =", round(prob_sum, 4), "(should be ~1.0)")
print("peak at (y, x):", (int(peak_y), int(peak_x)))
assert ld.shape == image.shape[:2] and abs(prob_sum - 1.0) < 1e-2
print("SMOKE TEST PASSED")
