"""Fit a chosen attempt to the original post size without stretching: crop the bottom, or pad the sides."""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from src.redesign.records import FinalRecord, sha256_of

logger = logging.getLogger(__name__)

# Width of the border strip sampled for the padding colour; wide enough to ignore anti-aliasing.
BORDER_PX = 8


def border_colour(image: Image.Image) -> tuple[int, int, int]:
    """Return the median RGB of the image's outer border, used to pad without a visible seam."""
    pixels = np.asarray(image.convert("RGB"))
    border = np.concatenate(
        [
            pixels[:BORDER_PX].reshape(-1, 3),
            pixels[-BORDER_PX:].reshape(-1, 3),
            pixels[:, :BORDER_PX].reshape(-1, 3),
            pixels[:, -BORDER_PX:].reshape(-1, 3),
        ]
    )
    red, green, blue = (int(value) for value in np.median(border, axis=0))
    return red, green, blue


def fit_and_pad(source: Path, out: Path, width: int, height: int) -> FinalRecord:
    """Scale `source` to fit inside `width x height` (aspect kept), centre it, pad with its border colour."""
    image = Image.open(source).convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    pad_colour = border_colour(image)
    canvas = Image.new("RGB", (width, height), pad_colour)
    offset = ((width - resized.width) // 2, (height - resized.height) // 2)
    canvas.paste(resized, offset)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    record = FinalRecord(
        source_file=str(source),
        source_sha256=sha256_of(source),
        source_size=image.size,
        output_file=str(out),
        output_sha256=sha256_of(out),
        output_size=canvas.size,
        method="fit inside, LANCZOS resize, centred, padded with median border colour",
        scale=scale,
        offset=offset,
        pad_colour=pad_colour,
    )
    logger.info("finalized source=%s out=%s scale=%.4f offset=%s", source, out, scale, offset)
    return record


def crop_bottom(source: Path, out: Path, width: int, height: int) -> FinalRecord:
    """Cut rows off the bottom of `source` to reach the `width:height` shape, then resize to `width x height`.

    Only the bottom is cut, so the top-aligned layout (logo, headline) keeps its position; see D18.
    """
    image = Image.open(source).convert("RGB")
    kept_height = round(image.width * height / width)
    if kept_height > image.height:
        msg = f"{source} is wider than {width}:{height}; bottom cropping cannot reach that shape"
        raise ValueError(msg)
    crop_box = (0, 0, image.width, kept_height)
    canvas = image.crop(crop_box).resize((width, height), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    record = FinalRecord(
        source_file=str(source),
        source_sha256=sha256_of(source),
        source_size=image.size,
        output_file=str(out),
        output_sha256=sha256_of(out),
        output_size=canvas.size,
        method=f"crop {image.height - kept_height} px from the bottom, LANCZOS resize",
        scale=width / image.width,
        crop_box=crop_box,
    )
    logger.info("finalized source=%s out=%s crop_box=%s scale=%.4f", source, out, list(crop_box), record.scale)
    return record
