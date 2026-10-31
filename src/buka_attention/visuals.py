"""Heatmap and AOI overlays drawn with numpy and PIL only."""

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from src.buka_attention.schemas import AoiStats, AoiType, PostSpec

# Inferno-like colour stops; inlined so the visuals need no matplotlib at runtime.
COLORMAP_STOPS = np.array(
    [
        [0, 0, 4],
        [40, 11, 84],
        [101, 21, 110],
        [159, 42, 99],
        [212, 72, 66],
        [245, 125, 21],
        [250, 193, 39],
        [252, 255, 164],
    ],
    float,
)
# Box and label colour per AOI type; white for anything unlisted.
TYPE_COLOURS: dict[AoiType, tuple[int, int, int]] = {
    AoiType.LOGO: (0, 200, 255),
    AoiType.CTA: (0, 255, 120),
    AoiType.OFFER: (255, 230, 0),
    AoiType.FACE: (255, 80, 200),
    AoiType.HEADLINE: (255, 255, 255),
    AoiType.BODY: (180, 180, 180),
    AoiType.VISUAL: (160, 120, 255),
}
# Bold fonts tried in order; the names cover Linux, Windows and macOS.
FONT_CANDIDATES = (
    "DejaVuSans-Bold.ttf",
    "arialbd.ttf",
    "Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)


def colormap(values: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Map values in 0..1 to RGB by linear interpolation between the colour stops."""
    scaled = np.clip(values, 0, 1) * (len(COLORMAP_STOPS) - 1)
    index = np.minimum(scaled.astype(int), len(COLORMAP_STOPS) - 2)
    fraction = (scaled - index)[..., None]
    return (COLORMAP_STOPS[index] * (1 - fraction) + COLORMAP_STOPS[index + 1] * fraction).astype(np.uint8)


def heat_overlay(image: Image.Image, density: npt.NDArray[np.float64], alpha: float = 0.8) -> Image.Image:
    """Blend the density heatmap over a dimmed copy of the image (tuning in DECISIONS.md D11)."""
    values = density / density.max()
    values = values**0.4  # gamma so mid-level attention stays visible
    rgb = colormap(values).astype(float)
    base = np.asarray(image.convert("RGB")).astype(float) * 0.6
    opacity = (alpha * np.clip(values * 2.0, 0, 1))[..., None]
    return Image.fromarray((base * (1 - opacity) + rgb * opacity).astype(np.uint8))


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return the first available bold font at `size`, or PIL's built-in font."""
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def aoi_overlay(heat: Image.Image, spec: PostSpec, rows: list[AoiStats]) -> Image.Image:
    """Draw each AOI's boxes and a `#rank name share%` label on a copy of the heatmap."""
    canvas = heat.copy()
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(16, canvas.width // 40))
    rows_by_name = {row.aoi: row for row in rows}
    for name, aoi in spec.aois.items():
        colour = TYPE_COLOURS.get(aoi.type, (255, 255, 255))
        row = rows_by_name[name]
        for box in aoi.boxes:
            draw.rectangle(box, outline=colour, width=4)
        x0, y0 = aoi.boxes[0][:2]
        text = f"#{row.first_look_rank} {name} {row.share_pct:.1f}%"
        text_w, text_h = draw.textbbox((0, 0), text, font=font)[2:]
        # Label sits above the first box, or inside it when there is no room above.
        text_y = y0 - text_h - 8 if y0 - text_h - 8 > 0 else y0 + 4
        draw.rectangle([x0, text_y, x0 + text_w + 10, text_y + text_h + 6], fill=(0, 0, 0))
        draw.text((x0 + 5, text_y + 2), text, fill=colour, font=font)
    return canvas
