"""The model's structured reply, its conversion to an aois.json entry, and the checks a reviewer relies on."""

from itertools import combinations
from typing import Literal

from pydantic import BaseModel

from src.buka_attention.schemas import Aoi, Box
from src.buka_attention.stats import aoi_mask

# The AOI types the prompt allows; `other` is reserved for the rest-of-image row.
type ReplyType = Literal["logo", "headline", "face", "offer", "cta", "body", "visual"]


class ReplyBox(BaseModel):
    """One rectangle in the pixel space of the image the model was shown."""

    x0: int
    y0: int
    x1: int
    y1: int


class ReplyAoi(BaseModel):
    """One named AOI with its type and rectangles."""

    name: str
    type: ReplyType
    boxes: list[ReplyBox]


class AoiReply(BaseModel):
    """The whole reply; a list rather than a name-keyed object, which structured outputs can't express."""

    aois: list[ReplyAoi]


def to_aois(reply: AoiReply, scale: float, width: int, height: int) -> dict[str, Aoi]:
    """Map boxes from the shown image back to original pixels (divide by `scale`) and clamp them to the image."""

    def original(box: ReplyBox) -> Box:
        x0, y0, x1, y1 = (round(value / scale) for value in (box.x0, box.y0, box.x1, box.y1))
        return (max(0, min(x0, width)), max(0, min(y0, height)), max(0, min(x1, width)), max(0, min(y1, height)))

    return {
        aoi.name: Aoi.model_validate({"type": aoi.type, "boxes": [original(box) for box in aoi.boxes]})
        for aoi in reply.aois
    }


def find_problems(aois: dict[str, Aoi], width: int, height: int) -> list[str]:
    """Return human-readable problems: empty, inverted or out-of-image boxes, duplicate names and AOI overlaps."""
    problems = [f"no boxes for {name}" for name, aoi in aois.items() if not aoi.boxes]
    for name, aoi in aois.items():
        for x0, y0, x1, y1 in aoi.boxes:
            if x0 >= x1 or y0 >= y1:
                problems.append(f"{name}: box {[x0, y0, x1, y1]} has zero or negative size")
            if min(x0, y0) < 0 or x1 > width or y1 > height:
                problems.append(f"{name}: box {[x0, y0, x1, y1]} extends past the {width}x{height} image")
    masks = {name: aoi_mask(aoi.boxes, height, width) for name, aoi in aois.items()}
    for (first, first_mask), (second, second_mask) in combinations(masks.items(), 2):
        overlap = int((first_mask & second_mask).sum())
        if overlap:
            problems.append(f"{first} and {second} overlap by {overlap} px")
    return problems
