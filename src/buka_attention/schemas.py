"""Typed shapes for the AOI spec file and the statistics written to disk."""

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class AoiType(StrEnum):
    """Category of an area of interest; drives colours and the per-type summary shares."""

    LOGO = "logo"
    HEADLINE = "headline"
    FACE = "face"
    OFFER = "offer"
    CTA = "cta"
    BODY = "body"
    VISUAL = "visual"
    OTHER = "other"


type Box = tuple[int, int, int, int]


class Aoi(BaseModel):
    """One area of interest: a type and one or more `[x0, y0, x1, y1]` pixel boxes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: AoiType
    boxes: list[Box] = Field(min_length=1)


class PostSpec(BaseModel):
    """One post: its image file, a display label and its AOIs in drawing order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: str
    label: str | None = None
    aois: dict[str, Aoi]


class AoiStats(BaseModel):
    """One row of `stats_aoi.csv`; field order is the CSV column order."""

    post: str
    aoi: str
    type: AoiType
    share_pct: float
    area_pct: float
    peak: float
    lift: float = 0.0
    first_look_rank: int | None = None


class PostSummary(BaseModel):
    """One row of `stats_summary.csv`; a `None` share means the element is absent from the post."""

    post: str
    label: str
    entropy_norm: float
    clarity: float
    top10_pct: float
    logo_share_pct: float | None
    cta_share_pct: float | None
    offer_share_pct: float | None
    face_share_pct: float | None
    headline_share_pct: float | None
    first_look_order: str


class CompareRow(BaseModel):
    """One row of `compare.csv`: a metric before and after a redesign."""

    before: str
    after: str
    metric: str
    before_val: float | str | None
    after_val: float | str | None
    change: float | None


def load_specs(path: Path) -> dict[str, PostSpec]:
    """Load and validate the AOI file; top-level keys starting with `_` are notes and are skipped."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {key: PostSpec.model_validate(spec) for key, spec in raw.items() if not key.startswith("_")}
