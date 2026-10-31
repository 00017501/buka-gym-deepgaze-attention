"""CSV and Markdown output, and the before/after comparison of redesigns."""

import csv
import logging
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel

from src.buka_attention.schemas import AoiStats, CompareRow, PostSummary

logger = logging.getLogger(__name__)

# Decimal places in the CSVs; enough to reproduce the Markdown tables exactly.
CSV_DECIMALS = 4
# Summary metrics compared for each before:after pair, in table order.
COMPARE_METRICS = (
    "cta_share_pct",
    "offer_share_pct",
    "logo_share_pct",
    "face_share_pct",
    "headline_share_pct",
    "clarity",
    "top10_pct",
)


class Column(NamedTuple):
    """One Markdown table column: header, model field, decimals and the text shown for `None`."""

    header: str
    field: str
    decimals: int = 1
    missing: str = "n/a"


SUMMARY_COLUMNS = [
    Column("Post", "label"),
    Column("Logo %", "logo_share_pct"),
    Column("Headline %", "headline_share_pct"),
    Column("Face %", "face_share_pct"),
    Column("Offer %", "offer_share_pct"),
    Column("CTA %", "cta_share_pct"),
    Column("Clarity (0-1)", "clarity", 2),
    Column("Top-10% px share", "top10_pct"),
    Column("First-look order (by peak)", "first_look_order"),
]
AOI_COLUMNS = [
    Column("AOI", "aoi"),
    Column("Type", "type"),
    Column("% attention", "share_pct"),
    Column("% area", "area_pct"),
    Column("Lift", "lift", 2),
    # The rest-of-image row has no rank and shows an empty cell.
    Column("First-look rank", "first_look_rank", missing=""),
]
COMPARE_COLUMNS = [
    Column("Before", "before"),
    Column("After", "after"),
    Column("Metric", "metric"),
    Column("Before", "before_val", 2),
    Column("After", "after_val", 2),
    Column("Change", "change", 2),
]


def format_cell(value: object, decimals: int = 1, missing: str = "n/a") -> str:
    """Format a table cell: floats to `decimals` places, `None` as `missing`."""
    if value is None:
        return missing
    return f"{value:.{decimals}f}" if isinstance(value, float) else str(value)


def write_csv(path: Path, rows: list[BaseModel]) -> None:
    """Write rows as CSV with model fields as columns, floats rounded and `None` left empty."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(type(rows[0]).model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: round(value, CSV_DECIMALS) if isinstance(value, float) else ("" if value is None else value)
                    for field, value in row.model_dump().items()
                }
            )


def md_table(rows: list[BaseModel], columns: list[Column]) -> str:
    """Return a Markdown table of `rows` with the given columns."""
    lines = ["| " + " | ".join(column.header for column in columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = (format_cell(getattr(row, c.field), c.decimals, c.missing) for c in columns)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compare(pairs: list[str], summaries: list[PostSummary]) -> list[CompareRow]:
    """Return before/after rows for each `before:after` key pair; pairs with an unprocessed key are skipped."""
    summaries_by_post = {summary.post: summary for summary in summaries}
    rows = []
    for pair in pairs:
        before_key, after_key = pair.split(":")
        if before_key not in summaries_by_post or after_key not in summaries_by_post:
            logger.warning("compare_pair_not_found pair=%s", pair)
            continue
        before, after = summaries_by_post[before_key], summaries_by_post[after_key]
        for metric in COMPARE_METRICS:
            before_val, after_val = getattr(before, metric), getattr(after, metric)
            has_both = before_val is not None and after_val is not None
            rows.append(
                CompareRow(
                    before=before_key,
                    after=after_key,
                    metric=metric,
                    before_val=before_val,
                    after_val=after_val,
                    change=after_val - before_val if has_both else None,
                )
            )
        for key in (before_key, after_key):
            rows.append(
                CompareRow(
                    before=before_key,
                    after=after_key,
                    metric=f"first_look_order ({key})",
                    before_val=before.first_look_order if key == before_key else "",
                    after_val=after.first_look_order if key == after_key else "",
                    change=None,
                )
            )
    return rows


def build_markdown(summaries: list[PostSummary], aoi_rows: list[AoiStats], comparison: list[CompareRow]) -> str:
    """Return `stats.md`: the summary table, one AOI table per post, then the comparison if any."""
    lines = [
        "# Predicted attention (DeepGaze IIE) - Buka Gym posts",
        "",
        "_Predicted free-viewing attention from DeepGaze IIE (Linardos et al., 2021). Not real eye-tracking._",
        "",
        "## Summary per post",
        "",
        md_table(summaries, SUMMARY_COLUMNS),
        "",
        "n/a = element not present in the post.",
        "",
        "## Per AOI",
        "",
    ]
    for summary in summaries:
        post_rows = [row for row in aoi_rows if row.post == summary.post]
        lines += [f"### {summary.label}", "", md_table(post_rows, AOI_COLUMNS), ""]
    if comparison:
        lines += ["## Before / after", "", md_table(comparison, COMPARE_COLUMNS), ""]
    return "\n".join(lines)
