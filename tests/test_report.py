"""Unit tests for table formatting, CSV output and the before/after comparison."""

import csv
from pathlib import Path

import pytest

from src.buka_attention.report import Column, compare, format_cell, md_table, write_csv
from src.buka_attention.schemas import AoiStats, AoiType, PostSummary


def _summary(post: str, cta: float | None) -> PostSummary:
    return PostSummary(
        post=post,
        label=post,
        entropy_norm=0.8,
        clarity=0.2,
        top10_pct=30.0,
        logo_share_pct=10.0,
        cta_share_pct=cta,
        offer_share_pct=None,
        face_share_pct=None,
        headline_share_pct=None,
        first_look_order=f"{post} order",
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "n/a"), (1.234, "1.23"), (3, "3"), ("x", "x"), (AoiType.LOGO, "logo")],
)
def test_format_cell(value: object, expected: str) -> None:
    assert format_cell(value, decimals=2) == expected


def test_md_table_uses_per_column_missing_text() -> None:
    row = AoiStats(post="p", aoi="(rest of image)", type=AoiType.OTHER, share_pct=1.0, area_pct=2.0, peak=0.0)
    table = md_table([row], [Column("AOI", "aoi"), Column("Rank", "first_look_rank", missing="")])
    assert table.splitlines() == ["| AOI | Rank |", "|---|---|", "| (rest of image) |  |"]


def test_write_csv_rounds_floats_and_blanks_none(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    write_csv(path, [_summary("a", None)])
    with path.open(encoding="utf-8") as file:
        (row,) = csv.DictReader(file)
    assert row["cta_share_pct"] == ""
    assert row["entropy_norm"] == "0.8"


def test_compare_computes_change_and_skips_missing_pairs() -> None:
    rows = compare(["a:b", "a:missing"], [_summary("a", 5.0), _summary("b", 8.0)])
    by_metric = {row.metric: row for row in rows}
    assert by_metric["cta_share_pct"].change == pytest.approx(3.0)
    assert by_metric["offer_share_pct"].change is None
    assert by_metric["first_look_order (b)"].after_val == "b order"
    assert {row.after for row in rows} == {"b"}
