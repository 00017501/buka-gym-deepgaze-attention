"""Tests for the pipeline run without the model: table completeness and AOI box warnings."""

import csv
import json
import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.buka_attention.cli import build_parser, run
from src.buka_attention.stats import Density


def _write_posts(tmp_path: Path, density: Density, keys: list[str], boxes: list[list[int]]) -> Path:
    """Create images, an AOI file and saved densities for `keys`; return the output folder."""
    out = tmp_path / "output"
    (tmp_path / "images").mkdir()
    (out / "density").mkdir(parents=True)
    spec = {}
    for key in keys:
        Image.new("RGB", (40, 40)).save(tmp_path / "images" / f"{key}.png")
        np.save(out / "density" / f"{key}.npy", density)
        spec[key] = {"image": f"{key}.png", "aois": {"logo": {"type": "logo", "boxes": boxes}}}
    (tmp_path / "aois.json").write_text(json.dumps(spec), encoding="utf-8")
    return out


def _run(tmp_path: Path, *extra: str) -> None:
    base = ["--images", str(tmp_path / "images"), "--aois", str(tmp_path / "aois.json")]
    run(build_parser().parse_args([*base, "--out", str(tmp_path / "output"), "--skip-model", *extra]))


def _posts_in(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as file:
        return [row["post"] for row in csv.DictReader(file)]


def test_only_keeps_other_posts_in_tables(tmp_path: Path, density: Density) -> None:
    out = _write_posts(tmp_path, density, ["a", "b", "c"], [[0, 0, 10, 10]])
    _run(tmp_path, "--only", "b")
    assert _posts_in(out / "stats_summary.csv") == ["a", "b", "c"]
    assert sorted(path.name for path in (out / "heatmaps").iterdir()) == ["b_aoi.png", "b_heatmap.png"]


def test_only_leaves_out_posts_without_saved_density(
    tmp_path: Path, density: Density, caplog: pytest.LogCaptureFixture
) -> None:
    out = _write_posts(tmp_path, density, ["a", "b"], [[0, 0, 10, 10]])
    (out / "density" / "a.npy").unlink()
    with caplog.at_level(logging.WARNING):
        _run(tmp_path, "--only", "b")
    assert _posts_in(out / "stats_summary.csv") == ["b"]
    assert "density_missing post=a" in caplog.text


def test_box_past_image_edge_is_warned(tmp_path: Path, density: Density, caplog: pytest.LogCaptureFixture) -> None:
    _write_posts(tmp_path, density, ["a"], [[30, 30, 50, 40]])
    with caplog.at_level(logging.WARNING):
        _run(tmp_path)
    assert "aoi_box_outside_image post=a aoi=logo box=[30, 30, 50, 40] size=40x40" in caplog.text
