"""Unit tests for the AOI statistics."""

import numpy as np
import pytest

from src.buka_attention.schemas import AoiType, PostSpec
from src.buka_attention.stats import (
    REST_AOI_NAME,
    Density,
    aoi_mask,
    boxes_outside_image,
    compute_stats,
    entropy_norm,
    top_k_share,
)


def test_aoi_mask_clips_boxes_to_image() -> None:
    mask = aoi_mask([(-5, -5, 3, 3), (8, 8, 50, 50)], 10, 10)
    assert mask.sum() == 3 * 3 + 2 * 2
    assert mask[0, 0] and mask[9, 9]


def test_aoi_mask_unions_overlapping_boxes() -> None:
    assert aoi_mask([(0, 0, 4, 4), (2, 2, 6, 6)], 10, 10).sum() == 16 + 16 - 4


def test_entropy_norm_is_one_for_uniform_density() -> None:
    assert entropy_norm(np.full((40, 40), 1 / 1600)) == pytest.approx(1.0)


def test_entropy_norm_is_zero_when_all_mass_is_in_one_cell() -> None:
    grid = np.zeros((40, 40))
    grid[:10, :10] = 1 / 100
    assert entropy_norm(grid) == pytest.approx(0.0)


def test_top_k_share_of_uniform_density_equals_fraction() -> None:
    assert top_k_share(np.full((10, 10), 0.01)) == pytest.approx(0.10)


def test_compute_stats_shares_cover_whole_image(density: Density, spec: PostSpec) -> None:
    rows, _ = compute_stats("p", spec, density)
    assert sum(row.share_pct for row in rows) == pytest.approx(100.0)
    assert sum(row.area_pct for row in rows) == pytest.approx(100.0)


def test_compute_stats_ranks_by_peak_and_puts_rest_last(density: Density, spec: PostSpec) -> None:
    rows, summary = compute_stats("p", spec, density)
    assert [(row.aoi, row.first_look_rank) for row in rows] == [("logo", 1), ("cta", 2), (REST_AOI_NAME, None)]
    assert rows[-1].type is AoiType.OTHER
    assert summary.first_look_order == "logo > cta"


def test_compute_stats_lift_compares_share_with_area(density: Density, spec: PostSpec) -> None:
    rows, _ = compute_stats("p", spec, density)
    logo = rows[0]
    assert logo.share_pct == pytest.approx(50.0)
    assert logo.lift == pytest.approx(50.0 / logo.area_pct)


def test_compute_stats_summary_marks_absent_types_as_none(density: Density, spec: PostSpec) -> None:
    _, summary = compute_stats("p", spec, density)
    assert summary.logo_share_pct == pytest.approx(50.0)
    assert summary.offer_share_pct is None
    assert summary.clarity == pytest.approx(1 - summary.entropy_norm)
    assert summary.label == "Synthetic"


def test_boxes_outside_image_lists_only_boxes_past_the_edge() -> None:
    spec = PostSpec.model_validate(
        {
            "image": "synthetic.png",
            "aois": {
                "inside": {"type": "logo", "boxes": [[0, 0, 40, 40]]},
                "faces": {"type": "face", "boxes": [[5, 5, 10, 10], [30, 30, 45, 38]]},
                "negative": {"type": "cta", "boxes": [[-1, 0, 5, 5]]},
            },
        }
    )
    assert boxes_outside_image(spec, 40, 40) == [("faces", (30, 30, 45, 38)), ("negative", (-1, 0, 5, 5))]
