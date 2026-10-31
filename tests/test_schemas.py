"""Unit tests for loading and validating AOI spec files."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.buka_attention.schemas import PostSpec, load_specs
from tests.conftest import REPO_ROOT

# The five posts whose AOIs are published and never replaced (D17); redesigns are added alongside them.
PUBLISHED_POSTS = {
    "post1_brand_flag",
    "post2_taekwondo_kids",
    "post3_trainer_rental",
    "post4_boxing_open_ring",
    "post5_special_offer",
}


@pytest.mark.parametrize("relative_path", ["aois.json", "docs/aois_v0_before_overlap_fix.json"])
def test_repository_aoi_files_load(relative_path: str) -> None:
    specs = load_specs(REPO_ROOT / relative_path)
    assert specs.keys() >= PUBLISHED_POSTS
    assert all(spec.aois for spec in specs.values())


def test_load_specs_skips_note_keys(tmp_path: Path) -> None:
    path = tmp_path / "aois.json"
    path.write_text(json.dumps({"_note": "free text", "p": {"image": "p.png", "aois": {}}}), encoding="utf-8")
    assert list(load_specs(path)) == ["p"]


@pytest.mark.parametrize(
    "aoi",
    [
        {"type": "banner", "boxes": [[0, 0, 1, 1]]},
        {"type": "logo", "boxes": [[0, 0, 1]]},
        {"type": "logo", "boxes": []},
        {"type": "logo", "boxes": [[0, 0, 1, 1]], "colour": "red"},
    ],
    ids=["unknown-type", "three-coordinates", "no-boxes", "extra-field"],
)
def test_invalid_aoi_is_rejected(aoi: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PostSpec.model_validate({"image": "p.png", "aois": {"a": aoi}})
