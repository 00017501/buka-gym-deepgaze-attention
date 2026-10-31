"""Tests for the redesign generator and finaliser; the OpenAI client is faked, so no network is used."""

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.redesign.finalize import border_colour, crop_bottom, fit_and_pad
from src.redesign.generate import EditRequest, crop_logo, generate_attempt


def _png_bytes(size: tuple[int, int], colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeImages:
    """Records `edit` calls and returns a small PNG, like the SDK's base64 response."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def edit(self, **kwargs: object) -> SimpleNamespace:
        """Record the call and return one fake image."""
        self.calls.append({**kwargs, "image": [file.name for file in kwargs["image"]]})
        item = SimpleNamespace(
            b64_json=base64.b64encode(_png_bytes((64, 96), (10, 10, 10))).decode(), revised_prompt=None
        )
        return SimpleNamespace(data=[item], created=1_793_000_000, usage=None, _request_id="req_test")


@pytest.fixture
def request_files(tmp_path: Path) -> EditRequest:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Make the offer bigger.", encoding="utf-8")
    images = []
    for name in ("original.png", "logo.png"):
        path = tmp_path / name
        path.write_bytes(_png_bytes((32, 32), (200, 0, 0)))
        images.append(path)
    return EditRequest("post2_v1", prompt, images, "model-x", "1024x1536", "high", None)


def test_generate_attempt_saves_image_and_sidecar(tmp_path: Path, request_files: EditRequest) -> None:
    client = SimpleNamespace(images=FakeImages())
    record = generate_attempt(client, request_files, tmp_path / "out")
    sidecar = json.loads((tmp_path / "out" / "post2_v1_attempt1.json").read_text(encoding="utf-8"))
    assert (tmp_path / "out" / "post2_v1_attempt1.png").exists()
    assert sidecar["prompt_text"] == "Make the offer bigger."
    assert sidecar["model"] == "model-x"
    assert sidecar["request_id"] == "req_test"
    assert [entry["path"] for entry in sidecar["inputs"]] == [str(path) for path in request_files.images]
    assert record.attempt == 1


def test_generate_attempt_sends_all_images_and_omits_unset_fidelity(tmp_path: Path, request_files: EditRequest) -> None:
    images = FakeImages()
    generate_attempt(SimpleNamespace(images=images), request_files, tmp_path / "out")
    (call,) = images.calls
    assert call["image"] == [str(path) for path in request_files.images]
    assert "input_fidelity" not in call
    assert call["n"] == 1


def test_generate_attempt_numbers_attempts_and_stops_at_limit(tmp_path: Path, request_files: EditRequest) -> None:
    client = SimpleNamespace(images=FakeImages())
    for expected in (1, 2):
        assert generate_attempt(client, request_files, tmp_path, max_attempts=2).attempt == expected
    with pytest.raises(RuntimeError, match="2 attempts already exist"):
        generate_attempt(client, request_files, tmp_path, max_attempts=2)


def test_sidecar_never_contains_the_api_key(
    tmp_path: Path, request_files: EditRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-value")
    generate_attempt(SimpleNamespace(images=FakeImages()), request_files, tmp_path)
    assert "sk-test-secret-value" not in (tmp_path / "post2_v1_attempt1.json").read_text(encoding="utf-8")


def test_crop_logo_cuts_the_given_box(tmp_path: Path) -> None:
    source = tmp_path / "post.png"
    Image.new("RGB", (100, 100), (0, 0, 255)).save(source)
    out = crop_logo(source, (10, 20, 40, 60), tmp_path / "inputs" / "logo.png")
    assert Image.open(out).size == (30, 40)


def test_fit_and_pad_keeps_aspect_and_pads_with_border_colour(tmp_path: Path) -> None:
    source = tmp_path / "attempt.png"
    Image.new("RGB", (1024, 1536), (20, 30, 40)).save(source)
    record = fit_and_pad(source, tmp_path / "final.png", 900, 1199)
    assert record.output_size == (900, 1199)
    assert record.scale == pytest.approx(1199 / 1536)
    assert record.offset[1] == 0
    assert record.offset[0] > 0
    assert record.pad_colour == (20, 30, 40)


def test_border_colour_ignores_the_centre() -> None:
    image = Image.new("RGB", (50, 50), (5, 5, 5))
    image.paste(Image.new("RGB", (30, 30), (255, 255, 255)), (10, 10))
    assert border_colour(image) == (5, 5, 5)


def test_crop_bottom_keeps_the_top_and_reaches_the_post_shape(tmp_path: Path) -> None:
    source = tmp_path / "attempt.png"
    image = Image.new("RGB", (1024, 1536), (0, 0, 0))
    image.paste(Image.new("RGB", (1024, 100), (255, 0, 0)), (0, 0))
    image.paste(Image.new("RGB", (1024, 100), (0, 0, 255)), (0, 1436))
    image.save(source)
    record = crop_bottom(source, tmp_path / "final.png", 900, 1199)
    final = Image.open(tmp_path / "final.png")
    assert final.size == (900, 1199)
    assert record.crop_box == (0, 0, 1024, 1364)
    assert final.getpixel((450, 5)) == (255, 0, 0)
    assert final.getpixel((450, 1195)) == (0, 0, 0)


def test_crop_bottom_rejects_images_wider_than_the_target(tmp_path: Path) -> None:
    source = tmp_path / "wide.png"
    Image.new("RGB", (1536, 1024)).save(source)
    with pytest.raises(ValueError, match="wider than"):
        crop_bottom(source, tmp_path / "final.png", 900, 1199)
