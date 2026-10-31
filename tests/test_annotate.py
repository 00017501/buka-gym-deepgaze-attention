"""Tests for the AOI annotator; the model is a local function, so no network is used."""

import json
import re
from pathlib import Path

import pytest
from PIL import Image
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

from src.annotate.annotate import MAX_EDGE_PX, AnnotationRequest, annotate, prepare_image
from src.annotate.apply import apply_entry
from src.annotate.reply import AoiReply, find_problems, to_aois
from src.buka_attention.schemas import Aoi, load_specs
from tests.conftest import REPO_ROOT

PROMPT_FILE = REPO_ROOT / "prompts" / "aoi_annotation.txt"


def _fake_model(reply: dict[str, object], seen: list[str] | None = None) -> FunctionModel:
    """Return a model that answers every request with `reply` and records the prompt text it was given."""

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if seen is not None:
            seen.extend(part.content[0] for part in messages[-1].parts if hasattr(part, "content"))
        return ModelResponse(parts=[TextPart(json.dumps(reply))], model_name="fake-model")

    return FunctionModel(respond, model_name="fake-model", profile=ModelProfile(supports_json_schema_output=True))


def _image(tmp_path: Path, size: tuple[int, int]) -> Path:
    path = tmp_path / "post9_test_original.png"
    Image.new("RGB", size, (30, 30, 30)).save(path)
    return path


REPLY = {
    "aois": [
        {"name": "headline", "type": "headline", "boxes": [{"x0": 10, "y0": 10, "x1": 300, "y1": 100}]},
        {"name": "cta", "type": "cta", "boxes": [{"x0": 10, "y0": 600, "x1": 200, "y1": 700}]},
    ]
}


def test_prompt_file_matches_the_published_prompt() -> None:
    published = (REPO_ROOT / "docs" / "AI_USE_AND_PROMPTS.md").read_text(encoding="utf-8")
    block = re.search(r"### 1b\..*?```\n(.*?)```", published, re.DOTALL)
    assert block is not None
    assert block.group(1) == PROMPT_FILE.read_text(encoding="utf-8")


def test_prepare_image_downscales_only_large_images(tmp_path: Path) -> None:
    _, original, sent, scale = prepare_image(_image(tmp_path, (900, 1800)))
    assert original == (900, 1800)
    assert max(sent) == MAX_EDGE_PX
    assert scale == pytest.approx(MAX_EDGE_PX / 1800)
    assert prepare_image(_image(tmp_path, (900, 1199)))[3] == 1.0


def test_to_aois_scales_boxes_back_and_clamps() -> None:
    reply = AoiReply.model_validate(
        {"aois": [{"name": "logo", "type": "logo", "boxes": [{"x0": 50, "y0": 50, "x1": 2000, "y1": 100}]}]}
    )
    assert to_aois(reply, 0.5, 900, 1800)["logo"].boxes == [(100, 100, 900, 200)]


def test_find_problems_reports_overlap_size_and_bounds() -> None:
    aois = {
        "a": Aoi.model_validate({"type": "logo", "boxes": [[0, 0, 20, 20]]}),
        "b": Aoi.model_validate({"type": "cta", "boxes": [[10, 10, 30, 30], [5, 5, 5, 9], [90, 90, 120, 95]]}),
    }
    problems = find_problems(aois, 100, 100)
    assert "b: box [5, 5, 5, 9] has zero or negative size" in problems
    assert "b: box [90, 90, 120, 95] extends past the 100x100 image" in problems
    assert "a and b overlap by 100 px" in problems


def test_annotate_writes_entry_and_sidecar_with_filled_prompt(tmp_path: Path) -> None:
    seen: list[str] = []
    request = AnnotationRequest("post9_test", _image(tmp_path, (900, 1199)), PROMPT_FILE, "high")
    record = annotate(_fake_model(REPLY, seen), request, tmp_path / "staging")
    entry = json.loads((tmp_path / "staging" / "post9_test.json").read_text(encoding="utf-8"))
    meta = json.loads((tmp_path / "staging" / "post9_test.meta.json").read_text(encoding="utf-8"))
    assert entry["aois"]["cta"] == {"type": "cta", "boxes": [[10, 600, 200, 700]]}
    assert "The image is 900 x 1199 pixels." in seen[0]
    assert meta["model_used"] == "fake-model"
    assert meta["effort"] == "high"
    assert meta["problems"] == []
    assert record.scale == 1.0


def test_annotate_never_overwrites_a_staging_file(tmp_path: Path) -> None:
    request = AnnotationRequest("post9_test", _image(tmp_path, (900, 1199)), PROMPT_FILE, "high")
    annotate(_fake_model(REPLY), request, tmp_path)
    with pytest.raises(FileExistsError):
        annotate(_fake_model(REPLY), request, tmp_path)


def test_sidecar_never_contains_the_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-secret")
    request = AnnotationRequest("post9_test", _image(tmp_path, (900, 1199)), PROMPT_FILE, "high")
    annotate(_fake_model(REPLY), request, tmp_path)
    assert "sk-ant-test-secret" not in (tmp_path / "post9_test.meta.json").read_text(encoding="utf-8")


def test_apply_appends_entry_and_keeps_existing_text(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (900, 1199)).save(images / "new.png")
    aois_path = tmp_path / "aois.json"
    original = '{\n  "old": {"image": "old.png", "aois": {"logo": {"type": "logo", "boxes": [[0, 0, 5, 5]]}}}\n}\n'
    aois_path.write_text(original, encoding="utf-8")
    entry = tmp_path / "post9_new.json"
    entry.write_text(json.dumps({"image": "new.png", "aois": {"cta": {"type": "cta", "boxes": [[1, 1, 9, 9]]}}}))
    apply_entry(entry, aois_path, images, "New post")
    text = aois_path.read_text(encoding="utf-8")
    assert text.startswith(original.rstrip()[:-1].rstrip())
    specs = load_specs(aois_path)
    assert list(specs) == ["old", "post9_new"]
    assert specs["post9_new"].label == "New post"


def test_apply_refuses_published_keys_and_failing_boxes(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (100, 100)).save(images / "x.png")
    aois_path = tmp_path / "aois.json"
    aois_path.write_text(json.dumps({"post1": {"image": "x.png", "aois": {}}}), encoding="utf-8")
    published = tmp_path / "post1.json"
    published.write_text(json.dumps({"image": "x.png", "aois": {}}))
    with pytest.raises(ValueError, match="never replaced"):
        apply_entry(published, aois_path, images, None)
    overlapping = tmp_path / "post2.json"
    boxes = {"a": {"type": "logo", "boxes": [[0, 0, 50, 50]]}, "b": {"type": "cta", "boxes": [[40, 40, 90, 90]]}}
    overlapping.write_text(json.dumps({"image": "x.png", "aois": boxes}))
    with pytest.raises(ValueError, match="overlap"):
        apply_entry(overlapping, aois_path, images, None)


def test_real_anthropic_request_shape(tmp_path: Path) -> None:
    import httpx2 as httpx
    from anthropic import AsyncAnthropic
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    sent: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        body = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": json.dumps(REPLY)}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        return httpx.Response(200, json=body)

    client = AsyncAnthropic(
        api_key="sk-ant-test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    model = AnthropicModel("claude-opus-5-5", provider=AnthropicProvider(anthropic_client=client))
    request = AnnotationRequest("post9_test", _image(tmp_path, (900, 1199)), PROMPT_FILE, "high")
    record = annotate(model, request, tmp_path)
    (payload,) = sent
    assert payload["model"] == "claude-opus-5-5"
    assert payload["output_config"]["effort"] == "high"
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert not {"temperature", "top_p", "top_k"} & payload.keys()
    assert payload.get("tool_choice", {"type": "auto"})["type"] in {"auto", "none"}
    assert payload.get("thinking", {"type": "adaptive"})["type"] == "adaptive"
    assert any(block.get("type") == "image" for block in payload["messages"][0]["content"])
    assert record.request_id == "msg_test"
    assert record.model_used == "claude-opus-5-5"


def test_run_explains_a_missing_image(tmp_path: Path) -> None:
    from src.annotate.cli import _run, build_parser

    args = build_parser().parse_args(["run", str(tmp_path / "post2_v1.png")])
    with pytest.raises(SystemExit, match=re.escape("run `python -m src.redesign finalize <attempt>` first")):
        _run(args)
