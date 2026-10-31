"""Ask Claude for AOI boxes on one image and write a staging entry plus an audit sidecar."""

import hashlib
import io
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from PIL import Image
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent, NativeOutput
from pydantic_ai.models import Model

from src.annotate.reply import AoiReply, find_problems, to_aois

logger = logging.getLogger(__name__)

# Longest edge sent to the model; larger images are downscaled by the API, which would shift the coordinates.
MAX_EDGE_PX = 1568
# Room for the boxes JSON with adaptive thinking on; well under the non-streaming timeout.
MAX_TOKENS = 16000
# Image paths in aois.json are relative to this folder.
DATA_DIR = Path("data")


@dataclass(frozen=True)
class SentImage:
    """How the image was sent: original and sent sizes, and the scale between them."""

    original_size: tuple[int, int]
    sent_size: tuple[int, int]
    scale: float


@dataclass(frozen=True)
class AnnotationRequest:
    """What to annotate and how: post key, image, prompt file and the model's effort level."""

    key: str
    image: Path
    prompt_file: Path
    effort: str


class AnnotationRecord(BaseModel):
    """Sidecar for one annotation: enough to audit the boxes and repeat the request."""

    key: str
    created_utc: str
    model_requested: str
    model_used: str
    effort: str
    sdk_versions: dict[str, str]
    prompt_file: str
    prompt_sha256: str
    prompt_text: str
    image: str
    image_sha256: str
    original_size: tuple[int, int]
    sent_size: tuple[int, int]
    scale: float
    request_id: str | None
    usage: dict[str, int]
    raw_reply: dict[str, object]  # the model's reply before scaling, exactly as returned
    problems: list[str]


def prepare_image(path: Path) -> tuple[bytes, tuple[int, int], tuple[int, int], float]:
    """Return PNG bytes with the long edge at most `MAX_EDGE_PX`, the original and sent sizes, and the scale."""
    image = Image.open(path).convert("RGB")
    scale = min(1.0, MAX_EDGE_PX / max(image.size))
    sent = image if scale == 1.0 else image.resize((round(image.width * scale), round(image.height * scale)))
    buffer = io.BytesIO()
    sent.save(buffer, format="PNG")
    return buffer.getvalue(), image.size, sent.size, scale


def image_ref(path: Path) -> str:
    """Return the image path as aois.json stores it: relative to `DATA_DIR`, or just the file name outside it."""
    try:
        return path.resolve().relative_to(DATA_DIR.resolve()).as_posix()
    except ValueError:
        return path.name


def annotate(model: Model | str, request: AnnotationRequest, out_dir: Path) -> AnnotationRecord:
    """Annotate one image into `<out_dir>/<key>.json` (the aois.json entry) and `<key>.meta.json`.

    1. Refuse to overwrite an earlier staging file, so every reply stays on record.
    2. Downscale the image if needed and fill the prompt with the size the model actually sees.
    3. Ask the model for a structured reply (native structured output, so no forced tool call).
    4. Scale the boxes back to original pixels, check them, and write the entry and the sidecar.
    """
    # 1. Never overwrite.
    entry_path, meta_path = out_dir / f"{request.key}.json", out_dir / f"{request.key}.meta.json"
    if entry_path.exists():
        msg = f"{entry_path} already exists; review or remove it before annotating {request.key} again"
        raise FileExistsError(msg)

    # 2. Image and prompt.
    png, original_size, sent_size, scale = prepare_image(request.image)
    template = request.prompt_file.read_text(encoding="utf-8")
    prompt = template.replace("{WIDTH}", str(sent_size[0])).replace("{HEIGHT}", str(sent_size[1]))
    logger.info("annotation_started key=%s sent=%dx%d scale=%.4f", request.key, *sent_size, scale)

    # 3. Model call.
    agent = Agent(model, output_type=NativeOutput(AoiReply))
    settings = {"anthropic_effort": request.effort, "max_tokens": MAX_TOKENS}
    result = agent.run_sync([prompt, BinaryContent(data=png, media_type="image/png")], model_settings=settings)

    # 4. Back to original pixels, checks, files.
    aois = to_aois(result.output, scale, *original_size)
    problems = find_problems(aois, *original_size)
    for problem in problems:
        logger.warning("aoi_problem key=%s problem=%s", request.key, problem)
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "image": image_ref(request.image),
        "aois": {name: aoi.model_dump(mode="json") for name, aoi in aois.items()},
    }
    entry_path.write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    record = _record(request, model, result, prompt, SentImage(original_size, sent_size, scale))
    record.problems = problems
    meta_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    logger.info("annotation_saved entry=%s problems=%d", entry_path, len(problems))
    return record


def _record(
    request: AnnotationRequest,
    model: Model | str,
    result: object,  # pydantic-ai run result
    prompt: str,
    sent: SentImage,
) -> AnnotationRecord:
    """Build the audit sidecar from the request and the run result; `problems` is filled in by the caller."""
    response = result.response
    usage = result.usage
    return AnnotationRecord(
        key=request.key,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        model_requested=model if isinstance(model, str) else model.model_name,
        model_used=response.model_name or "",
        effort=request.effort,
        sdk_versions={name: version(name) for name in ("pydantic-ai-slim", "anthropic")},
        prompt_file=str(request.prompt_file),
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        prompt_text=prompt,
        image=str(request.image),
        image_sha256=hashlib.sha256(request.image.read_bytes()).hexdigest(),
        original_size=sent.original_size,
        sent_size=sent.sent_size,
        scale=sent.scale,
        request_id=response.provider_response_id,
        usage={"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
        raw_reply=result.output.model_dump(),
        problems=[],
    )
