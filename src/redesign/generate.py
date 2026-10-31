"""One OpenAI image-edit request per attempt, saved with an audit sidecar."""

import base64
import hashlib
import logging
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from PIL import Image

from src.redesign.records import AttemptRecord, InputFile, sha256_of

logger = logging.getLogger(__name__)

# Logo source: the cleanest Buka Gym wordmark among the posts (post 4, top centre).
LOGO_SOURCE = Path("data/posts/post4_boxing_open_ring_original.png")
# Logo box in that post, [x0, y0, x1, y1] pixels (same as its AOI in aois.json).
LOGO_BOX = (385, 45, 505, 135)
# Where the crop is written; it is the second input image of the edit request.
LOGO_PATH = Path("provenance/redesigns/inputs/buka_logo.png")


@dataclass(frozen=True)
class EditRequest:
    """What to send: prompt file, input images (first one is edited) and model settings."""

    name: str
    prompt_file: Path
    images: list[Path]
    model: str
    size: str
    quality: str
    input_fidelity: str | None


def crop_logo(source: Path = LOGO_SOURCE, box: tuple[int, int, int, int] = LOGO_BOX, out: Path = LOGO_PATH) -> Path:
    """Crop the logo reference image from an existing post; deterministic, so it can be regenerated."""
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.open(source).convert("RGB").crop(box).save(out)
    logger.info("logo_cropped source=%s box=%s out=%s", source, list(box), out)
    return out


def next_attempt(out_dir: Path, name: str, max_attempts: int) -> int:
    """Return the next attempt number; refuse once `max_attempts` attempts exist (all are kept)."""
    done = len(list(out_dir.glob(f"{name}_attempt*.png")))
    if done >= max_attempts:
        msg = f"{done} attempts already exist for {name} (limit {max_attempts}); choose one or raise the limit"
        raise RuntimeError(msg)
    return done + 1


def _call_api(client: Any, request: EditRequest, prompt: str) -> Any:  # Any: OpenAI client / response objects
    """Send the edit request; `input_fidelity` is only sent when set, as not every model accepts it."""
    options = {"input_fidelity": request.input_fidelity} if request.input_fidelity else {}
    with ExitStack() as stack:
        files = [stack.enter_context(path.open("rb")) for path in request.images]
        return client.images.edit(
            image=files, prompt=prompt, model=request.model, size=request.size, quality=request.quality, n=1, **options
        )


def generate_attempt(client: Any, request: EditRequest, out_dir: Path, max_attempts: int = 3) -> AttemptRecord:
    """Run one attempt and save `<name>_attempt<k>.png` with a `.json` sidecar.

    1. Pick the attempt number (refuses past the limit, so earlier attempts are never overwritten).
    2. Send the prompt and input images to the image-edit endpoint.
    3. Save the returned image and an audit record of the request and response.
    """
    # 1. Attempt number.
    out_dir.mkdir(parents=True, exist_ok=True)
    attempt = next_attempt(out_dir, request.name, max_attempts)
    prompt = request.prompt_file.read_text(encoding="utf-8")

    # 2. API call.
    logger.info("attempt_started name=%s attempt=%d model=%s", request.name, attempt, request.model)
    response = _call_api(client, request, prompt)

    # 3. Image and sidecar.
    output = out_dir / f"{request.name}_attempt{attempt}.png"
    output.write_bytes(base64.b64decode(response.data[0].b64_json))
    usage = getattr(response, "usage", None)
    record = AttemptRecord(
        name=request.name,
        attempt=attempt,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        sdk_version=version("openai"),
        model=request.model,
        size=request.size,
        quality=request.quality,
        input_fidelity=request.input_fidelity,
        prompt_file=str(request.prompt_file),
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        prompt_text=prompt,
        inputs=[InputFile(path=str(path), sha256=sha256_of(path)) for path in request.images],
        request_id=getattr(response, "_request_id", None),
        response_created=getattr(response, "created", None),
        revised_prompt=getattr(response.data[0], "revised_prompt", None),
        usage=usage.model_dump() if usage is not None else None,
        output_file=str(output),
        output_sha256=sha256_of(output),
    )
    output.with_suffix(".json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    logger.info("attempt_saved file=%s", output)
    return record
