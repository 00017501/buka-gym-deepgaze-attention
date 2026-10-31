"""Audit records written next to every generated and finalised image."""

import hashlib
from pathlib import Path

from pydantic import BaseModel


class InputFile(BaseModel):
    """An input image sent to the API, identified by content hash."""

    path: str
    sha256: str


class AttemptRecord(BaseModel):
    """Sidecar for one generation attempt: everything needed to audit or repeat the request."""

    name: str
    attempt: int
    created_utc: str
    sdk_version: str
    model: str
    size: str
    quality: str
    input_fidelity: str | None
    prompt_file: str
    prompt_sha256: str
    prompt_text: str
    inputs: list[InputFile]
    request_id: str | None
    response_created: int | None
    revised_prompt: str | None
    usage: dict[str, object] | None  # token usage; its shape is defined by the API
    output_file: str
    output_sha256: str


class FinalRecord(BaseModel):
    """Sidecar for the chosen attempt after fitting it to the post size."""

    source_file: str
    source_sha256: str
    source_size: tuple[int, int]
    output_file: str
    output_sha256: str
    output_size: tuple[int, int]
    method: str
    scale: float
    offset: tuple[int, int] | None = None  # pad only: where the resized image sits on the canvas
    pad_colour: tuple[int, int, int] | None = None  # pad only
    crop_box: tuple[int, int, int, int] | None = None  # crop only: kept region of the source, [x0, y0, x1, y1]


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
