"""Add a reviewed staging entry to aois.json without touching the existing entries."""

import json
import logging
from pathlib import Path

from PIL import Image

from src.annotate.reply import find_problems
from src.buka_attention.schemas import PostSpec, load_specs

logger = logging.getLogger(__name__)


def apply_entry(entry_path: Path, aois_path: Path, images_dir: Path, label: str | None) -> None:
    """Append the entry in `entry_path` to `aois_path` under its key; refuse existing keys and failing checks.

    The new entry is appended as text before the final brace, so the existing entries keep their formatting.
    """
    key = entry_path.stem
    if key in load_specs(aois_path):
        msg = f"{key} is already in {aois_path}; published AOIs are never replaced (D17)"
        raise ValueError(msg)
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    if label is not None:
        entry = {"image": entry["image"], "label": label, "aois": entry["aois"]}
    spec = PostSpec.model_validate(entry)
    width, height = Image.open(images_dir / spec.image).size
    problems = find_problems(spec.aois, width, height)
    if problems:
        msg = f"{entry_path} fails the checks; fix the boxes first: " + "; ".join(problems)
        raise ValueError(msg)

    text = aois_path.read_text(encoding="utf-8").rstrip()
    if not text.endswith("}"):
        msg = f"{aois_path} does not end with a closing brace"
        raise ValueError(msg)
    block = json.dumps({key: entry}, indent=2, ensure_ascii=False)[1:-1].rstrip()
    aois_path.write_text(text[:-1].rstrip() + "," + block + "\n}\n", encoding="utf-8")
    load_specs(aois_path)
    logger.info("entry_applied key=%s aois=%s", key, aois_path)
