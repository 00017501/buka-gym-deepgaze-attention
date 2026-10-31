"""Command line: `run` asks Claude for AOI boxes into a staging file; `apply` adds a reviewed entry to aois.json."""

import argparse
import logging
import os
from pathlib import Path

from src.annotate.annotate import AnnotationRequest, annotate
from src.annotate.apply import apply_entry

logger = logging.getLogger(__name__)

# Current Opus model ID; the sidecar also records `model_used`, the model that actually answered.
DEFAULT_MODEL = "claude-opus-5-5"
# Opus 5.5 defaults to `medium`; box placement is careful visual work, so ask for more.
DEFAULT_EFFORT = "high"
STAGING_DIR = Path("provenance/aois")


def build_parser() -> argparse.ArgumentParser:
    """Return the parser with the `run` and `apply` subcommands."""
    parser = argparse.ArgumentParser(prog="python -m src.annotate", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run", help="annotate one image into provenance/aois/<key>.json (needs ANTHROPIC_API_KEY)"
    )
    run.add_argument("image", type=Path, help="e.g. data/redesigns/post2_v1.png")
    run.add_argument("--key", help="post key (default: image name without _original)")
    run.add_argument("--prompt", type=Path, default=Path("prompts/aoi_annotation.txt"))
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    run.add_argument("--out", type=Path, default=STAGING_DIR, help="staging folder (robustness runs: a subfolder)")

    apply = commands.add_parser("apply", help="add a reviewed staging entry to aois.json")
    apply.add_argument("entry", type=Path, help="e.g. provenance/aois/post2_v1.json")
    apply.add_argument("--label", help="display label for the tables")
    apply.add_argument("--aois", type=Path, default=Path("data/aois.json"))
    apply.add_argument("--images", type=Path, default=Path("data"), help="folder the entry's image path is relative to")
    return parser


def _run(args: argparse.Namespace) -> None:
    """Check the image exists and the key is set (its value is never read here), then annotate the image."""
    if not args.image.exists():
        msg = f"{args.image} does not exist. For a redesign, run `python -m src.redesign finalize <attempt>` first."
        raise SystemExit(msg)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        msg = "ANTHROPIC_API_KEY is not set; export it in your own terminal first"
        raise SystemExit(msg)
    key = args.key or args.image.stem.removesuffix("_original")
    model = f"anthropic:{args.model}"
    annotate(model, AnnotationRequest(key, args.image, args.prompt, args.effort), args.out)


def main() -> None:
    """Parse arguments, configure logging and run the subcommand."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    if args.command == "run":
        _run(args)
    else:
        apply_entry(args.entry, args.aois, args.images, args.label)
