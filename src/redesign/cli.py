"""Command line: `generate` an AI redesign attempt, or `finalize` a chosen attempt to the post size."""

import argparse
import logging
import os
from pathlib import Path

from src.redesign.finalize import crop_bottom, fit_and_pad
from src.redesign.generate import LOGO_PATH, EditRequest, crop_logo, generate_attempt

logger = logging.getLogger(__name__)

# Pinned snapshot so the run can be repeated with the same model.
DEFAULT_MODEL = "gpt-image-2.5-sunburst-2026-09-08"
# Portrait output closest to the posts' 3:4 shape.
DEFAULT_SIZE = "1024x1536"
DEFAULT_ORIGINAL = Path("data/posts/post2_taekwondo_kids_original.png")


def build_parser() -> argparse.ArgumentParser:
    """Return the parser with the `generate` and `finalize` subcommands."""
    parser = argparse.ArgumentParser(prog="python -m src.redesign", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="one OpenAI image-edit attempt (needs OPENAI_API_KEY)")
    generate.add_argument("--name", default="post2_v1", help="redesign key; attempts are <name>_attempt<k>.png")
    generate.add_argument("--prompt", type=Path, default=Path("prompts/post2_v1.txt"))
    generate.add_argument(
        "--images", type=Path, nargs="+", help="input images, first is edited (default: post 2 + logo)"
    )
    generate.add_argument("--model", default=DEFAULT_MODEL)
    generate.add_argument("--size", default=DEFAULT_SIZE)
    generate.add_argument("--quality", default="high")
    generate.add_argument("--input-fidelity", choices=["high", "low"], help="only for models that accept it")
    generate.add_argument("--out", type=Path, default=Path("provenance/redesigns"))
    generate.add_argument("--max-attempts", type=int, default=3)

    finalize = commands.add_parser("finalize", help="fit a chosen attempt to the post size into data/redesigns/")
    finalize.add_argument("attempt", type=Path, help="e.g. provenance/redesigns/post2_v1_attempt1.png")
    finalize.add_argument("--out", type=Path, help="default: data/redesigns/<name>.png")
    finalize.add_argument("--width", type=int, default=900)
    finalize.add_argument("--height", type=int, default=1199)
    finalize.add_argument(
        "--fit", choices=["crop", "pad"], default="crop", help="crop the bottom (D18) or pad the sides"
    )
    return parser


def _run_generate(args: argparse.Namespace) -> None:
    """Check the key is set (never read or logged here), prepare inputs and run one attempt."""
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        msg = "OPENAI_API_KEY is not set; export it in your own terminal first"
        raise SystemExit(msg)
    images = args.images or [DEFAULT_ORIGINAL, LOGO_PATH if LOGO_PATH.exists() else crop_logo()]
    request = EditRequest(args.name, args.prompt, images, args.model, args.size, args.quality, args.input_fidelity)
    generate_attempt(OpenAI(), request, args.out, args.max_attempts)


def _run_finalize(args: argparse.Namespace) -> None:
    """Fit the chosen attempt to the post size and write its sidecar next to the attempt."""
    name = args.attempt.stem.rsplit("_attempt", 1)[0]
    out = args.out or Path("data/redesigns") / f"{name}.png"
    fit = crop_bottom if args.fit == "crop" else fit_and_pad
    record = fit(args.attempt, out, args.width, args.height)
    args.attempt.with_name(f"{name}_final.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")


def main() -> None:
    """Parse arguments, configure logging and run the subcommand."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    if args.command == "generate":
        _run_generate(args)
    else:
        _run_finalize(args)
