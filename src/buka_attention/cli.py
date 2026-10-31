"""Command line: run DeepGaze IIE on every post in the AOI file and write heatmaps and statistics."""

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from src.buka_attention.model import DeepGazeRunner, pick_device
from src.buka_attention.report import build_markdown, compare, write_csv
from src.buka_attention.schemas import AoiStats, PostSpec, PostSummary, load_specs
from src.buka_attention.stats import Density, compute_stats
from src.buka_attention.visuals import aoi_overlay, heat_overlay

logger = logging.getLogger(__name__)

EPILOG = "Usage examples: README.md. Outputs and metric definitions: docs/METHOD.md."


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser; defaults match the layout of this repository."""
    parser = argparse.ArgumentParser(prog="python -m src.buka_attention", description=__doc__, epilog=EPILOG)
    parser.add_argument("--images", type=Path, default=Path("images"), help="folder with the post images")
    parser.add_argument("--aois", type=Path, default=Path("aois.json"), help="AOI spec file")
    parser.add_argument("--out", type=Path, default=Path("output"), help="output folder")
    parser.add_argument("--centerbias", type=Path, default=Path("centerbias_mit1003.npy"), help="MIT1003 prior")
    parser.add_argument("--only", nargs="*", help="process only these post keys")
    parser.add_argument("--skip-model", action="store_true", help="reuse saved density/*.npy instead of the model")
    parser.add_argument("--compare", nargs="*", default=[], help="before:after post key pairs")
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available, else cpu)")
    return parser


def _load_density(key: str, spec: PostSpec, args: argparse.Namespace, runner: DeepGazeRunner | None) -> Density:
    """Return the post's density, from the model (and saved) or from the previous run's `.npy`."""
    density_path = args.out / "density" / f"{key}.npy"
    if runner is None:
        return np.load(density_path)
    density = runner.predict_density_from_file(args.images / spec.image)
    np.save(density_path, density)
    return density


def _process_post(
    key: str, spec: PostSpec, args: argparse.Namespace, runner: DeepGazeRunner | None
) -> tuple[list[AoiStats], PostSummary]:
    """Compute one post's statistics and save its two heatmap images."""
    logger.info("post_started post=%s", key)
    density = _load_density(key, spec, args, runner)
    image = Image.open(args.images / spec.image).convert("RGB")
    rows, summary = compute_stats(key, spec, density)
    heat = heat_overlay(image, density)
    heat.save(args.out / "heatmaps" / f"{key}_heatmap.png")
    aoi_overlay(heat, spec, rows).save(args.out / "heatmaps" / f"{key}_aoi.png")
    return rows, summary


def run(args: argparse.Namespace) -> None:
    """Run the pipeline over the selected posts.

    1. Load the AOI file and keep the posts named by `--only` (all when omitted).
    2. For each post, get the density (model or saved `.npy`), compute stats and draw heatmaps.
    3. Write the CSVs and `stats.md`. Tables hold only this run's posts, so a comparison
       needs both keys of each pair in `--only`.
    """
    # 1. Select posts and prepare output folders.
    specs = load_specs(args.aois)
    keys = [key for key in specs if not args.only or key in args.only]
    for subfolder in ("density", "heatmaps"):
        (args.out / subfolder).mkdir(parents=True, exist_ok=True)
    runner = None if args.skip_model else DeepGazeRunner(args.centerbias, args.device or pick_device())

    # 2. Per-post density, statistics and images.
    aoi_rows: list[AoiStats] = []
    summaries: list[PostSummary] = []
    for key in keys:
        rows, summary = _process_post(key, specs[key], args, runner)
        aoi_rows += rows
        summaries.append(summary)

    # 3. Tables.
    write_csv(args.out / "stats_aoi.csv", aoi_rows)
    write_csv(args.out / "stats_summary.csv", summaries)
    comparison = compare(args.compare, summaries)
    if comparison:
        write_csv(args.out / "compare.csv", comparison)
    (args.out / "stats.md").write_text(build_markdown(summaries, aoi_rows, comparison), encoding="utf-8")
    logger.info("run_finished out=%s posts=%d", args.out, len(keys))


def main() -> None:
    """Parse arguments, configure logging and run the pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(build_parser().parse_args())
