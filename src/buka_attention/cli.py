"""Command line: run DeepGaze IIE on every post in the AOI file and write heatmaps and statistics."""

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from src.buka_attention.model import DeepGazeRunner, pick_device
from src.buka_attention.report import build_markdown, compare, write_csv
from src.buka_attention.schemas import AoiStats, PostSpec, PostSummary, load_specs
from src.buka_attention.stats import Density, boxes_outside_image, compute_stats
from src.buka_attention.visuals import aoi_overlay, heat_overlay

logger = logging.getLogger(__name__)

EPILOG = "Usage examples: README.md. Outputs and metric definitions: docs/METHOD.md."


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser; defaults match the layout of this repository."""
    parser = argparse.ArgumentParser(prog="python -m src.buka_attention", description=__doc__, epilog=EPILOG)
    parser.add_argument(
        "--images", type=Path, default=Path("data"), help="folder the image paths in the AOI file are relative to"
    )
    parser.add_argument("--aois", type=Path, default=Path("data/aois.json"), help="AOI spec file")
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


def _warn_boxes_outside(key: str, spec: PostSpec, image: Image.Image) -> None:
    """Log every AOI box that extends past the image edge; it still counts, clipped to the image."""
    for name, box in boxes_outside_image(spec, image.height, image.width):
        logger.warning(
            "aoi_box_outside_image post=%s aoi=%s box=%s size=%dx%d", key, name, list(box), image.width, image.height
        )


def _stats_from_saved(key: str, spec: PostSpec, out: Path) -> tuple[list[AoiStats], PostSummary] | None:
    """Return a post's statistics from its saved density, or `None` when it has not been run yet."""
    density_path = out / "density" / f"{key}.npy"
    if not density_path.exists():
        logger.warning("density_missing post=%s path=%s left_out_of_tables=true", key, density_path)
        return None
    return compute_stats(key, spec, np.load(density_path))


def _process_post(
    key: str, spec: PostSpec, args: argparse.Namespace, runner: DeepGazeRunner | None
) -> tuple[list[AoiStats], PostSummary]:
    """Compute one post's statistics and save its two heatmap images."""
    logger.info("post_started post=%s", key)
    density = _load_density(key, spec, args, runner)
    image = Image.open(args.images / spec.image).convert("RGB")
    _warn_boxes_outside(key, spec, image)
    rows, summary = compute_stats(key, spec, density)
    heat = heat_overlay(image, density)
    heat.save(args.out / "heatmaps" / f"{key}_heatmap.png")
    aoi_overlay(heat, spec, rows).save(args.out / "heatmaps" / f"{key}_aoi.png")
    return rows, summary


def run(args: argparse.Namespace) -> None:
    """Run the pipeline over the selected posts.

    1. Load the AOI file and select the posts named by `--only` (all when omitted).
    2. For each selected post, get the density (model or saved `.npy`), compute stats and draw
       heatmaps. Other posts are recomputed from their saved densities so the tables stay complete.
    3. Write the CSVs and `stats.md` for every post that has a density.
    """
    # 1. Select posts and prepare output folders.
    specs = load_specs(args.aois)
    keys = [key for key in specs if not args.only or key in args.only]
    for subfolder in ("density", "heatmaps"):
        (args.out / subfolder).mkdir(parents=True, exist_ok=True)
    runner = None if args.skip_model else DeepGazeRunner(args.centerbias, args.device or pick_device())

    # 2. Per-post density, statistics and images, in aois.json order.
    aoi_rows: list[AoiStats] = []
    summaries: list[PostSummary] = []
    for key, spec in specs.items():
        post_stats = _process_post(key, spec, args, runner) if key in keys else _stats_from_saved(key, spec, args.out)
        if post_stats is None:
            continue
        rows, summary = post_stats
        aoi_rows += rows
        summaries.append(summary)

    # 3. Tables.
    write_csv(args.out / "stats_aoi.csv", aoi_rows)
    write_csv(args.out / "stats_summary.csv", summaries)
    comparison = compare(args.compare, summaries)
    if comparison:
        write_csv(args.out / "compare.csv", comparison)
    (args.out / "stats.md").write_text(build_markdown(summaries, aoi_rows, comparison), encoding="utf-8")
    logger.info("run_finished out=%s posts_run=%d posts_in_tables=%d", args.out, len(keys), len(summaries))


def main() -> None:
    """Parse arguments, configure logging and run the pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(build_parser().parse_args())
