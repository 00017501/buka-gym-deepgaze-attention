"""AOI statistics over a fixation density map (see docs/METHOD.md §4 for definitions)."""

import numpy as np
import numpy.typing as npt

from src.buka_attention.schemas import Aoi, AoiStats, AoiType, Box, PostSpec, PostSummary

type Density = npt.NDArray[np.float64]
type Mask = npt.NDArray[np.bool_]

# Pooling block for the entropy; 10 px keeps single-pixel noise from inflating the entropy.
ENTROPY_GRID_PX = 10
# Fraction of most-attended pixels used by the top-share focus measure.
TOP_PIXEL_FRACTION = 0.10
# Peak density is scaled by 1e6 so the table shows readable numbers.
PEAK_SCALE = 1e6
# Rows for the part of the image no AOI covers carry this name.
REST_AOI_NAME = "(rest of image)"
# AOI types that get their own share column in the summary.
SUMMARY_TYPES = (AoiType.LOGO, AoiType.CTA, AoiType.OFFER, AoiType.FACE, AoiType.HEADLINE)


def aoi_mask(boxes: list[Box], height: int, width: int) -> Mask:
    """Return a boolean mask of the union of `boxes`, clipped to the image."""
    mask = np.zeros((height, width), bool)
    for x0, y0, x1, y1 in boxes:
        mask[max(0, int(y0)) : min(height, int(y1)), max(0, int(x0)) : min(width, int(x1))] = True
    return mask


def entropy_norm(density: Density) -> float:
    """Return the Shannon entropy of the grid-pooled density divided by its maximum; 0..1, lower = more focused."""
    height, width = density.shape
    cropped_h, cropped_w = height // ENTROPY_GRID_PX * ENTROPY_GRID_PX, width // ENTROPY_GRID_PX * ENTROPY_GRID_PX
    cells = (
        density[:cropped_h, :cropped_w]
        .reshape(cropped_h // ENTROPY_GRID_PX, ENTROPY_GRID_PX, cropped_w // ENTROPY_GRID_PX, ENTROPY_GRID_PX)
        .sum(axis=(1, 3))
        .ravel()
    )
    cells = cells / cells.sum()
    nonzero = cells[cells > 0]
    return float(-(nonzero * np.log(nonzero)).sum() / np.log(cells.size))


def top_k_share(density: Density, fraction: float = TOP_PIXEL_FRACTION) -> float:
    """Return the share (0..1) of attention falling in the `fraction` most-attended pixels."""
    flat = np.sort(density.ravel())[::-1]
    k = max(1, int(len(flat) * fraction))
    return float(flat[:k].sum() / flat.sum())


def _masked_peak(density: Density, mask: Mask) -> float:
    """Return the scaled maximum density inside `mask`, or 0 for an empty mask."""
    return PEAK_SCALE * density[mask].max() if mask.any() else 0.0


def _aoi_rows(key: str, aois: dict[str, Aoi], density: Density) -> tuple[list[AoiStats], Mask]:
    """Return one row per AOI, ranked by peak, plus the union mask of all AOIs."""
    height, width = density.shape
    union = np.zeros((height, width), bool)
    rows = []
    for name, aoi in aois.items():
        mask = aoi_mask(aoi.boxes, height, width)
        union |= mask
        share_pct, area_pct = 100 * density[mask].sum(), 100 * mask.mean()
        rows.append(
            AoiStats(
                post=key,
                aoi=name,
                type=aoi.type,
                share_pct=share_pct,
                area_pct=area_pct,
                peak=_masked_peak(density, mask),
                lift=share_pct / area_pct if area_pct else 0.0,
            )
        )
    # Stable sort: AOIs with equal peaks keep their aois.json order.
    for rank, row in enumerate(sorted(rows, key=lambda row: -row.peak), 1):
        row.first_look_rank = rank
    return rows, union


def _rest_row(key: str, density: Density, union: Mask) -> AoiStats:
    """Return the row for every pixel outside all AOIs."""
    rest = ~union
    return AoiStats(
        post=key,
        aoi=REST_AOI_NAME,
        type=AoiType.OTHER,
        share_pct=100 * density[rest].sum(),
        area_pct=100 * rest.mean(),
        peak=_masked_peak(density, rest),
        lift=(density[rest].sum() / rest.mean()) if rest.any() else 0.0,
    )


def _type_share(spec: PostSpec, aoi_type: AoiType, density: Density) -> float | None:
    """Return the % attention on all AOIs of `aoi_type`, or `None` when the post has none."""
    height, width = density.shape
    masks = [aoi_mask(aoi.boxes, height, width) for aoi in spec.aois.values() if aoi.type == aoi_type]
    if not masks:
        return None
    return 100 * density[np.logical_or.reduce(masks)].sum()


def compute_stats(key: str, spec: PostSpec, density: Density) -> tuple[list[AoiStats], PostSummary]:
    """Return per-AOI rows (rest of image last) and the post summary."""
    aoi_rows, union = _aoi_rows(key, spec.aois, density)
    entropy = entropy_norm(density)
    shares = {aoi_type: _type_share(spec, aoi_type, density) for aoi_type in SUMMARY_TYPES}
    summary = PostSummary(
        post=key,
        label=spec.label if spec.label is not None else key,
        entropy_norm=entropy,
        clarity=1 - entropy,
        top10_pct=100 * top_k_share(density),
        logo_share_pct=shares[AoiType.LOGO],
        cta_share_pct=shares[AoiType.CTA],
        offer_share_pct=shares[AoiType.OFFER],
        face_share_pct=shares[AoiType.FACE],
        headline_share_pct=shares[AoiType.HEADLINE],
        first_look_order=" > ".join(row.aoi for row in sorted(aoi_rows, key=lambda row: row.first_look_rank or 0)),
    )
    return [*aoi_rows, _rest_row(key, density, union)], summary
