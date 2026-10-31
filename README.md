# Buka Gym – Predicted Visual Attention (DeepGaze IIE)

Supporting repository for the CW1 individual report (Neuromarketing & AI).
It contains the code, inputs, settings and outputs used to estimate where
viewers are predicted to look on five Buka Gym Instagram posts, so every
number and heatmap in the report can be traced and reproduced.

> **Caveat.** All attention values are *predicted* free-viewing attention from
> DeepGaze IIE (Linardos et al., 2021). They are not eye-tracking measurements
> of Buka Gym's audience and ignore viewing goals, scrolling, captions and the
> Instagram interface.

## Repository layout

| Path | What it is |
|---|---|
| `images/` | The five original posts analysed (`*_original.png`, native resolution) |
| `aois.json` | Areas of interest (AOIs) drawn on each post – final version used for the results |
| `src/buka_attention/` | Pipeline package: `cli` (entry point), `model` (DeepGaze IIE), `stats` (AOI statistics), `visuals` (heatmaps), `report` (tables), `schemas` (AOI file and output rows) |
| `tests/` | Unit tests, plus `test_smoke.py`: installation check on a synthetic image |
| `results/stats.md` | Summary and per-AOI tables (same data as the CSVs) |
| `results/stats_summary.csv`, `results/stats_aoi.csv` | Machine-readable results |
| `results/heatmaps/` | `<post>_heatmap.png` and `<post>_aoi.png` (heatmap + AOI boxes + % attention) |
| `docs/METHOD.md` | Step-by-step method, metric definitions, limitations |
| `docs/DECISIONS.md` | Decision log: what we chose, why, and the alternatives |
| `docs/aois_v0_before_overlap_fix.json` | First AOI version, kept for the audit trail |
| `patches/` | One-line compatibility patch applied to the DeepGaze package |
| `requirements.txt`, `setup.sh` | Exact environment used |
| `requirements-dev.txt`, `ruff.toml`, `pytest.ini` | Lint and test tooling (not needed to reproduce) |

## Results at a glance

| Post | Logo % | Headline % | Face % | Offer % | CTA % | Clarity (0-1) | First-look order (by peak) |
|---|---|---|---|---|---|---|---|
| Brand flag (no offer) | 43.7 | 27.4 | 14.2 | n/a | n/a | 0.19 | tagline > logo > people |
| Taekwondo Kids free trial | n/a | 29.9 | 9.7 | 9.1 | 4.8 | 0.11 | face > benefits > headline > offer > cta |
| Trainer space rental | 0.9 | 52.6 | n/a | 21.6 | 1.5 | 0.12 | headline > offer > equipment > logo > cta |
| Boxing open ring | 2.6 | 21.8 | 15.9 | n/a | 0.6 | 0.13 | faces > body > logo > headline > cta |
| Membership special offer | 7.0 | 27.0 | n/a | 32.1 | 16.2 | 0.14 | logo > offer > headline > cta > dumbbell |

Full tables: [`results/stats.md`](results/stats.md). Metric definitions: [`docs/METHOD.md`](docs/METHOD.md#4-metrics).

## Reproduce

Tested on macOS (Apple M4 Pro, CPU inference), Python 3.13.

```bash
./setup.sh                                   # venv, pinned packages, DeepGaze at a fixed commit, centre-bias file
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest -m slow                     # installation check on a synthetic image
.venv/bin/python -m src.buka_attention --images images --aois aois.json \
    --out output --centerbias centerbias_mit1003.npy
```

The first run downloads about 1–2 GB of pretrained weights into `~/.cache/torch/hub`.
Outputs land in `output/`; compare with `results/`.

### Other runs

```bash
# Only some posts (e.g. a redesign added to aois.json later)
.venv/bin/python -m src.buka_attention --only post2_taekwondo_kids post2_v1

# Recompute stats and overlays after editing AOI boxes, without re-running the model
.venv/bin/python -m src.buka_attention --skip-model

# Before/after table for redesigns (pairs of keys from aois.json); writes compare.csv
.venv/bin/python -m src.buka_attention --skip-model --compare post2_taekwondo_kids:post2_v1
```

Tables hold only the posts processed in that run, so include both keys of a pair
when combining `--only` with `--compare`. `--help` lists every flag.

| Output (in `--out`) | Content |
|---|---|
| `density/<post>.npy` | Fixation density (H × W, sums to 1) |
| `heatmaps/<post>_heatmap.png`, `heatmaps/<post>_aoi.png` | Heatmap, and heatmap + AOI boxes with % attention |
| `stats_aoi.csv` | One row per post × AOI |
| `stats_summary.csv` | One row per post (entropy, clarity, per-type shares, first-look order) |
| `stats.md` | The same tables in Markdown |
| `compare.csv` | Only with `--compare` |

## Development

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/pytest -m "not slow"               # unit tests, no model download
.venv/bin/pytest                             # everything, including the DeepGaze smoke test
```

Code conventions: [`.claude/rules/python.md`](.claude/rules/python.md).

## Licence

Code and documentation: MIT (see [`LICENSE`](LICENSE)).
**The post images in `images/` and the heatmaps derived from them in
`results/heatmaps/` belong to Buka Gym. The MIT licence does not cover them.**
They are included only so the analysis can be checked and reproduced.

## References

- Linardos, A., Kümmerer, M., Press, O., & Bethge, M. (2021). DeepGaze IIE: Calibrated prediction in and out-of-domain for state-of-the-art saliency modeling. *ICCV 2021*. arXiv:2105.12441
- DeepGaze code and weights: https://github.com/matthias-k/DeepGaze (commit `c7db17e`)
- Judd, T., Ehinger, K., Durand, F., & Torralba, A. (2009). Learning to predict where humans look. *ICCV 2009* (MIT1003 dataset, source of the centre-bias prior)
