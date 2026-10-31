# Method

How the predicted-attention results for the five Buka Gym posts were produced.
Each step lists what was done and where the evidence is in this repository.
The reasons behind each choice are in [DECISIONS.md](DECISIONS.md).

## 1. Stimuli

Five Buka Gym Instagram posts, saved as PNG at their native export resolution
(900 px wide) in `images/`:

| Key | Post | Size (w × h) |
|---|---|---|
| `post1_brand_flag` | Brand flag image, no offer | 900 × 1199 |
| `post2_taekwondo_kids` | Taekwondo Kids free trial | 900 × 1199 |
| `post3_trainer_rental` | Trainer space rental | 900 × 1185 |
| `post4_boxing_open_ring` | Boxing open ring event | 900 × 1189 |
| `post5_special_offer` | Membership special offer (story format) | 900 × 1605 |

Images were not resized or cropped. Small interface elements that were part of
the captured screenshots (a carousel arrow on post 1, an account tag on post 2,
a mute icon on post 5) were left in place (see Limitations).

## 2. Model

- **DeepGaze IIE** (Linardos et al., 2021), pretrained weights from the official
  repository, loaded with `deepgaze_pytorch.DeepGazeIIE(pretrained=True)`.
  It is an ensemble of four ImageNet backbones (ShapeNet-ResNet50, EfficientNet-B5,
  DenseNet-201, ResNeXt-50) with a readout trained on human free-viewing fixations.
- **Output**: a log fixation density over the image. After exponentiating,
  every pixel holds the probability that a free-viewing fixation lands there,
  and the map sums to 1. That's why AOI attention shares can be read directly as
  percentages.
- **Centre-bias prior**: `centerbias_mit1003.npy` (1024 × 1024 log-density from
  the MIT1003 eye-tracking dataset), resized to each image with nearest-neighbour
  interpolation and renormalised (log-sum-exp = 0), as in the official example.
- **Inference**: CPU, batch of one image, no test-time augmentation.

## 3. Areas of interest (AOIs)

AOIs were drawn by hand as rectangles in original-image pixel coordinates
(`[x0, y0, x1, y1]`) and stored in `aois.json`. Each AOI has a type:
`logo`, `headline`, `face`, `offer`, `cta`, `body` or `visual`. One AOI can have
several boxes (e.g. the three heads in post 4). Pixels outside all AOIs form
"(rest of image)".

Quality check: every `results/heatmaps/<post>_aoi.png` was inspected to confirm
each box covers its element. The check also confirmed that for every post, the AOI
shares plus "(rest of image)" sum to 100%, so no pixel is counted twice.
The first AOI version had two overlaps; it was corrected before the final run
(DECISIONS.md, D8).

## 4. Metrics

Computed in `src/buka_attention.py` from the density map `d` (sums to 1):

| Metric | Definition | Reading |
|---|---|---|
| `share_pct` | 100 × Σ d inside the AOI | % of predicted attention on the element |
| `area_pct` | 100 × AOI pixels / image pixels | how big the element is |
| `lift` | share_pct / area_pct | > 1: the element attracts more than its size alone would predict |
| `peak` | max d inside the AOI (×10⁶) | intensity of the strongest hotspot |
| `first_look_rank` | AOIs ranked by `peak` | proxy for what is likely fixated first |
| `entropy_norm` | Shannon entropy of d pooled on a 10 × 10 px grid ÷ log(number of cells) | 0–1, lower = more focused |
| `clarity` | 1 − entropy_norm | higher = attention concentrated on fewer places |
| `top10_pct` | % of attention in the 10% most-attended pixels | second focus measure |
| `<type>_share_pct` | share of the union of all AOIs of that type | e.g. `cta_share_pct` |

## 5. Procedure (as run)

1. Created a Python 3.13 virtual environment and installed pinned packages
   (`requirements.txt`).
2. Cloned DeepGaze at commit `c7db17e`, reviewed the source (the only network
   access is downloading pretrained weights from official release URLs),
   applied `patches/deepgaze_optional_msdb_import.patch`, and installed it from
   the local clone.
3. Downloaded the MIT1003 centre-bias file and verified it (1024 × 1024, float64).
4. Ran `src/smoke_test.py`: a synthetic grey image with one red square. Pass
   criteria: the density matches the image size, sums to 1.0, and peaks inside the
   square. Result: peak at (y = 260, x = 716), inside the square → passed.
5. Ran the full pipeline on the five posts (`buka_attention.py`, ~19 s on CPU).
6. Inspected the AOI overlays, corrected two overlapping boxes, and re-computed
   the statistics with `--skip-model` (same density maps, new AOIs).
7. Exported the tables (`results/stats*.csv`, `results/stats.md`) and heatmaps.

## 6. Limitations

- **Predicted, not measured.** DeepGaze IIE is trained on lab free-viewing of
  natural images. It does not model the audience's goals, feed-scrolling speed,
  captions, or competition from neighbouring posts.
- **First-look order is a proxy.** The order is ranked by peak density, not by a
  sequential scanpath model (e.g. DeepGaze III).
- **AOIs are subjective.** Box placement affects shares. To mitigate this,
  `aois.json` and the earlier version are published so readers can redraw boxes and
  re-run with `--skip-model`.
- **Text is not "read".** The model responds to text as a visual feature
  (contrast, size, position), not to its meaning or language (Russian/Cyrillic).
- **Small sample.** Five posts, one model run each. The model is deterministic, so
  there is no sampling error to test, but the results describe these posts and
  should not be generalised statistically.
- **Interface artefacts.** Small UI overlays left in the screenshots may attract a
  little predicted attention.
