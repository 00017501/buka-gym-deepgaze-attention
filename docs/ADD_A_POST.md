# Analysing a new post (or a redesign)

A step-by-step guide to adding another image and getting the same attention
statistics as the five original posts. Run every command from the repository root,
after `./setup.sh` (see the README).

> **Rules**
> - Fix the AOI boxes **before** you look at the heatmap. Afterwards, correct them
>   only for clear placement errors (a box missing its element, or an overlap), and
>   record the correction.
> - Never hand-edit numbers or images in `output/` or `results/`. Regenerate them.
> - The output is **predicted** free-viewing attention (DeepGaze IIE), not
>   eye-tracking.

In the examples below, `post6_class_schedule` is a new post and `post2_v1` is a
redesign of post 2. Replace them with your own names.

## 1. Prepare the image

- Export or screenshot the post at native resolution, about 900 px wide like the
  others. Do not upscale.
- Crop away the Instagram interface (header, buttons, caption) so only the post
  itself remains. If a small overlay can't be removed, leave it and note it (see
  D10).
- Save it as PNG in `images/`:
  - new post: `images/<postN>_<short_name>_original.png`, e.g. `post6_class_schedule_original.png`
  - redesign: `images/<postN>_v<k>.png`, e.g. `post2_v1.png`

## 2. Get the image size

```bash
.venv/bin/python -c "from PIL import Image; print(Image.open('images/post6_class_schedule_original.png').size)"
```

This prints `(width, height)`.

## 3. Generate the AOIs with the AI prompt

1. Copy the prompt from [AI_USE_AND_PROMPTS.md §1b](AI_USE_AND_PROMPTS.md#1b-reusable-prompt-to-replicate-the-annotation)
   and replace `{WIDTH}` and `{HEIGHT}` with the numbers from step 2.
2. Give the prompt and the image to a multimodal AI assistant (one image per
   request).
3. Keep a record for the AI log: the tool and model name, the date, the exact
   prompt, and the raw reply.

The reply is JSON like this:

```json
{"aois": {
   "headline": {"type": "headline", "boxes": [[60, 120, 840, 380]]},
   "cta":      {"type": "cta",      "boxes": [[150, 1040, 750, 1120]]}
}}
```

## 4. Add the entry to `aois.json`

Add a new top-level key: the post name without `_original`. Its value holds the
image file name, a short display label, and the `aois` object from the reply:

```json
"post6_class_schedule": {
  "image": "post6_class_schedule_original.png",
  "label": "Class schedule",
  "aois": {
    "headline": {"type": "headline", "boxes": [[60, 120, 840, 380]]},
    "cta":      {"type": "cta",      "boxes": [[150, 1040, 750, 1120]]}
  }
}
```

- Allowed types: `logo`, `headline`, `face`, `offer`, `cta`, `body`, `visual`.
- Boxes are `[x0, y0, x1, y1]` integer pixels in the original image (top-left, bottom-right).
- An AOI can have several boxes, e.g. one per face.

Validate the file:

```bash
.venv/bin/python -c "from pathlib import Path; from src.buka_attention.schemas import load_specs; print(list(load_specs(Path('aois.json'))))"
```

It prints all post keys. A typo, such as an unknown type, a missing field or a
non-integer coordinate, stops with a `ValidationError` naming the problem.

## 5. Run the model and check the boxes

```bash
.venv/bin/python -m src.buka_attention --only post6_class_schedule
```

- The first run downloads the model weights, about 1–2 GB.
- `--only` rewrites `output/stats*.csv` and `output/stats.md` with **only** the
  posts named in this run. `results/` is not touched.

Open `output/heatmaps/post6_class_schedule_aoi.png`, and check that each box
covers its element and that no boxes overlap. Overlaps show up as the shares
adding to more than 100%:

```bash
.venv/bin/python -c "import csv; print(round(sum(float(r['share_pct']) for r in csv.DictReader(open('output/stats_aoi.csv')) if r['post']=='post6_class_schedule'), 2))"
```

The result must be `100.0`. If a box is wrong, fix it in `aois.json` and
recompute without re-running the model:

```bash
.venv/bin/python -m src.buka_attention --only post6_class_schedule --skip-model
```

Record every correction (old box → new box, and why), as was done for D8.

## 6. Redesigns: compare with the original

Name both keys in `--only`:

```bash
.venv/bin/python -m src.buka_attention --only post2_taekwondo_kids post2_v1 \
    --skip-model --compare post2_taekwondo_kids:post2_v1
```

`--skip-model` needs the saved densities of both posts in `output/density/`. Drop
it the first time a post is run. The before/after table goes to
`output/compare.csv` and the end of `output/stats.md`.

## 7. Update the published results

To refresh `results/` with every post, including the new one, run everything:

```bash
.venv/bin/python -m src.buka_attention --images images --aois aois.json \
    --out output --centerbias centerbias_mit1003.npy
```

Check that the original posts' rows in `output/` are unchanged against
`results/`. Then copy `output/stats*.csv`, `output/stats.md` (and
`output/compare.csv` if used) and `output/heatmaps/*.png` into `results/`.

## 8. Document it

- [AI_USE_AND_PROMPTS.md](AI_USE_AND_PROMPTS.md): add the AOI record from step 3
  (tool, model, date, prompt, raw reply). For a redesign, also add the
  image-generation tool and the exact prompt for each variant to §4.
- [DECISIONS.md](DECISIONS.md): add a new `Dn` entry for anything decided
  (a new post, AOI corrections, a redesign brief). Never edit an existing entry.
- `docs/METHOD.md`: add the post to the stimuli table in §1.

## 9. Commit and push checklist

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/pytest
grep -rniE "<your name>|/Users/|/home/" $(git ls-files) || echo "no names or local paths"
git config user.name && git config user.email   # must be the student ID and its GitHub noreply address
git add images/ aois.json results/ docs/ && git commit -m "Add <post key>: AOIs and results"
git push origin main
```

`output/` and the `.npy` density files are git-ignored on purpose. Commit only
the copies in `results/`.
