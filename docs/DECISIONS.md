# Decision log

Choices made while producing the results. Each entry lists the decision, the
reason, and the alternatives considered.

### D0 – History: from illustrative maps to a named model
- **25 Oct 2026:** The in-class presentation used saliency maps from OpenCV's
  spectral-residual method. They were meant only as illustrations and carried
  no statistics.
- **31 Oct 2026:** Tutor feedback asked for a named, citable attention model with
  statistics. The spectral-residual maps were therefore replaced by DeepGaze IIE
  (D1) with AOI statistics (D7). All entries below date from 31 Oct 2026.

### D0b – Commercial predictive tools rejected
- **Attention Insight** was considered and rejected. It is a black box with an
  unpublished method, its free trial limits the number of images, and its
  results cannot be reproduced by a reader.
- **Neurons Predict** was rejected for the same reason.

### D1 – Saliency model: DeepGaze IIE
- **Decision:** Use DeepGaze IIE with the official pretrained weights.
- **Why:** It is one of the top-ranked free-viewing models on the MIT/Tübingen
  saliency benchmark. It outputs a *calibrated probability density*, so "x% of
  attention on the CTA" is a meaningful quantity. The code and weights are open,
  which makes the work reproducible.
- **Alternatives:** Commercial predictive-eye-tracking tools (e.g. attention-heatmap
  SaaS) are closed and not reproducible. Classic saliency (Itti–Koch) is less
  accurate. DeepGaze III models scanpaths but needs fixation history, which
  adds complexity a single-glance feed post doesn't need.

### D2 – Centre-bias prior: MIT1003
- **Decision:** Use the MIT1003 centre-bias log-density recommended by the
  DeepGaze authors.
- **Why:** This prior matches how the model was evaluated. People tend to look
  near the centre of a frame, and the prior encodes that.
- **Alternative:** A uniform prior (`np.zeros`), which removes the centre
  tendency. We didn't use it as the main setting because the model was not
  calibrated that way.

### D3 – Runtime: local CPU, Python 3.13
- **Decision:** Run locally on a MacBook (Apple M4 Pro) on CPU, in a Python 3.13
  virtual environment.
- **Why:** With five images, inference takes about 19 s, so a GPU is unnecessary.
  The machine's default Python 3.14 was skipped to avoid gaps in package support.
  Running locally keeps the images and results on our own machine.

### D4 – Installing DeepGaze from a reviewed, pinned clone
- **Decision:** Clone the repository, read the source, and install from the local
  copy at commit `c7db17e` instead of `pip install git+https://...`.
- **Why:** This lets us check what the code does before running it, and it pins
  the exact version for reproducibility.

### D5 – Make the DeepGaze MSDB import optional (patch)
- **Decision:** In `deepgaze_pytorch/__init__.py`, wrap the import of the newer
  MSDB model in `try/except ImportError` (`patches/deepgaze_optional_msdb_import.patch`).
- **Why:** The package imports MSDB at start-up, and MSDB needs OpenAI's CLIP
  library. DeepGaze IIE doesn't use CLIP. With the patch, IIE loads without
  installing an unrelated dependency.
- **Effect on results:** None. The IIE code and weights are untouched.

### D6 – Validate the installation with a synthetic smoke test
- **Decision:** Before analysing real posts, run the model on a synthetic image
  with one salient object.
- **Why:** This confirms the pipeline produces a valid probability map (correct
  size, sums to 1) and that the peak lands on the obvious salient object rather
  than only the centre. Result: passed (`src/smoke_test.py`, METHOD.md §5).

### D7 – AOI scheme: hand-drawn rectangles with typed categories
- **Decision:** Mark each marketing element (logo, headline, face, offer, CTA,
  body text, visual) with rectangles in original pixel coordinates, and give each
  a type.
- **Why:** The typed categories map directly to the report's questions: does the
  logo/offer/CTA get seen? Can posts be compared by element type? Rectangles are
  simple to audit and to redraw.
- **Alternative:** Pixel-precise segmentation masks were rejected as unnecessary
  precision for flat graphic layouts.
- **Timing:** The AOIs were defined in `aois.json` *before* the model was first
  run, so the boxes were not fitted to the heatmaps. The only later change was the
  overlap fix (D8). It removed double counting and did not move any box towards a
  hotspot.

### D8 – Remove AOI overlaps
- **Decision:** Two boxes in the first AOI version overlapped. They were corrected:
  - Post 2 headline: `[55, 90, 645, 520]` → `[55, 90, 570, 520]`. It overlapped
    the coach's face box, so the face hotspot was double-counted.
  - Post 3 equipment: `[425, 590, 900, 1080]` → `[425, 660, 900, 1080]`. It
    overlapped the bottom-right of the headline box.
- **Why:** With overlaps, post 2's shares summed to about 105% and the face peak
  was credited to the headline as well. Non-overlapping AOIs make each share an
  exclusive part of 100%.
- **Effect:** Post 2's headline share went from 36.1% to 29.9%, and its first-look
  order changed from headline > face > … to face > benefits > headline > ….
  Post 3's equipment share went from 8.8% to 7.3%. Other posts were unchanged.
  The original file is kept as `docs/aois_v0_before_overlap_fix.json`.

### D9 – First-look order from peak density
- **Decision:** Rank AOIs by their highest density value as a proxy for what is
  likely to be looked at first.
- **Why:** This is simple and transparent with a single-image model. The
  limitation is recorded in METHOD.md §6. A scanpath model (DeepGaze III) would
  be the next step if the order became central to the argument.

### D10 – Keep images at native resolution, unedited
- **Decision:** Analyse the posts as exported (900 px wide), without cropping out
  small interface overlays.
- **Why:** Resizing or editing would change what the model sees. The overlays are
  small and are noted as a limitation.

### D11 – Heatmap images re-drawn brighter, same numbers
- **Decision:** The heatmap overlays were made brighter for legibility in the
  report. In `heat_overlay()` the opacity went from 0.6 to 0.8, the gamma from 0.5
  to 0.4, the background brightness from 0.55 to 0.6, and the alpha-mask gain
  from 1.6 to 2.0.
- **Why:** The first version was too dark to read in print.
- **Effect:** This changes the visual only. The images were regenerated from the
  same saved density maps (`--skip-model`), and `stats.md` and both CSVs are
  byte-identical to the previous run.

### D12 – No personal names in the repository
- **Decision:** The repository contains no personal names or local file paths.
  Authorship is the student ID 00017501.
- **Why:** The coursework is marked anonymously and identified by student ID
  only.

### D13 – Code restructured, results unchanged
- **Decision:** The pipeline script became the package `src/buka_attention/`. It
  now has type hints, validated AOI files (Pydantic), ruff linting and pytest tests.
  The smoke test (D6) is now `tests/test_smoke.py`.
- **Why:** The code is easier to check and extend, and the AOI file is validated on
  load. The methodology is unchanged.
- **Effect:** None on the numbers. A full rerun gives densities, CSVs, `stats.md`
  and heatmaps byte-identical to `results/` and to the previous script.

### D14 – Correction: AOIs were AI-proposed, not hand-drawn
- **Decision:** Record how the AOI boxes were actually made. D7 originally called
  them "hand-drawn rectangles". In fact an AI assistant proposed the coordinates by
  visual inspection of each post. The boxes were fixed before the model was run,
  checked against the overlays by the researcher, and corrected once (D8).
- **Why:** The module requires any AI-generated material, and the prompts behind it,
  to be disclosed. The instruction, a reusable annotation prompt and the
  replication notes are in [AI_USE_AND_PROMPTS.md](AI_USE_AND_PROMPTS.md).
  METHOD.md §3 is updated to match. D7 itself is left unedited, following the
  rule that log entries are never rewritten.
- **Effect:** None on the numbers. `aois.json` is unchanged.

### D15 – Restructure (D13) independently verified
- **Decision:** Before publishing D13, the restructured code was checked against
  the original script.
- **Evidence (31 Oct 2026):**
  - The code review found no change to the metric definitions (share, area, lift,
    first-look rank by peak, 10 px-grid entropy and clarity, top-10% share,
    per-type shares with n/a when absent, rest of image), the overlay settings (D11),
    the centre-bias handling, or the CLI flags (`--skip-model`, `--only`, `--compare`).
  - `ruff check` and `ruff format --check` were clean. `pytest`: 25 passed,
    including the DeepGaze smoke test.
  - A full model re-run on the five posts into a fresh folder gave
    `stats.md`, `stats_aoi.csv`, `stats_summary.csv` and all 10 heatmaps
    byte-identical to `results/`. All 5 density maps were byte-identical to the
    original script's run.
  - A `--skip-model` re-run was also identical, and `--only` with `--compare`
    produced `compare.csv` as expected.
- **Effect:** None on the numbers.

### D16 – Complete tables with `--only`, and a warning for boxes past the edge
- **Decision:** Two changes to support adding more posts (see ADD_A_POST.md):
  1. A run with `--only` still runs the model and draws heatmaps only for the named
     posts. The tables (`stats_*.csv`, `stats.md`) now list **every** post,
     recomputing the others from their saved density maps. A post with no saved
     density is left out with a `density_missing` warning. Before this change, the
     tables held only the posts named in that run.
  2. An AOI box that extends past the image edge now logs an
     `aoi_box_outside_image` warning. The box is still clipped to the image, as
     before.
- **Why:** With more posts and redesigns, single-post runs are common, and they
  should not leave partial tables. A box past the edge is almost always an
  annotation mistake, and before this change it was clipped silently.
- **Effect:** None on the numbers. The densities are the saved model outputs, so the
  other posts' rows are exact. A full run, and an `--only post3_trainer_rental`
  run, both give `stats.md` and both CSVs byte-identical to `results/`, and all 10
  heatmaps are identical. None of the existing boxes triggers the new warning.
  Tests: 29 passed, including new tests for both changes.

### D17 – Scripted AOI annotation; published AOIs stay fixed
- **Decision:** AOIs for new posts and redesigns are proposed by a script,
  `python -m src.annotate run`. It calls Claude (`claude-opus-5-5`, effort `high`)
  through the API with `pydantic-ai` and native structured output, using the
  1b prompt from `prompts/aoi_annotation.txt`. The proposal lands in
  `aois_generated/` with an audit record. A person reviews it, and `apply` then
  appends it to `aois.json`.
- **Policy:** The five published posts keep their existing AOIs, because their
  numbers are already published and those boxes were fixed before the model ran.
  `apply` refuses any key already in `aois.json`. A re-annotation of a published
  post is allowed only as a separate robustness check (`--out
  aois_generated/robustness/`) and never replaces the published boxes.
- **Details:**
  - Images with a long edge over 1568 px are downscaled before sending, so the
    model's coordinates match what it sees. Boxes are scaled back to original
    pixels, and the scale is logged.
  - Boxes are checked for overlaps, inverted boxes and boxes past the edge.
  - Sampling parameters are not sent, and tool use is not forced; both are
    rejected by this model.
  - The API key is read only from `ANTHROPIC_API_KEY` and is never logged or
    stored.
- **Why:** This replaces copy-pasting into a chat with a run that records the
  exact prompt, model, settings and raw reply, and keeps a human review step.
- **Effect:** None on existing results.

### D18 – Variant B of post 2: generation, acceptance and cropping
- **Decision:**
  - **Generation:** variant B (`post2_v1`) was generated with
    `python -m src.redesign generate`, from the prompt `prompts/post2_v1.txt`. The
    model was `gpt-image-2.5-sunburst-2026-09-08` (`gpt-image-1` had been shut
    down), quality `high`, size 1024×1536. The inputs were the original post 2 and
    a crop of the Buka Gym logo from post 4.
  - **Acceptance:** attempt 1 was checked against rules fixed before generation,
    and passed all of them. Every Cyrillic line is spelled exactly. The coach's
    identity, pose and uniform are unchanged; the model re-renders the photo, so
    it is not pixel-identical. The logo is not redrawn into something else, the
    benefits are cut to one line, and there is no extra text, people or
    watermark. Attempt 1 is therefore used. The evidence is in
    `redesigns/post2_v1_attempt1_check.jpg`, and no further attempts were made.
  - **Fitting to the post size:** the 2:3 output was cropped to 3:4 by removing
    172 px from the bottom only, then resized to 900×1199 (`--fit crop`,
    record `redesigns/post2_v1_final.json`).
- **Why crop rather than pad:** padding the sides would have shrunk every element
  to 78% and added dark bands, a size confound in the before/after comparison.
  Cropping keeps elements at 88% and changes nothing else in the layout. The
  padded version's record is kept as
  `redesigns/post2_v1_final_padded_superseded.json`.
- **Known deviation:** the removed strip contained the embroidered belt name
  "KARASEV", which the original post also shows. Variant B therefore lacks this
  small, incidental text. It is not marketing copy and not an AOI, and it sits in
  a low-attention area of the original. Check image:
  `redesigns/post2_v1_crop_strip_check.png`.
- **Timing:** the redesign prompt, the acceptance rules and the crop were all
  decided before any annotation or heatmap of variant B existed.

### D19 – AOIs for variant B: AI proposal, corrections, and the result
- **Proposal:** `python -m src.annotate run images/post2_v1.png` (model
  `claude-opus-5-5`, effort `high`, request `msg_011Cga95YuQ3Btdmd54YxUYu`). The
  unedited reply is kept as `aois_generated/post2_v1.ai_original.json`, with its
  record in `aois_generated/post2_v1.meta.json`. The checks flagged two overlaps.
- **Corrections** (made before any heatmap of variant B existed, by the D8 rule:
  the face keeps its full box, and text boxes are trimmed or split at the
  visible boundary):
  - `logo` `[48, 32, 206, 138]` → `[48, 32, 206, 140]`. It now covers the whole
    "Train and Win" line and ends where the headline starts.
  - `headline` first box `[48, 128, 648, 322]` → two boxes,
    `[48, 142, 648, 212]` (above the face) and `[48, 212, 574, 322]` (beside the
    face). This removes the 1,580 px overlap with the logo and the 8,140 px
    overlap with the face.
  - `age` `[48, 514, 420, 572]` (type `body`) merged into `headline`. In the
    published AOIs of the original post 2, "ДЛЯ ДЕТЕЙ ОТ 5 ЛЕТ" is part of the
    headline box, so it is kept the same way for a like-for-like comparison.
  - Names match the original post 2 (`headline`, `face`, `benefits`, `offer`,
    `cta`), plus `logo`, which the original post does not have.
- **Result** (`--only post2_taekwondo_kids post2_v1 --compare
  post2_taekwondo_kids:post2_v1`, shares sum to 100% for both posts):
  - offer 9.1% → 27.7%
  - CTA 4.8% → 12.6%
  - offer + CTA 13.9% → 40.3%
  - benefits 26.5% → 0.9%
  - headline 29.9% → 31.4%
  - face 9.7% → 10.9%
  - clarity 0.11 → 0.11, top-10% share 53.0 → 53.2
  - first look: face > benefits > headline > offer > cta → face > headline > cta
    > offer > logo > benefits
  - Both pre-set criteria are met: offer + CTA above 13.9%, and the offer
    ranked ahead of the benefits in first-look order.
- **Reading the result:** much of the offer's gain comes from its larger panel
  (its area grew from 3.9% to 13.0% of the image; lift 2.36 → 2.13). The CTA
  gained both area (3.1% → 5.4%) and pull per pixel (lift 1.57 → 2.35). The
  benefits block, which drew a quarter of the attention, is now a single line.
  These are predicted attention shares, not eye-tracking.

### D20 – Guide renamed
- **Decision:** `docs/ADD_A_POST.md` (named in D16) was renamed to
  `docs/ANALYSIS_GUIDE.md`, because it now covers redesigns as well as new posts.
  The content is unchanged; the details are in `AOI_GENERATION.md` and
  `REDESIGN.md`.

### D21 – Files reorganised: inputs, outputs, audit trail and doc images separated
- **Decision:** Research inputs, outputs, the audit trail and doc-only images now
  live in separate folders. Files were moved with `git mv`, so their history is
  kept. The code defaults (`--images data`, `--aois data/aois.json`, staging and
  attempt folders), the docs and the tests point to the new paths. Image paths in
  `data/aois.json` are now relative to `data/`.
- **Mapping (old → new):**
  - `images/*_original.png` → `data/posts/`
  - `images/post2_v1.png` → `data/redesigns/post2_v1.png`
  - `aois.json` → `data/aois.json`
  - `redesigns/` → `provenance/redesigns/`
  - `aois_generated/` → `provenance/aois/`
  - `docs/aois_v0_before_overlap_fix.json` → `provenance/aois/`
- **Deleted** (seven resized copies of research files; the docs now link the
  originals): `docs/img/guide/01_input.jpg`, `03_heatmap.jpg` and `04_aoi.jpg`,
  and `docs/img/redesign/a_vs_b.jpg`, `attempt1_check.jpg`, `crop_strip.jpg` and
  `final.jpg`. Nothing from the audit trail was deleted.
- **Not changed:** the records inside `provenance/` and earlier D entries keep
  the paths they had when written. Read them with the mapping above.
- **Effect:** None on the numbers. A full re-run gives `results/` byte-identical.

### D22 – Prompts sorted by purpose
- **Decision:** Prompt files are sorted into folders by purpose, so the names
  say what each prompt is for:
  - `prompts/post2_v1.txt` → `prompts/redesign/post2_v1.txt`
  - `prompts/aoi_annotation.txt` → `prompts/aoi/annotation.txt`
  - `prompts/README.md` indexes them; `post2_v1` is variant B.
- **Defaults:** `src.redesign generate` now defaults to
  `prompts/redesign/<name>.txt`. The post key `post2_v1` is unchanged.
- **Records:** the texts and their SHA-256 hashes are unchanged, so records in
  `provenance/` still match them, though they show the old paths.
