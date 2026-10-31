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
