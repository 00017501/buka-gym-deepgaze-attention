# AI use and prompts

This file records where generative AI was used in the attention analysis, and the
prompts involved, so that readers can replicate or audit each step. The attention
model itself (DeepGaze IIE) is a published, deterministic neural network and is
described in [METHOD.md](METHOD.md). This file covers the **assistant** AI:
Claude (Anthropic), used through Claude Code.

## Summary of AI use

| Step | What the AI did | Human check |
|---|---|---|
| AOI definition | Proposed the AOI rectangles (pixel coordinates and types) for the 5 posts by looking at each image, and wrote `aois.json` | The boxes were checked on the overlay images. Two overlapping boxes were corrected after the first run on the researcher's instruction ([DECISIONS.md](DECISIONS.md) D8) |
| Analysis code | Wrote the original single-file pipeline (statistics, heatmaps, overlays) and the smoke test | Tested on synthetic data before use. The metric definitions are documented in METHOD.md |
| Code restructure review | Reviewed the researcher's restructure into `src/buka_attention/` (D13): checked that every metric, the overlay settings, the centre-bias handling and the CLI flags are unchanged, ran ruff and the tests, and re-ran the full model to confirm byte-identical outputs (D15) | The restructure itself was directed by the researcher, with Claude Code assistance (see the commit co-author lines) |
| Environment setup | Installed DeepGaze IIE on the researcher's laptop, made the compatibility patch, ran the model | The smoke test checked the output (shape, sum to 1, off-centre peak) |
| Documentation | Drafted README, METHOD.md, DECISIONS.md and this file | Reviewed by the researcher and approved for publication |

## 1. AOI definition

### 1a. The instruction actually given (31 Oct 2026)

The AOIs were made as part of this task instruction:

> For each post, define AOIs (CTA, logo, price/offer, face, headline, where each is present) and write a stats script that computes the AOI % attention share, the first-look order (ranked by peak attention), the map entropy/clarity, and the logo share.

The assistant viewed each original image at full resolution (900 px wide) and
estimated rectangles in original-image pixel coordinates by eye, without a grid
or measuring tool. It then drew the boxes over the images to check them, before
the model was run. It added two more types, `body` and `visual`, for the large
text blocks and product images, so that these would not be hidden inside
"rest of image".

### 1b. Reusable prompt (to replicate the annotation)

Give a multimodal AI model **one post image** and this prompt. Then paste its
output into `aois.json` under a new key.

```
You are annotating areas of interest (AOIs) on a social-media marketing image for an
attention study. The image is {WIDTH} x {HEIGHT} pixels.

Find each of these elements if present, and give a tight bounding rectangle for each
in ORIGINAL image pixel coordinates [x0, y0, x1, y1] (x0,y0 = top-left; x1,y1 = bottom-right):
- logo: the brand logo or wordmark (include a tagline only if it is part of the logo lock-up)
- headline: the main title text
- face: each human face or head (one rectangle per face, grouped under one AOI)
- offer: the value proposition / price / promotion text (e.g. "free trial", "special offer")
- cta: the call to action (e.g. "details in description", "book now", a button or arrow)
- body: supporting text blocks (benefit lists, event descriptions)
- visual: the main product or equipment image, if it is a distinct object

Rules:
1. Omit any element that is not in the image. Do not invent one.
2. Rectangles must not overlap. If two elements touch, split the boundary between them.
3. Include a text element's own background panel or banner, but not empty space around it.
4. Name each AOI with a short lowercase name and give its type from the list above.
5. Output ONLY valid JSON in this format, with no commentary:

{"aois": {
   "<name>": {"type": "<type>", "boxes": [[x0, y0, x1, y1], ...]},
   ...
}}
```

**Replication note.** AI annotation is not deterministic: a different model, or
the same model on another day, will give slightly different coordinates. That is
why the **exact boxes used are published** in `aois.json`, with the pre-fix
version in `docs/aois_v0_before_overlap_fix.json`. Those files, not the prompt,
are what reproduce the numbers. Anyone can redraw the boxes with another tool (or
by hand) and re-run the statistics with `--skip-model` to test how sensitive the
results are to AOI placement.

### 1c. Scripted annotation (new posts and redesigns)

`python -m src.annotate run <image>` sends the 1b prompt to Claude (Anthropic)
through the API, with `pydantic-ai` native structured output.

- **Prompt:** loaded from `prompts/aoi_annotation.txt`. A test keeps that file
  identical to 1b, and `{WIDTH}`/`{HEIGHT}` are filled with the size actually sent.
- **Defaults:** model `claude-opus-5-5`, effort `high`.
- **Output:** the proposal and an audit record (model used, hashes, request ID,
  tokens, raw reply) in `aois_generated/`.
- **Policy:** the five published posts keep the 1a boxes (D17).

Commands: [REDESIGN.md §4–6](REDESIGN.md#4-propose-aois).

## 2. Analysis code

The original pipeline script was written from the instruction in 1a, plus:

> Make the script reusable for the AI redesigns.

It was later restructured, by the researcher with Claude Code assistance, into the package `src/buka_attention/` without changing any output (D13, D15). All numbers in the report come from running this code. No figures were typed in
or edited by an AI.

## 3. Instructions given to the local assistant (Claude Code)

These instructions drove the setup, the run and the publication of this
repository, in order. They are lightly shortened and personal names are removed.

1. **Setup:** "Set up DeepGaze IIE locally and confirm it runs. Report OS,
   Python version, and whether a GPU is present. Create a project folder with a
   Python virtual environment. Install PyTorch + torchvision, scipy, numpy,
   pandas, pillow, opencv-python, matplotlib, and DeepGaze from
   https://github.com/matthias-k/DeepGaze. Download the centre-bias template
   (centerbias_mit1003.npy) and run a smoke test: load
   `deepgaze_pytorch.DeepGazeIIE(pretrained=True)`, run it on any test image, and
   confirm a log-density map is produced."
2. **Run:** "Read buka_attention.py first. Then run
   `python buka_attention.py --images images --aois aois.json --out output --centerbias centerbias_mit1003.npy`.
   Report the full contents of output/stats.md and any errors. Look at each
   output/heatmaps/*_aoi.png and say whether any AOI box clearly misses its
   element. Do not edit aois.json yourself."
3. **AOI overlap fix:** "Apply both overlap fixes in aois.json: post2 headline
   boxes → [[55, 90, 570, 520]]; post3 equipment boxes → [[425, 660, 900, 1080]].
   Re-run with --skip-model. Check that every post's AOI shares plus '(rest of
   image)' now sum to 100%."
4. **Documentation:** "We need to document and explain the steps of how we did
   it in the report, create a GitHub repository where we store the outcomes, and
   document the decisions we made."
5. **Heatmap legibility and decision log:** "In heat_overlay(): alpha 0.8, gamma
   0.4, base brightness 0.6, alpha-mask gain 2.0. Re-run with --skip-model and
   check the stats stay byte-identical. Add decision-log entries (history,
   rejected commercial tools, AOIs defined before the model ran, brighter
   heatmaps, no personal names). Add an MIT licence and a note that the post
   images belong to Buka Gym."
6. **Publish:** "Create a proper commit and push it to the main branch."
7. **AI disclosure:** "Provide the prompts that we used so readers can replicate
   our case."
8. **Restructure check:** "My changes are done. Now please check it out. Verify
   if everything is alright and according to our plans."

## 4. AI-generated redesigns (Step 2)

Variant B of post 2 (offer and CTA made most salient) was made with
`python -m src.redesign generate` (OpenAI image edits) from
`prompts/post2_v1.txt`, and accepted by checks fixed in advance. Procedure:
[REDESIGN.md](REDESIGN.md).

The run (31 Oct 2026) went as follows:

- **Attempt 1:** model `gpt-image-2.5-sunburst-2026-09-08`, quality `high`,
  1024×1536, request `req_8951c45d948d4224835fc41d84962804`. It passed every
  check (`redesigns/post2_v1_attempt1_check.jpg`) and was used. No further
  attempts were made.
- **Fit to the post size:** cropped 172 px from the bottom and resized to
  900×1199 (`redesigns/post2_v1_final.json`, D18).
- **AOIs:** proposed by Claude with `python -m src.annotate run` (1c). The raw
  reply is in `aois_generated/post2_v1.ai_original.json`, and the reviewed
  corrections are listed in D19.
- **Result:** `results/post2_A_vs_B.jpg` and `results/compare.csv`.
