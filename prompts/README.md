# Prompts

- `redesign/<key>.txt`: image-edit prompt for `python -m src.redesign generate --name <key>`. `post2_v1` is **variant B** of post 2 in the report (D18).
- `aoi/annotation.txt`: AOI annotation prompt for `python -m src.annotate run`. It is identical to [AI_USE_AND_PROMPTS.md §1b](../docs/AI_USE_AND_PROMPTS.md#1b-reusable-prompt-to-replicate-the-annotation), and a test enforces this.
- Records in `provenance/` keep the prompt paths used at the time (D22).
