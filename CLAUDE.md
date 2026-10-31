# CLAUDE.md

Entrypoint for Claude Code in this repo. Code conventions live in [`.claude/rules/python.md`](.claude/rules/python.md).

---

## Project

Supporting repository for a coursework report: predicted visual attention (DeepGaze IIE) and AOI
statistics on five Buka Gym Instagram posts. Every number and heatmap in the report must be traceable
to and reproducible from this repo. Values are *predicted* attention, not eye-tracking.

## Tech stack

- Python 3.13, CPU inference, PyTorch + DeepGaze IIE (pinned commit, installed by `setup.sh`)
- numpy, PIL, Pydantic v2
- Exact versions: [`requirements.txt`](requirements.txt); lint/test tools: [`requirements-dev.txt`](requirements-dev.txt)

## Commands

```bash
./setup.sh                                    # venv, pinned packages, DeepGaze, centre-bias file
.venv/bin/pip install -r requirements-dev.txt # ruff + pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/pytest -m "not slow"                # unit tests (seconds, no model)
.venv/bin/pytest                              # + DeepGaze smoke test
.venv/bin/python -m src.buka_attention --images data --aois data/aois.json --out output --centerbias centerbias_mit1003.npy
```

## Layout

```
.
├── src/buka_attention/
│   ├── cli.py        # argparse + run() orchestration, logging setup
│   ├── schemas.py    # Pydantic models: aois.json (PostSpec/Aoi) and output rows
│   ├── model.py      # centre-bias prior, DeepGazeRunner
│   ├── stats.py      # AOI masks, shares, entropy, first-look ranks
│   ├── visuals.py    # heatmap + AOI overlays (numpy/PIL)
│   └── report.py     # CSV, Markdown tables, before/after comparison
├── src/annotate/     # Claude AOI proposals (run) and adding reviewed entries (apply)
├── src/redesign/     # OpenAI image-edit redesigns (generate) and fitting to post size (finalize)
├── tests/            # unit tests; test_smoke.py is marked `slow`
├── data/             # research inputs
│   ├── posts/        # the five original posts (Buka Gym's, not MIT-licensed)
│   ├── redesigns/    # finalised AI redesigns that are analysed
│   └── aois.json     # AOI boxes per image (final version)
├── prompts/          # exact AI prompts (redesign, AOI annotation)
├── provenance/       # audit trail, never deleted: AI attempts, raw replies, records, superseded files
├── results/          # committed outputs cited by the report
├── docs/             # METHOD, DECISIONS (D0..Dn), ANALYSIS_GUIDE, AOI_GENERATION, REDESIGN, AI_USE_AND_PROMPTS
│   └── img/          # images made only for the docs
└── patches/          # patch applied to the DeepGaze clone
```

## Boundaries

### ✅ Always

- Read [`.claude/rules/python.md`](.claude/rules/python.md) before non-trivial changes
- After any change to `src/`, rerun the full pipeline and diff `output/` against `results/` - a refactor must be byte-identical
- Record methodology or output-affecting changes as a new `Dn` entry in `docs/DECISIONS.md`
- Keep `docs/METHOD.md` §4 in sync with what `stats.py` computes
- Run ruff and `pytest` before declaring work done

### ⚠️ Ask first

- Any change that alters numbers or images in `results/`
- New dependency, or changing a pinned version in `requirements.txt`
- Editing `setup.sh`, `patches/`, `data/aois.json`, or the DeepGaze commit
- Editing `ruff.toml` or this `CLAUDE.md`

### 🚫 Never

- Hand-edit files in `results/` - regenerate them
- Delete or edit anything in `provenance/`
- Edit or delete an existing `Dn` entry - add a new one
- Add personal names or local file paths anywhere (D12)
- Modify `data/posts/`; posts are analysed as exported (D10)
- Add `# noqa` without an inline reason
