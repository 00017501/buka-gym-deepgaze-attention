# Python Conventions

Ruff ([`ruff.toml`](../../ruff.toml)) enforces what it can; this file covers the rest.

---

## Naming

| Element              | Convention                      | Example                                   |
| -------------------- | ------------------------------- | ----------------------------------------- |
| Variables, functions | `snake_case`                    | `share_pct`, `compute_stats()`            |
| Classes, models      | `PascalCase`                    | `PostSpec`, `DeepGazeRunner`              |
| Constants            | `SCREAMING_SNAKE_CASE`          | `ENTROPY_GRID_PX`, `CSV_DECIMALS`         |
| Private              | Leading underscore              | `_aoi_rows()`, `_model`                   |
| Type aliases         | `PascalCase`, 3.12+ syntax      | `type Density = npt.NDArray[np.float64]`  |
| Booleans             | `is_`, `has_`, `should_` prefix | `has_both`, `is_salient`                  |
| Test functions       | `test_` + scenario              | `test_aoi_mask_clips_boxes_to_image`      |

- Describe purpose, not type: `density`, not `d` or `arr`. Collections are plural.
- Spell names out. Avoid `data`, `info`, `result`, `item`, `obj`, and single letters beyond loop indices and `x0, y0, x1, y1`.
- Functions: action + subject (`load_centerbias`, `build_markdown`). Not `handle`, `process`, `do`.

---

## Type Hints

Mandatory on every signature (params + return).

- Modern syntax: `list[X]`, `X | None`. Never `Optional`, `List`, `Union`.
- Arrays as `npt.NDArray[np.float64]` (or the `Density` / `Mask` aliases in `stats.py`).
- `TYPE_CHECKING` for import-only types, e.g. `torch` in `model.py`, which is imported lazily so
  `--skip-model` runs never load it.
- `Any` only at a genuinely dynamic edge, with a comment.

---

## Data shapes (Pydantic v2)

Anything crossing a module boundary is a typed model, never a raw dict.

- Input files (`aois.json`) are validated on load: `ConfigDict(frozen=True, extra="forbid")`.
- Fixed value sets are `StrEnum` (`AoiType`), never raw strings.
- Output rows are models whose field order is the CSV column order; write with `model_dump()`.
- `Field(...)` for constraints (`min_length=1` on boxes). No logic in validators - shape only.

---

## Functions and modules

- Function length target ≤30 lines, hard limit 50. Params target ≤3, hard limit 5.
- Guard clauses over nesting.
- Modules: hard limit 250 lines; split by concern into a sibling module.
- In a module that defines one class, a helper only that class uses is a private method.
- A constant lives with what owns it; keep only genuinely module-wide constants at the top.

---

## Numerical invariance

`results/` is cited by the report. A refactor must not change a single byte of output.

- Don't reorder floating-point operations, change dtypes (`float32` vs `float64`), resampling
  filters, or the CSV rounding - each can shift the last digit.
- Keep row order (`aois.json` order, rest-of-image last) and stable sorts.
- Verify by rerunning the full pipeline and `cmp`-ing every file in `output/` against `results/`.

---

## Errors

- Never bare `except` / `except Exception`. Catch specific exceptions (`OSError` for font lookup).
- Preserve the chain: `raise ... from err`.
- Never swallow silently - a handled fallback logs a warning (missing centre-bias file).
- Invalid input fails fast at load (Pydantic `ValidationError`), not halfway through a run.

---

## Logging

stdlib `logging`, one `logger = logging.getLogger(__name__)` per module; `cli.main` configures it.

```python
logger.info("post_started post=%s", key)
logger.warning("compare_pair_not_found pair=%s", pair)
```

- Event name first (`snake_case`), then `key=value` context via `%s` args - no f-strings, no sentences.
- No `print()` (Ruff `T20`).
- `INFO` for progress, `WARNING` for a handled problem a human should look at.

---

## Comments & Documentation

**Be brief.** Prose is the part of a file nobody updates, so every line of it is a future lie.

- Explain WHY, not WHAT. Code shows what.
- **One-line docstring is the default**, on public and private alike.
- **Inline comment blocks: 4 lines maximum.** Longer explanations go in `docs/METHOD.md` or a
  `docs/DECISIONS.md` entry.
- Every constant has a one-line why-comment.
- TODOs need a reason and an owner or issue: `# TODO(#12): ...`. Never bare `# TODO: fix later`.
- Section-header comments, when a file groups distinct concerns: `# ----- Title -----`, centered,
  one width per file.

### Docstring sections

Google style. `Args:` / `Returns:` / `Raises:` only for what the signature does **not** carry: a
unit, a range, a convention, a non-obvious raise.

```python
# Good - the signature already says everything.
def colormap(values: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Map values in 0..1 to RGB by linear interpolation between the colour stops."""


# Good - the one non-obvious fact, as prose.
def top_k_share(density: Density, fraction: float = TOP_PIXEL_FRACTION) -> float:
    """Return the share (0..1) of attention falling in the `fraction` most-attended pixels."""


# Bad - three sections, zero information.
def top_k_share(density: Density, fraction: float = 0.1) -> float:
    """Top k share.

    Args:
        density: The density.
        fraction: The fraction.

    Returns:
        The share.
    """
```

An `Args:` block is all-or-nothing (ruff `D417`). When one parameter needs a note, write a sentence.

### Orchestration docstrings

A function that runs **3+ steps** across **2+ collaborators** gets a numbered-step docstring and
matching `# N.` markers in the body (see `cli.run`).

- Each step line says what happens and, if non-obvious, why.
- Docstring steps and `# N.` markers match 1:1; one blank line before each marker (none before the first statement).
- A marker is a single line. Pure functions and single-collaborator helpers keep the one-liner.

### No provenance

Say what the code does and why. Never where it came from or what it used to be. Don't name other
repositories or projects in source, and don't write diff history (`# was a dict before`) - git
records that. Methodology history belongs in `docs/DECISIONS.md`.

---

## Imports

stdlib → third-party → local (Ruff `I`). Absolute imports from `src.buka_attention...`.
Explicit names, never `import *`. Aliases: `numpy as np`, `numpy.typing as npt`, `pandas as pd`.
`pathlib.Path` over `os.path` (Ruff `PTH`).

---

## Tests

- Pure logic gets unit tests on small synthetic arrays - no model, no network.
- Anything that runs DeepGaze is `@pytest.mark.slow` and skips cleanly when `setup.sh` hasn't run.
- Test names state the scenario; no docstrings needed.
