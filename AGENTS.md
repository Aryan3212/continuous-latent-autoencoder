# Agent Instructions

## Plans

- Follow the plan index in `/.plans/uncertainties_index.md` top-to-bottom.
- When an item is completed, update `UNCERTAINTIES.md` and add a minimal validation (unit test or smoke script).
- Keep plan artifacts in `/.plans/` as the source of step-by-step execution.
- Use `CODEBASE.md` as the quick reference; avoid rescanning the repo unless necessary.

## Current state (quick)

- See `CODEBASE.md` for the canonical snapshot of modules, entrypoints, and scripts.
- Issue log lives in `CHANGELOG.md`.

## Environment + verification

- Use the uv-managed environment: run `uv sync` before running scripts or tests.
- Run verification via uv, e.g. `uv run python scripts/smoke_encoder_mhc.py`.
- Prefer uv for any python invocation to ensure correct dependencies and interpreter.
