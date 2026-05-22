# Simplification Plan — Remaining Work

Most of the original plan was executed in a single subagent-driven pass on
branch `simplification` (one commit, off `main` so the Zipformer-era main is
preserved). What's below is **only** the tasks still open, plus caveats /
follow-ups produced during that pass.

For context on what was already done, see the git history on the
`simplification` branch and `.static-analysis/DIAGNOSIS.md`.

---

## Caveats from the simplification pass (act on these first)

### C1 — Old checkpoints will not load
The Zipformer → Conformer rewrite (`models/conformer.py` + rewritten
`models/encoder.py`) renames every encoder parameter. Any `.pt` saved against
`Zipformer2EncoderLayer` is incompatible. **Restart training from scratch on
this branch.** No state_dict shim is provided.

### C2 — Manifest paths are placeholders
`configs/exp0.yaml`, `configs/exp0_test.yaml`, etc. were rewritten to point at
JSONL manifests (`data/manifests/train.jsonl`, `val.jsonl`,
`asr_probe_train.jsonl`, `asr_probe_val.jsonl`). These are placeholder paths
— correct them to the real manifest locations on the training box before the
first run. `configs/exp0_20pct.yaml` / `configs/exp0_20pct_merged.yaml` were
left untouched because they already referenced JSONL paths.

### C3 — Two configs will fail the new pydantic schema
`utils/schema.py` validates configs at startup with `extra="forbid"` on every
nested model. Two configs contain fields the schema rejects:

- `configs/exp0_20pct_merged.yaml`:
  - `model.predictor.{kind, hidden_dim, n_layers, dropout}` — legacy
    predictor block; no Python reads it.
  - `loss.sigreg.{reduction, clip_value}` — orphan fields; no Python reads
    them.
  - The file also appears to be two configs concatenated (the
    `data` / `train` / `run` blocks reappear at the bottom). May not have
    been intended for direct use.
- `configs/exp0_test.yaml`:
  - Same `model.predictor` block.
  - Same `loss.sigreg.{reduction, clip_value}` orphans.

**Pick one** for each:
- delete the offending blocks from the YAML, or
- add `PredictorCfg` + the two `sigreg` extras to `utils/schema.py`.

`configs/exp0.yaml` (canonical) validates cleanly.

### C4 — `scripts/pack_webdataset.py` is now orphan
The WebDataset → JSONL switch (§3.2) means nothing consumes WebDataset shards
anymore. The shard-packing script still imports `webdataset` and is the only
remaining producer. Either delete the script or keep it as a one-off
utility — flag for explicit decision; it was left in place during the pass.

### C5 — Style nit in `train.py:455` (semicolon)
Two statements on one line:
```python
v_sig_f, _ = sigreg(fp, step=step); v_sig_f = v_sig_f / max(1, fp.size(0))
```
Carried over from the §1.5 `_validate_one` rewrite. Ruff flags it (`E702`).
Trivial split when convenient.

### C6 — `eval/run_probes.py:35` `proc` left unused
`subprocess.run(...)` assigned to `proc` and never read. The §2.3 subagent
left it deliberately so a future error handler can reach the process result;
revisit if you decide you want the cleaner form.

### C7 — `CODEBASE.md` was updated; `CHANGELOG.md` was not
`agents.md` says to update both on major changes. `CODEBASE.md` was patched
(Conformer reference, removed dead-file mentions). `CHANGELOG.md` was left to
you because the entries depend on how you want to frame the squashed history
of this branch.

---

## §3.8 (continued) — call-site migration of config access

The pydantic schema was added (`utils/schema.py`) and is enforced at startup
by `utils/config.py`, but `load_config` still returns `model.model_dump()` —
a plain dict. Every existing `cfg["a"]["b"]` access site in `train.py`,
`eval/`, `scripts/` continues to work unchanged.

The full migration to `cfg.a.b` attribute access (the upside: drops the
remaining `int(cfg["..."])` / `float(...)` / dict-key-filter ceremony in
`train.py` and the projector/sigreg construction sites) was deferred because
it touches hundreds of lines and needs Python in the loop to verify.

**Plan when ready:**
- Change `load_config` to return the `Config` pydantic model instead of
  `model.model_dump()`.
- Walk every `cfg["..."]` / `cfg.get("...")` site in `train.py`, `eval/`,
  `scripts/`, `eval/run_probes.py` and convert to attribute access.
- Drop the `int(...)`, `float(...)`, `bool(...)` casts at config-access
  sites — the schema already coerces types.
- Remove the dict-key filter `{k: v for k, v in proj_cfg_raw.items() if k in
  {...}}` (and the sigreg equivalent) in `train.py` — pydantic forbids
  extras on those nested models so the filter is redundant.
- Run training end-to-end after the migration; expect to chase a few
  `AttributeError` / `KeyError` mismatches that grep can't find.

This is a one-evening task with a running Python; do not attempt it without.

---

## §3.3 — MHC ablation (decision-deferred experiment)

MHC machinery (`models/mhc.py`, the wrapper plumbing in `models/encoder.py`,
the `model.encoder.mhc` config block) is **kept on this branch** but the
ablation decision is still open.

**Plan:**
- Run two training jobs side by side (same seed, same data):
  - MHC on:  `model.encoder.mhc.enabled=true`  (current config).
  - MHC off: `model.encoder.mhc.enabled=false` (or `num_streams=1`).
- Compare ASR-probe WER, JEPA loss curves, SIGReg variance.
- Decide:
  - **No clear win for MHC** → delete `models/mhc.py`, strip the
    `mhc_wrappers` / `_mhc_layers` / `_apply_per_stream` machinery from
    `models/encoder.py`, drop the `MHCCfg` from `utils/schema.py`, remove the
    `mhc:` block from every config.
  - **MHC wins** → document what it's winning on so the keeper rationale is
    in the repo.

No code changes until the experiment runs.

---

## §3.6 — Eval surface (decision-deferred)

`eval/eval_emotion.py`, `eval/eval_gender.py`, `eval/extract_embeddings.py`
are config-disabled but still on disk. Left in place during the pass — they
work as written, just unused.

`iter_embeddings` and `iter_frame_features` in `eval/common.py` are ~90%
duplicate. Could be unified with a `pool: bool` flag and one helper. Left
for later.

**Decision needed:** keep these eval entrypoints (re-enable later) or delete
once the project commits to the inline-probe-only path?

---

## §3.7 — GAN code (decision-deferred)

`gan.enabled: false` in every config. The GAN code path (discriminators,
adaptive weight, two `torch.autograd.grad` calls, hinge loss) is gated and
inert when disabled. Cost when off is zero.

**Decision needed:** keep GAN path for future enabling, or delete entirely
to slim `train.py` by another ~150 lines? No code changes yet.

---

## §5 — Static-analysis cadence (re-run when convenient)

The `.static-analysis/` reports under git are from before the pass and are
now stale. Re-run after this branch lands to get a clean baseline for
future work:

```bash
uvx ruff check --select F,ARG,ERA,RUF --output-format=concise \
    train.py models/ data/ losses/ optim/ eval/ utils/ tests/ \
    > .static-analysis/ruff.txt

uvx vulture --min-confidence 60 \
    train.py models/ data/ losses/ optim/ eval/ utils/ tests/ \
    .static-analysis/vulture-allowlist.py \
    > .static-analysis/vulture.txt

npx -y pyright --project .static-analysis/pyrightconfig.json --outputjson \
    > .static-analysis/pyright.json
```

Expected: ~zero F-class ruff findings (a final F401 sweep ran at the end of
the simplification pass), one F841 `proc` survivor (see C6), a much smaller
vulture report now that the Zipformer tree is gone, and pyright clean
modulo whatever the Conformer attention `del pos_emb, ...` shadowing
produces.

Diff each report against the previous baseline (saved under git) to find
cascading orphans.
