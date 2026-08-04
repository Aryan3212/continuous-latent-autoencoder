# Sequential ablation runs

Run all six 30k conditions, one at a time, from the repository root:

```bash
scripts/run_ablation_suite.sh
```

The fixed order is full objective, reconstruction-only, representation-only,
mHC-off, 25 Hz, and decoder-corruption-off. A condition starts only after the
previous condition produces an intact `step_030000.pt`. The filenames and run IDs
retain their historical `_50k` suffix so an already-completed full run remains
discoverable and can be skipped.

The launcher gives every condition a stable run ID under `runs/ablations/`.
Re-run the same command after an interruption or machine restart. Completed
conditions are skipped; the interrupted condition resumes from the newest
intact `last.pt` or `step_<N>.pt`. ZIP-format checkpoints receive a full archive
CRC check when `unzip` is available. `train.py` then performs the authoritative
model/optimizer/config validation while loading the selected checkpoint.

For two GPUs per run:

```bash
ABLATION_GPUS=2 scripts/run_ablation_suite.sh
```

To use another output volume, keep the same value on every invocation:

```bash
ABLATION_OUT_DIR=/mnt/training/clae-ablations scripts/run_ablation_suite.sh
```

Additional dotted config overrides are forwarded to every condition:

```bash
scripts/run_ablation_suite.sh run.wandb.enabled=false
```

The launcher rejects overrides for the stable run ID/output root, 30k stopping
point, 1k save cadence, and 100k scheduler horizon because its completion and
resume guarantees depend on those values. Use `ABLATION_OUT_DIR` to relocate
outputs.

If training exits unsuccessfully, or exits cleanly before creating the 30k
checkpoint (for example because of `--max_hours` added outside this launcher),
the suite stops rather than advancing to the next condition. It does not retry
failures in a loop; after correcting transient hardware, storage, or data
problems, re-run the same launcher command. If a run directory contains no
checkpoint yet, that condition restarts from step zero and its existing log may
contain a duplicated initial step range.

Do not change `ABLATION_OUT_DIR`, the config files, GPU count, or forwarded
training overrides between the initial invocation and a resume. In particular,
changing GPU count changes the effective global batch size and packed-data
assignment, so it would no longer be a matched continuation.

## Generate the matched diagnostic plots

After the suite finishes, generate the five-condition comparison and the
separate 25 Hz supplement with:

```bash
uv run python scripts/plot_ablation_diagnostics.py \
  --full runs/ablations/large-2kh-ablation-full-r-j-visreg-50k \
  --reconstruction-only runs/ablations/large-2kh-ablation-recon-only-r-50k \
  --representation-only runs/ablations/large-2kh-ablation-repr-only-j-visreg-50k \
  --no-mhc runs/ablations/large-2kh-ablation-no-mhc-50k \
  --no-decoder-corruption runs/ablations/large-2kh-ablation-no-decoder-corruption-50k \
  --supplementary-25hz runs/ablations/large-2kh-ablation-25hz-full-50k \
  --output-dir runs/ablations/diagnostic-plots
```

If `ABLATION_OUT_DIR` was changed, replace the `runs/ablations` prefixes. The
plotter requires all logged metrics and common 10k, 20k, and 30k steps; it fails
instead of silently omitting an incomplete condition.
