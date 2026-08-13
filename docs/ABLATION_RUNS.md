# Sequential ablation runs

Run all six 25k training conditions, then evaluate them once, from the
repository root. Prepare the normal validation manifest first and pass it to
the suite:

```bash
ABLATION_EVAL_MANIFEST=staging/manifests/val.jsonl \
  ABLATION_EVAL_SUBESCO_DIR=datasets/SUBESCO \
  scripts/run_ablation_suite.sh
```

For the stated remote layout, the exact absolute form is:

```bash
ABLATION_EVAL_MANIFEST=/home/msc022/continuous-latent-autoencoder/staging/manifests/val.jsonl \
  ABLATION_EVAL_SUBESCO_DIR=/home/msc022/continuous-latent-autoencoder/datasets/SUBESCO \
  scripts/run_ablation_suite.sh
```

`ABLATION_EVAL_MANIFEST` must be a normal JSONL manifest whose
`audio_filepath` entries are accessible on the evaluation machine. The same
ordered validation samples and fixed first segments are used for every CLAE
condition and Mimi. They are also used for the listening package when that
optional package is enabled.

The fixed order is reconstruction-only, representation-only, mHC-off, 25 Hz,
decoder-corruption-off, and the original packed-base full objective. A run starts
only after the previous run produces an intact `step_025000.pt`. After all six
training runs are complete, the suite calls `scripts/run_ablation_evals.sh`
once. There is no separate matched-full training run. The five ablation
filenames and run IDs retain their historical `_50k` suffix.

The launcher gives every run a stable run ID under `runs/ablations/`.
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

The launcher rejects overrides for the stable run ID/output root, 25k stopping
point, save cadence, and 100k scheduler horizon because its completion and
resume guarantees depend on those values. Use `ABLATION_OUT_DIR` to relocate
outputs.

If training exits unsuccessfully, or exits cleanly before creating the 25k
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
  --full runs/ablations/large-2kh-packed-25k \
  --reconstruction-only runs/ablations/large-2kh-ablation-recon-only-r-50k \
  --representation-only runs/ablations/large-2kh-ablation-repr-only-j-visreg-50k \
  --no-mhc runs/ablations/large-2kh-ablation-no-mhc-50k \
  --no-decoder-corruption runs/ablations/large-2kh-ablation-no-decoder-corruption-50k \
  --supplementary-25hz runs/ablations/large-2kh-ablation-25hz-full-50k \
  --output-dir runs/ablations/diagnostic-plots
```

If `ABLATION_OUT_DIR` was changed, replace the `runs/ablations` prefixes. The
plotter requires all logged metrics at the common available steps; it fails
instead of silently omitting an incomplete condition.

## Evaluation outputs

The suite evaluates packed-base, reconstruction-only, representation-only,
no-mHC, 25 Hz, and no-decoder-corruption checkpoints. It
writes each raw `eval.run_all` summary to
`runs/ablations/evaluations/<condition>/step_25000/summary.json`, and collects
all outcomes in `runs/ablations/evaluations/results.json` and
`runs/ablations/evaluations/results.md`. It continues through other conditions
after a missing checkpoint or failed evaluation, then returns a nonzero status
if any condition needs attention.

`eval.run_all` runs the shared reconstruction set (MR-STFT, waveform L1,
SI-SDR, STOI, ESTOI, and wide-band PESQ), the configured basic probes, both
SUBESCO temporal emotion heads, and the UTMOS-colored t-SNE/UMAP plot. SUBESCO
is never discovered implicitly: set `ABLATION_EVAL_SUBESCO_DIR`, otherwise the
three report tasks are explicitly recorded as skipped. The current packed
configs do not set basic probe manifests, so those probes remain explicit skips
until their `eval.*` paths are configured.

This is not yet the entire journal evaluation matrix. The automatic launcher
does not run the standalone speaker-ID, speaker-verification, Common Voice age,
attention-ASR, or Kathbath entry points. Run those separately when their fixed
manifests and protocols are ready; do not describe `results.json` as containing
them.

The launcher selects the `reconstruction-metrics` project extra, which pins
`pystoi==0.4.1` and, on Linux, `pesq==0.0.4`. PESQ is excluded from non-Linux
environments because its compiled extension is platform-sensitive. Missing
libraries or per-clip metric errors are recorded in metric coverage; they do
not stop later conditions, but the launcher returns nonzero at the end so
incomplete paper results are visible.

All CLAE conditions and Mimi use one evaluation-only MR-STFT contract: FFT
sizes 256/512/1024/2048, 0.25 hop ratio, Hann windows, centered STFT, and
weights 0.1/1/1 for spectral-convergence/magnitude/log-magnitude. A condition
is incomplete if any requested metric fails on any source item. Partial means
are retained for diagnosis but suppressed from the consolidated paper table.

After all six CLAE conditions, the evaluator runs the pinned Mimi revision at
exactly eight quantizers (the paper-style 1.1 kbps point). It refuses to run if
the installed Transformers API cannot enforce or verify eight codebooks. It
then creates `evaluations/listening_study/public/` with randomized stimuli and
an empty rating template, while preserving the condition key under
`listening_study/private/` only when `ABLATION_EVAL_LISTENING=1`. This prepares
a study; it does not collect ratings or analyze a human experiment. The public
stimulus manifest and private key both include SHA-256 hashes.

## TACL repeated probes and paired statistics

After the six checkpoints exist, the dedicated evidence launcher keeps one
data/split seed, runs probe seeds 0, 1, and 2, and computes paired bootstrap
intervals from saved item/trial predictions:

```bash
TACL_RECON_MANIFEST=staging/manifests/val.jsonl \
TACL_ASR_TRAIN_MANIFEST=/path/to/fixed-asr-train.jsonl \
TACL_ASR_DEV_MANIFEST=/path/to/fixed-asr-dev.jsonl \
TACL_SUBESCO_DIR=datasets/SUBESCO \
TACL_CV_ROOT=/path/to/common-voice-bn \
scripts/run_tacl_statistics.sh
```

Unset optional task inputs to skip those task families. The launcher never
starts autoencoder training. It evaluates reconstruction and verification once,
repeats stochastic probes at seeds 0/1/2, reuses frozen feature caches, and
resamples the same IDs jointly across all six conditions. A configured task
failure leaves its artifacts for diagnosis and suppresses the combined
bootstrap file until the full configured matrix succeeds.

UTMOSv2 package code is pinned to commit
`cc2700db57bb83ee13dc31ebe1b868c254e15d09`, but its default upstream weight
download uses an unversioned Hugging Face `main` URL. The evaluator hashes the
exact loaded state dictionary, includes that hash/config/fold/seed in plot
metadata, and uses it to invalidate the MOS cache. Set
`UTMOSV2_CHECKPOINT=/absolute/path/to/checkpoint.pth` to force a preserved local
weight file.

To intentionally skip reconstruction, Mimi, and listening stimuli while still
running report/probe tasks, all three settings must agree:

```bash
ABLATION_EVAL_SKIP_RECON=1 \
  ABLATION_EVAL_MIMI=0 \
  ABLATION_EVAL_LISTENING=0 \
  scripts/run_ablation_evals.sh
```

To rerun evaluation without training, call the evaluator directly with the same
validation manifest:

```bash
ABLATION_EVAL_MANIFEST=staging/manifests/val.jsonl \
  ABLATION_EVAL_SUBESCO_DIR=datasets/SUBESCO \
  scripts/run_ablation_evals.sh
```

The evaluator continues after a missing checkpoint or failed condition, writes
the consolidated result files, and returns nonzero if anything needs attention.
Kathbath and SUBESCO preparation remain separate from this launcher.
