# Ablation logging requirements

Implemented logging and plotting contract for the 25k TACL ablation runs. This
document does not authorize starting a run. All six configs inherit the packed
TAR backend from `large_2kh_packed.yaml`.

## Scope and invariants

- Log at `train.log_interval_steps`, reduce metrics across DDP ranks, and write
  the same keys to local JSONL and W&B.
- Compute covariance/eigenvalue diagnostics in FP32 under `torch.no_grad()` at
  log boundaries only. Do not add them to every microbatch.
- Use the globally gathered representation population for distribution
  diagnostics so single-GPU and DDP runs measure the same quantity.
- Preserve existing metric names. New names should use the namespaces below.
- Record both an interval mean and an interval maximum for unstable quantities
  such as gradient norms; record counts as cumulative counters plus interval
  fractions where applicable.
- Verify that logging does not retain autograd graphs or change the objective.

## Already logged; retain these

- Objective and reconstruction: `loss`, `l_mel` or `l_stft`, `l_wav`,
  `l_jepa`, `l_vis`, and the active reconstruction-loss breakdown.
- Representation change/collapse: `z_diff_rms`, `z_to_norm_ratio`, `z_rank`,
  `z_rank_utt`, `z_rank_res`, `p_rank_utt`, and the clean/augmented positive and
  negative similarity probe metrics.
- Decoder health: input/output RMS and peak, tanh-saturation fraction,
  `decoder_grad_norm`, `decoder_grad_norm_max`, and `decoder_update_norm`.
- Runtime/provenance: step, LR, saved resolved config, Git hash, manifests, and
  packed-data epoch/assignment details when the TAR backend is active.

## Required additions

### 1. Objective terms and VISReg decomposition

Log the raw and weighted contribution of every active term:

- `objective/recon_raw`, `objective/recon_weighted`
- `objective/jepa_raw`, `objective/jepa_weighted`
- `objective/reg_raw`, `objective/reg_weighted`
- `objective/total`

Expose VISReg's three internal terms without changing their sum or gradients:

- `visreg/center_loss`
- `visreg/scale_loss`
- `visreg/shape_loss`

These fields must still be emitted when a term's configured weight is zero. That
makes the reconstruction-only and representation-only runs auditable and shows
whether an inactive branch was accidentally optimized.

### 2. Latent and projector distribution diagnostics

Compute these separately for clean encoder frames (`z`) and the projector
population consumed by VISReg (`p`):

- `*/dim_std_mean`, `*/dim_std_min`, `*/dim_std_p05`, `*/dim_std_median`,
  `*/dim_std_p95`, and `*/dim_std_max`
- `*/collapsed_dim_frac`: fraction of dimensions with standard deviation below
  a documented threshold (default `1e-3`)
- `*/mean_abs`: mean absolute per-dimension population mean
- `*/cov_diag_mean`
- `*/cov_offdiag_abs_mean` and `*/cov_offdiag_rms`
- `*/effective_rank`: covariance participation ratio
- `*/top_eigenvalue_fraction`: largest covariance eigenvalue divided by trace
- `*/isotropy_ratio`: smallest eigenvalue divided by largest eigenvalue, with a
  documented numerical epsilon

Use prefixes `latent/z_*` and `projector/p_*`, or another single consistent
namespace. Keep the existing rank keys as backward-compatible aliases if the
new implementation consolidates the calculation.

### 3. Optimizer and stability diagnostics

- `optim/total_grad_norm_preclip`
- `optim/total_grad_norm_preclip_max`
- `optim/clip_applied`: 0/1 at the log-boundary update
- `optim/clip_fraction`: fraction of optimizer steps in the interval whose
  pre-clip norm exceeded `optim.grad_clip`
- Per-module pre-clip gradient norms for `frontend`, `encoder`, `projector`, and
  `decoder`
- Per-module update norms for those same four modules at log boundaries
- `amp/scale`
- `amp/skipped_updates_total`
- `amp/skipped_update_fraction`: fraction in the interval
- `optim/finite_grad`: 0/1, plus a cumulative non-finite-gradient count

AMP skip detection should compare the scaler state before and after
`scaler.update()`; do not infer it from a zero parameter-update norm because a
zero-weight branch can legitimately have no update.

### 4. mHC-specific diagnostics

For every `MHCWrapper`, log the following under a stable layer-qualified name:

- sigmoid of `H_res_alpha_logit`
- `tanh(branch_scale)`
- entropy of the softmax `H_pre` distribution
- entropy of the softmax `H_post` distribution
- mean diagonal and mean absolute off-diagonal of the effective residual matrix
- Frobenius distance of the effective residual matrix from identity
- maximum row-sum and column-sum error of the Sinkhorn-projected matrix
- gradient norms of `H_res_logits`, `H_pre_logits`, `H_post_logits`,
  `H_res_alpha_logit`, and `branch_scale`

The no-mHC run should record `mhc/enabled: 0`; mHC-enabled runs should record
`mhc/enabled: 1` and the number of wrappers. This makes missing mHC fields
distinguishable from a logging failure.

### 5. Throughput and resource accounting

- elapsed wall-clock seconds
- optimizer steps/second and audio samples/second over the interval
- source-audio hours processed cumulatively
- CUDA allocated/reserved memory and peak allocated/reserved memory
- world size, per-device batch size, gradient accumulation, segment duration,
  and effective batch size as run metadata
- GPU model names and visible GPU count as run metadata

The source-audio exposure counter is:

`completed_steps * batch_size * grad_accum_steps * world_size * segment_seconds / 3600`

Do not multiply it by the six augmented encoder views. Report the number of
views separately because it affects compute, not the number of source-audio
hours sampled.

## Required representation-diagnostic plots

The logging implementation must also provide a small post-processing entrypoint
that reads the JSONL logs and generates comparable plots for these conditions:

- full `R + J + VISReg`
- reconstruction-only `R`
- representation-only `J + VISReg`
- no mHC
- no decoder corruption

Plot the following over optimizer step:

- effective rank
- per-dimension standard-deviation distribution
- covariance off-diagonal magnitude and isotropy
- collapsed-dimension fraction
- clean/augmented positive similarity versus different-utterance negative
  similarity

Use the same checkpoint/log steps and identical axis ranges for every condition.
Use the common available optimizer steps; the 1k checkpoint cadence permits
additional post-hoc points when needed. Do not auto-scale each run independently,
because that can visually exaggerate or conceal differences.

Produce separate panels or figures for:

- projector-space diagnostics (`p`), which show whether VISReg shaped the
  distribution it directly optimizes; and
- encoder-latent diagnostics (`z`), which characterize the representation used
  by the decoder and downstream probes.

Do not combine `p` and `z` into one curve or use projector-space health as a
substitute for encoder-latent/downstream evidence. The 25 Hz run may be shown in
a separate supplementary panel because its different number of frames changes
the sampled representation population.

Each generated artifact should record the input run directories, config names,
metric keys, selected steps, collapsed-dimension threshold, and creation time.

## Acceptance checks

- A short hardware smoke run produces finite values for every applicable key.
- Zero-weight objective contributions are exactly zero while their raw losses
  remain visible.
- Full, no-mHC, and no-decoder-corruption runs share an identical metric schema;
  only explicitly inapplicable mHC fields may be absent.
- DDP and single-GPU metrics use the same definitions.
- JSONL and W&B values agree at the same step.
- Added diagnostic tensors are detached and do not alter gradients.
- The plotter fails clearly when a requested run or metric is missing rather
  than silently dropping that condition.
- Cross-run plots use common steps and fixed axes, and keep projector and
  encoder-latent diagnostics visibly separate.
