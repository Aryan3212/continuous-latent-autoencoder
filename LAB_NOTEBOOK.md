# Lab Notebook

## Experiment Log

| Date | Run ID | Hypothesis | Outcome | Notes |
|------|--------|------------|---------|-------|
| 2026-02-10 | exp0_sweep | Hyperparameter sweep for baseline | Pending | Sweeping lr, d_model, latent_dim, loss weights |
| 2026-02-11 | n/a | Numerical instability in STFT loss during silence | Resolved | Increased `logmag_eps` to 1e-3 in `losses/multires_stft.py` and `configs/exp0.yaml` to prevent Spectral Convergence explosion on near-zero targets. |
| 2026-06-10 | local_6gb, 8,280 steps | Projector-space SIGReg + JEPA would keep encoder z healthy on its own | Refuted — z rank-collapsed to ~4 of 64 dims while every projector-space metric looked healthy | z_rank 3.62, z_rank_res 3.36, z_rank_utt 0.54 (mathematically impossible — metric bug, see human log). Rank plateaued by ~4k steps; losses kept falling. Fix set: SIGReg directly on z + on pooled p, projector output_dim ≥ d_model, weight_decay 1e-5. |


## Human log
1. We are not doing latent normalization that was introduced in the RAE paper, we are also not using the Vision Transformer(ViT)-like decoder. Or audio transformer like setup, but maybe we can explore that later. The paper added noise to the latents from the pretrained encoder, which we are adding in hopes that the noise helps the decoder be invariant to slight perturbations to the features and can create smooth data manifold that has a semantic understanding of the data.
2. RAE requires high dimensionality for reconstruction we are preserving that here.
3. We stick with the Convolutional/FiLM decoder for now, as it is efficient for audio. Not ViT as previously mentioned.

### 2026-06-10 — the 8.2k-step local run collapsed, and our dashboards told us it was healthy

We ran the local_6gb config (d_model 64, 3 Conformer layers, projector 64→128→32, batch 96 × 1.5 s segments, jepa weight 3, sigreg weight 0.2 on the projector output, stft 0.005, AdamW wd 0.01, cosine 30k) for 8,280 steps on the training PC. The losses looked like a textbook run: l_jepa 0.322 and falling (l_global 0.019, l_predict 0.139, l_context 0.164), l_sig 4.69, l_stft 6.92. Meanwhile the encoder output z had quietly collapsed: **z_rank 3.62 out of 64**, z_rank_res 3.36, and z_rank_utt **0.54** — which is mathematically impossible (the participation ratio of a PSD covariance is ≥ 1). The impossible number was its own clue: the utterance-mean covariance was so close to zero that `eigvalsh` returned small negative eigenvalues, which the unclamped PR formula happily turned into 0.54. In plain terms: every utterance was pooling to nearly the same embedding. The rank metrics had plateaued by ~step 4k while the losses kept improving, so running longer would not have fixed anything.

**Why we were deceived.** Everything we watched for collapse — z_var_min/med/max (≈ 0.93/0.99/1.07), p_a_rms (0.997) — is computed on the *projector* output p, and SIGReg directly forces p toward N(0, I), so those stats are healthy *by construction*. They are blind to the backbone. jepa_to_norm_ratio was 0.43 and declining, which we read as "views converging" rather than "everything converging".

**The mechanism, as we now understand it: the projector absorbs the regularization.** All anti-collapse pressure (SIGReg) and all training signal (JEPA) act on p = projector(z). The projector output was 32-dim against a 64-dim z, so half of z was structurally invisible to every loss that cared about structure. The only loss that demands full-information z is the waveform reconstruction, and we had deliberately weighted it 30× below JEPA (stft 0.005 vs jepa 3). AdamW weight decay at 0.01 then does exactly its job: it deletes whatever no loss demands. Result: z keeps ~4 effective dimensions — enough to satisfy a 32-dim projector through a wide MLP, and nothing more. We audited the JEPA loss math itself and it is correct; not detaching the globals-center is the LeJEPA design (SIGReg is supposed to replace stop-grad/EMA heuristics) — but we had only applied SIGReg in projector space, so the backbone had no protection.

**The fix set** (going into the next run; exact values in `configs/local_6gb.yaml` and `configs/exp0.yaml`):
- Frame-level SIGReg directly on z (`loss.sigreg.z_weight=0.02`) — anti-collapse pressure the projector cannot filter out — plus utterance-level SIGReg on time-pooled p (`loss.sigreg.utt_weight=0.05`, logged as `l_sig_utt`) so pooled embeddings can't all converge to one point.
- Projector output_dim ≥ d_model (32→64 local, 64→192 exp0) so no z dimension is invisible to the losses; hidden widened to match.
- weight_decay 0.01 → 1e-5 — at this scale it was the deletion mechanism.
- Eigenvalue clamp in the three z_rank metrics so they can never report < 1 again.
- Secondary: chunk-mask target_ratio 0.15→0.25 (harder prediction task), segments 1.5 s→2.5 s with batch 96→64, lowpass_min_freq raised to 2700 Hz so we stop training away the timbre band the gender/emotion probes need, gender/emotion probes enabled in-config, ASR probe steps 1000→8000, and ASR probe feasibility fixes (12.5 Hz frames vs ~8–15 Bengali chars/sec made many CTC samples infeasible and `zero_infinity=True` silently zeroed them — the probe was measuring its own handicap).

**Lesson for us:** never judge a self-supervised run by metrics computed in the same space the regularizer acts on. The backbone rank metrics (z_rank, z_rank_utt) are the ones the probes actually feel, and they are now the primary gauges for what "healthy" must look like before we commit to a full 30k run.

### 2026-06-11 — open decisions carried over from the simplification pass

The simplification pass (see `CHANGELOG.md`) deleted a lot of dead code and
folded the old planning/history docs into git. These are the only items from
those docs that are still *open* and actionable:

**mHC ablation (decision deferred).** mHC machinery (`models/mhc.py`, the
wrapper plumbing in `models/encoder.py`, the `model.encoder.mhc` config block)
is kept on this branch but its value is unproven. Plan: run two training jobs
side by side, same seed/data — mHC on (`model.encoder.mhc.enabled=true`) vs off
(`enabled=false` / `num_streams=1`). Compare ASR-probe WER, JEPA loss curves,
and the z_rank gauges. If no clear win: delete `models/mhc.py` and strip the
wrapper/`MHCCfg`/config blocks. If it wins: document what it wins on so the
keeper rationale is in the repo. No code changes until the experiment runs.

**Bring-back-later (removed, recoverable from git history):**
- *Inline / offline CTC probe.* The old inline probe was deleted (it wasn't
  torch.compile/DDP-safe). If brought back, pick one shape: a clean
  nn.Module head that participates in the main forward (don't toggle
  requires_grad mid-step), OR a separate launcher that finds the latest
  `last.pt` and runs `eval/eval_asr.py` out-of-process on a cadence.
- *Best-checkpoint tracking.* Only `last.pt` is saved now (`best_asr.pt` /
  `best_composite.pt` are gone — the gating metric came from the deleted
  eval-on-save block). Re-add best-tracking only when there's an actual metric
  driving it and you'll use those checkpoints downstream.
- *CodeCarbon emissions tracking.* Removed (~10 lines). Re-add if ever wanted.

**On the old history docs (HISTORICAL_CHANGES / RESEARCH_SUMMARY_2026_04_21,
now deleted).** They documented the superseded Zipformer + GAN + InfoNCE +
RAE-noise era. None of it is still actionable for the current
reconstruction + JEPA + SIGReg pipeline — those design choices were all reverted
or replaced. The few durable findings already live where they matter: the STFT
spectral-convergence silence fix is in `CHANGELOG.md` and the loss code; the
"high-dimensional latents help reconstruction" RAE rationale is in the Human log
above. Full commit-by-commit detail is in git history if ever needed.
