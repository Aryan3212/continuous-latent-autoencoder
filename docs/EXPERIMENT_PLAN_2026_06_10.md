# Experiment Plan — 2026-06-10 (post-collapse recovery)

Plan for validating the anti-collapse change set after the 8,280-step
local_6gb run rank-collapsed (z_rank 3.62/64, z_rank_utt 0.54; diagnosis in
`LAB_NOTEBOOK.md`, change list in `CHANGELOG.md` 2026-06-10). The governing
lesson: judge runs by **backbone** metrics (`z_rank`, `z_rank_utt`,
`z_rank_res`), not by metrics computed in the space SIGReg regularizes
(`z_var_*`, `p_a_rms` are healthy by construction).

Configs: `configs/local_6gb.yaml` (d_model 64, this PC, 6 GB VRAM) and
`configs/exp0.yaml` (d_model 192, ~6M params, needs a bigger card).

---

## 1. Success gauges for the next run

Last time the rank metrics plateaued by ~step 4k while losses kept falling,
so a short run is enough to falsify the fix. Check these at every probe of
the dashboard; "collapsed run" column is the step-8280 wandb summary.

| Gauge | Collapsed run | Success looks like | Failure looks like |
| --- | --- | --- | --- |
| `z_rank` (frame PR, max 64) | 3.62, flat from ~4k | climbing well into the tens (20+ and rising by mid-run) | plateau < 10 by 5k steps |
| `z_rank_utt` (pooled PR) | 0.54 (metric bug; truly ≈ 1) | > 5–10 and rising | pinned near 1 (all utterances pool to one point) |
| `z_rank_res` | 3.36 | tracks `z_rank` upward | flat single digits |
| `l_sig_z` (new, frame SIGReg on z) | n/a | decreasing **without** `l_jepa` stalling | `l_sig_z` falls but `l_jepa` stops improving (SIGReg fighting JEPA), or `l_sig_z` flat (weight too small) |
| `l_sig_utt` (new, pooled-p SIGReg) | n/a | decreasing; pooled-rank gauge in probe JSON rising | flat / rising |
| Gender probe accuracy | not run | clearly off chance (50%) and improving across checkpoints | hovering at chance — pooled embedding still uninformative |
| ASR probe WER (after feasibility fixes) | ~1.0 (probe measured its own handicap) | < ~0.9 with infeasible-sample count near zero | ≥ 1.0, or large infeasible fraction (feasibility fix didn't take) |
| `jepa_to_norm_ratio` | 0.43 and declining | stable in roughly 0.15–0.5 | trending to < 0.1 |

Caveats:
- `z_rank_utt` is only trustworthy once the eigenvalue clamp lands in
  `train.py` (PR ≥ 1 by construction afterwards). Any value < 1 means the
  clamp is missing, not that the run is "below floor".
- `z_rank_utt` is computed on one batch of utterance means, so it is noisy;
  judge the trend over a few hundred steps, not single points.
- The healthy ranges in `docs/LOG_INTERPRETATION.md` were written for
  d_model 192; for local_6gb (64 dims) scale expectations down accordingly.

## 2. Run sequencing

1. **Diagnostic run, 5–8k steps, local_6gb** (`--max_steps 8000` CLI
   override; everything else from `configs/local_6gb.yaml`). The collapsed
   run's rank metrics had flatlined by ~4k, so this is a cheap, decisive
   test. Gate: the gauges above, especially `z_rank` trajectory.
   - If `z_rank` is still single-digit and flat by 5k: stop, do not burn the
     full 30k. First lever is `loss.sigreg.z_weight` (see ablation grid
     below), second is the JEPA:recon weight ratio.
   - If memory is tight at 64 × 2.5 s (est. ~5.6 GB): batch 48 +
     `grad_accum_steps: 2` per the config comment.
2. **Full 30k-step run, local_6gb**, only after the diagnostic passes. Run
   the offline probes (`eval/eval_asr.py` with the new upsampling/feasibility
   path, gender/emotion probes) on the final checkpoint; record WER,
   accuracies, and the pooled-embedding rank gauge in the experiment-log
   table in `LAB_NOTEBOOK.md`.
3. **Scale the proven recipe to exp0** (`configs/exp0.yaml`, d_model 192) on
   a ≥16 GB card (Kaggle/Colab T4) for the headline numbers in the course
   writeup. Same gauge discipline: a short diagnostic before the long run.
   Do not change recipe and scale at the same time — port the local_6gb
   hyperparameters that worked, only growing the dims.

Old checkpoints cannot be resumed: the projector changed shape
(32→64 / 64→192 output) and the configs changed, so `load_state_dict`
(strict) will fail. All runs in this plan start from scratch.

## 3. Ablations for the course writeup

Both ablations are cheap at local_6gb scale (5–8k diagnostic-length runs are
sufficient since collapse expresses itself by ~4k) and give the writeup an
actual experimental claim rather than just "we fixed our run".

### 3a. Stop-grad on the JEPA center (`center.detach()`)

V-JEPA-family methods use stop-grad/EMA targets to prevent collapse; LeJEPA
claims SIGReg makes those heuristics unnecessary. Our setup can test that
claim directly: in `_global_local_jepa_loss` (`train.py`), the globals-only
center is *not* detached — the LeJEPA design. One-line ablation: detach it.

- Arms: (i) current code (no detach, SIGReg on p + z), (ii) `center.detach()`
  with the same SIGReg settings, and ideally (iii) no detach with
  `z_weight=0` — the collapsed configuration, as the negative control.
- Compare `z_rank` / `z_rank_utt` trajectories, plus `l_jepa` components.
- Interesting outcome either way: if (i) ≈ (ii), SIGReg indeed substitutes
  for stop-grad at this scale; if (ii) rescues runs that (i) does not, the
  LeJEPA claim doesn't transfer to a 6M-param audio model where SIGReg was
  needed on the backbone too.

### 3b. `loss.sigreg.z_weight` sweep ∈ {0, 0.02, 0.1}

- 0 isolates how much of the rescue comes from the other fixes (projector
  width ≥ d_model, wd 1e-5) versus SIGReg-on-z itself.
- 0.02 is the configured default.
- 0.1 probes the over-regularization side: watch for `l_jepa` stalling or
  `z_rank` rising while ASR/gender probes get *worse* (isotropy at the cost
  of task structure).
- Report `z_rank` trajectory + probe metrics per arm.

## 4. Open risks

- **SIGReg-on-z is a rank-preserver, not a strict N(0, I) constraint.** Raw
  Conformer output is LayerNorm-shaped and resists Gaussianization — that is
  exactly why the BatchNorm projector exists (see `models/projector.py`
  docstring). Expect `l_sig_z` to plateau well above `l_sig` (the p-space
  term) and do not chase it to zero; its job is to keep variance spread
  across the 64 dims, not to make z Gaussian. If `z_weight` is pushed high
  enough to force Gaussianity, it will likely fight the encoder's useful
  structure.
- **Checkpoint/resume incompatibility.** Projector shape changes alone break
  old checkpoints; if the code agents add any new module state, resume of
  *mid-change-set* checkpoints breaks too. Treat everything before
  2026-06-10 as unloadable; budget runs accordingly.
- **Three SIGReg calls share one `SlicingUnivariateTest` instance** whose
  internal `global_step` buffer seeds the random slices and increments per
  call. Single-GPU this is fine (each branch just gets different slices);
  under any future DDP use, seed sync relies on identical call order/count
  across ranks — keep the `z_weight > 0` / `utt_weight > 0` branching
  identical on all ranks.
- **The probes can still mislead in the other direction.** The ASR probe
  fixes (×4 upsampling, duration filtering, infeasibility accounting) change
  the measurement, so WER is not comparable to earlier numbers; record the
  infeasible-sample count alongside WER every time. Gender accuracy can be
  driven by trivial cues (energy, pitch range) even from a low-rank
  embedding — read it jointly with the pooled-embedding rank gauge.
- **Two regularizers, one budget.** `weight` (p-space), `z_weight`, and
  `utt_weight` now all pull toward isotropy in different spaces. If the
  diagnostic run shows `l_jepa` refusing to fall, suspect the combined
  SIGReg pressure before suspecting the JEPA recipe.
