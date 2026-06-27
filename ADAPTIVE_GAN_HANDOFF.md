# Handoff: VQGAN-style adaptive adversarial weight (and the resume/multi-GPU changes around it)

Audience: another Claude agent picking up debugging of the adaptive GAN.
Branch: `simplification`. All code paths cited are in `train.py` unless noted.
This doc reflects state as of HEAD `c2f7005`.

---

## 0. TL;DR

The adversarial loss was overwhelming reconstruction once the GAN turned on at
step 20000, so an **adaptive adversarial weight** (VQGAN / Esser et al. style) was
added: each optimizer step, scale `l_adv` by `lambda = ||grad_recon|| / ||grad_adv||`
measured at the decoder's last conv, so `adv_weight` becomes a clean
relative-strength knob (1.0 = gradient parity with reconstruction). Toggle:
`loss.adv.adaptive` (default `false`; set `true` in `configs/exp_3m_gan.yaml`).

It was reviewed and judged correct by an Opus subagent, **but the user is hitting
issues running it** — see §5 for the most likely causes and how to isolate them.
The fastest triage is to set `loss.adv.adaptive=false` and confirm whether the
problem disappears; that tells you instantly whether it's the adaptive path or the
GAN generally.

---

## 1. Why this exists (the diagnosis)

Config `configs/exp_3m_gan.yaml` (inherited by `configs/kaggle_3m_gan.yaml`):
the generator loss is

```
loss = stft_w*l_stft + wav_l1_w*l_wav + jepa_w*l_jepa + sig_w*sig_scale*l_sig
       + adv_w*l_adv + fm_w*l_fm
```

with `stft_weight=0.1`, `wav_l1_weight=0.0`, `jepa.weight=6.0`, `adv.adv_weight=1.0`,
`adv.fm_weight=2.0`. The GAN (MPD discriminator, LSGAN) activates at step 20000
(`adv.adv_start_step` / `adv.fm_start_step`).

Observed after 20k: `stft_mag`, `stft_log`, `l_wav` all started diverging upward
with growing spikes. Root cause: at the LSGAN equilibrium `l_adv ≈ 1.25` (5
sub-discriminators × `(1−0.5)²`), so `adv_w*l_adv ≈ 1.25` **dwarfs**
`stft_w*l_stft ≈ 0.13` by ~10×. The generator optimized "fool D" over
reconstruction. (`l_disc` sits at ~2.5 = 5×`(0.25+0.25)`, also the equilibrium —
that part is healthy and expected.)

`l_jepa` kept trending down throughout — representation learning was fine; only
reconstruction was being crowded out.

---

## 2. What was implemented

### 2a. Schema (`schema.py`, `AdvCfg`)
```python
adaptive: bool = False
adaptive_max: float = 1.0e4   # clamp on the adaptive lambda (VQGAN default)
```

### 2b. Config (`configs/exp_3m_gan.yaml`, under `loss.adv`)
```yaml
adaptive: true     # rescale l_adv to gradient-parity with recon at the decoder's last layer
adv_weight: 1.0    # with adaptive on, this is a relative-strength knob (1.0 = parity)
```

### 2c. Training loop (`train.py`)
- Reads the toggles near the other loss weights:
  ```python
  adaptive_adv = acfg.adaptive
  adaptive_max = acfg.adaptive_max
  ```
- Resets a per-step cache before the grad-accum microbatch loop:
  ```python
  lam_adv_cached: Optional[torch.Tensor] = None
  ```
- Core block (currently ~`train.py:738-760`), right after `l_adv`/`l_fm` are
  computed and before `loss` is assembled:
  ```python
  lam_adv = x_hat.new_ones(())
  if disc_active and adaptive_adv:
      if lam_adv_cached is None:
          last_w = _unwrap(model["decoder"]).out_conv.weight
          gs = 1.0e3
          rec_g = torch.autograd.grad(
              gs * (stft_w * l_stft + wav_l1_w * l_wav), last_w, retain_graph=True
          )[0]
          adv_g = torch.autograd.grad(gs * l_adv, last_w, retain_graph=True)[0]
          lam_adv_cached = (
              rec_g.float().norm() / (adv_g.float().norm() + 1e-4)
          ).clamp(0.0, adaptive_max).detach()
      lam_adv = lam_adv_cached
  ```
- Applied in the loss as `... + adv_w * lam_adv * l_adv + fm_w * l_fm`.
- Logged as `lam_adv` (W&B) only when `disc_active and adaptive_adv`.

### Key design points
- **Last layer** = `out_conv.weight` of `WaveformDecoder` (`models/decoder_generator.py:90`,
  the final `Conv1d(in_ch, 1, k=7)` before `tanh`). `x_hat` comes from
  `_decode(model, z_a, ...)` → `model["decoder"](...)`, so `out_conv.weight` is the
  genuine, reachable last-layer tensor in the graph that produced `x_hat`.
- **Computed once per optimizer step** (first disc-active microbatch) and cached
  for the whole accumulation window — lambda varies slowly, and per-microbatch
  would add two partial backward passes *each* microbatch.
- **`gs=1e3` prescale** cancels exactly in the ratio but lifts fp16 grads off the
  AMP underflow floor (GradScaler does NOT scale `autograd.grad` output).
- **`clamp(0, adaptive_max)` + `+1e-4`** bound a vanishing-adv-grad blowup;
  `optim.grad_clip=1.0` is the final backstop.
- `l_adv` is built from `_unwrap(disc)(...)` (raw, **not** the DDP-wrapped disc),
  with disc params set `requires_grad_(False)` during the generator step.

---

## 3. Semantics / how to tune

- With `adaptive=true`, `adv_weight` is a **relative strength** knob: `1.0` =
  adversarial gradient at parity with the (weighted) reconstruction gradient at the
  last layer. Raise it (2–4) to let the GAN push fidelity harder; lower it (<1) if
  it's still too hot.
- Expect `lam_adv` to settle **small** (~0.1), because reconstruction is lightly
  weighted (`stft_w=0.1`). The effective adversarial contribution is
  `adv_w * lam_adv * l_adv ≈ 1.0 * 0.1 * 1.25 ≈ 0.13`, i.e. parity with `stft_w*l_stft`.
- **Watch `lam_adv`**: pinned at `adaptive_max` (1e4) = the adv gradient collapsed
  to ~0 (dead/saturated discriminator) — that's a real signal, not a crash.
- `fm_weight` (feature matching) is intentionally left static — it's a stabilizer,
  not what was overwhelming reconstruction.

---

## 4. Verification already done

An Opus subagent reviewed the change and confirmed (read-only, plus torch experiments):
- `autograd.grad(..., last_w)` returns grads directly and does **not** fire DDP's
  AccumulateGrad hooks, so the per-rank reducer still sees exactly one backward
  (the full `loss.backward()`); no "marked ready twice" *in principle*.
- AMP-safe (gs prescale + clamp + grad_clip), retain_graph correct (no double
  counting; `autograd.grad` doesn't write `.grad`), per-step caching correct,
  resume reloads both optimizers + both GradScalers + disc, the `adv_w=1.0`=parity
  semantics hold, and `out_conv.weight` is the right reachable tensor.

Caveat: that review was static + small torch experiments, **not** a real multi-GPU
DDP+AMP run. The user's issues may be exactly the gap between "safe in principle"
and "works on 2× T4 under torchrun." Treat §5 as the live debugging surface.

---

## 5. LIKELY ISSUES the user is hitting (start here)

The change runs two extra partial backward passes through a **DDP-wrapped**
submodule before the main backward, under AMP. Most probable failure modes,
roughly in order:

1. **DDP backward error under torchrun.** Despite the "doesn't fire hooks"
   reasoning, `autograd.grad` that backprops through the DDP-wrapped `model["decoder"]`
   (and, for `adv_g`, through the disc into the decoder) can still trip the reducer
   in some torch/DDP versions — symptoms like `RuntimeError: Expected to mark a
   variable ready only once`, or a hang at the first GAN step (step 20000). Each
   param-bearing submodule is wrapped **separately** in DDP (`train.py` ~505-512),
   which is an unusual setup worth scrutinizing.
   - **Isolate:** run with `loss.adv.adaptive=false` (static weight). If the error
     vanishes, it's the adaptive partial-backward × DDP interaction.
   - **Possible fixes to evaluate:** wrap the lambda computation so it doesn't go
     through the DDP module (e.g. compute grads via the unwrapped decoder forward —
     but `x_hat` already came from the wrapped one, so this needs care), or guard
     with `model["decoder"].no_sync()` context during the partial grads, or compute
     lambda only on rank 0 and broadcast. None of these are implemented yet.

2. **Higher peak VRAM / OOM.** `retain_graph=True` on the two partial backwards
   keeps the autograd graph alive longer (through to the main `loss.backward()`),
   raising peak memory during the GAN phase. On a tight GPU this can OOM right at
   step 20000 when the disc + adaptive path both switch on.
   - **Isolate:** lower `train.batch_size` (raise `grad_accum_steps` to keep eff
     batch — see §6); or `adaptive=false` to confirm.

3. **`lam_adv` exploding to `adaptive_max`.** If `adv_g.norm()` underflows or the
   disc gives ~no gradient, lambda → clamp at 1e4, and `adv_w*1e4*l_adv` becomes a
   huge loss term. `grad_clip=1.0` should bound the *update*, but the loss/W&B will
   look unstable. Check the `lam_adv` panel. If this happens, lower `adaptive_max`
   (e.g. to 10–100) and/or investigate disc health.

4. **Slowdown.** Two extra partial backwards per *step* (cached, so not per
   microbatch) — cost is modest but non-zero; on a time-boxed Kaggle session it
   eats into the budget. Expected, not a bug.

5. **Lambda noisiness.** It's computed from only the **first** disc-active
   microbatch of each step and reused. If batches are very heterogeneous, lambda is
   noisier than a full-window estimate. Usually fine; mentioned for completeness.

When debugging, the single most useful first action is the `adaptive=false` A/B —
it cleanly separates "GAN/DDP problem" from "adaptive-weight problem."

---

## 6. Related changes that affect running/resuming (context you'll need)

### 6a. Multi-GPU effective batch (IMPORTANT)
`scripts/kaggle_session.sh` auto-detects GPUs and launches `torchrun
--nproc_per_node=N`. Under DDP, **`train.batch_size` is per-GPU**, so
`eff_batch = batch_size × grad_accum_steps × N`. The checkpoint was trained at
**eff 200** (10×20×1). On 2 GPUs you MUST pass `train.grad_accum_steps=10` to keep
eff 200 — nothing divides it automatically. Launch e.g.:
```bash
bash scripts/kaggle_session.sh -H 11.5 -- train.grad_accum_steps=10
```

### 6b. Scheduler resume fast-forward (committed in `c2f7005`)
The LR scheduler is `SequentialLR([LinearLR warmup, CosineAnnealingLR])` with
`T_max = total_steps - warmup_steps`. On resume the code **no longer restores
scheduler state**; instead it fast-forwards a freshly-built scheduler:
```python
step = int(state.get("step", 0))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _ in range(step):
        scheduler.step()
```
Why: `scheduler.load_state_dict()` baked in the checkpoint's old `T_max` AND left
CosineAnnealingLR's recurrent `get_lr` anchored to the old curve's amplitude, so a
raised `optim.scheduler.total_steps` silently trained the extension at ~0.18×
(≈5.5× too low) LR. Fast-forward re-derives the exact curve for the current
`total_steps` (verified ratio 1.000 vs a from-scratch schedule).

**Status:** a final independent Opus re-verification of this fast-forward was
in-flight when this doc was written. It tested as correct locally; if you touch the
resume path, re-confirm off-by-one vs ground truth and the unchanged/lowered
`total_steps` cases.

To extend a run, raise BOTH together (they must match):
`train.max_steps=60000 optim.scheduler.total_steps=60000`.

---

## 7. How to run / resume / toggle

- **Resume on 2 GPUs, GAN + adaptive on:**
  `bash scripts/kaggle_session.sh -H 11.5 -- train.grad_accum_steps=10`
  (script pulls `last.pt` from HF, reloads model + both optimizers + both
  GradScalers + disc + step, fast-forwards the scheduler).
- **Turn adaptive OFF (triage):** add `-- loss.adv.adaptive=false`.
- **Extend the run:** add `train.max_steps=N optim.scheduler.total_steps=N`.
- Checkpoint repo (HF): `aryanrahman/clae-bengali-encoder`; fixed `run_id:
  clae_3m_kaggle` so sessions resume the same W&B run + `runs/` dir.

---

## 8. Verifying scheduler/loss math locally (no GPU)

This box is edit-only for ML; **do not** use the project `.venv` (CUDA-pinned) or
`uv run`. A CPU venv with torch 2.12.1 + yaml + pydantic exists at
`.venv-recon/bin/python` — use it for config-override and scheduler experiments
(e.g. reproducing `apply_overrides`, or the fast-forward vs ground-truth LR check).
Do not run full training here.

---

## 9. Git state

- Branch `simplification`, HEAD `c2f7005` "learning rate scheduler fixes" (pushed;
  contains the scheduler fast-forward).
- `22cb7e1` = the adaptive adversarial weight (schema + train.py + config).
- `4c1147d` = an earlier, **superseded/incorrect** scheduler fix (set `T_max` only;
  ran the extension at ~5.5× too-low LR). It's in history but the current code in
  `c2f7005` is the fast-forward replacement — don't reintroduce the T_max-only idea.
- This doc (`ADAPTIVE_GAN_HANDOFF.md`) is uncommitted.
