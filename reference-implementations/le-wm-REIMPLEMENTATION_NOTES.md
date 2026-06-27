# LeWorldModel (le-wm) — Reimplementation Notes

Source: https://github.com/lucas-maes/le-wm (cloned into this directory).

## Architecture (encoder + AR predictor + SIGReg)

- **Encoder** (`train.py:82-88`): HuggingFace ViT (variable scale), CLS token only.
- **Projector** (`train.py:104-109`): 2-layer MLP with **BatchNorm1d** → maps CLS to `embed_dim`. BatchNorm (not LayerNorm) is critical for SIGReg to reshape the distribution to isotropic Gaussian.
- **Action encoder** (`module.py:189-214`): conv1d patch + MLP that maps `(B,T,A)` → `(B,T,embed_dim)`.
- **Predictor** (`module.py:244-285`): AR Transformer with **AdaLN-zero conditioning** on actions. AdaLN params zero-initialised so training starts as a no-op. Causal self-attention. Has its own MLP projector head.

## Loss (the entire training objective)

`train.py:18-46` — `lejepa_forward`:

```
emb     = encoder(pixels)                       # (B,T,D), passed via projector
tgt_emb = emb[:, n_preds:]                      # future frames as labels
pred_emb = predictor(emb[:, :ctx_len], act)     # context + action -> next

pred_loss   = MSE(pred_emb, tgt_emb)            # frame-by-frame
sigreg_loss = SIGReg(emb.transpose(0,1))        # treat T as sample-dim → (T,B,D)
loss        = pred_loss + lambd * sigreg_loss   # lambd ≈ 0.1 in paper
```

Two terms only. **No EMA, no stop-grad, no centring, no covariance regulariser, no VICReg.** Anti-collapse is purely SIGReg.

## SIGReg details (`module.py:10-36`)

Sketch Isotropic Gaussian Regulariser. Epps-Pulley test statistic on `num_proj` 1-D random projections.

- `proj` shape `(T, B, D)`: SIGReg treats `T` as independent batches; for each timestep it runs the test on `B` samples. So the statistic averages over both time and projections.
- Random unit-norm directions `A ∈ R^{D×num_proj}` sampled fresh every call.
- `statistic = err @ weights * proj.size(-2)` — **multiplies by N=B** to give the test its paper-specified power. **Don't divide by N afterwards.**

## Hyperparameters that mattered

- `lambd` (SIGReg weight): ~0.1. Only sensitive knob.
- `num_proj`: 1024 (robust).
- AdaLN-zero init: required for stability.
- BatchNorm in projector: enables Gaussian reshape.

## Application to audio

The core idea transfers directly:
1. Replace ViT encoder with audio encoder (Zipformer), keep CLS-like token if doing world-model planning, otherwise keep per-frame tokens.
2. **Per-frame** prediction MSE (V-JEPA 2.1 style) — *not* utterance-pooled.
3. SIGReg on frame embeddings reshaped `(T, B, D)` so each frame slot is a sample of `B` independent draws. Exactly LeWM's recipe.
