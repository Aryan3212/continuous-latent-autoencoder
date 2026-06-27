# V-JEPA 2.1 — Reimplementation Notes (audio-relevant subset)

Source: https://github.com/facebookresearch/vjepa2 (cloned into this directory).
Code path of interest: `app/vjepa_2_1/`.

## The single change that fixes "dense" features: Dense Predictive Loss

V-JEPA-1 only supervised masked patches. V-JEPA 2.1 supervises **all** patches: masked **and** visible/context. This is the L_predict + L_context split.

`app/vjepa_2_1/train.py:677-703`:

```python
h = forward_target(clips)                    # teacher (EMA) targets for ALL tokens
z_pred, z_context = forward_context(clips)   # student outputs masked + visible

# (1) Predict loss — same as V-JEPA 1
loss_pred = loss_fn(z_pred, h, masks_pred, d_weights=None)
loss = loss_pred

# (2) Context loss — NEW: supervise visible tokens too
if predict_all:
    distance_weights = compute_mask_distance(masks_pred, masks_enc, grid_size, ...)
    d_weights = distance_weights if weight_distance_loss else None
    loss_context = loss_fn(z_context, h, masks_enc, cls_loss=False, d_weights=d_weights)
    loss += loss_context * lambda_value
```

### Distance-weighted context (`app/vjepa_2_1/models/utils/masks_dist.py:44-`)

For every context token `i`, compute its spatio-temporal distance `d_min(i, M)` to the nearest masked token. Then:

```
λ_i = λ / sqrt(d_min(i, M))
```

Implemented as `loss_n = |z - h|^p * (1 / d_i)` — i.e. **closer to a masked boundary ⇒ higher weight**. Forces local spatial coherence between visible features and the predictions for masked neighbours.

**Why it works** (paper's intuition): without context supervision, visible tokens drift into "global aggregator" roles (like register tokens) and lose local spatial identity. Supervising them, especially near mask boundaries, keeps them spatially grounded.

## Deep self-supervision

`app/vjepa_2_1/train.py` (forward_target / forward_context wrappers): the predictor sees outputs from 4 equally-spaced intermediate encoder blocks, channel-concatenated through a tiny MLP. This pushes local info through the whole stack. Optional but cheap.

## Other moving pieces

- EMA target encoder (`train.py:736-747`): standard JEPA momentum update.
- Heavy predictor (24 blocks).
- 3D RoPE positional embeddings; for images, interpolate temporal frequencies.
- Image / video mixed batches with a learnable modality token.

## Application to audio (1-D time)

Audio is V-JEPA 2.1 with the spatial axes collapsed:
- 1-D time grid `t = 1..T'`. Distance is `|t_ctx - t_mask|`.
- "Masked" frames = positions zeroed by SpecAug-style time masking.
- "Context" frames = unmasked positions.

The transfer recipe:
1. Two augmented views → encode both with shared encoder (or EMA teacher).
2. Per-frame prediction MSE on the masked positions (L_predict).
3. Per-frame MSE on the context positions, weighted by `1 / sqrt(d_min(t))` (L_context).
4. SIGReg as the *only* anti-collapse (LeJEPA / LeWM style); drop utterance-pooled regularisation.

This recovers V-JEPA 2.1's core inductive bias — **local features must reconstruct local targets** — while keeping LeJEPA's hyperparameter parsimony.
