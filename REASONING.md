# Training Stability Improvements

Based on the analysis of run `20260310_021048`, we have implemented three key fixes to stabilize the latent space and improve reconstruction quality.

## 1. Weight Rebalancing
- **Changes**: `sigreg.weight: 0.05`, `jepa.weight: 0.8`, `stft_weight: 0.2`.
- **Reason**: In the previous run, the ratio of SIGReg to JEPA was 1:300 (0.05 vs 15.0). This was too weak to enforce the isotropic Gaussian constraint. The new ratio (0.05 vs 0.8, approximately 1:16) aligns with the LeJEPA paper's recommendation of $\lambda = 0.05$, where the ratio is $0.05 : 0.95$. This ensures the Gaussian regularization is strong enough to compete with the predictive loss.

## 2. Numerical Stability (SIGReg Bandwidth)
- **Changes**: `sigreg.t_max: 1.0` (reduced from 3.0).
- **Reason**: The Empirical Characteristic Function matching in SIGReg suffers from vanishing gradients when latent variance is high. The gradient magnitude is proportional to $e^{-\sigma^2 t^2 / 2}$. By reducing $t_{max}$, we keep the matching points in a region where the gradient remains significant even if the variance drifts slightly, allowing the loss to "pull" the distribution back to the target.

## 3. Structural Fix (Decoder Latent Normalization)
- **Changes**: `latent_norm: true` in the Decoder config.
- **Reason**: Without normalization, the decoder might "prefer" high-variance latents to achieve better reconstruction, creating an adversarial relationship with the SIGReg loss. Enabling `latent_norm` decouples the latent scale from the decoder's requirements, allowing the encoder to stay within the $N(0, 1)$ target without sacrificing reconstruction performance.
