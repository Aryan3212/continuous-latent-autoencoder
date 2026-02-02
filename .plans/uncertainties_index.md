# UNCERTAINTIES.md resolution plan (index)

Source of truth for open items: `UNCERTAINTIES.md`

## Items (work through top-to-bottom)

- [ ] Master plan (with reference repos): `/.plans/uncertainties_resolution_with_refs.md`
- [ ] Encoder / Zipformer + mHC (`models/encoder.py`) → `/.plans/encoder_zipformer_mhc.md`
- [ ] ScaledAdam (`optim/scaled_adam.py`) → `/.plans/scaled_adam.md`
- [ ] LeJEPA objective wiring (`train.py`) → `/.plans/objective_lejepa.md`
- [ ] SIGReg (`models/sigreg.py`) → `/.plans/sigreg.md`
- [ ] Training hyperparams (CALM-like, ScaledAdam) (`configs/*.yaml`) → `/.plans/hparams_calm_like.md`
- [ ] Decoder (RAE-inspired) (`models/decoder_generator.py`) → `/.plans/decoder_rae_inspired.md`
- [ ] GAN adversarial training (`train.py`, `models/discriminators.py`) → `/.plans/gan_adversarial.md`
- [ ] Multi-Res STFT loss details (`losses/multires_stft.py`) → `/.plans/multires_stft_loss.md`
- [ ] Evaluation probes / CTC probe (`eval/eval_asr.py`) → `/.plans/eval_probes.md`
- [ ] Evaluation + baselines (EnCodec/HuBERT, PESQ, Prism) → `/.plans/eval_benchmarks.md`

## Definition of done (per item)

- Spec/paper reference identified (or an explicit “local spec” written down).
- Implementation updated to match the reference/spec.
- Minimal validation added (unit test / smoke script) proving expected behavior.
- Item removed or rewritten in `UNCERTAINTIES.md` with concrete confirmation.
