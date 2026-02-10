# Research Experiment Guide

This guide outlines a rigorous workflow for conducting research experiments with the Continuous Latent Autoencoder, ensuring reproducibility, valid comparisons, and organized results.

## 1. Experimental Philosophy

A "proper" research experiment follows the scientific method:
1.  **Hypothesis**: Define what you expect to improve (e.g., "Adding MMD loss improves latent disentanglement").
2.  **Control**: Establish a strong baseline (Exp0).
3.  **Treatment**: Run the new model with the proposed change.
4.  **Observation**: Compare Control vs. Treatment using objective metrics and downstream probes.
5.  **Ablation**: Remove individual components to verify their necessity.
6.  **Significance**: Run multiple seeds to ensure results are not noise.

## 2. Directory Structure

Organize your experiments by hypothesis, not just date.

```
runs/
  ├── exp0_baseline/          # The control
  │     ├── seed_42/
  │     └── seed_1337/
  ├── exp_new_idea/           # The treatment
  │     ├── seed_42/
  │     └── seed_1337/
  └── ablations/              # Supporting evidence
        ├── no_mhc/
        └── lower_dim/
```

## 3. Workflow Steps

### Phase 1: Establish the Baseline (Exp0)

Before testing your new idea, you must have a trusted baseline.

```bash
# Run 3 seeds for the baseline
uv run python train.py --config configs/exp0.yaml run.run_id=exp0_baseline_s42 run.seed=42
uv run python train.py --config configs/exp0.yaml run.run_id=exp0_baseline_s43 run.seed=43
uv run python train.py --config configs/exp0.yaml run.run_id=exp0_baseline_s44 run.seed=44
```

### Phase 2: Train the New Model

Create a new config `configs/exp_new.yaml` (inheriting from or copying `exp0.yaml`) or use command-line overrides for rapid prototyping.

```bash
# Example: Increasing latent dimension to 512
uv run python train.py \
  --config configs/exp0.yaml \
  model.bottleneck.latent_dim=512 \
  run.run_id=exp_high_dim_s42 \
  run.seed=42
```

### Phase 3: Evaluation (Probing)

Once training is complete, run the unified evaluation suite. This runs:
1.  **Reconstruction**: STFT loss, Wav L1.
2.  **Probes**: ASR, Emotion, Gender classification on frozen embeddings.
3.  **Baselines**: EnCodec / HuBERT (if configured).

```bash
# Evaluate the best checkpoint
uv run python eval/run_all.py \
  --config configs/exp0.yaml \
  --ckpt runs/exp0_baseline_s42/checkpoints/best_composite.pt \
  --manifest data/manifests/val.jsonl \
  --out_dir runs/exp0_baseline_s42/final_eval
```

### Phase 4: Ablations

If your new model has multiple components (e.g., A + B + C), you must show that all are necessary.

1.  **Full**: A + B + C
2.  **No A**: B + C
3.  **No B**: A + C
4.  **No C**: A + B

Use `configs/ablations/` for these configurations.

```bash
uv run python train.py --config configs/exp0.yaml configs/ablations/no_jepa.yaml run.run_id=ablation_no_jepa
```

## 4. Best Practices

1.  **Commit Your Code**: Ensure `train.py` logs the git hash (it does automatically). Don't run experiments on uncommitted changes if possible.
2.  **Monitor Logs**: Use the `jsonl` logs in `runs/<id>/logs/` or WandB. Look for loss spikes or "NaN" values.
3.  **Fixed Splits**: NEVER change the validation/test manifests during a study.
4.  **Budgeting**: If resources are tight, run 1 seed for development, but 3 seeds for the final paper numbers.

## 5. Artifacts Checklist

For a paper/report, you need:
- [ ] Table 1: Main Comparison (Exp0 vs. ExpNew) on all metrics.
- [ ] Table 2: Ablation Study (Component analysis).
- [ ] Figure 1: Reconstruction Spectrograms (Visual proof).
- [ ] Figure 2: t-SNE/UMAP of Latent Space (Visualizing structure).

## 6. Data Preparation

**Crucial**: To ensure valid comparisons, you must use **identical** train/val/test splits for all experiments.

1.  **Generate Fixed Splits**: Use the provided script to scan your audio directory and create immutable `.jsonl` manifests.
    ```bash
    uv run python scripts/create_dataset_splits.py \
        --data_dir /path/to/your/audio/files \
        --out_dir data/manifests/experiment_v1 \
        --min_duration 2.0 \
        --val_frac 0.05 \
        --test_frac 0.05 \
        --seed 42
    ```

2.  **Lock Them In**:
    - Update your `exp0.yaml` to point to these files:
      ```yaml
      data:
        train_manifest: "data/manifests/experiment_v1/train.jsonl"
        val_manifest: "data/manifests/experiment_v1/val.jsonl"
      ```
    - Commit these manifests to git (if small enough) or checksum them to ensure they don't change.

## 7. Deep Sanity Checks (The "Think Hard" List)

Before trusting any result, perform these checks:

### 1. Latent Space Topology
Low reconstruction loss $\ne$ Good representation. You can learn an identity mapping or high-frequency noise hiding.
-   **Check**: Use `scripts/visualize_latents.py` to plot PCA/UMAP of the validation set.
-   **Expectation**: If you have speaker labels (even if not used in training), they should cluster. If the plot is a single blob, the model hasn't learned disentangled features.

### 2. The "Speaker Leakage" Trap
-   **Danger**: If Speaker A is in both Train and Val, you are measuring memorization, not generalization.
-   **Action**: Ensure your splits in `create_dataset_splits.py` respect `speaker_id` if available. (The provided script is file-based; verify your data source doesn't have speaker overlaps).

### 3. Gradient & Parameter Health
-   **Danger**: "Dead" neurons (ReLU=0 forever) or dimensional collapse.
-   **Action**: Monitor `z_var_min` (from SIGReg stats) and `z_std` in WandB.
    -   `z_std` $\approx$ 0: **Total Collapse**. The model ignored the latent code completely.
    -   `z_var_min` $\approx$ 0: **Dimensional Collapse**. The model is only using a few dimensions and ignoring the rest. Ideally, `z_var_min` $\approx$ `z_var_max` $\approx$ 1.0 (Isotropy).
    -   `grad_norm` spikes: **Exploding Gradients**.

### 4. Qualitative "Golden Ear" Test
-   **Danger**: Metrics (STFT, WER) don't capture "robotic" or "phasiness" artifacts.
-   **Action**: Listen to the reconstructed audio!
    -   Does it sound metallic?
    -   Is the background noise preserved or hallucinated?
    -   Does the pitch wobble?

### 5. The "Shift Invariance" Test
-   **Test**: Encode `audio`, decode `out1`. Shift `audio` by 1 sample, encode-decode `out2`.
-   **Expectation**: `out2` should be `out1` shifted by 1 sample. If they are wildly different, your model is aliasing (bad strides).

## 9. Pre-Flight Checklist (The "Partner Protocol")

Before considering an experiment "valid" or ready for publication, ensure it passes this checklist.

### Before First Experiment
- [ ] **Fix & Log Seeds**: `train.py` must set `seed_all()` and log it.
- [ ] **Pin Dependencies**: Use `uv.lock` or `requirements.txt`.
- [ ] **WandB Logging**: Ensure every run logs to the cloud.
- [ ] **Sample Rate Strategy**: Decide on a target SR (e.g., 16kHz) and ensure `AudioManifestDataset` resamples correctly.
- [ ] **Dataset Splits**: Use `scripts/create_dataset_splits.py` to make immutable Train/Val/Test JSONL files.
- [ ] **Documentation**: Create `DATASET_README.md` listing sources, hours, and domains.
- [ ] **Lab Notebook**: Start `LAB_NOTEBOOK.md` (Date + Hypothesis + Outcome).

### During Training
- [ ] **Hyperparameters**: Log full config to WandB (`wandb.config`).
- [ ] **Loss Granularity**: If using multiple datasets, log loss per-dataset (requires code update).
- [ ] **Probes**: Ensure `run_eval_on_save=True` is active to run ASR/Emotion probes every N steps.
- [ ] **Collapse Watch**: Monitor `z_std` in WandB. If it stays at 0, kill the run.
- [ ] **Negative Results**: Log failed runs in `LAB_NOTEBOOK.md`—they are data!

### Before Scaling / Publication
- [ ] **Speaker Overlap**: Verify no speakers leak between Train and Test (requires speaker-aware splitting).
- [ ] **Duplicate Check**: Run `scripts/audit_datasets.py` to find duplicates and audio issues.
- [ ] **Quality Scan**: Remove silence or clipped audio (flagged by audit script).
- [ ] **Ablations**: Run the "No A", "No B" configs to prove contribution.
- [ ] **Compute Cost**: Note GPU hours from WandB.
- [ ] **Final Eval**: Run `eval/run_all.py` on the test set with 3 seeds.

### Tool Stack Reference
- **Tracking**: WandB
- **Config**: YAML + Argparse (`utils/config.py`)
- **Sweeps**: WandB Sweeps
- **Data**: Custom `AudioManifestDataset` (Note: Lhotse is an alternative if complex mixing is needed).
- **Versioning**: Git (Automatic hash logging in `train.py`).
