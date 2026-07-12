---
title: CLAE Bengali Speech Autoencoder
emoji: 🗣️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
python_version: "3.12"
pinned: false
---

# CLAE Bengali Speech Autoencoder — checkpoint comparison demo

Web version of `scripts/reconstruct_live.py`: upload or record audio, the ~2.5M-param
continuous-latent autoencoder encodes it to a 12.5 Hz latent and decodes back to a
waveform. To make training progress audible, the same input is decoded through **two
checkpoints** side by side — `step-30320.pt` (early) and `last.pt` (= `step-48000.pt`,
latest). Runs on CPU (free Spaces tier is fine).

## Deploy to Hugging Face Spaces

1. Create a new Space → SDK **Gradio**, hardware **CPU basic** (free).
2. Upload the three files in this `demo/` folder to the Space root: `app.py`,
   `requirements.txt`, `README.md` (the YAML header above is what configures the Space).

   ```bash
   # or push via git:
   git clone https://huggingface.co/spaces/<you>/clae-bengali-demo
   cp demo/{app.py,requirements.txt,README.md} clae-bengali-demo/
   cd clae-bengali-demo && git add . && git commit -m "CLAE demo" && git push
   ```
3. If the checkpoint repo `aryan3212/clae-bengali-encoder` is **private**, add an
   `HF_TOKEN` secret in the Space settings (Settings → Variables and secrets). If it's
   public, no secret is needed.
4. The Space builds, clones the model code from GitHub at startup, downloads both
   checkpoints (`step-30320.pt` and `last.pt`), and serves the UI. First boot takes
   ~1–2 min (clone + ckpt downloads + model loads).

## Notes

- Audio is processed in independent **3 s windows** (the encoder's training segment
  length); longer clips may have mild seam artifacts.
- Output is **intelligible but robotic** — that's the current model.
- To change which checkpoints are compared, edit the `CKPTS` dict in `app.py` (any
  number of `label -> filename` entries renders that many side-by-side outputs); to
  change the model config, change `CONFIG_PATH`.
