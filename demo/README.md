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

# CLAE Bengali Speech Autoencoder

This Gradio demo reconstructs uploaded or recorded audio with the `last.pt`
checkpoint from [`aryan3212/clae-bengali-encoder`](https://huggingface.co/aryan3212/clae-bengali-encoder).

It uses `configs/large_2kh.yaml` and clones the model code from the `main`
branch at startup. Audio is processed in independent 3-second windows.
