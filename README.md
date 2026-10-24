# SDXL_Research

One notebook for the Google Colab free tier (T4): caption a dataset, train identity LoRAs for SDXL and
Z-Image, compare them, and generate or edit images in a web app you open over Tailscale.

- **Generate**: Z-Image Turbo (GGUF Q8) or any SDXL checkpoint, Quality/Speed/Fast modes, LoRA mixes with
  trigger chips, live previews.
- **Edit**: paint the part to change, change the whole image, extend the canvas, upscale (with an optional
  Z-Image ControlNet "better blend" and tile model). Pixels outside the painted area stay exactly as they were.
- **Train**: SDXL (kohya sd-scripts) or Z-Image (musubi-tuner) LoRAs from your photos, with Qwen3-VL or WD
  captions; compare epochs and strengths in a grid.
- Built for 15 GB VRAM and 12.7 GB RAM: the image engine runs in its own process with a RAM guard, so the
  Colab session survives memory trouble.

Open `notebook.ipynb` in Colab and follow its two cells (Setup, Launch). Notes for developers: `docs/`.

## Privacy

Code only. Datasets, LoRAs and outputs live on the Colab VM and in your own **private** Hugging Face repos.
