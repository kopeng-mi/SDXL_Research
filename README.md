# SDXL_Research

Five Colab notebooks for the free tier (T4): caption a dataset, train and test SDXL and Z-Image LoRAs,
and generate with Fooocus (SDXL) or our Z-Image app, both opened over Tailscale. Datasets and LoRAs live
on your Google Drive (`MyDrive/SDXLR`).

| notebook | what it does |
|---|---|
| `notebooks/captioning.ipynb` | import photos to Drive, caption them (Qwen3-VL incl. abliterated, JoyCaption, WD tags), review |
| `notebooks/train-test-sdxl.ipynb` | SDXL LoRA (kohya): Fast/Best presets, auto repeats, v-pred bases, resume from Drive, epoch grid |
| `notebooks/train-test-zimage.ipynb` | Z-Image LoRA (musubi): Base or Turbo+adapter, text encoder choice, resume, epoch grid |
| `notebooks/sdxl.ipynb` | Fooocus 2.5.5 with your checkpoints and LoRAs at http://sdxl:7865 |
| `notebooks/zimage.ipynb` | Z-Image generate, edit, upscale, "Write a detailed prompt", custom checkpoints, at http://zimage:7860 |

Open one in Colab (from the `step5-notebooks` branch), run **Setup**, then the cells in order.
Notes for developers: `docs/` (plan: `docs/NOTEBOOKS_PLAN.md`); cells are built from `cells/` by `tools/build_notebook.py`.

## Privacy

Code only. Datasets, LoRAs and outputs live on the Colab VM and in your own Google Drive.
