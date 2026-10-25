# %% [md]
# ## 1. Models (optional)
# Downloads now so the app loads fast; you can also pick everything inside the app.
# A custom Z-Image checkpoint (HF file, Civitai or https link, or Drive path) is loaded as 4-bit the first time
# and cached. Models in `SDXLR/models/zimage` and LoRAs in `SDXLR/loras/zimage` are listed in the app when Drive is on.
# %%
#@title Models
MODEL = "turbo 4-bit (unsloth)"  #@param ["turbo 4-bit (unsloth)", "turbo Q8", "turbo Q6", "turbo Q5", "turbo Q4"]
CUSTOM_MODEL = ""  #@param {type:"string"}
TE = "abliterated"  #@param ["abliterated", "official", "heretic", "engineer"]
#@markdown TE: use the one your LoRA was trained with. `abliterated`/`heretic` follow prompts without refusing;
#@markdown `engineer` also writes the best prompts with **Write a detailed prompt**.
LORAS = ""  #@param {type:"string"}
#@markdown LORAS: comma list of LoRA links or .zip files.
from sdxlr import paths
if CUSTOM_MODEL.strip():
    nb.run(["-m", "sdxlr.fetch", "zimage", CUSTOM_MODEL.strip()])
else:
    nb.run(["-m", "sdxlr.fetch", "prefetch-zimage", paths.Z_MODELS[MODEL], paths.Z_ENCODERS[TE]])
srcs = [x.strip() for x in LORAS.split(",") if x.strip()]
if srcs:
    nb.run(["-m", "sdxlr.fetch", "loras", *srcs])
