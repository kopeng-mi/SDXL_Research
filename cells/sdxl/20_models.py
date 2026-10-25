# %% [md]
# ## 2. Models
# Pick a base or paste any SDXL checkpoint: an HF file `org/repo/file.safetensors`, a Civitai or https link,
# or a Drive path. Checkpoints in `SDXLR/models/sdxl` and LoRAs in `SDXLR/loras/sdxl` (your trained ones)
# show up in Fooocus by themselves when Drive is on.
# %%
#@title Models
MODEL = "sdxl-base"  #@param ["sdxl-base", "pony-v6", "illustrious-0.1", "noobai-eps-1.1"]
CUSTOM_MODEL = ""  #@param {type:"string"}
EXTRA_LORAS = ""  #@param {type:"string"}
#@markdown EXTRA_LORAS: comma list of LoRA links or .zip files (HF, Civitai, https, Drive).
from sdxlr import fetch, loraformat, paths
src = CUSTOM_MODEL.strip() or paths.SDXL_BASES[MODEL]
ckpt = fetch.fetch(src, fooocus.CKPTS)
loraformat.check_sdxl_checkpoint(ckpt)
if loraformat.is_vpred(ckpt):
    raise SystemExit(f"{ckpt.name} is a v-prediction model; Fooocus 2.5.5 cannot run those (the images come out grey). "
                     "Use an eps model here, or the train-test-sdxl notebook to test v-pred LoRAs.")
for s in [x.strip() for x in EXTRA_LORAS.split(",") if x.strip()]:
    print("LoRAs:", [p.name for p in fetch.fetch_loras(s)])
CKPT_DIRS = list(dict.fromkeys([fooocus.CKPTS, ckpt.parent] + ([ROOT / "models" / "sdxl"] if ROOT else [])))
LORA_DIRS = [fooocus.LORAS, paths.LORAS] + ([ROOT / "loras" / "sdxl"] if ROOT else [])
print(f"default model: {ckpt.name}")
print(f"checkpoints: {len(fooocus.checkpoints(CKPT_DIRS))}, LoRAs: {len(fooocus.checkpoints(LORA_DIRS))}")
