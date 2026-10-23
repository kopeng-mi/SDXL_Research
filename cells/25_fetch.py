# %% [md]
# ## Models and LoRAs
# Downloads to `/content/work`. LoRAs you train in this session are picked up
# automatically. Comma-separate several sources; `.zip` files of LoRAs are unpacked.
# %%
#@title Fetch base model + LoRAs
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"  #@param {type:"string"}
#@markdown org/repo/file on HF, an https URL, or a local path
LORAS = ""  #@param {type:"string"}
#@markdown e.g. `me/loras/set1.zip, me/loras/face_b.safetensors`

nb.run(["-m", "sdxlr.fetch", "model", BASE_MODEL])
if LORAS.strip():
    nb.run(["-m", "sdxlr.fetch", "loras", *[s.strip() for s in LORAS.split(",") if s.strip()]])
