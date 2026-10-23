# %% [md]
# ## Generate (UI)
# Opens the app; the cell keeps running while you use it. Stop the cell to
# close the app and free all VRAM (do that before training).
# %%
#@title Start the app
SHARE_LINK = False  #@param {type:"boolean"}
#@markdown off: Colab's private proxy link (only you). on: public gradio.live link as a fallback.
LOAD_AT_START = "SDXL"  #@param ["SDXL", "Z-Image", "none"]
#@markdown SDXL uses BASE_MODEL from the fetch cell. Z-Image: pick the model and text encoder (or paste any org/repo/file.gguf / org/repo).
Z_MODEL = "turbo Q8"  #@param ["turbo Q8", "turbo Q6", "turbo Q5", "turbo Q4"] {allow-input: true}
Z_TEXT_ENCODER = "official"  #@param ["official", "abliterated"] {allow-input: true}
Z_ENCODER_PRECISION = "nf4"  #@param ["nf4", "int8", "fp16"]

try:
    from google.colab.output import eval_js
    print("App link (opens when the app says it is running):", eval_js("google.colab.kernel.proxyPort(7860)"))
except Exception:
    pass
args = ["-m", "sdxlr.app", "--port", 7860]
if SHARE_LINK: args += ["--share"]
if LOAD_AT_START == "SDXL":
    args += ["--model", BASE_MODEL]
elif LOAD_AT_START == "Z-Image":
    args += ["--zimage", Z_MODEL, "--te", Z_TEXT_ENCODER, "--te-quant", Z_ENCODER_PRECISION]
nb.run(args)
