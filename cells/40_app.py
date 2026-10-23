# %% [md]
# ## Generate (UI)
# Opens the app; the cell keeps running while you use it. Stop the cell to
# close the app and free all VRAM (do that before training).
# %%
#@title Start the app
SHARE_LINK = False  #@param {type:"boolean"}
#@markdown off: Colab's private proxy link (only you). on: public gradio.live link as a fallback.
LOAD_MODEL = True  #@param {type:"boolean"}

try:
    from google.colab.output import eval_js
    print("App link (opens when the app says it is running):", eval_js("google.colab.kernel.proxyPort(7860)"))
except Exception:
    pass
args = ["-m", "sdxlr.app", "--port", 7860]
if SHARE_LINK: args += ["--share"]
if LOAD_MODEL: args += ["--model", BASE_MODEL]
nb.run(args)
