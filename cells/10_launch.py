# %% [md]
# ## Launch
# Opens the app. Everything else (datasets, captions, training, compare,
# generation, downloads, presets) happens in the app. The cell keeps running
# while the app is open; stop it to close the app.
# %%
#@title Launch the app
PRIVATE_LINK = True  #@param {type:"boolean"}
#@markdown on: Colab's private link (only you). off: public gradio.live link (use if the private link does not open).
PULL_FROM_SYNC_REPO = True  #@param {type:"boolean"}
#@markdown pulls your presets + LoRAs from the sync repo saved in Library → Sync

args = ["-m", "sdxlr.app", "--port", 7860]
if not PRIVATE_LINK:
    args += ["--share"]
if PULL_FROM_SYNC_REPO:
    args += ["--sync"]
if PRIVATE_LINK:
    try:
        from google.colab.output import eval_js
        print("Open the app (ready when the log below says 'Running on'):", eval_js("google.colab.kernel.proxyPort(7860)"))
    except Exception:
        print("Not in Colab: open http://127.0.0.1:7860")
nb.run(args)
