# %% [md]
# ## Launch
# Opens the app. Everything else (datasets, captions, training, compare,
# generation, downloads, presets) happens in the app. The cell keeps running
# while the app is open and shows the app's log; stop it to close the app.
# %%
#@title Launch the app
PULL_FROM_SYNC_REPO = True  #@param {type:"boolean"}
#@markdown pulls your presets + LoRAs from the sync repo saved in Library
#@markdown
#@markdown Open **http://sdxlr:7860** on a device signed in to your Tailscale (the address is also
#@markdown printed below). Needs TS_AUTHKEY in Secrets. Logs are saved in /content/work/logs.
#@markdown
#@markdown **Keep this Colab tab open** and click in it now and then: Colab can stop a session whose notebook
#@markdown sits idle (often after about 90 minutes), even while you use the app. Everything on the VM is gone then;
#@markdown trained LoRAs survive in your HF sync repo.

args = ["-m", "sdxlr.webui", "--port", 7860]
if PULL_FROM_SYNC_REPO:
    args += ["--sync"]
nb.serve(args)  # starts the app again if it crashes
