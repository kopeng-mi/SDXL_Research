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
#@markdown The link (trycloudflare.com) is printed below in a few seconds and changes every launch.
#@markdown Anyone with the link can open the app. Logs are also saved in /content/work/logs.

args = ["-m", "sdxlr.webui", "--port", 7860, "--tunnel", "cloudflare"]
if PULL_FROM_SYNC_REPO:
    args += ["--sync"]
nb.run(args)
