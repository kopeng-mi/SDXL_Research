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
#@markdown The address is printed below in a few seconds. Cloudflare links change every launch and
#@markdown anyone with one can open the app. Logs are also saved in /content/work/logs.

ACCESS = "auto"  #@param ["auto", "tailscale", "cloudflare"]
#@markdown auto: Tailscale (fixed private address http://sdxlr:7860) when TS_AUTHKEY is in Secrets, else Cloudflare
args = ["-m", "sdxlr.webui", "--port", 7860, "--access", ACCESS]
if PULL_FROM_SYNC_REPO:
    args += ["--sync"]
nb.run(args)
