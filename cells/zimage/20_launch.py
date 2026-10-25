# %% [md]
# ## 2. Launch
# Open **http://zimage:7860** on a device signed in to your Tailscale (needs TS_AUTHKEY in Secrets).
# Pages: Generate (with edit and upscale), Library, History, Activity. Pick the same TE in the app as above.
# **Keep this Colab tab open** and click in it now and then: Colab can stop an idle notebook (often after about
# 90 minutes) even while you use the app. The cell keeps running while the app is open; stop it to close the app.
# %%
#@title Launch the app
args = ["-m", "sdxlr.webui", "--port", 7860, "--host-name", "zimage", "--zimage-only"]
if ROOT:
    args += ["--drive", ROOT]
nb.serve(args)  # starts the app again if it crashes
