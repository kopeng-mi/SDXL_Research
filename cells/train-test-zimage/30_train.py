# %% [md]
# ## 3. Train
# Caches latents and text-encoder outputs first, then trains. Every epoch goes to `SDXLR/loras/zimage/<RUN>/`
# (the `_diffusers` files are what the app loads) and the newest state to `SDXLR/state/<RUN>/`.
# If Colab disconnects, run Setup, Dataset, Settings and this cell again: it continues where it stopped.
# %%
#@title Train
args = ["-m", "sdxlr.train_zimage", "--name", RUN, "--images", LOCAL, "--preset", PRESET, "--te", TE,
        "--subject", SUBJECT, "--trigger", TRIGGER, "--class-word", CLASS_WORD, "--set", *sets]
if ROOT:
    args += ["--drive", ROOT] + (["--resume"] if RESUME else [])
nb.run(args)
