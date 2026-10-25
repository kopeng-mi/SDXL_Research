# %% [md]
# ## 3. Train
# Every epoch is copied to `SDXLR/loras/sdxl/<RUN>/` and the newest state to `SDXLR/state/<RUN>/` as soon as it is saved.
# If Colab disconnects, run Setup, Dataset, Settings and this cell again: it continues where it stopped.
# %%
#@title Train
args = ["-m", "sdxlr.train", "--name", RUN, "--images", LOCAL, "--preset", PRESET, "--model", BASE_SRC,
        "--subject", SUBJECT, "--trigger", TRIGGER, "--class-word", CLASS_WORD, "--set", *sets]
if ROOT:
    args += ["--drive", ROOT] + (["--resume"] if RESUME else [])
nb.run(args)
