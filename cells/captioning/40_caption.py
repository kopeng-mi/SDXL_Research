# %% [md]
# ## 4. Caption all
# Photos that already have a caption are skipped unless OVERWRITE is on. Each caption is copied to Drive
# as soon as it is written; stop and run again to continue.
# %%
#@title Caption all
args = cap_args + ["--images", LOCAL]
if OVERWRITE:
    args += ["--overwrite"]
if ROOT:
    args += ["--mirror", DRIVE_DS]
nb.run(args)
