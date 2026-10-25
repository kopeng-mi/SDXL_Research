# %% [md]
# ## 2. Import
# Copies the Drive dataset to the VM (fast reads), adds SRC (duplicates skipped, phone photos turned upright),
# copies the result back to Drive and prints the checks.
# %%
#@title Import
CHECK_RES = 1024  #@param {type:"integer"}
#@markdown CHECK_RES: training size to check against (SDXL 1024, Z-Image 512 or 768).
LOCAL.mkdir(parents=True, exist_ok=True)
if ROOT and DRIVE_DS.exists():
    print(f"from Drive: {drive.pull(DRIVE_DS, LOCAL)} files copied")
srcs = [s.strip() for s in SRC.split(",") if s.strip()]
nb.run(["-m", "sdxlr.data", "--dest", LOCAL, *srcs, "--res", CHECK_RES, "--kind", TARGET])
if ROOT:
    print(f"to Drive: {drive.pull(LOCAL, DRIVE_DS)} files copied")
