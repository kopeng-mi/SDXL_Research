# %% [md]
# ## Train (Z-Image)
# LoRA for Z-Image via musubi-tuner, trained on the Z-Image **base** model and
# used on **Turbo** in the app. Defaults: `configs/train/zimage_person.yaml`.
# On the T4 this trains at 512px (1024px runs out of memory).
# %%
#@title Train Z-Image LoRA
RUN_NAME = "z_test_run"  #@param {type:"string"}
RESOLUTION = 512  #@param [512, 768] {type:"raw"}
EPOCHS = 10  #@param {type:"integer"}
REPEATS = 4  #@param {type:"integer"}
DIM = 16  #@param {type:"integer"}
LR = 1e-4  #@param {type:"number"}
EXTRA = ""  #@param {type:"string"}
#@markdown EXTRA: more overrides as `key=value key=value` (see the yaml)

nb.run(["-m", "sdxlr.train_zimage", "--name", RUN_NAME, "--images", DATASET_DIR,
        "--set", f"resolution={RESOLUTION}", f"epochs={EPOCHS}", f"repeats={REPEATS}",
        f"network_dim={DIM}", f"network_alpha={DIM}", f"lr={LR}", *EXTRA.split()])
