# %% [md]
# ## Train
# Defaults: `configs/train/sdxl_person.yaml` (proven T4 settings).
# With `HF_REPO` set, LoRAs + resume state upload to that **private** repo each
# epoch; tick `RESUME` after a disconnect to continue.
# %%
#@title Train SDXL LoRA
RUN_NAME = "test_run"  #@param {type:"string"}
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"  #@param {type:"string"}
#@markdown org/repo/file on HF, an https URL, or a local path
HF_REPO = ""  #@param {type:"string"}
RESUME = False  #@param {type:"boolean"}
EPOCHS = 15  #@param {type:"integer"}
REPEATS = 6  #@param {type:"integer"}
BATCH = 3  #@param {type:"integer"}
LORA_TYPE = "LoRA"  #@param ["LoRA", "LoCon"]
DIM = 24  #@param {type:"integer"}
ALPHA = 16  #@param {type:"integer"}
UNET_LR = 2e-4  #@param {type:"number"}
TE_LR = 2e-5  #@param {type:"number"}
EXTRA = ""  #@param {type:"string"}
#@markdown EXTRA: more overrides as `key=value key=value` (see the yaml)

args = ["-m", "sdxlr.train", "--name", RUN_NAME, "--images", DATASET_DIR, "--model", BASE_MODEL,
        "--set", f"epochs={EPOCHS}", f"repeats={REPEATS}", f"batch_size={BATCH}",
        f"lora_type={LORA_TYPE}", f"network_dim={DIM}", f"network_alpha={ALPHA}",
        f"unet_lr={UNET_LR}", f"text_encoder_lr={TE_LR}", *EXTRA.split()]
if HF_REPO: args += ["--hf-repo", HF_REPO]
if RESUME: args += ["--resume"]
nb.run(args)
