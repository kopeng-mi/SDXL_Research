# %% [md]
# ## 2. Settings
# **PRESET**: `best` = the measured settings (1024 px, about 30 min for any photo count);
# `fast` = 768 px, text encoders frozen, 10 epochs (an estimate until measured).
# Leave EPOCHS / REPEATS / DIM / LR at 0 for the preset values; repeats 0 = auto from your photo count.
# %%
#@title Settings
BASE = "sdxl-base"  #@param ["sdxl-base", "pony-v6", "illustrious-0.1", "noobai-eps-1.1", "noobai-vpred-1.0"]
CUSTOM_BASE = ""  #@param {type:"string"}
#@markdown CUSTOM_BASE: an HF file `org/repo/file.safetensors`, a Civitai or https link, or a Drive path; overrides BASE.
SUBJECT = "person"  #@param ["person", "character/object", "style", "concept/pose"]
PRESET = "best"  #@param ["best", "fast"]
RUN = "my_lora"  #@param {type:"string"}
EPOCHS = 0  #@param {type:"integer"}
REPEATS = 0  #@param {type:"integer"}
DIM = 0  #@param {type:"integer"}
LR = 0.0  #@param {type:"number"}
RESUME = True  #@param {type:"boolean"}
#@markdown RESUME continues a stopped run with the same RUN name from its state on Drive.
import yaml
cfg = yaml.safe_load((REPO / "configs/train" / ("sdxl_person.yaml" if PRESET == "best" else "sdxl_fast.yaml")).read_text())
BASE_SRC = CUSTOM_BASE or BASE
sets = []
if EPOCHS: sets.append(f"epochs={EPOCHS}")
sets.append(f"repeats={REPEATS}")  # 0 = auto
if DIM: sets += [f"network_dim={DIM}", f"network_alpha={round(DIM * cfg['network_alpha'] / cfg['network_dim'])}"]  # keeps alpha/dim
if LR: sets.append(f"unet_lr={LR}")
n = len(sorted(p for p in LOCAL.iterdir() if p.suffix.lower() in IMGS))
p = plan.train("sdxl", SUBJECT, n, cfg["resolution"], cfg["batch_size"], EPOCHS or cfg["epochs"], REPEATS)
print(p.summary())
print(f"LoRA scale = alpha/dim x strength = {plan.lora_scale(cfg['network_alpha'], cfg['network_dim']):.2f} x strength")
