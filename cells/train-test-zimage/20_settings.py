# %% [md]
# ## 2. Settings
# Defaults are the lightest setup: **fast** (Turbo + ostris training adapter, 512 px, fp8 weights, 6 epochs) with the
# **abliterated** text encoder; tests run on Turbo 4-bit (about 6 GB VRAM). `best` = Z-Image Base, 10 epochs (measured).
# Any photo count works: repeats and epochs scale so the total stays near the proven amount. For 100 photos that is
# 1 repeat x 6 epochs = 600 steps on fast (about 62 min est), or 1 x 4 = 400 steps on best (about 43 min).
# **TE**: the text encoder used for training. Generate with the same one (the Z-Image app has the same list).
# Leave EPOCHS / REPEATS / DIM / LR at 0 for the preset values; repeats 0 = auto from your photo count.
# %%
#@title Settings
SUBJECT = "person"  #@param ["person", "character/object", "style", "concept/pose"]
PRESET = "fast"  #@param ["fast", "best"]
TE = "abliterated"  #@param ["abliterated", "official", "heretic", "engineer"]
RUN = "my_zlora"  #@param {type:"string"}
RESOLUTION = 512  #@param [512, 768]
EPOCHS = 0  #@param {type:"integer"}
REPEATS = 0  #@param {type:"integer"}
DIM = 0  #@param {type:"integer"}
LR = 0.0  #@param {type:"number"}
RESUME = True  #@param {type:"boolean"}
#@markdown RESUME continues a stopped run with the same RUN name from its state on Drive. 768 px is about 2.3x slower (est).
import yaml
cfg = yaml.safe_load((REPO / "configs/train" / ("zimage_person.yaml" if PRESET == "best" else "zimage_fast.yaml")).read_text())
sets = [f"repeats={REPEATS}", f"resolution={RESOLUTION}"]
if EPOCHS: sets.append(f"epochs={EPOCHS}")
if DIM: sets += [f"network_dim={DIM}", f"network_alpha={DIM}"]  # alpha = dim, as the preset
if LR: sets.append(f"lr={LR}")
n = len(sorted(p for p in LOCAL.iterdir() if p.suffix.lower() in IMGS))
p = plan.train("zimage", SUBJECT, n, RESOLUTION, cfg["batch_size"], EPOCHS or cfg["epochs"], REPEATS)
print(p.summary())
