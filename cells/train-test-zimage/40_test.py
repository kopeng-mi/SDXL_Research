# %% [md]
# ## 4. Test
# A grid: rows = the last epochs (x prompts), columns = LoRA strengths, same seed everywhere.
# Runs on Z-Image Turbo 4-bit with the training TE: about 36 s per image plus the model load.
# %%
#@title Test
EPOCHS_TO_TEST = "last:3"  #@param {type:"string"}
#@markdown last:K, all, or a list like 4,7,10
STRENGTHS = "0.6 0.8 1.0"  #@param {type:"string"}
PROMPT_1 = "A photo of {t} {c} sitting by a window, soft daylight on the face, relaxed expression, shallow depth of field"  #@param {type:"string"}
PROMPT_2 = "A full-body photo of {t} {c} walking down a busy city street at dusk, neon signs, candid"  #@param {type:"string"}
STEPS = 9  #@param {type:"integer"}
SEED = 42  #@param {type:"integer"}
prompts = [x.replace("{t}", TRIGGER).replace("{c}", CLASS_WORD) for x in (PROMPT_1, PROMPT_2) if x.strip()]
run_dir = paths.WORK / "runs" / RUN
if ROOT and not any(run_dir.glob("*.safetensors")):  # a new VM: test what Drive has
    drive.pull(ROOT / "loras" / "zimage" / RUN, run_dir)
args = ["-m", "sdxlr.compare", "--name", RUN, "--zimage", "--model", "turbo 4-bit (unsloth)", "--te", TE, "--cfg", 0,
        "--lora-dir", run_dir, "--epochs", EPOCHS_TO_TEST, "--strengths", *STRENGTHS.split(),
        "--prompts", *prompts, "--steps", STEPS, "--seed", SEED]
if ROOT:
    args += ["--drive-out", ROOT / "compare" / RUN]
nb.run(args)
from IPython.display import Image as _Img, display
display(_Img(filename=str(paths.WORK / "compare" / RUN / "grid.png")))
