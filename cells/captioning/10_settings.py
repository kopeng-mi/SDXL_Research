# %% [md]
# ## 1. Settings
# The dataset lives on Drive in `SDXLR/datasets/<DS_NAME>`; captions are written there as they finish.
# %%
#@title Settings
DS_NAME = "my_dataset"  #@param {type:"string"}
#@markdown **SRC**: where the photos come from (comma list): a Drive folder, a .zip link, an HF file
#@markdown `org/repo/file.zip`, or `test` (5 public photos). Leave empty to use the Drive dataset as it is.
SRC = ""  #@param {type:"string"}
MODEL = "qwen3vl-4b"  #@param ["qwen3vl-4b", "qwen3vl-2b", "qwen3vl-4b-abl", "qwen3vl-8b-abl", "qwen35-4b-abl (exp)", "qwen35-9b-abl (exp)", "joy", "wd"] {allow-input: true}
#@markdown **STYLE**: `id` for a person/character/object LoRA (no face or body words), `detailed` for style or
#@markdown concept LoRAs, `tags` for booru tags (WD tagger, ignores MODEL).
STYLE = "id"  #@param ["id", "detailed", "tags"]
TARGET = "sdxl"  #@param ["sdxl", "zimage"]
#@markdown TARGET sets the length: sdxl 30 to 50 words, zimage 80 to 150.
TRIGGER = "ohwx"  #@param {type:"string"}
CLASS_WORD = "woman"  #@param {type:"string"}
MAX_WORDS = 0  #@param {type:"integer"}
CUSTOM_PROMPT = ""  #@param {type:"string"}
#@markdown CUSTOM_PROMPT replaces the style prompt; `{cls}` and `{words}` are filled in.
QUANT = "auto"  #@param ["auto", "fp16", "nf4"]
BATCH = 0  #@param {type:"integer"}
OVERWRITE = False  #@param {type:"boolean"}

from sdxlr import drive, plan, paths
LOCAL = paths.DATASETS / DS_NAME
DRIVE_DS = (ROOT or drive.ROOT) / "datasets" / DS_NAME
repo, params_b, loader = paths.CAPTIONERS.get(MODEL, (MODEL, 4.0, "qwen"))
q = plan.caption_quant(params_b) if QUANT == "auto" else QUANT
print(f"dataset: {LOCAL}  (Drive: {DRIVE_DS if ROOT else 'off'})")
print(f"captioner: {repo}, {q}, about {plan.caption_vram(params_b, q):.1f} GB of the T4's 15 GB")
cap_args = ["-m", "sdxlr.caption", "--model", MODEL, "--style", STYLE, "--target", TARGET, "--trigger", TRIGGER,
            "--class-word", CLASS_WORD, "--max-words", MAX_WORDS, "--quant", QUANT, "--batch", BATCH]
if CUSTOM_PROMPT:
    cap_args += ["--prompt", CUSTOM_PROMPT]
