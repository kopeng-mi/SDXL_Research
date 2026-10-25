#!/usr/bin/env bash
# One T4 session that runs what the five notebooks run, with short epochs, and times each step.
# Replaces tools/train_check.py for step 5. Drive is a plain folder here (FAKE_DRIVE): drive.py only copies files.
#
#   bash tools/t4_nb_check.sh /content/t4_imgs   (repo cloned at /content/SDXL_Research, any branch)
#
# Writes /content/logs/nb_check.md (one line per step: seconds, ok/FAILED) and each step's log next to it.
set -u
IMGS=${1:?images folder}
REPO=/content/SDXL_Research
PY=/content/venv/bin/python
LOGS=/content/logs
DRV=/content/fake_drive/SDXLR
DS=/content/work/datasets/nbcheck
export PYTHONPATH=$REPO/src PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$LOGS" "$DRV"
REPORT=$LOGS/nb_check.md
echo "| step | seconds | result |" > "$REPORT"; echo "|---|---|---|" >> "$REPORT"

step() {  # step NAME CMD...
  local name=$1; shift
  local t0=$SECONDS
  echo "== $name" | tee -a "$LOGS/nb_check.log"
  if (cd "$REPO" && "$@") > "$LOGS/$name.log" 2>&1; then r=ok; else r=FAILED; fi
  echo "| $name | $((SECONDS - t0)) | $r |" | tee -a "$REPORT"
}

step bootstrap python3 -m sdxlr.bootstrap --groups caption,train-sdxl,train-z,app
step import $PY -m sdxlr.data --dest $DS "$IMGS" --res 1024 --kind sdxl

# captioning: 4 images per model (fp16 batch 4, nf4 batch 2, Qwen3.5 exp)
for m in qwen3vl-4b qwen3vl-2b qwen3vl-4b-abl qwen3vl-8b-abl "qwen35-4b-abl (exp)"; do
  slug=$(echo "$m" | tr -cd 'a-z0-9-')
  step "caption-$slug" $PY -m sdxlr.caption --images $DS --out /content/work/cap_$slug --model "$m" --limit 4 --overwrite \
       --trigger ohwx --class-word woman
done
step caption-zimage-style $PY -m sdxlr.caption --images $DS --out /content/work/cap_z --target zimage --limit 2 --overwrite
step caption-all $PY -m sdxlr.caption --images $DS --trigger ohwx --class-word woman --mirror $DRV/datasets/nbcheck

# SDXL: fast preset, 1 epoch with Drive, then resume to 2 (must train only epoch 2)
step sdxl-fast-e1 $PY -m sdxlr.train --name nbs --images $DS --preset fast --drive $DRV --resume --set epochs=1 repeats=1
step sdxl-fast-e2-resume $PY -m sdxlr.train --name nbs --images $DS --preset fast --drive $DRV --resume --set epochs=2 repeats=1
step sdxl-compare $PY -m sdxlr.compare --name nbs --model stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors --lora-dir /content/work/runs/nbs --epochs all \
     --strengths 0.8 1.0 --prompts "photo of ohwx woman, portrait" --steps 20 --drive-out $DRV/compare/nbs

# Z-Image: fast preset (Turbo + ostris) with the abliterated TE, 1 epoch
step z-fast-e1 $PY -m sdxlr.train_zimage --name nbz --images $DS --preset fast --te abliterated --drive $DRV --resume \
     --set epochs=1 repeats=1
step z-compare $PY -m sdxlr.compare --name nbz --zimage --model "turbo 4-bit (unsloth)" --te abliterated --cfg 0 --steps 9 \
     --lora-dir /content/work/runs/nbz --epochs all --strengths 1.0 --prompts "A photo of ohwx woman by a window"

# prompt writer on the loaded TE (official and abliterated)
step enhance $PY - <<'EOF'
import time
from sdxlr.engine.models import Engine
from sdxlr.engine import enhance
from sdxlr.paths import Z_ENCODERS, Z_MODELS
e = Engine()
for te in ("official", "abliterated"):
    print(e.load_zimage(Z_MODELS["turbo 4-bit (unsloth)"], Z_ENCODERS[te], "nf4"))
    for mode in ("official", "engineer"):
        t = time.time()
        out = enhance.write(e.pipe, "a woman reading in a cafe at night", mode)
        n = len(out.split())
        print(f"{te}/{mode}: {n} words in {time.time() - t:.1f}s ({n * 1.3 / (time.time() - t):.1f} tok/s est)\n{out}\n")
EOF

# Fooocus: install, launch, wait for the page, stop
step fooocus-setup python3 -c "import sys; sys.path.insert(0, '$REPO/src'); from sdxlr import fooocus; fooocus.setup()"
step fooocus-launch bash -c "
  python3 -c \"import sys; sys.path.insert(0, '$REPO/src'); from sdxlr import fetch, fooocus, paths;
c = fetch.fetch(paths.SDXL_BASES['sdxl-base'], fooocus.CKPTS);
fooocus.write_config(c.name, [fooocus.CKPTS], [fooocus.LORAS], fooocus.OUTPUTS)\" &&
  cd /content/Fooocus && (MPLBACKEND=Agg /content/fooocus-venv/bin/python launch.py --listen --port 7865 --always-high-vram \
     --disable-analytics --disable-preset-download > $LOGS/fooocus-server.log 2>&1 &) &&
  ok=0; for i in \$(seq 1 120); do curl -sf -o /dev/null http://127.0.0.1:7865 && ok=1 && echo up after \$((i*5))s && break; sleep 5; done;
  pkill -f 'launch.py --listen'; [ \$ok = 1 ]"

echo "== done"; cat "$REPORT"
