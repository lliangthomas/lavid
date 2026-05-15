#!/bin/bash
set -e

SHORT_DS="/home/ubuntu/CUB_200_2011"
SHORT_DS_NAME="cub200"
WATERBIRD=""
WATERBIRD_NAME="waterbird"
TEACHER_CKPT_DIR=""

SEED=27
TEACHER=qwen2.5-7b
FEATURE_TYPE=chat_50-5


# 1) resnet18 baseline
uv run train.py \
    --data-dir $SHORT_DS \
    --dataset $SHORT_DS_NAME \
    --model resnet50 \
    --seed $SEED \
    --wandb-name "smoke_resnet18_${SEED}"


# 2) resnet18 + LaViD KD
uv run train.py \
    --data-dir $SHORT_DS \
    --dataset $SHORT_DS_NAME \
    --model resnet18 \
    --teacher $TEACHER \
    --kd lavid \
    --gamma 1 --beta 40 \
    --feature-type $FEATURE_TYPE \
    --seed $SEED \
    --wandb-name "smoke_resnet18_lavid_${SEED}"


# 3) resnet18 + traditional KD  (the OTHER get_best_checkpoint call site)
uv run train.py \
    --data-dir $SHORT_DS \
    --dataset $SHORT_DS_NAME \
    --model resnet18 \
    --teacher resnet50 \
    --teacher-path $TEACHER_CKPT_DIR \
    --kd dkd \
    --gamma 1 --beta 1 \
    --seed $SEED \
    --wandb-name "smoke_resnet18_dkd_${SEED}"


# 4) resnet18 on Waterbird
uv run train.py \
    --data-dir $WATERBIRD \
    --dataset $WATERBIRD_NAME \
    --model resnet18 \
    --seed $SEED \
    --wandb-name "smoke_resnet18_waterbird_${SEED}"


# 5) CLIP baseline
uv run train.py \
    --data-dir $SHORT_DS \
    --dataset $SHORT_DS_NAME \
    --model clip \
    --seed $SEED \
    --wandb-name "smoke_clip_${SEED}"


# 6) CLIP + LaViD loss
uv run train.py \
    --data-dir $SHORT_DS \
    --dataset $SHORT_DS_NAME \
    --model clip \
    --teacher $TEACHER \
    --kd lavid \
    --gamma 1 --beta 40 \
    --feature-type $FEATURE_TYPE \
    --seed $SEED \
    --wandb-name "smoke_clip_lavid_${SEED}"


# 7) CLIP + open-world
uv run train.py \
    --data-dir $SHORT_DS \
    --dataset $SHORT_DS_NAME \
    --model clip \
    --open-world \
    --seed $SEED \
    --wandb-name "smoke_clip_openworld_${SEED}"


# 8) standalone evaluate.py
CLIP_CKPT_DIR=$(ls -td output/smoke_clip_${SEED}-* | head -1)
uv run evaluate.py \
    --weights-path "$CLIP_CKPT_DIR" \
    --data-dir $SHORT_DS \
    --dataset $SHORT_DS_NAME \
    --model clip \
    --wandb-name "smoke_clip_eval_${SEED}"


# 9) get_best_checkpoint unit test  (no GPU, no wandb)
uv run python - <<'PY'
import os, tempfile
from lavid.utils import get_best_checkpoint
with tempfile.TemporaryDirectory() as d:
    for name in ["epoch=1-val_loss=0.5000.ckpt",
                 "epoch=2-val_loss=0.3000.ckpt",
                 "epoch=3-val_loss=0.4000.ckpt",
                 "random-name.ckpt"]:
        open(os.path.join(d, name), "w").close()
    best = get_best_checkpoint(d)
    assert best.endswith("val_loss=0.3000.ckpt"), best
    print("get_best_checkpoint OK:", os.path.basename(best))
PY
