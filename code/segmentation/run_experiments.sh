#!/usr/bin/env bash
set -euo pipefail

# Runs CamVid segmentation experiments (PSPNet / FCN / DeepLabv3+)
#
# Usage:
#   bash code/segmentation/run_experiments.sh <run_id>
#
# Examples:
#   bash code/segmentation/run_experiments.sh seg-20251218
#   EPOCHS=100 bash code/segmentation/run_experiments.sh seg-long
#
# Overrides (env vars):
#   DATA_ROOT   default: data/camvid
#   EPOCHS      default: 50
#   CROP_H/W    default: 360/480
#   LR          default: 0.01
#   WD          default: 1e-4
#   SEED        default: 42
#   NUM_WORKERS default: 4
#   COMPILE     default: 1 (set to 0 to disable torch.compile)
#   BACKBONE    default: resnet50 (resnet18/resnet34/resnet50)
#   PRETRAIN    default: 1 (set to 0 to disable ImageNet backbone weights)
#
# Per-model batch size (env vars):
#   BS_PSPNET   default: 8
#   BS_FCN      default: 4
#   BS_DEEPLAB  default: 4

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <run_id>"
  exit 2
fi

RUN_ID="$1"

DATA_ROOT="${DATA_ROOT:-data/camvid}"
EPOCHS="${EPOCHS:-50}"
CROP_H="${CROP_H:-360}"
CROP_W="${CROP_W:-480}"
LR="${LR:-0.01}"
WD="${WD:-1e-4}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
COMPILE="${COMPILE:-1}"
BACKBONE="${BACKBONE:-resnet50}"
PRETRAIN="${PRETRAIN:-1}"

BS_PSPNET="${BS_PSPNET:-8}"
BS_FCN="${BS_FCN:-4}"
BS_DEEPLAB="${BS_DEEPLAB:-4}"

NO_COMPILE_FLAG=""
if [[ "$COMPILE" != "1" ]]; then
  NO_COMPILE_FLAG="--no-compile"
fi

echo "[INFO] run_id=$RUN_ID"
echo "[INFO] data_root=$DATA_ROOT epochs=$EPOCHS crop=${CROP_H}x${CROP_W} lr=$LR wd=$WD seed=$SEED workers=$NUM_WORKERS compile=$COMPILE backbone=$BACKBONE pretrain=$PRETRAIN"

run_one () {
  local model="$1"
  local bs="$2"
  local rid="${RUN_ID}-${model}-${BACKBONE}"

  local pretrain_flag="--no-backbone-pretrained"
  if [[ "$PRETRAIN" == "1" ]]; then
    pretrain_flag="--backbone-pretrained"
  fi

  echo
  echo "===== TRAIN: $model (run_id=$rid) ====="
  pixi run python code/segmentation/train_camvid.py \
    --data-root "$DATA_ROOT" \
    --model "$model" \
    --backbone "$BACKBONE" \
    $pretrain_flag \
    --epochs "$EPOCHS" \
    --batch-size "$bs" \
    --lr "$LR" \
    --weight-decay "$WD" \
    --crop-h "$CROP_H" \
    --crop-w "$CROP_W" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --run-id "$rid" \
    $NO_COMPILE_FLAG

  echo
  echo "===== CURVES: $model (run_id=$rid) ====="
  pixi run python code/segmentation/plot_training_curves.py \
    --run-dir "experiments/segmentation/${rid}"

  echo
  echo "===== EVAL+VIS (val): $model (run_id=$rid) ====="
  pixi run python code/segmentation/eval_camvid.py \
    --data-root "$DATA_ROOT" \
    --model "$model" \
    --backbone "$BACKBONE" \
    --split val \
    --batch-size 2 \
    --num-workers 2 \
    --checkpoint "experiments/segmentation/${rid}/best.pth" \
    --out-dir "report/figures/segmentation/${rid}/eval_${model}"
}

run_one "pspnet" "$BS_PSPNET"
run_one "fcn" "$BS_FCN"
run_one "deeplabv3plus" "$BS_DEEPLAB"

echo
echo "[DONE] Outputs:"
echo "  - experiments/segmentation/${RUN_ID}-*/"
echo "  - report/figures/segmentation/${RUN_ID}-*/"
