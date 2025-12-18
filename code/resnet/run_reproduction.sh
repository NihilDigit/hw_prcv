#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="experiments/resnet/reproduction/${RUN_ID}"

EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SHORTCUT_TYPE="${SHORTCUT_TYPE:-A}"
NO_COMPILE="${NO_COMPILE:-1}" # default off for stability; set 0 to enable torch.compile

echo "Run ID: ${RUN_ID}"
echo "Out root: ${OUT_ROOT}"
echo "epochs=${EPOCHS} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} shortcut_type=${SHORTCUT_TYPE} no_compile=${NO_COMPILE}"

mkdir -p "${OUT_ROOT}"

has_completed () {
  local model="$1"
  local metrics="${OUT_ROOT}/${model}/metrics.json"
  if [[ ! -f "${metrics}" ]]; then
    return 1
  fi
  pixi run python - <<PY
import json
from pathlib import Path
p = Path(${metrics@Q})
rows = json.loads(p.read_text())
print(len(rows))
raise SystemExit(0 if (rows and int(rows[-1].get("epoch", -1)) == int(${EPOCHS})) else 1)
PY
}

run_one () {
  local model="$1"
  local out_dir="${OUT_ROOT}/${model}"
  mkdir -p "${out_dir}"
  if has_completed "${model}" >/dev/null 2>&1; then
    echo "=== Skip ${model} (already has ${EPOCHS} epochs) ==="
    return 0
  fi
  echo "=== Training ${model} -> ${out_dir} ==="
  local extra_args=()
  if [[ "${NO_COMPILE}" == "1" ]]; then
    extra_args+=(--no-compile)
  fi
  pixi run python code/resnet/train_cifar.py \
    --model "${model}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --shortcut-type "${SHORTCUT_TYPE}" \
    --out-dir "${out_dir}" \
    "${extra_args[@]}"
}

run_one plain20
run_one plain56
run_one resnet20
run_one resnet56

echo "=== Plotting curves ==="
pixi run python code/resnet/plot_reproduction.py --run-dir "${OUT_ROOT}"
echo "Done. Outputs: ${OUT_ROOT} and report/figures/resnet/${RUN_ID}"
