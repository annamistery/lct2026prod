#!/usr/bin/env bash
set -uo pipefail

OLD_ROOT="${1:-$HOME/LCT2026}"
OUTPUT="${2:-model_inventory.txt}"

exec >"$OUTPUT" 2>&1

section() {
  printf '\n================================================================================\n%s\n================================================================================\n' "$1"
}

list_files() {
  local root="$1"
  shift
  if [[ ! -d "$root" ]]; then
    echo "NOT FOUND: $root"
    return
  fi
  find "$root" -type f "$@" -printf '%p\t%k KB\n' 2>/dev/null | sort
}

section "REPORT"
date --iso-8601=seconds 2>/dev/null || date
printf 'host=%s\n' "$(hostname)"
printf 'old_root=%s\n' "$OLD_ROOT"
printf 'output=%s\n' "$OUTPUT"

section "SYSTEM"
uname -a
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "nvidia-smi: NOT FOUND"
command -v docker >/dev/null && docker --version || echo "docker: NOT FOUND"
docker compose version 2>/dev/null || echo "docker compose: NOT FOUND"
git lfs version 2>/dev/null || echo "git lfs: NOT FOUND"
python3 --version 2>/dev/null || true

section "OLD PROJECT"
if [[ -d "$OLD_ROOT" ]]; then
  printf 'resolved=%s\n' "$(readlink -f "$OLD_ROOT")"
  du -sh "$OLD_ROOT" 2>/dev/null || true
  find "$OLD_ROOT" -maxdepth 2 -type d -printf '%p\n' 2>/dev/null | sort
else
  echo "NOT FOUND: $OLD_ROOT"
fi

section "EXPECTED MODEL DIRECTORIES"
for directory in \
  "$OLD_ROOT/test_dino_v5/models/dino_aug_116" \
  "$OLD_ROOT/train_v5/dinov2_label_finetuned" \
  "$OLD_ROOT/train/dinov2_label_finetuned"
do
  echo "--- $directory"
  if [[ -d "$directory" ]]; then
    du -shL "$directory" 2>/dev/null || true
    find -L "$directory" -maxdepth 2 -type f -printf '%p\t%k KB\n' 2>/dev/null | sort
  else
    echo "NOT FOUND"
  fi
done

section "ALL MODEL FILES"
list_files "$OLD_ROOT" \( \
  -name 'adapter_config.json' -o \
  -name 'adapter_model.safetensors' -o \
  -name 'adapter_model.bin' -o \
  -name 'model.safetensors' -o \
  -name 'pytorch_model.bin' -o \
  -name 'config.json' -o \
  -name 'preprocessor_config.json' -o \
  -name 'processor_config.json' \
\)

section "ADAPTER CONFIG SUMMARIES"
while IFS= read -r -d '' file; do
  echo "--- $file"
  python3 - "$file" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        data = json.load(stream)
    for key in ("base_model_name_or_path", "peft_type", "task_type", "r", "lora_alpha", "target_modules"):
        print(f"{key}={data.get(key)!r}")
except Exception as exc:
    print(f"ERROR: {type(exc).__name__}: {exc}")
PY
done < <(find "$OLD_ROOT" -type f -name 'adapter_config.json' -print0 2>/dev/null)

section "INDEX AND CATALOG ARTIFACTS"
list_files "$OLD_ROOT" \( \
  -name 'index.faiss' -o \
  -name 'vectors.npy' -o \
  -name 'gallery_ids.json' -o \
  -name 'products.json' -o \
  -name 'meta.json' -o \
  -name 'twins_pairs.json' -o \
  -name 'dataset.json' -o \
  -name '*manifest.json' \
\)

section "INDEX META SUMMARIES"
while IFS= read -r -d '' file; do
  echo "--- $file"
  python3 - "$file" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, dict):
        for key in ("model", "head_path", "dim", "count", "index_type", "metric"):
            if key in data:
                print(f"{key}={data[key]!r}")
    else:
        print(f"json_type={type(data).__name__} length={len(data)}")
except Exception as exc:
    print(f"ERROR: {type(exc).__name__}: {exc}")
PY
done < <(find "$OLD_ROOT" -type f -name 'meta.json' -print0 2>/dev/null)

section "EXPECTED V5 INDEX"
V5_INDEX="$OLD_ROOT/test_dino_v5/indexes/index_aug_116"
if [[ -d "$V5_INDEX" ]]; then
  du -sh "$V5_INDEX" 2>/dev/null || true
  find "$V5_INDEX" -maxdepth 1 -type f -printf '%p\t%k KB\n' 2>/dev/null | sort
else
  echo "NOT FOUND: $V5_INDEX"
fi

section "HUGGING FACE DINO BASE"
HF_DIR="$HOME/.cache/huggingface/hub/models--facebook--dinov2-small"
if [[ -f "$HF_DIR/refs/main" ]]; then
  REVISION="$(cat "$HF_DIR/refs/main")"
  SNAPSHOT="$HF_DIR/snapshots/$REVISION"
  printf 'revision=%s\n' "$REVISION"
  printf 'snapshot=%s\n' "$SNAPSHOT"
  du -shL "$SNAPSHOT" 2>/dev/null || true
  find -L "$SNAPSHOT" -maxdepth 1 -type f -printf '%p\t%k KB\n' 2>/dev/null | sort
else
  echo "NOT FOUND: $HF_DIR/refs/main"
  find "$HF_DIR" -maxdepth 3 -type d -printf '%p\n' 2>/dev/null | sort
fi

section "IMAGE COUNTS"
for directory in \
  "$OLD_ROOT/test_dino_v5/base_dataset/labels" \
  "$OLD_ROOT/test_dino_v5/real_crops" \
  "$OLD_ROOT/dataset_v5/labels" \
  "$OLD_ROOT/dataset_all/images"
do
  if [[ -d "$directory" ]]; then
    count="$(find "$directory" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) 2>/dev/null | wc -l)"
    size="$(du -sh "$directory" 2>/dev/null | cut -f1)"
    printf '%s\tfiles=%s\tsize=%s\n' "$directory" "$count" "${size:-unknown}"
  else
    echo "NOT FOUND: $directory"
  fi
done

section "RESULT"
echo "Inventory completed. This report intentionally excludes .env files, process environments, credentials, and file contents other than selected non-secret JSON metadata fields."
