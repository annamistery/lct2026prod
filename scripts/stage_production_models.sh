#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
OLD_ROOT="${1:-$HOME/LCT2026}"
HF_DIR="$HOME/.cache/huggingface/hub/models--facebook--dinov2-small"
ADAPTER_SOURCE="$OLD_ROOT/test_dino_v5/models/dino_aug_116"
BASE_DESTINATION="$REPO_ROOT/models/dinov2-small"
ADAPTER_DESTINATION="$REPO_ROOT/models/dinov2_label_finetuned"
MANIFEST="$REPO_ROOT/models/MODEL_MANIFEST.sha256"

cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked working tree is not clean. Commit, discard, or stash existing changes first." >&2
  exit 1
fi

if ! git lfs version >/dev/null 2>&1; then
  echo "ERROR: Git LFS is not installed." >&2
  exit 1
fi

if [[ ! -f "$HF_DIR/refs/main" ]]; then
  echo "ERROR: DINOv2 base revision is missing: $HF_DIR/refs/main" >&2
  exit 1
fi

REVISION="$(cat "$HF_DIR/refs/main")"
BASE_SOURCE="$HF_DIR/snapshots/$REVISION"

for file in config.json preprocessor_config.json model.safetensors; do
  if [[ ! -e "$BASE_SOURCE/$file" ]]; then
    echo "ERROR: base model file is missing: $BASE_SOURCE/$file" >&2
    exit 1
  fi
done

for file in adapter_config.json adapter_model.bin preprocessor_config.json; do
  if [[ ! -f "$ADAPTER_SOURCE/$file" ]]; then
    echo "ERROR: adapter file is missing: $ADAPTER_SOURCE/$file" >&2
    exit 1
  fi
done

if [[ -e "$BASE_DESTINATION" || -e "$ADAPTER_DESTINATION" ]]; then
  echo "ERROR: destination model directories already exist. Refusing to overwrite them." >&2
  exit 1
fi

mkdir -p "$BASE_DESTINATION" "$ADAPTER_DESTINATION"
cp -aL "$BASE_SOURCE/." "$BASE_DESTINATION/"
cp -aL "$ADAPTER_SOURCE/." "$ADAPTER_DESTINATION/"

for file in \
  "$BASE_DESTINATION/config.json" \
  "$BASE_DESTINATION/preprocessor_config.json" \
  "$BASE_DESTINATION/model.safetensors" \
  "$ADAPTER_DESTINATION/adapter_config.json" \
  "$ADAPTER_DESTINATION/adapter_model.bin" \
  "$ADAPTER_DESTINATION/preprocessor_config.json" \
  "$REPO_ROOT/models/yolo_label.pt" \
  "$REPO_ROOT/web/models/yolov8_label.onnx"
do
  if [[ ! -f "$file" ]]; then
    echo "ERROR: required production model file is missing after copy: $file" >&2
    exit 1
  fi
done

{
  find models/dinov2-small models/dinov2_label_finetuned -type f -print0
  printf '%s\0' models/yolo_label.pt web/models/yolov8_label.onnx
} | sort -z | xargs -0 sha256sum > "$MANIFEST"

BASE_SIZE="$(du -sh "$BASE_DESTINATION" | cut -f1)"
ADAPTER_SIZE="$(du -sh "$ADAPTER_DESTINATION" | cut -f1)"

echo "DINO base: $BASE_SIZE"
echo "DINO adapter: $ADAPTER_SIZE"
echo "Checksums: $MANIFEST"
sha256sum --check "$MANIFEST"

git add .gitattributes models/dinov2-small models/dinov2_label_finetuned models/MODEL_MANIFEST.sha256

echo
git status --short
echo
read -r -p "Commit and push all production models to origin? [y/N] " answer
if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
  echo "Models are copied and staged, but not committed."
  exit 0
fi

git commit \
  -m "Bundle complete production model set." \
  -m $'Store the local DINOv2 base model and final dino_aug_116 LoRA adapter in Git LFS so a clean clone can run without external caches or downloads.\n\nGenerated with [Devin](https://devin.ai)\n\nCo-Authored-By: Devin <158243242+devin-ai-integration[bot]@users.noreply.github.com>'
git push

echo "Production models committed and pushed successfully."
