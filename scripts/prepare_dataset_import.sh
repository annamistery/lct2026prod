#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <archive.zip|archive.tar.gz> <batch_id>" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
ARCHIVE="$(readlink -f "$1")"
BATCH_ID="$2"
DESTINATION="$REPO_ROOT/imports/staging/$BATCH_ID"

if [[ ! "$BATCH_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$ ]]; then
  echo "ERROR: batch_id may contain only letters, digits, underscore, and hyphen." >&2
  exit 1
fi
if [[ ! -f "$ARCHIVE" ]]; then
  echo "ERROR: archive not found: $ARCHIVE" >&2
  exit 1
fi
if [[ -e "$DESTINATION" ]]; then
  echo "ERROR: destination already exists: $DESTINATION" >&2
  exit 1
fi

mkdir -p "$DESTINATION"
trap 'rm -rf -- "$DESTINATION"' ERR
python3 - "$ARCHIVE" "$DESTINATION" <<'PY'
import shutil
import stat
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

archive = Path(sys.argv[1])
destination = Path(sys.argv[2]).resolve()


def target(name: str) -> Path:
    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"unsafe archive path: {name}")
    result = destination.joinpath(*normalized.parts).resolve()
    if result != destination and destination not in result.parents:
        raise ValueError(f"archive path escapes destination: {name}")
    return result


if zipfile.is_zipfile(archive):
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            output = target(member.filename)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"symbolic links are forbidden: {member.filename}")
            if member.is_dir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as input_stream, output.open("xb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream)
elif tarfile.is_tarfile(archive):
    with tarfile.open(archive, "r:*") as source:
        for member in source.getmembers():
            output = target(member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"links and devices are forbidden: {member.name}")
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                output.parent.mkdir(parents=True, exist_ok=True)
                input_stream = source.extractfile(member)
                if input_stream is None:
                    raise ValueError(f"cannot read archive member: {member.name}")
                with input_stream, output.open("xb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream)
else:
    raise ValueError("archive must be ZIP or TAR")
PY
trap - ERR

MANIFESTS="$(find "$DESTINATION" -maxdepth 1 -type f \( -iname '*.json' -o -iname '*.csv' \) -printf '%f\n' | sort)"
IMAGE_COUNT="$(find "$DESTINATION" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l)"

echo "Batch ready: $BATCH_ID"
echo "Directory: $DESTINATION"
echo "Images: $IMAGE_COUNT"
echo "Manifests:"
printf '%s\n' "$MANIFESTS"
