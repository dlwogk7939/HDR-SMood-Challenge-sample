#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

required_files=(
  "model.py"
  "requirements.txt"
  "weights/model.pt"
)

for f in "${required_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required file: $f" >&2
    exit 1
  fi
done

rm -f submission.zip
zip -q -r submission.zip model.py requirements.txt weights

echo "Created submission.zip"
echo "Contents:"
zipinfo -1 submission.zip
