#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/home/lhc/data/gudsen/asr/dataset"
mkdir -p "$OUT_DIR"

while f="$(find "$OUT_DIR" -type f \( -name '*.tgz' -o -name '*.tar.gz' -o -name '*.tar' \) | head -n 1)"; [[ -n "${f:-}" ]]; do
  d="$(dirname "$f")"
  case "$f" in
    *.tar) tar -xf  "$f" -C "$d" ;;
    *)     tar -xzf "$f" -C "$d" ;;
  esac
  rm -f -- "$f"
done
