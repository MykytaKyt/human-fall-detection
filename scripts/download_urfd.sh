#!/usr/bin/env bash
# Downloads the UR Fall Detection (URFD) dataset: RGB frames (cam0) for all
# 30 fall sequences and 40 ADL sequences, plus the per-frame label CSVs.
#
# Source: http://fenix.ur.edu.pl/~mkepski/ds/uf.html
# Citation: Kwolek & Kepski (2014), Computer Methods and Programs in
# Biomedicine, 117(3), 489-501. https://doi.org/10.1016/j.cmpb.2014.09.005
#
# Total size: ~4.3 GB (unpacked). Usage:
#   bash scripts/download_urfd.sh [data_dir]
set -euo pipefail

BASE="http://fenix.ur.edu.pl/~mkepski/ds/data"
DATA_DIR="${1:-data}"
FRAMES_DIR="$DATA_DIR/frames"
FALL_N=30
ADL_N=40

mkdir -p "$FRAMES_DIR"

echo "Downloading label CSVs..."
curl -s -L -o "$DATA_DIR/urfall-cam0-falls.csv" "$BASE/urfall-cam0-falls.csv"
curl -s -L -o "$DATA_DIR/urfall-cam0-adls.csv" "$BASE/urfall-cam0-adls.csv"

download_sequence() {
  local name="$1"
  local zip="$DATA_DIR/${name}-cam0-rgb.zip"
  local dir="$FRAMES_DIR/${name}"
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "skip $name (already downloaded)"
    return
  fi
  curl -s -L --max-time 300 -o "$zip" "$BASE/${name}-cam0-rgb.zip"
  mkdir -p "$dir"
  unzip -q -o "$zip" -d "$dir"
  rm -f "$zip"
  echo "done $name"
}

for i in $(seq -w 1 "$FALL_N"); do download_sequence "fall-${i}"; done
for i in $(seq -w 1 "$ADL_N"); do download_sequence "adl-${i}"; done

echo "URFD download complete: $FRAMES_DIR"
