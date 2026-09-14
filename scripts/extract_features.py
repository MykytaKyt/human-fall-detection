#!/usr/bin/env python3
"""CLI: run YOLOv8-Pose + DDL feature extraction over the downloaded URFD
frames and save the result as data/features.parquet.

Usage:
    python scripts/extract_features.py --data-dir data --device 0
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urfd_cascade.extract import extract_dataset_features  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory with URFD CSVs")
    parser.add_argument("--frames-dir", default=None, help="Directory with downloaded frames (default: <data-dir>/frames)")
    parser.add_argument("--out", default=None, help="Output parquet path (default: <data-dir>/features.parquet)")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Ultralytics pose model")
    parser.add_argument("--device", default=0, help="Torch device (0 for GPU 0, or 'cpu')")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data_dir = Path(args.data_dir)
    frames_dir = Path(args.frames_dir) if args.frames_dir else data_dir / "frames"
    out_path = Path(args.out) if args.out else data_dir / "features.parquet"

    device = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    df = extract_dataset_features(data_dir, frames_dir, model_name=args.model, device=device)
    df.to_parquet(out_path)
    print(f"\nSaved {out_path}: {len(df)} rows, valid={df['valid'].sum()}, "
          f"falls={(df['label'] == 1).sum()}, not_fall={(df['label'] == 0).sum()}")


if __name__ == "__main__":
    main()
