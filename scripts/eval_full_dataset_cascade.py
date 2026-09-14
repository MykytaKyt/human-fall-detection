#!/usr/bin/env python3
"""Full-dataset, apples-to-apples evaluation of the two-cascade system:
the SAME trained LTC model used in the main Results table (out-of-fold
over all 11,936 URFD frames via 5-fold sequence-grouped CV), routed to a
live VLM when the LTC's own predictive entropy is high, fused via
Dempster-Shafer.

Unlike eval_full_cascade.py (which used a threshold-score stand-in on a
deliberately adversarial 200-frame sample), this script:
  - uses the real trained LTC's out-of-fold probability as P(fall|x),
  - uses the LTC's binary entropy as the uncertainty signal driving
    routing (a direct, non-invented measure: 0 = certain, 1 = coin flip),
  - evaluates on ALL frames above the entropy threshold, comparing the
    LTC-only prediction vs. the VLM-fused prediction for exactly those
    frames, then reports the resulting dataset-wide P/R/F1.

ponytail: routing here is "call VLM iff entropy > threshold" rather than
the full R(x,VLM)=(1-lambda)*P_error-lambda*Cost formula from cascade.py
-- entropy IS P_error's natural proxy from a calibrated classifier, and
skipping the lambda/cost bookkeeping keeps this script (and its runtime)
tractable; cascade.py's routing_risk is still what's used in
run_cascade_demo.py for the single-frame/deployment-facing path.

Usage:
    python scripts/eval_full_dataset_cascade.py --entropy-threshold 0.6
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urfd_cascade.dataset import list_sequence_frames  # noqa: E402
from urfd_cascade.dst import combine, decide, kinematic_evidence, pignistic, semantic_evidence  # noqa: E402
from urfd_cascade.evaluate import get_oof_ltc_predictions  # noqa: E402
from urfd_cascade.vlm import VLMClient  # noqa: E402


def binary_entropy(p: np.ndarray) -> np.ndarray:
    eps = 1e-9
    return -(p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps))


def frame_image_path(frames_dir: Path, seq: str, frame: int) -> str | None:
    for path in list_sequence_frames(frames_dir, seq):
        if path.endswith(f"-{frame:03d}.png"):
            return path
    return None


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="data/features.parquet")
    parser.add_argument("--frames-dir", default="/home/mykyta/urfd_exp/data/frames")
    parser.add_argument("--entropy-threshold", type=float, default=0.6)
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--model", default="google/gemma-3-4b")
    parser.add_argument("--out", default="results/full_dataset_cascade_eval.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of routed frames (debug/testing)")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    print("Computing out-of-fold LTC predictions over all frames (same as main Results table)...")
    seqs, frames, mos, y, prob = get_oof_ltc_predictions(df, device="cuda", seed=args.seed)
    entropy = binary_entropy(prob)
    ltc_pred = (prob >= 0.5).astype(int)  # LTC-only decision (matches the classifier's own boundary)

    route_mask = entropy > args.entropy_threshold
    n_routed = int(route_mask.sum())
    print(f"Total frames: {len(y)} | LTC-only F1 sanity: "
          f"{prf(y, ltc_pred)}")
    print(f"Routing to VLM: {n_routed} frames ({100 * n_routed / len(y):.1f}%) "
          f"with entropy > {args.entropy_threshold}")

    client = VLMClient(base_url=args.base_url, model=args.model)
    if not client.ping():
        print(f"ERROR: VLM server not reachable at {args.base_url}", file=sys.stderr)
        sys.exit(1)

    frames_dir = Path(args.frames_dir)
    fused_pred = ltc_pred.copy()  # start from LTC's own decision everywhere
    routed_indices = np.where(route_mask)[0]
    if args.limit:
        routed_indices = routed_indices[: args.limit]

    t0 = time.time()
    n_ok = 0
    for count, idx in enumerate(routed_indices, 1):
        img = frame_image_path(frames_dir, str(seqs[idx]), int(frames[idx]))
        if img is None:
            continue
        m_y = kinematic_evidence(
            lyapunov_v=float(prob[idx]) * 5.0,  # ponytail: same rescaling as eval_full_cascade.py
            stability_gamma=2.5,
            mos=float(mos[idx]) if not np.isnan(mos[idx]) else 0.0,
        )
        try:
            judgement = client.judge_image(img)
        except Exception as e:
            print(f"WARN: VLM call failed for {seqs[idx]} frame {frames[idx]}: {e}")
            continue
        m_v = semantic_evidence(judgement.p_anomaly, judgement.entropy)
        fused, _ = combine(m_y, m_v)
        betp = pignistic(fused)
        fused_pred[idx] = 1 if decide(betp) == "A" else 0
        n_ok += 1
        if count % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {count}/{len(routed_indices)} routed frames processed "
                  f"({elapsed:.0f}s elapsed)")

    ltc_metrics = prf(y, ltc_pred)
    fused_metrics = prf(y, fused_pred)

    print(f"\n===== Full-dataset evaluation ({len(y)} frames total, "
          f"{n_ok}/{n_routed} routed frames successfully queried) =====")
    print(f"LTC-only        : P={ltc_metrics['precision']:.3f} "
          f"R={ltc_metrics['recall']:.3f} F1={ltc_metrics['f1']:.3f}")
    print(f"LTC + VLM + DST : P={fused_metrics['precision']:.3f} "
          f"R={fused_metrics['recall']:.3f} F1={fused_metrics['f1']:.3f}")
    print(f"VLM call rate   : {n_routed}/{len(y)} ({100 * n_routed / len(y):.1f}%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(
            n_total=len(y), n_routed=n_routed, n_routed_ok=n_ok,
            entropy_threshold=args.entropy_threshold,
            ltc_only=ltc_metrics, fused=fused_metrics,
        ), f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
