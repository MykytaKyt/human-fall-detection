#!/usr/bin/env python3
"""Quantitative end-to-end evaluation of the FULL two-cascade pipeline
(kinematic filter -> router -> VLM -> DST fusion) on a sample of URFD
frames, compared against the kinematic-only physics threshold.

The sample is drawn from the physics threshold's own errors (false
positives on ADL, false negatives on falls) plus some correct
predictions for context -- i.e. exactly the "ambiguous zone" the
semantic cascade is meant to help with (see paper Sec. 5.3 discussion).

ponytail: we don't have a real Lyapunov V(x) from a trained LTC here
(that requires the full LTC forward pass wired to a stability manifold,
which is out of scope for this evaluation); we use the physics
threshold's own score as a stand-in for the calibrated kinematic
confidence that would normally come from the LTC stage. This is a
documented approximation, not a hidden one -- see README.

Usage:
    python scripts/eval_full_cascade.py --n-fp 80 --n-fn 40 --n-tp 40 --n-tn 40
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urfd_cascade.cascade import run_cascade  # noqa: E402
from urfd_cascade.dataset import list_sequence_frames  # noqa: E402
from urfd_cascade.evaluate import FEATURE_COLUMNS, make_windows, metrics_at_recall, threshold_scores  # noqa: E402
from urfd_cascade.vlm import VLMClient  # noqa: E402


def build_sample(features_path: str, n_fp: int, n_fn: int, n_tp: int, n_tn: int, seed: int = 0) -> pd.DataFrame:
    """Reproduce the physics-threshold predictions and sample frames
    stratified by error type (see module docstring)."""
    np.random.seed(seed)
    df = pd.read_parquet(features_path)
    df[FEATURE_COLUMNS] = df.groupby("seq")[FEATURE_COLUMNS].ffill().fillna(0.0)

    meta_rows = []
    for seq, g in df.groupby("seq"):
        for frame, label, valid, mos in zip(g["frame"], g["label"], g["valid"], g["mos"]):
            meta_rows.append((seq, frame, label, valid, mos))
    meta = pd.DataFrame(meta_rows, columns=["seq", "frame", "label", "valid", "mos"])

    x, y, _ = make_windows(df, window=8)
    assert len(meta) == len(x)
    scores = threshold_scores(x)
    _, _, _, thr = metrics_at_recall(y, scores, target=0.95)
    pred = (scores >= thr).astype(int)

    meta["pred"] = pred
    meta["score"] = scores

    valid_mask = meta.valid == 1
    fp = meta[valid_mask & (meta.pred == 1) & (meta.label == 0)]
    fn = meta[valid_mask & (meta.pred == 0) & (meta.label == 1)]
    tp = meta[valid_mask & (meta.pred == 1) & (meta.label == 1)]
    tn = meta[valid_mask & (meta.pred == 0) & (meta.label == 0)]

    parts = [
        fn.sample(min(n_fn, len(fn)), random_state=seed),
        fp.sample(min(n_fp, len(fp)), random_state=seed),
        tp.sample(min(n_tp, len(tp)), random_state=seed),
        tn.sample(min(n_tn, len(tn)), random_state=seed),
    ]
    return pd.concat(parts).reset_index(drop=True)


def frame_image_path(frames_dir: Path, seq: str, frame: int) -> str | None:
    paths = list_sequence_frames(frames_dir, seq)
    target = f"-{frame:03d}.png"
    for p in paths:
        if p.endswith(target):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="data/features.parquet")
    parser.add_argument("--frames-dir", default=None, help="Default: /home/mykyta/urfd_exp/data/frames")
    parser.add_argument("--n-fp", type=int, default=80)
    parser.add_argument("--n-fn", type=int, default=40)
    parser.add_argument("--n-tp", type=int, default=40)
    parser.add_argument("--n-tn", type=int, default=40)
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--model", default="google/gemma-3-4b")
    parser.add_argument("--out", default="results/full_cascade_eval.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir) if args.frames_dir else Path("/home/mykyta/urfd_exp/data/frames")

    sample = build_sample(args.features, args.n_fp, args.n_fn, args.n_tp, args.n_tn, seed=args.seed)
    print(f"Sample: {len(sample)} frames "
          f"(FN={((sample.pred==0)&(sample.label==1)).sum()}, "
          f"FP={((sample.pred==1)&(sample.label==0)).sum()}, "
          f"TP={((sample.pred==1)&(sample.label==1)).sum()}, "
          f"TN={((sample.pred==0)&(sample.label==0)).sum()})")

    client = VLMClient(base_url=args.base_url, model=args.model)
    if not client.ping():
        print(f"ERROR: VLM server not reachable at {args.base_url}", file=sys.stderr)
        sys.exit(1)

    records = []
    n_routed = 0
    t0 = time.time()
    for i, row in sample.iterrows():
        img = frame_image_path(frames_dir, row.seq, int(row.frame))
        if img is None:
            print(f"WARN: no image for {row.seq} frame {row.frame}, skipping")
            continue

        # kinematic confidence stand-in: rescale the physics score into a
        # pseudo-Lyapunov value around the decision threshold (ponytail:
        # documented approximation, see module docstring)
        v = float(row.score) * 5.0
        gamma = 5.0 * float(sample.score.median())

        try:
            result = run_cascade(
                lyapunov_v=v, stability_gamma=gamma, mos=float(row.mos),
                image_path=img, vlm=client,
            )
        except Exception as e:
            print(f"WARN: VLM call failed for {row.seq} frame {row.frame}: {e}")
            continue

        if result.routed_to_vlm:
            n_routed += 1

        records.append(dict(
            seq=row.seq, frame=int(row.frame), label=int(row.label),
            kinematic_only_pred=int(row.pred),
            routed_to_vlm=result.routed_to_vlm,
            fused_pred=1 if result.decision == "A" else 0,
            conflict_k=result.conflict_k,
        ))
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(sample)} done ({elapsed:.0f}s elapsed, "
                  f"{n_routed} routed to VLM)")

    out_df = pd.DataFrame(records)
    y_true = out_df["label"].to_numpy()
    kin_pred = out_df["kinematic_only_pred"].to_numpy()
    fused_pred = out_df["fused_pred"].to_numpy()

    def prf(y, pred):
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return dict(precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn)

    kin_metrics = prf(y_true, kin_pred)
    fused_metrics = prf(y_true, fused_pred)

    print(f"\n===== Full-cascade evaluation on {len(out_df)} sampled frames =====")
    print(f"Kinematic-only : P={kin_metrics['precision']:.3f} R={kin_metrics['recall']:.3f} F1={kin_metrics['f1']:.3f}")
    print(f"Fused (2-cascade): P={fused_metrics['precision']:.3f} R={fused_metrics['recall']:.3f} F1={fused_metrics['f1']:.3f}")
    print(f"Routed to VLM: {n_routed}/{len(out_df)} ({100*n_routed/len(out_df):.1f}%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(
            n_samples=len(out_df), n_routed=n_routed,
            kinematic_only=kin_metrics, fused=fused_metrics,
            records=records,
        ), f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
