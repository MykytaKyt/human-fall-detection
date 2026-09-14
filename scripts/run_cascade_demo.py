#!/usr/bin/env python3
"""Run the full two-cascade pipeline (kinematic filter -> router -> VLM ->
DST fusion) on a single image, using a local LM Studio server for the
semantic stage.

Setup (once):
    1. Install LM Studio: https://lmstudio.ai
    2. In LM Studio, download and load a vision-capable model, e.g.
       `lms get google/gemma-3-4b` then `lms load google/gemma-3-4b`
    3. Start the local server (LM Studio does this automatically, or:
       `lms server start`) -- default: http://localhost:1234

Usage:
    python scripts/run_cascade_demo.py --check          # just ping the server
    python scripts/run_cascade_demo.py --image path.png --v 2.0 --gamma 1.0 --mos -0.3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urfd_cascade.cascade import run_cascade  # noqa: E402
from urfd_cascade.vlm import VLMClient  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="Path to a frame image (required unless --check)")
    parser.add_argument("--v", type=float, default=1.0, help="LTC Lyapunov value V(x) from the kinematic filter")
    parser.add_argument("--gamma", type=float, default=1.0, help="Stability threshold gamma")
    parser.add_argument("--mos", type=float, default=0.0, help="Margin of Stability")
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--model", default="google/gemma-3-4b")
    parser.add_argument("--check", action="store_true", help="Only check the LM Studio server is reachable")
    args = parser.parse_args()

    client = VLMClient(base_url=args.base_url, model=args.model)

    if args.check:
        ok = client.ping()
        print(f"LM Studio at {args.base_url} serving {args.model}: {'OK' if ok else 'NOT REACHABLE'}")
        sys.exit(0 if ok else 1)

    if not args.image:
        parser.error("--image is required unless --check")

    result = run_cascade(
        lyapunov_v=args.v, stability_gamma=args.gamma, mos=args.mos,
        image_path=args.image, vlm=client,
    )

    print(f"Routed to VLM : {result.routed_to_vlm}  (risk={result.risk:.3f})")
    print(f"Kinematic mass: A={result.m_kinematic['A']:.3f} "
          f"N={result.m_kinematic['N']:.3f} Theta={result.m_kinematic['AN']:.3f}")
    if result.m_semantic:
        print(f"Semantic mass : A={result.m_semantic['A']:.3f} "
              f"N={result.m_semantic['N']:.3f} Theta={result.m_semantic['AN']:.3f}")
        print(f"Conflict K    : {result.conflict_k:.3f}")
    print(f"BetP          : A={result.betp['A']:.3f} N={result.betp['N']:.3f}")
    print(f"DECISION      : {'ANOMALY (fall)' if result.decision == 'A' else 'NORMAL'}")


if __name__ == "__main__":
    main()
