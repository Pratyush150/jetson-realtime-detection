#!/usr/bin/env python3
"""What adaptive frame skipping actually buys you, in numbers.

No hardware required: the skipper's cost model is driven with the kind of
timings you measure on real boards, and the resulting output frame rate is
printed for each case.

    python3 examples/02_adaptive_skip.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from edgevision.pipeline import AdaptiveFrameSkipper  # noqa: E402

# (label, inference seconds per frame, per-frame overhead seconds)
# Plug your own measured numbers in here from `edgevision-bench`.
CASES = [
    ("fast engine, cheap overlay", 0.012, 0.003),
    ("mid-range engine", 0.045, 0.004),
    ("heavy model on a small board", 0.100, 0.005),
    ("heavy model + expensive overlay", 0.100, 0.030),
    ("overlay alone blows the budget", 0.050, 0.045),
]

TARGET_FPS = 30.0


def main() -> int:
    header = f"{'case':<36} {'skip':>5} {'no-skip fps':>12} {'with-skip fps':>14}"
    print(header)
    print("-" * len(header))

    for label, inference_s, overhead_s in CASES:
        skipper = AdaptiveFrameSkipper(target_fps=TARGET_FPS, max_interval=12)
        for _ in range(80):
            skipper.update(inference_s=inference_s, overhead_s=overhead_s)

        naive_fps = 1.0 / (inference_s + overhead_s)
        print(
            f"{label:<36} {skipper.interval:>5} {naive_fps:>12.1f} "
            f"{skipper.projected_fps:>14.1f}"
            + ("   <- unreachable target" if skipper.budget_exceeded else "")
        )

    print()
    print(f"target: {TARGET_FPS:.0f} fps output.")
    print(
        "Skipping does not make inference faster. It amortises it across frames\n"
        "and lets the tracker's motion model carry the boxes in between, so the\n"
        "*output* keeps up with the camera and stays close to real time."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
