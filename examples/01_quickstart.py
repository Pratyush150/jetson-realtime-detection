#!/usr/bin/env python3
"""Smallest complete pipeline: detect, track, report where the time went.

Runs offline with MockBackend, so it works on any machine with numpy. Swap
``MockBackend()`` for ``create_backend("auto", model_path="yolov8n.onnx")``
and it is a real pipeline.

    python3 examples/01_quickstart.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from edgevision import MockBackend, Pipeline, SortTracker  # noqa: E402


def synthetic_frames(count: int, height: int = 480, width: int = 640):
    """Stand-in for a camera: a static background, deterministic content."""
    rng = np.random.default_rng(0)
    background = rng.integers(0, 60, size=(height, width, 3), dtype=np.uint8)
    for _ in range(count):
        yield background


def main() -> int:
    detector = MockBackend(
        num_objects=3, velocity=(2.5, 1.0), origin=(40.0, 40.0), spacing=(0.0, 120.0)
    )
    pipeline = Pipeline(
        detector,
        tracker=SortTracker(max_age=20, min_hits=3),
        target_fps=30.0,
        skip=0,            # 0 = adapt from measured timings
        annotate_frames=True,
    )

    last = None
    for frame in synthetic_frames(100):
        last = pipeline.process(frame)

    print(pipeline.format_report())
    print()
    if last is not None:
        print(f"tracks on the final frame: {[t.track_id for t in last.tracks]}")
        print(f"detector ran on that frame: {last.ran_inference}")
    pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
