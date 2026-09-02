#!/usr/bin/env python3
"""Tiled inference for small, distant objects (drone and long-range imagery).

    python3 examples/05_tiled_small_objects.py

Shows the tile layout and the cost multiplier for a given frame and tile size,
and demonstrates that boxes found inside a tile come back in full-frame
coordinates. Runs offline with MockBackend.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from edgevision import MockBackend, TiledInference, tile_regions  # noqa: E402

FRAME_W, FRAME_H = 1920, 1080


def main() -> int:
    print(f"frame: {FRAME_W}x{FRAME_H}\n")
    print(f"{'tile':>10} {'overlap':>8} {'tiles':>6} {'inference cost':>15} {'min object px':>14}")
    print("-" * 60)

    for tile in (1920, 960, 640, 480):
        for overlap in (0.0, 0.2):
            regions = tile_regions(FRAME_W, FRAME_H, (tile, tile), overlap)
            # An object N px wide in the source is N * (640/tile) px at the
            # network input; a detector needs roughly 16 px to fire.
            min_px = 16.0 * tile / 640.0
            print(
                f"{tile:>10} {overlap:>8.0%} {len(regions):>6} "
                f"{len(regions):>14.0f}x {min_px:>13.0f}"
            )

    print(
        "\nSmaller tiles find smaller objects, and cost linearly more inference.\n"
        "Pair tiling with frame skipping: run the tiled pass every Nth detection\n"
        "and let the tracker carry the boxes in between."
    )

    print("\n--- box remapping check ---")
    detector = MockBackend(num_objects=1, velocity=(0.0, 0.0), origin=(20.0, 30.0),
                           box_size=(24.0, 24.0))
    tiled = TiledInference(detector, tile_size=(640, 640), overlap=0.2)
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    detections = tiled.infer(frame)

    print(f"tiles run: {tiled.last_tile_count}")
    print(f"detections after cross-tile NMS: {len(detections)}")
    for det in detections[:5]:
        print(f"  {det.to_dict()['bbox']}  {det.class_name}")
    inside = all(0 <= d.x1 and d.x2 <= FRAME_W and 0 <= d.y1 and d.y2 <= FRAME_H
                 for d in detections)
    print(f"all boxes inside the full frame: {inside}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
