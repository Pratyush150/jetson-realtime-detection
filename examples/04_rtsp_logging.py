#!/usr/bin/env python3
"""RTSP camera with reconnect, a JSON-lines log and rate-limited snapshots.

    python3 examples/04_rtsp_logging.py --source rtsp://10.0.0.5:554/live \
        --model yolov8n.onnx --classes 0

RTSP is where the latest-frame buffer earns its keep. The server's jitter
buffer plus any network hiccup means a naive read loop drifts seconds behind
within a minute. The reader thread drains the socket continuously and keeps
only the newest frame; the drop counter tells you how far behind you would
have been.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from edgevision import FrameGrabber, JsonLinesSink, Pipeline, SnapshotSink, create_backend  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="rtsp:// URL")
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--classes", default=None, help="comma-separated class ids")
    parser.add_argument("--outdir", default="examples/output")
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    classes = [int(c) for c in args.classes.split(",")] if args.classes else None

    detector = create_backend(args.backend, model_path=args.model, classes=classes)
    detector.warmup(3)

    sinks = [
        JsonLinesSink(os.path.join(args.outdir, "detections.jsonl")),
        SnapshotSink(os.path.join(args.outdir, "snapshots"), min_interval_s=5.0),
    ]

    # max_reconnect_attempts=0 means "retry forever with exponential backoff",
    # which is what you want on a camera at the far end of a radio link.
    grabber = FrameGrabber(
        args.source, reconnect=True, reconnect_delay=1.0, max_reconnect_attempts=0
    ).start()
    pipeline = Pipeline(detector, sinks=sinks, target_fps=15.0, annotate_frames=False)

    try:
        pipeline.run(grabber, max_seconds=args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        grabber.stop()
        pipeline.close()

    stats = grabber.stats
    print(pipeline.format_report())
    print()
    print(f"frames read      : {stats.frames_read}")
    print(f"frames processed : {stats.frames_delivered}")
    print(f"frames dropped   : {stats.frames_dropped} ({stats.drop_rate * 100:.1f}%)")
    print(f"reconnects       : {stats.reconnects}")
    print(
        "\nA high drop rate is not a bug: it is the latency you did NOT accumulate.\n"
        "Every dropped frame is a frame you would otherwise have processed late."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
