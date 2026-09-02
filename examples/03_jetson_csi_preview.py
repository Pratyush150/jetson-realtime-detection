#!/usr/bin/env python3
"""Jetson CSI camera -> TensorRT -> MJPEG preview you can open from a laptop.

Needs real hardware. On the Jetson:

    python3 examples/03_jetson_csi_preview.py --model yolov8n.engine

then open http://<jetson-ip>:8090/ in a browser.

The CSI path matters here: ``nvarguscamerasrc`` keeps the capture, the
rescale and the colour conversion on the ISP and the VIC. A USB camera doing
the same job burns CPU on decode and conversion, which competes with the
Python side of your pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from edgevision import (  # noqa: E402
    FrameGrabber,
    MjpegPreviewServer,
    Pipeline,
    create_backend,
    csi_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="path to .engine / .onnx / .pt")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--duration", type=float, default=0.0, help="0 = run forever")
    args = parser.parse_args()

    source = csi_pipeline(
        sensor_id=args.sensor_id,
        display_width=args.width,
        display_height=args.height,
        framerate=int(args.fps),
    )
    print(f"gstreamer pipeline:\n  {source}\n")

    detector = create_backend(args.backend, model_path=args.model)
    print(f"backend: {detector.name}")
    detector.warmup(3)  # never time the first inference

    preview = MjpegPreviewServer(port=args.port).start()
    print(f"preview: {preview.url}")

    pipeline = Pipeline(detector, sinks=[preview], target_fps=args.fps)
    grabber = FrameGrabber(source, reconnect=True).start()
    try:
        pipeline.run(grabber, max_seconds=args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        grabber.stop()
        pipeline.close()

    print()
    print(pipeline.format_report())
    print(f"capture: {grabber.stats.to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
