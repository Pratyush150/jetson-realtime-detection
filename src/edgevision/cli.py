"""Command-line entry points: ``edgevision-run`` and ``edgevision-bench``.

Both parsers are built by functions so the tests can assert on argument
handling without running anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, List, Optional, Sequence

from . import __version__

__all__ = [
    "build_run_parser",
    "build_bench_parser",
    "main_run",
    "main_bench",
    "parse_classes",
]


def parse_classes(value: Optional[str]) -> Optional[List[int]]:
    """``"0,2,7"`` -> ``[0, 2, 7]``. Empty/None -> None (keep all classes)."""
    if not value:
        return None
    out: List[int] = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(int(chunk))
    return out or None


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        default="auto",
        help="tensorrt | hailo | onnxruntime | ultralytics | mock | auto (default)",
    )
    parser.add_argument("--model", default=None, help="path to weights/engine/hef")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="network input size, square (default 640)"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--classes", default=None, help="comma-separated class ids to keep, e.g. 0,2"
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="fail instead of silently dropping to a slower backend",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="print the capability probe table and exit",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--version", action="version", version=f"edgevision {__version__}")


def build_run_parser() -> argparse.ArgumentParser:
    """Parser for ``edgevision-run``."""
    parser = argparse.ArgumentParser(
        prog="edgevision-run",
        description=(
            "Run detection + tracking on a camera, RTSP stream, GStreamer "
            "pipeline or video file, with adaptive frame skipping."
        ),
        epilog=(
            "examples:\n"
            "  edgevision-run --source 0 --backend auto --model yolov8n.onnx\n"
            "  edgevision-run --source rtsp://cam/live --skip 3 --preview\n"
            "  edgevision-run --source clip.mp4 --json out.jsonl --record out.mp4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common(parser)
    parser.add_argument(
        "--source",
        default="0",
        help="camera index, file path, rtsp:// URL, or a GStreamer pipeline string",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="fixed detection interval; 0 (default) = adapt from measured timings",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="output frame rate the adaptive skipper aims at (default 30)",
    )
    parser.add_argument(
        "--max-interval", type=int, default=12, help="upper bound on the skip interval"
    )
    parser.add_argument("--tracker", default="sort", choices=("sort", "iou", "none"))
    parser.add_argument("--max-age", type=int, default=30, help="tracker max_age in frames")
    parser.add_argument("--min-hits", type=int, default=3, help="tracker min_hits")
    parser.add_argument(
        "--tile",
        type=int,
        default=0,
        help="run tiled inference with this tile size (for small distant objects)",
    )
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--record", default=None, help="write annotated video to this path")
    parser.add_argument("--record-fps", type=float, default=0.0,
                        help="fps metadata for --record; 0 = use --target-fps")
    parser.add_argument("--json", dest="jsonl", default=None,
                        help="write a JSON-lines detection log to this path")
    parser.add_argument("--snapshots", default=None, help="directory for event snapshots")
    parser.add_argument("--snapshot-interval", type=float, default=2.0)
    parser.add_argument("--preview", action="store_true", help="serve an MJPEG preview")
    parser.add_argument("--preview-port", type=int, default=8090)
    parser.add_argument("--preview-host", default="0.0.0.0")
    parser.add_argument("--no-annotate", action="store_true",
                        help="skip drawing entirely (saves time on headless runs)")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0.0, help="stop after N seconds")
    parser.add_argument("--stats-json", action="store_true",
                        help="print the final stats as JSON instead of a table")
    return parser


def build_bench_parser() -> argparse.ArgumentParser:
    """Parser for ``edgevision-bench``."""
    parser = argparse.ArgumentParser(
        prog="edgevision-bench",
        description=(
            "Benchmark a detection backend and report per-stage latency "
            "percentiles. Warmup runs are discarded."
        ),
        epilog=(
            "examples:\n"
            "  edgevision-bench --backend onnxruntime --model yolov8n.onnx --frames 200\n"
            "  edgevision-bench --list-backends\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common(parser)
    parser.add_argument("--frames", type=int, default=100, help="frames to time (default 100)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="warmup frames, discarded (default 10)")
    parser.add_argument("--resolution", default="1280x720",
                        help="synthetic source frame size, WxH (default 1280x720)")
    parser.add_argument("--source", default=None,
                        help="optional real source to pull benchmark frames from")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="emit the result as JSON instead of a table")
    parser.add_argument("--tile", type=int, default=0,
                        help="benchmark tiled inference with this tile size")
    return parser


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )


def _parse_resolution(text: str) -> Sequence[int]:
    try:
        width, height = (int(v) for v in str(text).lower().split("x"))
    except Exception as exc:
        raise SystemExit(f"bad --resolution {text!r}, expected WxH") from exc
    return (height, width, 3)


def _build_detector(args: argparse.Namespace):
    from .backends import create_backend

    return create_backend(
        args.backend,
        model_path=args.model,
        allow_fallback=not args.no_fallback,
        input_size=(args.imgsz, args.imgsz),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        classes=parse_classes(args.classes),
    )


def main_run(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``tools/edgevision-run``."""
    args = build_run_parser().parse_args(argv)
    _configure_logging(args.verbose)

    from .backends import format_backend_table, select_backend

    if args.list_backends:
        print(format_backend_table())
        return 0

    choice = select_backend(args.backend, args.model, allow_fallback=not args.no_fallback)
    for line in choice.log:
        logging.getLogger("edgevision").info("backend probe: %s", line)
    print(f"backend: {choice.name} ({choice.reason})", file=sys.stderr)

    detector = _build_detector(args)
    if args.tile:
        from .roi import TiledInference

        detector = TiledInference(
            detector, tile_size=(args.tile, args.tile), overlap=args.tile_overlap
        )

    from .pipeline import Pipeline
    from .sinks import JsonLinesSink, MjpegPreviewServer, SnapshotSink, VideoWriterSink
    from .tracker import build_tracker

    tracker = None if args.tracker == "none" else build_tracker(
        args.tracker, max_age=args.max_age, min_hits=args.min_hits
    )

    sinks: List[Any] = []
    preview: Optional[MjpegPreviewServer] = None
    if args.record:
        sinks.append(
            VideoWriterSink(args.record, fps=args.record_fps or args.target_fps)
        )
    if args.jsonl:
        sinks.append(JsonLinesSink(args.jsonl))
    if args.snapshots:
        sinks.append(SnapshotSink(args.snapshots, min_interval_s=args.snapshot_interval))
    if args.preview:
        preview = MjpegPreviewServer(host=args.preview_host, port=args.preview_port).start()
        print(f"preview: {preview.url}", file=sys.stderr)
        sinks.append(preview)

    pipeline = Pipeline(
        detector,
        tracker=tracker,
        sinks=sinks,
        target_fps=args.target_fps,
        skip=args.skip,
        max_interval=args.max_interval,
        annotate_frames=not args.no_annotate,
        track=args.tracker != "none",
    )

    try:
        pipeline.run(
            args.source, max_frames=args.max_frames, max_seconds=args.duration
        )
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
    finally:
        pipeline.close()

    if args.stats_json:
        print(json.dumps(pipeline.stats(), indent=2))
    else:
        print(pipeline.format_report())
    return 0


def main_bench(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``tools/edgevision-bench``."""
    args = build_bench_parser().parse_args(argv)
    _configure_logging(args.verbose)

    from .backends import format_backend_table, select_backend
    from .profiling import benchmark

    if args.list_backends:
        print(format_backend_table())
        return 0

    choice = select_backend(args.backend, args.model, allow_fallback=not args.no_fallback)
    print(f"backend: {choice.name} ({choice.reason})", file=sys.stderr)

    detector = _build_detector(args)
    notes = [choice.reason]
    if args.tile:
        from .roi import TiledInference

        detector = TiledInference(detector, tile_size=(args.tile, args.tile))
        notes.append(f"tiled inference, tile={args.tile}")

    frames = None
    if args.source:
        from .capture import FrameGrabber

        grabber = FrameGrabber(args.source).start()
        try:
            collected = []
            while len(collected) < args.frames:
                item = grabber.read_with_meta(timeout=2.0)
                if item is None:
                    break
                collected.append(item[0])
            frames = collected or None
        finally:
            grabber.stop()
        if frames:
            notes.append(f"frames captured from {args.source}")

    result = benchmark(
        detector,
        frames=frames,
        num_frames=args.frames,
        frame_shape=_parse_resolution(args.resolution),
        warmup=args.warmup,
        notes=notes,
    )

    if args.as_json:
        print(result.to_json(indent=2))
    else:
        print(result.format_table())
    return 0
