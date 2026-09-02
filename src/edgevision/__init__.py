"""edgevision - real-time detection and tracking on constrained edge hardware.

The package is deliberately importable with nothing but numpy installed.
Every heavy or platform-specific dependency (torch, ultralytics, onnxruntime,
tensorrt, hailo_platform, cv2, rclpy) is guarded at its point of use, so you
can read configuration, run the tests, and develop against
:class:`~edgevision.backends.mock.MockBackend` on any machine.

``edgevision.ros2_node`` is not imported here on purpose: importing it pulls
in rclpy, which is slow and only meaningful inside a sourced ROS 2 workspace.
Import it explicitly when you need it.
"""

from __future__ import annotations

from .backends import (
    Detector,
    HailoBackend,
    MockBackend,
    OnnxRuntimeBackend,
    TensorRTBackend,
    UltralyticsBackend,
    available_backends,
    create_backend,
    probe_backends,
    select_backend,
)
from .capture import FrameGrabber, LatestFrameBuffer, csi_pipeline, parse_source, rtsp_pipeline
from .pipeline import AdaptiveFrameSkipper, Pipeline, annotate
from .postprocess import batched_nms, iou_matrix, nms
from .preprocess import LetterboxParams, letterbox, unletterbox_boxes
from .profiling import Profiler, benchmark, percentile
from .roi import Region, TiledInference, tile_regions
from .sinks import JsonLinesSink, MjpegPreviewServer, SnapshotSink, VideoWriterSink
from .tracker import IoUTracker, SortTracker, hungarian
from .types import Detection, Track, TrackState

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # types
    "Detection", "Track", "TrackState",
    # capture
    "FrameGrabber", "LatestFrameBuffer", "parse_source", "csi_pipeline", "rtsp_pipeline",
    # backends
    "Detector", "MockBackend", "UltralyticsBackend", "OnnxRuntimeBackend",
    "TensorRTBackend", "HailoBackend", "create_backend", "select_backend",
    "probe_backends", "available_backends",
    # pre/post
    "letterbox", "unletterbox_boxes", "LetterboxParams", "nms", "batched_nms", "iou_matrix",
    # tracking
    "SortTracker", "IoUTracker", "hungarian",
    # pipeline
    "Pipeline", "AdaptiveFrameSkipper", "annotate",
    # profiling
    "Profiler", "benchmark", "percentile",
    # roi
    "Region", "TiledInference", "tile_regions",
    # sinks
    "VideoWriterSink", "SnapshotSink", "JsonLinesSink", "MjpegPreviewServer",
]
