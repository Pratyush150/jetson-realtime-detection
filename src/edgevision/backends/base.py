"""The ``Detector`` interface every backend implements.

The whole point of this abstraction is that the pipeline, the tracker and the
sinks never learn which accelerator is underneath. You can develop against
``MockBackend`` on a laptop, validate numerics against ``OnnxRuntimeBackend``,
and ship ``TensorRTBackend`` on the Jetson without touching a line of
pipeline code.

The shared preprocess/postprocess helpers live here on purpose. A recurring
class of bug on edge deployments is that the ONNX path and the TensorRT path
letterbox slightly differently, so the engine that "validated fine" scores
worse in production. If every backend calls the same two functions, that
cannot happen.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..postprocess import batched_nms, decode_yolo_v5, decode_yolo_v8
from ..preprocess import LetterboxParams, letterbox, unletterbox_boxes
from ..types import Detection

LOGGER = logging.getLogger(__name__)

__all__ = ["Detector", "Availability", "COCO_CLASSES"]


# Kept here so a backend that has no metadata in the model file can still emit
# readable labels. Order is the standard COCO-80 order used by YOLOv5/v8/v11.
COCO_CLASSES: Tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


@dataclass(frozen=True)
class Availability:
    """Result of a backend capability probe.

    ``reason`` is always populated, including on success, because "why did it
    pick ONNX Runtime instead of TensorRT?" is a question you will ask at
    2 a.m. on a robot, and the answer belongs in the log.
    """

    name: str
    available: bool
    reason: str
    priority: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.available


class Detector(abc.ABC):
    """Uniform detector interface: ``infer(frame) -> list[Detection]``.

    Subclasses implement :meth:`load` and :meth:`_forward`. The shared
    :meth:`infer` handles letterboxing, decoding, NMS and un-letterboxing so
    coordinates always come back in original-frame pixels.
    """

    #: Human-readable registry key, set by subclasses.
    name: str = "base"
    #: Higher wins during automatic selection.
    priority: int = 0

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: Optional[Sequence[str]] = None,
        classes: Optional[Sequence[int]] = None,
        max_detections: int = 300,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.class_names: Tuple[str, ...] = tuple(class_names or COCO_CLASSES)
        self.class_filter = set(int(c) for c in classes) if classes else None
        self.max_detections = int(max_detections)
        self.device = device
        self.options: Dict[str, Any] = dict(kwargs)
        self.loaded = False
        self._warmup_done = False

    # -- capability probing -------------------------------------------------

    @classmethod
    def probe(cls) -> Availability:
        """Report whether this backend can run here, and why."""
        return Availability(cls.name, True, "always available", cls.priority)

    # -- lifecycle ----------------------------------------------------------

    @abc.abstractmethod
    def load(self) -> None:
        """Load weights / build the engine. Must set ``self.loaded = True``."""

    def ensure_loaded(self) -> None:
        if not self.loaded:
            self.load()

    def close(self) -> None:
        """Release device resources. Safe to call more than once."""
        self.loaded = False

    def __enter__(self) -> "Detector":
        self.ensure_loaded()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- inference ----------------------------------------------------------

    @abc.abstractmethod
    def _forward(self, tensor: Any, params: LetterboxParams) -> Any:
        """Run the network on a preprocessed input, return raw output."""

    def _preprocess(self, frame: np.ndarray) -> Tuple[Any, LetterboxParams]:
        """Letterbox to the network input size. Override for custom layouts."""
        padded, params = letterbox(frame, self.input_size)
        return padded, params

    def _postprocess(
        self, raw: Any, params: LetterboxParams
    ) -> List[Detection]:
        """Decode -> class-aware NMS -> un-letterbox -> ``Detection`` objects."""
        boxes, scores, class_ids = self._decode(raw)
        return self.finalize(boxes, scores, class_ids, params)

    def _decode(self, raw: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Turn a raw network output into ``(boxes_xyxy, scores, class_ids)``.

        Defaults to the YOLOv8/YOLO11 layout; ``head='v5'`` selects the
        v5/v7 layout with a separate objectness column.
        """
        head = str(self.options.get("head", "v8")).lower()
        decoder = decode_yolo_v5 if head in ("v5", "v7") else decode_yolo_v8
        return decoder(raw, self.conf_threshold, num_classes=len(self.class_names))

    def finalize(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        params: LetterboxParams,
    ) -> List[Detection]:
        """Shared tail: filter classes, NMS, un-letterbox, wrap.

        Exposed publicly because backends whose runtime already does NMS
        internally (Ultralytics, some Hailo post-process nodes) still need
        the class filter and the coordinate mapping.
        """
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        class_ids = np.asarray(class_ids).reshape(-1).astype(np.int64)
        if len(boxes) == 0:
            return []

        if self.class_filter is not None:
            mask = np.isin(class_ids, list(self.class_filter))
            boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
            if len(boxes) == 0:
                return []

        keep = batched_nms(
            boxes, scores, class_ids, self.iou_threshold, self.max_detections
        )
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

        boxes = unletterbox_boxes(boxes, params)

        detections: List[Detection] = []
        for (x1, y1, x2, y2), score, cls in zip(boxes, scores, class_ids):
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                # A box entirely inside the letterbox padding clips to zero
                # area. Dropping it here keeps the tracker from being fed a
                # degenerate observation.
                continue
            detections.append(
                Detection(
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(score),
                    int(cls),
                    self.label(int(cls)),
                )
            )
        return detections

    def label(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return str(class_id)

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a BGR frame, returning original-frame coordinates."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        self.ensure_loaded()
        tensor, params = self._preprocess(frame)
        raw = self._forward(tensor, params)
        return self._postprocess(raw, params)

    __call__ = infer

    # -- warmup -------------------------------------------------------------

    def warmup(self, iterations: int = 3, frame: Optional[np.ndarray] = None) -> float:
        """Run a few throwaway inferences and return the last one's latency.

        You must do this, and you must discard the timings. The first call
        allocates device memory, JITs kernels, and on TensorRT can trigger
        tactic selection; it is routinely 10-100x slower than steady state.
        A benchmark that includes it reports a number nobody will ever see
        again, and a pipeline that skips it drops its first second of frames.
        """
        self.ensure_loaded()
        if frame is None:
            frame = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        elapsed = 0.0
        for _ in range(max(1, int(iterations))):
            start = time.perf_counter()
            self.infer(frame)
            elapsed = time.perf_counter() - start
        self._warmup_done = True
        LOGGER.info("%s warmup complete, last iteration %.1f ms", self.name, elapsed * 1e3)
        return elapsed

    # -- introspection ------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """Everything worth putting in a run log or a benchmark row."""
        return {
            "backend": self.name,
            "model": self.model_path,
            "input_size": list(self.input_size),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "loaded": self.loaded,
            "num_classes": len(self.class_names),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}(model={self.model_path!r}, "
            f"input_size={self.input_size}, device={self.device!r})"
        )
