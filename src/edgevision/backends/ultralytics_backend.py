"""Ultralytics backend: the convenient path, not the fast path.

Use this to establish a correctness baseline and to sanity-check labels, then
export to ONNX/TensorRT and compare. On a Jetson the PyTorch path typically
costs several times the latency of the same model as a TensorRT engine,
mostly in Python-side pre/post-processing and kernel launch overhead rather
than in the convolutions themselves.

Ultralytics already letterboxes and runs NMS internally, so this backend does
*not* re-do either; it only applies the class filter and wraps the results in
the common :class:`~edgevision.types.Detection` type.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

from .._compat import module_available
from ..preprocess import LetterboxParams
from ..types import Detection
from .base import Availability, Detector

LOGGER = logging.getLogger(__name__)

__all__ = ["UltralyticsBackend"]


def _identity_params(frame: np.ndarray) -> LetterboxParams:
    height, width = frame.shape[:2]
    return LetterboxParams(1.0, 0.0, 0.0, width, height, width, height)


class UltralyticsBackend(Detector):
    """Wraps ``ultralytics.YOLO`` (``.pt``, ``.onnx``, ``.engine``)."""

    name = "ultralytics"
    priority = 20

    def __init__(
        self,
        model_path: Optional[str] = "yolov8n.pt",
        input_size: Tuple[int, int] = (640, 640),
        half: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path=model_path, input_size=input_size, **kwargs)
        self.half = bool(half)
        self._model: Any = None

    @classmethod
    def probe(cls) -> Availability:
        if not module_available("ultralytics"):
            return Availability(
                cls.name, False, "ultralytics is not installed", cls.priority
            )
        if not module_available("torch"):
            return Availability(
                cls.name, False, "ultralytics requires torch, which is missing",
                cls.priority,
            )
        details = {}
        try:  # pragma: no cover - depends on environment
            import torch  # type: ignore

            details["torch"] = torch.__version__
            details["cuda"] = bool(torch.cuda.is_available())
        except Exception as exc:  # pragma: no cover
            return Availability(cls.name, False, f"torch import failed: {exc}", cls.priority)
        return Availability(
            cls.name, True, "ultralytics + torch importable", cls.priority, details
        )

    def load(self) -> None:  # pragma: no cover - requires ultralytics
        from ultralytics import YOLO  # type: ignore

        if not self.model_path:
            raise ValueError("UltralyticsBackend requires model_path")
        self._model = YOLO(self.model_path)

        device = self.device
        if device == "auto":
            try:
                import torch  # type: ignore

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        try:
            self._model.to(device)
        except Exception:
            LOGGER.debug("model.to(%s) not supported for this weight type", device)

        names = getattr(self._model, "names", None)
        if isinstance(names, dict) and names:
            self.class_names = tuple(names[i] for i in sorted(names))
        elif isinstance(names, (list, tuple)) and names:
            self.class_names = tuple(names)
        self.loaded = True
        LOGGER.info("ultralytics model %s loaded on %s", self.model_path, device)

    def close(self) -> None:  # pragma: no cover - requires ultralytics
        self._model = None
        self.loaded = False

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, LetterboxParams]:
        # Ultralytics owns preprocessing; hand it the raw frame.
        return frame, _identity_params(frame)

    def _forward(self, tensor: Any, params: LetterboxParams) -> Any:  # pragma: no cover
        return self._model.predict(
            tensor,
            imgsz=max(self.input_size),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            half=self.half,
            device=self.device,
            classes=sorted(self.class_filter) if self.class_filter else None,
            max_det=self.max_detections,
            verbose=False,
        )

    def _postprocess(  # pragma: no cover - requires ultralytics
        self, raw: Any, params: LetterboxParams
    ) -> List[Detection]:
        detections: List[Detection] = []
        if not raw:
            return detections
        boxes = getattr(raw[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), score, cls in zip(xyxy, confs, classes):
            if self.class_filter is not None and int(cls) not in self.class_filter:
                continue
            detections.append(
                Detection(
                    float(x1), float(y1), float(x2), float(y2),
                    float(score), int(cls), self.label(int(cls)),
                )
            )
        return detections
