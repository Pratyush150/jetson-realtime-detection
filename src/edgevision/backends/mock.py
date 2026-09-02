"""A deterministic, dependency-free detector used by every test.

``MockBackend`` is not a stub that returns a hard-coded list. It synthesises a
YOLOv8-shaped output tensor in *letterboxed network space* and pushes it
through the exact same decode -> NMS -> un-letterbox tail every real backend
uses. So the tests that use it are genuinely exercising the coordinate math,
not bypassing it.

It also models motion, which is what makes offline tracker tests meaningful:
objects translate by a fixed velocity per frame, so a correct tracker must
hold a stable ID and an incorrect one visibly will not.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ..preprocess import LetterboxParams, letterbox_boxes
from ..types import Detection
from .base import Availability, Detector

__all__ = ["MockBackend"]


class MockBackend(Detector):
    """Synthetic detector with reproducible, moving boxes.

    Parameters
    ----------
    num_objects:
        How many synthetic objects to emit per frame.
    velocity:
        ``(vx, vy)`` pixels per inference call. Objects wrap around the frame.
    box_size:
        ``(w, h)`` of each synthetic box in original-frame pixels.
    origin, spacing:
        Placement of object 0 and the offset between successive objects.
    duplicate_overlap:
        If > 0, emit a second, lower-scoring box shifted by this many pixels
        for every object. NMS must collapse it back to one detection — that is
        how the tests prove the suppression tail is actually wired in.
    jitter:
        Deterministic pseudo-random per-frame perturbation (seeded by frame
        index), for testing tracker robustness to noisy boxes.
    latency_s:
        Artificial sleep per call, so pipeline timing paths can be exercised.
    """

    name = "mock"
    priority = -100  # never auto-selected over a real backend

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (640, 640),
        num_objects: int = 2,
        velocity: Tuple[float, float] = (6.0, 0.0),
        box_size: Tuple[float, float] = (80.0, 60.0),
        origin: Tuple[float, float] = (40.0, 60.0),
        spacing: Tuple[float, float] = (0.0, 140.0),
        class_ids: Optional[Sequence[int]] = None,
        base_score: float = 0.9,
        duplicate_overlap: float = 0.0,
        jitter: float = 0.0,
        seed: int = 0,
        latency_s: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path=model_path, input_size=input_size, **kwargs)
        self.num_objects = int(num_objects)
        self.velocity = (float(velocity[0]), float(velocity[1]))
        self.box_size = (float(box_size[0]), float(box_size[1]))
        self.origin = (float(origin[0]), float(origin[1]))
        self.spacing = (float(spacing[0]), float(spacing[1]))
        self.class_ids = tuple(int(c) for c in (class_ids or range(self.num_objects)))
        self.base_score = float(base_score)
        self.duplicate_overlap = float(duplicate_overlap)
        self.jitter = float(jitter)
        self.seed = int(seed)
        self.latency_s = float(latency_s)
        self.frame_index = 0
        self.call_count = 0

    # -- capability ---------------------------------------------------------

    @classmethod
    def probe(cls) -> Availability:
        return Availability(
            cls.name,
            True,
            "always available (synthetic detections, no model executed)",
            cls.priority,
        )

    # -- lifecycle ----------------------------------------------------------

    def load(self) -> None:
        self.loaded = True

    def reset(self) -> None:
        """Rewind the motion model so a test can replay the same sequence."""
        self.frame_index = 0
        self.call_count = 0

    # -- geometry -----------------------------------------------------------

    def boxes_for_frame(
        self, frame_index: int, frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Ground-truth ``xyxy`` boxes in original-frame coordinates.

        Public so tests can compare what the tracker produced against what the
        synthetic world actually did.
        """
        height, width = frame_shape[:2]
        bw, bh = self.box_size
        rng = np.random.default_rng(self.seed + frame_index) if self.jitter else None

        boxes = np.zeros((self.num_objects, 4), dtype=np.float32)
        for i in range(self.num_objects):
            x = self.origin[0] + self.spacing[0] * i + self.velocity[0] * frame_index
            y = self.origin[1] + self.spacing[1] * i + self.velocity[1] * frame_index
            # Wrap so long runs never march the object off the frame.
            span_x = max(1.0, width - bw)
            span_y = max(1.0, height - bh)
            x = float(np.mod(x, span_x))
            y = float(np.mod(y, span_y))
            if rng is not None:
                x += float(rng.uniform(-self.jitter, self.jitter))
                y += float(rng.uniform(-self.jitter, self.jitter))
            boxes[i] = (x, y, x + bw, y + bh)
        return boxes

    def detections_for_frame(
        self, frame_index: int, frame_shape: Tuple[int, int]
    ) -> List[Detection]:
        """Ground truth as :class:`Detection` objects."""
        boxes = self.boxes_for_frame(frame_index, frame_shape)
        return [
            Detection(
                float(b[0]),
                float(b[1]),
                float(b[2]),
                float(b[3]),
                self._score(i),
                self._class_id(i),
                self.label(self._class_id(i)),
            )
            for i, b in enumerate(boxes)
        ]

    def _class_id(self, i: int) -> int:
        if not self.class_ids:
            return 0
        return int(self.class_ids[i % len(self.class_ids)])

    def _score(self, i: int) -> float:
        # Deterministic, distinct, and strictly descending so NMS ordering is
        # unambiguous in assertions.
        return float(max(0.05, self.base_score - 0.05 * i))

    # -- inference ----------------------------------------------------------

    def _forward(self, tensor: Any, params: LetterboxParams) -> np.ndarray:
        """Build a YOLOv8-shaped ``(1, 4 + nc, anchors)`` tensor."""
        if self.latency_s > 0:
            time.sleep(self.latency_s)
        self.call_count += 1

        boxes = self.boxes_for_frame(self.frame_index, (params.src_h, params.src_w))
        self.frame_index += 1

        rows: List[Tuple[np.ndarray, float, int]] = []
        for i, box in enumerate(boxes):
            rows.append((box, self._score(i), self._class_id(i)))
            if self.duplicate_overlap > 0:
                shifted = box + np.array(
                    [self.duplicate_overlap, self.duplicate_overlap] * 2,
                    dtype=np.float32,
                )
                rows.append((shifted, self._score(i) * 0.8, self._class_id(i)))

        num_classes = len(self.class_names)
        out = np.zeros((1, 4 + num_classes, max(1, len(rows))), dtype=np.float32)
        if not rows:
            return out

        for j, (box, score, cls) in enumerate(rows):
            lb = letterbox_boxes(box[None, :], params)[0]
            out[0, 0, j] = (lb[0] + lb[2]) / 2.0
            out[0, 1, j] = (lb[1] + lb[3]) / 2.0
            out[0, 2, j] = lb[2] - lb[0]
            out[0, 3, j] = lb[3] - lb[1]
            out[0, 4 + (cls % num_classes), j] = score
        return out

    def describe(self) -> dict:
        info = super().describe()
        info.update(
            {
                "num_objects": self.num_objects,
                "velocity": list(self.velocity),
                "frame_index": self.frame_index,
                "synthetic": True,
            }
        )
        return info
