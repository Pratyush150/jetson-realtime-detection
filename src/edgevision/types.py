"""Core value types shared by every stage of the pipeline.

Keeping these dependency-free (numpy only) is deliberate: the capture thread,
the backends, the tracker and the sinks all speak the same small vocabulary,
so swapping a TensorRT engine for a Hailo device changes nothing downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Detection",
    "Track",
    "TrackState",
    "detections_to_array",
    "array_to_detections",
]


@dataclass
class Detection:
    """A single axis-aligned detection in *original frame* pixel coordinates.

    Coordinates are always ``xyxy`` floats in the source frame's coordinate
    system. Backends are responsible for undoing their own letterbox padding
    before handing a :class:`Detection` out; nothing downstream knows or cares
    what input resolution the network ran at.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int = 0
    class_name: str = ""

    def __post_init__(self) -> None:
        # Normalise so that x1 <= x2 and y1 <= y2. Some exporters emit boxes
        # with flipped corners after a transpose; fixing it here stops the
        # bug from silently producing negative-area IoU downstream.
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1
        self.score = float(self.score)
        self.class_id = int(self.class_id)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def as_xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def as_xywh(self) -> np.ndarray:
        return np.array(
            [self.x1, self.y1, self.width, self.height], dtype=np.float32
        )

    def iou(self, other: "Detection") -> float:
        """Intersection-over-union with another detection."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        if union <= 0.0:
            return 0.0
        return float(inter / union)

    def scaled(self, sx: float, sy: float) -> "Detection":
        """Return a copy scaled about the origin (used by ROI remapping)."""
        return Detection(
            self.x1 * sx,
            self.y1 * sy,
            self.x2 * sx,
            self.y2 * sy,
            self.score,
            self.class_id,
            self.class_name,
        )

    def translated(self, dx: float, dy: float) -> "Detection":
        return Detection(
            self.x1 + dx,
            self.y1 + dy,
            self.x2 + dx,
            self.y2 + dy,
            self.score,
            self.class_id,
            self.class_name,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [
                round(float(self.x1), 2),
                round(float(self.y1), 2),
                round(float(self.x2), 2),
                round(float(self.y2), 2),
            ],
            "score": round(float(self.score), 4),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Detection":
        x1, y1, x2, y2 = payload["bbox"]
        return cls(
            x1,
            y1,
            x2,
            y2,
            payload.get("score", 0.0),
            payload.get("class_id", 0),
            payload.get("class_name", ""),
        )


class TrackState(str, Enum):
    """Lifecycle of a track.

    ``TENTATIVE`` tracks exist but are not reported: a single false positive
    should not spawn a visible ID. ``LOST`` tracks are coasting on the motion
    model only; they are still reported (optionally) so short occlusions do
    not break an ID, but they are deleted once ``max_age`` is exceeded.
    """

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class Track:
    """A tracked object with a stable integer ID."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 0.0
    class_id: int = 0
    class_name: str = ""
    state: TrackState = TrackState.TENTATIVE
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)

    @property
    def is_confirmed(self) -> bool:
        return self.state is TrackState.CONFIRMED

    def as_xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def as_detection(self) -> Detection:
        return Detection(
            self.x1, self.y1, self.x2, self.y2, self.score, self.class_id, self.class_name
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = self.as_detection().to_dict()
        payload.update(
            {
                "track_id": int(self.track_id),
                "state": self.state.value,
                "age": int(self.age),
                "hits": int(self.hits),
                "time_since_update": int(self.time_since_update),
            }
        )
        return payload


def detections_to_array(detections: Sequence[Detection]) -> np.ndarray:
    """Pack detections into an ``(N, 6)`` float array ``[x1,y1,x2,y2,score,cls]``."""
    if not detections:
        return np.zeros((0, 6), dtype=np.float32)
    return np.array(
        [
            [d.x1, d.y1, d.x2, d.y2, d.score, float(d.class_id)]
            for d in detections
        ],
        dtype=np.float32,
    )


def array_to_detections(
    array: np.ndarray, class_names: Optional[Sequence[str]] = None
) -> List[Detection]:
    """Inverse of :func:`detections_to_array`."""
    out: List[Detection] = []
    if array is None or len(array) == 0:
        return out
    array = np.asarray(array, dtype=np.float32).reshape(-1, array.shape[-1])
    for row in array:
        cls_id = int(row[5]) if row.shape[0] > 5 else 0
        name = ""
        if class_names is not None and 0 <= cls_id < len(class_names):
            name = class_names[cls_id]
        out.append(
            Detection(
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]) if row.shape[0] > 4 else 1.0,
                cls_id,
                name,
            )
        )
    return out
