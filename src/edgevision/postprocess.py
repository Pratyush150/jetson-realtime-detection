"""Detection decoding and non-maximum suppression, in plain numpy.

Every backend ends up needing the same three things: turn a raw tensor into
``xyxy + score + class``, drop the overlaps, and undo the letterbox. Doing it
once here means the TensorRT path and the ONNX Runtime path cannot silently
disagree about, say, whether YOLOv8 output is transposed.

NMS is implemented in numpy rather than deferred to ``torchvision.ops.nms``
because on an edge board you frequently do not have torch at all, and because
a 8400x84 head with a 0.25 confidence gate typically leaves a few dozen boxes
— at that size a vectorised numpy loop is not the bottleneck, the network is.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


__all__ = [
    "xywh2xyxy",
    "xyxy2xywh",
    "iou_matrix",
    "nms",
    "batched_nms",
    "decode_yolo_v8",
    "decode_yolo_v5",
]


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """``[cx, cy, w, h]`` -> ``[x1, y1, x2, y2]``."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    out = np.empty_like(boxes)
    half_w = boxes[:, 2] / 2.0
    half_h = boxes[:, 3] / 2.0
    out[:, 0] = boxes[:, 0] - half_w
    out[:, 1] = boxes[:, 1] - half_h
    out[:, 2] = boxes[:, 0] + half_w
    out[:, 3] = boxes[:, 1] + half_h
    return out


def xyxy2xywh(boxes: np.ndarray) -> np.ndarray:
    """``[x1, y1, x2, y2]`` -> ``[cx, cy, w, h]``."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    out = np.empty_like(boxes)
    out[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
    out[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``xyxy`` boxes -> ``(len(a), len(b))``."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    iw = np.clip(ix2 - ix1, 0.0, None)
    ih = np.clip(iy2 - iy1, 0.0, None)
    inter = iw * ih

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out.astype(np.float32)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    max_detections: Optional[int] = None,
) -> np.ndarray:
    """Greedy non-maximum suppression.

    Returns the indices to keep, ordered by descending score. The highest
    scoring box in any overlapping cluster always survives — that property is
    what makes NMS safe to run before the tracker, which would otherwise get
    two competing observations for one object and flip the ID between them.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")

    order = np.argsort(-scores, kind="stable")
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0.0, None
    )

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if max_detections is not None and len(keep) >= max_detections:
            break
        if order.size == 1:
            break
        rest = order[1:]

        ix1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        iy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        ix2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        iy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)
        union = areas[i] + areas[rest] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            ious = np.where(union > 0, inter / union, 0.0)

        order = rest[ious <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45,
    max_detections: Optional[int] = None,
    class_agnostic: bool = False,
) -> np.ndarray:
    """Class-aware NMS.

    A person standing in front of a car legitimately overlaps it; suppressing
    across classes throws away a real detection. The standard trick is to
    offset each class into its own coordinate band so a single NMS pass can
    never compare boxes of different classes.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    class_ids = np.asarray(class_ids).reshape(-1).astype(np.int64)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    if class_agnostic:
        return nms(boxes, scores, iou_threshold, max_detections)

    span = float(boxes.max() - boxes.min()) + 1.0 if len(boxes) else 1.0
    offsets = class_ids.astype(np.float32)[:, None] * span
    return nms(boxes + offsets, scores, iou_threshold, max_detections)


def _orient(array: np.ndarray, channels: Optional[int]) -> np.ndarray:
    """Return the ``(num_anchors, channels)`` orientation of a 2-D head output.

    Different exporters disagree about whether the channel axis comes first.
    When the caller knows ``4 + nc`` (or ``5 + nc``) we match it exactly;
    otherwise we fall back to "the anchor axis is the longer one", which is
    true for any real anchor count (8400 at 640x640) but not for the tiny
    synthetic tensors used in tests — hence the explicit hint.
    """
    if channels is not None:
        if array.shape[1] == channels:
            return array
        if array.shape[0] == channels:
            return array.T
    if array.shape[0] < array.shape[1]:
        return array.T
    return array


def decode_yolo_v8(
    output: np.ndarray,
    conf_threshold: float = 0.25,
    num_classes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a YOLOv8/YOLO11-style head.

    Accepts either ``(1, 4 + nc, num_anchors)`` (the raw export layout) or
    ``(1, num_anchors, 4 + nc)`` (already transposed by some runtimes). Pass
    ``num_classes`` when you know it; otherwise the orientation is inferred
    from which axis is longer.

    Returns ``(boxes_xyxy, scores, class_ids)`` in *network input* space; the
    caller still has to un-letterbox.
    """
    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError("only batch size 1 is supported")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"unexpected YOLOv8 output shape {np.shape(output)}")

    array = _orient(array, None if num_classes is None else 4 + int(num_classes))
    if array.shape[1] < 5:
        raise ValueError("YOLOv8 output needs at least 4 box values and 1 class")

    boxes = array[:, :4]
    class_scores = array[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]

    mask = scores >= conf_threshold
    boxes = xywh2xyxy(boxes[mask])
    return boxes, scores[mask].astype(np.float32), class_ids[mask].astype(np.int64)


def decode_yolo_v5(
    output: np.ndarray,
    conf_threshold: float = 0.25,
    num_classes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a YOLOv5/v7-style head ``(1, num_anchors, 5 + nc)``.

    v5 keeps a separate objectness column; the usable score is
    ``objectness * class_prob``. Forgetting the multiply is a classic port
    bug: recall looks fine but every score is inflated, so a confidence gate
    tuned on v8 lets a pile of junk through.
    """
    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError("only batch size 1 is supported")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"unexpected YOLOv5 output shape {np.shape(output)}")
    array = _orient(array, None if num_classes is None else 5 + int(num_classes))
    if array.shape[1] < 6:
        raise ValueError("YOLOv5 output needs box, objectness and >=1 class")

    objectness = array[:, 4]
    class_scores = array[:, 5:] * objectness[:, None]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]

    mask = scores >= conf_threshold
    boxes = xywh2xyxy(array[mask, :4])
    return boxes, scores[mask].astype(np.float32), class_ids[mask].astype(np.int64)
