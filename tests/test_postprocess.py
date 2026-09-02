"""NMS, IoU and YOLO head decoding."""

from __future__ import annotations

import numpy as np
import pytest

from edgevision.postprocess import (
    batched_nms,
    decode_yolo_v5,
    decode_yolo_v8,
    iou_matrix,
    nms,
    xywh2xyxy,
    xyxy2xywh,
)


def test_iou_matrix_known_values():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array(
        [
            [0, 0, 10, 10],    # identical      -> 1.0
            [5, 0, 15, 10],    # half overlap   -> 50 / 150
            [20, 20, 30, 30],  # disjoint       -> 0.0
        ],
        dtype=np.float32,
    )
    ious = iou_matrix(a, b)
    assert ious.shape == (1, 3)
    assert ious[0, 0] == pytest.approx(1.0)
    assert ious[0, 1] == pytest.approx(50.0 / 150.0)
    assert ious[0, 2] == pytest.approx(0.0)


def test_iou_matrix_handles_empty_inputs():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)
    assert iou_matrix(np.zeros((2, 4)), np.zeros((0, 4))).shape == (2, 0)


def test_nms_suppresses_overlaps_and_keeps_the_highest_score():
    """The core guarantee: one box per cluster, and it is the best one."""
    boxes = np.array(
        [
            [100, 100, 200, 200],  # score 0.60
            [105, 105, 205, 205],  # score 0.95  <- best of this cluster
            [102, 98, 198, 202],   # score 0.80
            [400, 400, 480, 480],  # score 0.70  <- separate object
        ],
        dtype=np.float32,
    )
    scores = np.array([0.60, 0.95, 0.80, 0.70], dtype=np.float32)

    keep = nms(boxes, scores, iou_threshold=0.5)

    assert len(keep) == 2, "the three overlapping boxes must collapse to one"
    assert keep[0] == 1, "highest-scoring box must survive and come first"
    assert 3 in keep, "the non-overlapping box must be kept"
    assert 0 not in keep and 2 not in keep


def test_nms_keeps_everything_when_nothing_overlaps():
    boxes = np.array(
        [[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]], dtype=np.float32
    )
    scores = np.array([0.3, 0.9, 0.6], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert sorted(keep.tolist()) == [0, 1, 2]
    # Output is ordered by descending score.
    assert keep.tolist() == [1, 2, 0]


def test_nms_threshold_controls_aggressiveness():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=np.float32)  # IoU = 1/3
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert len(nms(boxes, scores, iou_threshold=0.2)) == 1
    assert len(nms(boxes, scores, iou_threshold=0.5)) == 2


def test_nms_respects_max_detections():
    rng = np.random.default_rng(3)
    boxes = rng.uniform(0, 1000, size=(50, 2))
    boxes = np.concatenate([boxes, boxes + 10], axis=1).astype(np.float32)
    scores = rng.uniform(0, 1, size=50).astype(np.float32)
    assert len(nms(boxes, scores, 0.5, max_detections=5)) == 5


def test_nms_on_empty_input():
    assert nms(np.zeros((0, 4)), np.zeros((0,))).shape == (0,)


def test_nms_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        nms(np.zeros((3, 4)), np.zeros((2,)))


def test_batched_nms_does_not_suppress_across_classes():
    """A person in front of a car overlaps it; both are correct detections."""
    boxes = np.array([[0, 0, 100, 100], [2, 2, 98, 98]], dtype=np.float32)
    scores = np.array([0.9, 0.85], dtype=np.float32)

    same_class = batched_nms(boxes, scores, np.array([0, 0]), 0.5)
    other_class = batched_nms(boxes, scores, np.array([0, 2]), 0.5)

    assert len(same_class) == 1
    assert len(other_class) == 2
    assert batched_nms(boxes, scores, np.array([0, 2]), 0.5, class_agnostic=True).size == 1


def test_xywh_round_trip():
    boxes = np.array([[50, 60, 20, 10], [0, 0, 4, 4]], dtype=np.float32)
    assert np.allclose(xyxy2xywh(xywh2xyxy(boxes)), boxes)
    assert np.allclose(xywh2xyxy(np.array([[10, 10, 4, 6]])), [[8, 7, 12, 13]])


def test_decode_yolo_v8_handles_both_orientations():
    num_classes = 3
    # One anchor: box (cx=50, cy=60, w=20, h=10), class 2 at score 0.8.
    raw = np.zeros((1, 4 + num_classes, 1), dtype=np.float32)
    raw[0, :4, 0] = [50, 60, 20, 10]
    raw[0, 4 + 2, 0] = 0.8

    boxes, scores, classes = decode_yolo_v8(raw, 0.25, num_classes=num_classes)
    assert boxes.shape == (1, 4)
    assert np.allclose(boxes[0], [40, 55, 60, 65])
    assert scores[0] == pytest.approx(0.8)
    assert classes[0] == 2

    transposed = np.transpose(raw, (0, 2, 1))
    boxes_t, scores_t, classes_t = decode_yolo_v8(transposed, 0.25, num_classes=num_classes)
    assert np.allclose(boxes_t, boxes) and classes_t[0] == 2 and scores_t[0] == scores[0]


def test_decode_yolo_v8_applies_the_confidence_gate():
    raw = np.zeros((1, 4 + 2, 3), dtype=np.float32)
    raw[0, :4, :] = np.array([[10, 20, 30], [10, 20, 30], [4, 4, 4], [4, 4, 4]])
    raw[0, 4, :] = [0.1, 0.5, 0.9]
    boxes, scores, _ = decode_yolo_v8(raw, conf_threshold=0.4, num_classes=2)
    assert len(boxes) == 2
    assert scores.min() >= 0.4


def test_decode_yolo_v5_multiplies_objectness_by_class_score():
    """v5 keeps objectness separate; forgetting the multiply inflates scores."""
    raw = np.zeros((1, 1, 5 + 2), dtype=np.float32)
    raw[0, 0, :4] = [50, 60, 20, 10]
    raw[0, 0, 4] = 0.5      # objectness
    raw[0, 0, 5 + 1] = 0.6  # class 1 probability

    boxes, scores, classes = decode_yolo_v5(raw, conf_threshold=0.2, num_classes=2)
    assert classes[0] == 1
    assert scores[0] == pytest.approx(0.3), "score must be objectness * class prob"
    assert np.allclose(boxes[0], [40, 55, 60, 65])

    # With the gate above the product the detection disappears.
    assert len(decode_yolo_v5(raw, conf_threshold=0.45, num_classes=2)[0]) == 0


def test_decode_rejects_batched_input():
    with pytest.raises(ValueError):
        decode_yolo_v8(np.zeros((2, 84, 10), dtype=np.float32))
