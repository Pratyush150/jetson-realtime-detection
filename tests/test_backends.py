"""Backend registry, capability probing, and the shared inference tail."""

from __future__ import annotations

import numpy as np
import pytest

from edgevision.backends import (
    Availability,
    Detector,
    HailoBackend,
    MockBackend,
    OnnxRuntimeBackend,
    TensorRTBackend,
    UltralyticsBackend,
    available_backends,
    backend_names,
    create_backend,
    format_backend_table,
    get_backend_class,
    probe_backends,
    register_backend,
    select_backend,
)


def blank(height=360, width=640):
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Registry and probing
# ---------------------------------------------------------------------------


def test_all_backends_are_registered_in_priority_order():
    names = backend_names()
    assert set(names) == {"mock", "ultralytics", "onnxruntime", "tensorrt", "hailo"}
    assert names[0] == "tensorrt", "TensorRT must outrank everything on Jetson"
    assert names[-1] == "mock", "mock must never outrank a real backend"


def test_every_probe_returns_a_reason_even_on_success():
    probes = probe_backends()
    assert set(probes) == set(backend_names())
    for name, probe in probes.items():
        assert isinstance(probe, Availability)
        assert probe.reason, f"{name} gave no reason"
        assert bool(probe) == probe.available


def test_mock_is_always_available():
    probe = MockBackend.probe()
    assert probe.available is True
    assert "mock" in available_backends()


def test_probes_of_absent_backends_explain_themselves():
    for cls in (TensorRTBackend, HailoBackend, OnnxRuntimeBackend, UltralyticsBackend):
        probe = cls.probe()
        if not probe.available:
            assert "not installed" in probe.reason or "failed" in probe.reason


def test_select_backend_falls_back_to_mock_without_a_model():
    choice = select_backend("auto", model_path=None)
    assert choice.name == "mock"
    assert "no accelerated backend" in choice.reason or choice.reason
    assert any("selected mock" in line for line in choice.log)


def test_select_backend_will_not_pick_a_backend_that_cannot_load_the_model():
    """TensorRT being installed is irrelevant if all you have is a .pt file."""
    choice = select_backend("tensorrt", model_path="yolov8n.pt")
    assert choice.name != "tensorrt"
    assert any("unusable" in line for line in choice.log)


def test_select_backend_can_be_told_not_to_fall_back():
    with pytest.raises(RuntimeError):
        select_backend("hailo", model_path="model.hef", allow_fallback=False)


def test_select_backend_rejects_unknown_names():
    with pytest.raises(KeyError):
        select_backend("tflite")
    with pytest.raises(KeyError):
        get_backend_class("coreml")


def test_backend_table_lists_every_backend():
    table = format_backend_table()
    for name in backend_names():
        assert name in table
    assert "reason" in table


def test_create_backend_returns_a_working_detector():
    detector = create_backend("mock", num_objects=2)
    assert isinstance(detector, MockBackend)
    assert len(detector.infer(blank())) == 2


def test_registering_a_bad_class_is_rejected():
    class NotADetector:
        name = "nope"

    with pytest.raises(TypeError):
        register_backend(NotADetector)  # type: ignore[arg-type]

    class Unnamed(MockBackend):
        name = "base"

    with pytest.raises(ValueError):
        register_backend(Unnamed)


def test_detector_is_abstract():
    with pytest.raises(TypeError):
        Detector()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# The shared inference tail, exercised through MockBackend
# ---------------------------------------------------------------------------


def test_mock_backend_is_deterministic():
    a = MockBackend(num_objects=2, velocity=(5.0, 2.0))
    b = MockBackend(num_objects=2, velocity=(5.0, 2.0))
    frame = blank()
    for _ in range(5):
        assert [d.to_dict() for d in a.infer(frame)] == [d.to_dict() for d in b.infer(frame)]


def test_mock_backend_detections_match_the_synthetic_ground_truth():
    """Proves letterbox -> decode -> NMS -> un-letterbox round-trips exactly."""
    detector = MockBackend(num_objects=3, velocity=(7.0, 4.0), input_size=(416, 416))
    frame = blank(480, 854)

    truth = detector.boxes_for_frame(0, frame.shape[:2])
    detections = detector.infer(frame)

    assert len(detections) == 3
    for det, expected in zip(detections, truth):
        assert det.as_xyxy() == pytest.approx(expected, abs=1.0)


def test_mock_backend_moves_objects_between_calls():
    detector = MockBackend(num_objects=1, velocity=(12.0, 0.0))
    frame = blank()
    first = detector.infer(frame)[0]
    second = detector.infer(frame)[0]
    assert second.x1 - first.x1 == pytest.approx(12.0, abs=1.0)

    detector.reset()
    assert detector.infer(frame)[0].x1 == pytest.approx(first.x1, abs=1.0)


def test_nms_runs_inside_the_backend_tail():
    """Duplicated boxes from the synthetic head must be collapsed to one each."""
    detector = MockBackend(num_objects=2, duplicate_overlap=6.0, velocity=(0.0, 0.0))
    detections = detector.infer(blank())

    assert len(detections) == 2, "the shifted duplicates were not suppressed"
    assert detections[0].score > detections[1].score
    assert detections[0].score == pytest.approx(0.9)


def test_class_filter_is_applied():
    detector = MockBackend(num_objects=3, class_ids=(0, 2, 5), classes=[2])
    detections = detector.infer(blank())
    assert len(detections) == 1 and detections[0].class_id == 2
    assert detections[0].class_name == "car"


def test_confidence_threshold_is_applied():
    detector = MockBackend(num_objects=3, base_score=0.9, conf_threshold=0.87)
    # Scores are 0.90, 0.85, 0.80 -> only the first survives.
    assert len(detector.infer(blank())) == 1


def test_backend_handles_an_empty_frame():
    assert MockBackend().infer(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_warmup_runs_and_reports_a_latency():
    detector = MockBackend()
    elapsed = detector.warmup(iterations=3)
    assert elapsed >= 0.0
    assert detector.call_count == 3
    assert detector.loaded is True


def test_describe_carries_enough_for_a_benchmark_row():
    info = MockBackend(input_size=(320, 320)).describe()
    assert info["backend"] == "mock"
    assert info["input_size"] == [320, 320]
    assert info["num_classes"] == 80
    assert info["synthetic"] is True


def test_context_manager_loads_and_closes():
    with MockBackend() as detector:
        assert detector.loaded is True
        detector.infer(blank())
    assert detector.loaded is False


# ---------------------------------------------------------------------------
# Hailo's on-device NMS decoding (pure function, no hardware needed)
# ---------------------------------------------------------------------------


def test_hailo_nms_decode_handles_normalised_y_first_boxes():
    """HailoRT emits [y_min, x_min, y_max, x_max, score] normalised to [0,1]."""
    raw = [
        np.zeros((0, 5), dtype=np.float32),                       # class 0: nothing
        np.array([[0.25, 0.10, 0.75, 0.60, 0.9]], np.float32),    # class 1
    ]
    boxes, scores, classes = HailoBackend.decode_hailo_nms(raw, 0.25, (640, 640))

    assert len(boxes) == 1
    assert classes[0] == 1 and scores[0] == pytest.approx(0.9)
    # x from column 1, y from column 0 -> x1=64, y1=160, x2=384, y2=480
    assert boxes[0] == pytest.approx([64.0, 160.0, 384.0, 480.0])


def test_hailo_nms_decode_applies_the_confidence_gate():
    raw = [np.array([[0.1, 0.1, 0.2, 0.2, 0.10]], np.float32)]
    boxes, _, _ = HailoBackend.decode_hailo_nms(raw, 0.25, (640, 640))
    assert len(boxes) == 0
