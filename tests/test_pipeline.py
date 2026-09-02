"""Pipeline orchestration and adaptive frame skipping.

The skipper is tested against its own cost model with synthetic timings, so
the assertions are exact and do not depend on how fast the CI machine is.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgevision.backends import MockBackend
from edgevision.pipeline import (
    AdaptiveFrameSkipper,
    Pipeline,
    ThermalMonitor,
    ThroughputWatchdog,
    annotate,
    track_color,
)
from edgevision.sinks import Sink
from edgevision.tracker import SortTracker
from edgevision.types import Track, TrackState


def blank(height: int = 240, width: int = 320) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


class RecordingSink(Sink):
    def __init__(self) -> None:
        self.calls = []

    def write(self, frame, tracks=(), detections=(), meta=None):
        self.calls.append((frame, list(tracks), list(detections), dict(meta or {})))


class ExplodingSink(Sink):
    def write(self, frame, tracks=(), detections=(), meta=None):
        raise RuntimeError("SD card full")


# ---------------------------------------------------------------------------
# Adaptive frame skipping
# ---------------------------------------------------------------------------


def test_adaptive_skip_converges_to_the_target_fps_budget():
    """100 ms inference + 5 ms overhead, 25 FPS target -> detect every 3rd frame.

    cost(N) = overhead + inference/N must be <= 1/25 = 40 ms.
      N=2 -> 5 + 50   = 55 ms -> 18.2 fps  (too slow)
      N=3 -> 5 + 33.3 = 38.3 ms -> 26.1 fps (fits)
    """
    skipper = AdaptiveFrameSkipper(target_fps=25.0, min_interval=1, max_interval=12)
    for _ in range(60):
        skipper.update(inference_s=0.100, overhead_s=0.005)

    assert skipper.interval == 3
    assert skipper.projected_fps >= 25.0
    assert skipper.projected_frame_cost(2) > skipper.budget_s, "3 must be the minimum"
    assert skipper.budget_exceeded is False
    assert "skip=3" in skipper.explain()


def test_adaptive_skip_stops_skipping_when_inference_is_cheap():
    skipper = AdaptiveFrameSkipper(target_fps=30.0)
    for _ in range(40):
        skipper.update(inference_s=0.005, overhead_s=0.002)
    assert skipper.interval == 1
    assert skipper.projected_fps > 30.0


def test_adaptive_skip_reacts_when_inference_gets_slower():
    """Thermal throttling makes inference slower mid-run; the interval must grow."""
    skipper = AdaptiveFrameSkipper(target_fps=30.0, alpha=0.5)
    for _ in range(30):
        skipper.update(inference_s=0.030, overhead_s=0.003)
    fast = skipper.interval

    for _ in range(30):
        skipper.update(inference_s=0.150, overhead_s=0.003)
    slow = skipper.interval

    assert slow > fast
    assert skipper.projected_fps >= 30.0 * 0.95


def test_adaptive_skip_clamps_to_max_interval():
    skipper = AdaptiveFrameSkipper(target_fps=60.0, max_interval=4)
    for _ in range(40):
        skipper.update(inference_s=2.0, overhead_s=0.001)
    assert skipper.interval == 4


def test_adaptive_skip_flags_an_unreachable_budget():
    """If per-frame overhead alone blows the budget, skipping cannot help."""
    skipper = AdaptiveFrameSkipper(target_fps=60.0, max_interval=8)
    for _ in range(30):
        skipper.update(inference_s=0.050, overhead_s=0.050)

    assert skipper.budget_exceeded is True
    assert skipper.interval == 8
    assert "overhead alone exceeds the budget" in skipper.explain()


def test_fixed_interval_overrides_adaptation():
    skipper = AdaptiveFrameSkipper(target_fps=30.0, fixed_interval=4)
    for _ in range(30):
        skipper.update(inference_s=0.5, overhead_s=0.001)
    assert skipper.is_fixed and skipper.interval == 4
    assert [skipper.should_detect(i) for i in range(8)] == [
        True, False, False, False, True, False, False, False
    ]


def test_skipper_waits_for_evidence_before_adapting():
    skipper = AdaptiveFrameSkipper(target_fps=30.0, warmup_updates=5)
    skipper.update(inference_s=1.0, overhead_s=0.001)
    assert skipper.interval == 1, "one slow sample must not swing the interval"


def test_skipper_rejects_nonsense_configuration():
    with pytest.raises(ValueError):
        AdaptiveFrameSkipper(target_fps=0)
    with pytest.raises(ValueError):
        AdaptiveFrameSkipper(min_interval=0)
    with pytest.raises(ValueError):
        AdaptiveFrameSkipper(min_interval=5, max_interval=2)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_pipeline_runs_the_detector_only_on_scheduled_frames():
    detector = MockBackend(num_objects=2, velocity=(6.0, 0.0))
    pipeline = Pipeline(detector, target_fps=30, skip=4, annotate_frames=False)

    results = [pipeline.process(blank()) for _ in range(12)]
    ran = [r.ran_inference for r in results]

    assert ran == [i % 4 == 0 for i in range(12)]
    assert pipeline.inference_count == 3
    assert detector.call_count == 3, "the detector must not run on skipped frames"
    assert pipeline.inference_ratio == pytest.approx(0.25)


def test_tracker_propagates_boxes_on_skipped_frames():
    """The whole point of skipping: boxes keep moving without the detector."""
    detector = MockBackend(num_objects=1, velocity=(10.0, 0.0), origin=(20.0, 60.0))
    pipeline = Pipeline(
        detector,
        tracker=SortTracker(max_age=30, min_hits=2),
        target_fps=30,
        skip=3,
        annotate_frames=False,
    )

    xs = []
    for _ in range(15):
        result = pipeline.process(blank())
        if result.tracks:
            xs.append((result.ran_inference, result.tracks[0].x1))

    detection_frames = sum(1 for ran, _ in xs if ran)
    assert detection_frames < len(xs), "some reported frames had no detector run"
    assert detection_frames <= 5, "with skip=3 at most every 3rd frame runs inference"

    positions = [x for _, x in xs]
    assert positions[-1] > positions[0], "boxes must advance across skipped frames"
    # Every frame reports a track, including the ones with no inference.
    assert len(xs) >= 12


def test_pipeline_records_a_timing_for_every_stage():
    pipeline = Pipeline(MockBackend(), target_fps=30, skip=1)
    for _ in range(6):
        result = pipeline.process(blank())

    assert set(result.timings) >= {"inference", "track", "draw", "sink", "total"}
    summary = pipeline.profiler.summary()
    assert {"inference", "track", "draw", "sink", "total"} <= set(summary)
    assert summary["total"].count == 6
    assert summary["total"].p99_ms >= summary["total"].p50_ms
    # Stage times must add up to no more than the measured total.
    assert result.timings["total"] >= result.timings["track"]


def test_pipeline_feeds_sinks_and_survives_a_failing_one():
    good = RecordingSink()
    pipeline = Pipeline(
        MockBackend(num_objects=1),
        sinks=[ExplodingSink(), good],
        target_fps=30,
        skip=1,
        annotate_frames=False,
    )
    for _ in range(4):
        pipeline.process(blank())

    assert len(good.calls) == 4, "a broken sink must not stop the others"
    assert len(pipeline.sinks.sinks) == 1
    assert pipeline.sinks.failed and "SD card full" in pipeline.sinks.failed[0][1]
    meta = good.calls[-1][3]
    assert meta["ran_inference"] is True and meta["skip_interval"] == 1


def test_pipeline_run_over_an_iterable_of_frames():
    frames = [blank() for _ in range(20)]
    pipeline = Pipeline(MockBackend(), target_fps=30, skip=2, annotate_frames=False)
    profiler = pipeline.run(frames, max_frames=10)

    assert pipeline.frame_index == 10
    assert profiler.frames == 10
    stats = pipeline.stats()
    assert stats["inference_frames"] == 5
    assert stats["skipper"]["interval"] == 2
    assert "detector" in stats and stats["detector"] == "mock"


def test_pipeline_on_frame_callback_can_stop_the_run():
    pipeline = Pipeline(MockBackend(), target_fps=30, skip=1, annotate_frames=False)
    seen = []

    def callback(result):
        seen.append(result.index)
        return len(seen) < 3

    pipeline.run([blank() for _ in range(50)], on_frame=callback)
    assert seen == [0, 1, 2]


def test_pipeline_without_a_tracker_still_detects():
    pipeline = Pipeline(MockBackend(num_objects=2), track=False, skip=1, annotate_frames=False)
    result = pipeline.process(blank())
    assert len(result.detections) == 2
    assert result.tracks == []


def test_frame_result_serialises():
    pipeline = Pipeline(MockBackend(num_objects=1), skip=1, annotate_frames=False)
    payload = pipeline.process(blank()).to_dict()
    assert payload["frame"] == 0 and payload["ran_inference"] is True
    assert "timings_ms" in payload and "total" in payload["timings_ms"]
    assert isinstance(payload["tracks"], list)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def test_annotate_draws_without_touching_the_source_frame():
    frame = blank(120, 160)
    tracks = [Track(1, 20, 30, 90, 100, 0.9, 0, "person", TrackState.CONFIRMED)]
    out = annotate(frame, tracks, overlay=["12.0 fps"])

    assert out.shape == frame.shape
    assert out.any(), "something must have been drawn"
    assert not frame.any(), "copy=True must leave the input untouched"


def test_annotate_in_place_modifies_the_frame():
    frame = blank(120, 160)
    annotate(frame, [Track(2, 10, 10, 50, 50, 0.5)], copy=False)
    assert frame.any()


def test_annotate_clips_boxes_outside_the_frame():
    frame = blank(60, 80)
    annotate(frame, [Track(3, -500, -500, 5000, 5000, 0.4)], copy=False)
    assert frame.any()


def test_track_colours_are_stable_and_distinct():
    assert track_color(5) == track_color(5)
    assert track_color(1) != track_color(2)


# ---------------------------------------------------------------------------
# Thermal / throughput awareness
# ---------------------------------------------------------------------------


def test_thermal_monitor_reads_sysfs_style_layout(tmp_path):
    zone = tmp_path / "thermal_zone0"
    zone.mkdir()
    (zone / "temp").write_text("85300\n")
    (zone / "type").write_text("CPU-therm\n")

    monitor = ThermalMonitor(warn_celsius=80.0, root=str(tmp_path))
    assert monitor.available()
    assert monitor.read()["CPU-therm"] == pytest.approx(85.3)
    assert monitor.hottest()[0] == "CPU-therm"

    warning = monitor.check()
    assert warning is not None and "85.3" in warning
    assert monitor.check() is None, "the warning must not repeat every frame"


def test_thermal_monitor_is_silent_without_sysfs(tmp_path):
    monitor = ThermalMonitor(root=str(tmp_path / "missing"))
    assert monitor.available() is False
    assert monitor.read() == {}
    assert monitor.check() is None


def test_throughput_watchdog_reports_a_sustained_collapse():
    watchdog = ThroughputWatchdog(baseline_frames=10, drop_ratio=0.75, patience=5)
    for _ in range(10):
        assert watchdog.update(30.0) is None
    assert watchdog.baseline_fps == pytest.approx(30.0)

    assert watchdog.update(29.0) is None, "small variation is not a collapse"
    messages = [watchdog.update(12.0) for _ in range(5)]

    assert messages[:4] == [None, None, None, None], "must be sustained, not a blip"
    assert messages[4] is not None and "throttling" in messages[4]
    assert watchdog.alerts == 1
