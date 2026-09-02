"""Percentile math, per-stage collection and the benchmark harness."""

from __future__ import annotations

import numpy as np
import pytest

from edgevision.backends import MockBackend
from edgevision.profiling import (
    BenchmarkResult,
    Profiler,
    RollingRate,
    StageStats,
    benchmark,
    percentile,
)


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


def test_percentile_interpolates_between_samples():
    data = [10, 20, 30, 40]
    # rank = 0.5 * 3 = 1.5 -> halfway between 20 and 30.
    assert percentile(data, 50) == pytest.approx(25.0)
    assert percentile(data, 0) == 10.0
    assert percentile(data, 100) == 40.0
    assert percentile(data, 25) == pytest.approx(17.5)


def test_percentile_matches_numpy_on_random_data():
    rng = np.random.default_rng(5)
    for _ in range(20):
        data = rng.uniform(0, 100, size=int(rng.integers(2, 200))).tolist()
        for q in (0, 1, 25, 50, 90, 95, 99, 100):
            assert percentile(data, q) == pytest.approx(float(np.percentile(data, q)))


def test_percentile_is_order_independent():
    data = [5, 1, 9, 3, 7]
    assert percentile(data, 50) == percentile(sorted(data, reverse=True), 50) == 5.0


def test_percentile_edge_cases():
    assert percentile([42.0], 99) == 42.0
    assert np.isnan(percentile([], 50))
    with pytest.raises(ValueError):
        percentile([1, 2], 101)


def test_mean_hides_the_tail_that_percentiles_expose():
    """The motivating example: a good mean with five visible freezes."""
    samples_s = [0.030] * 95 + [0.300] * 5
    stats = StageStats.from_samples("inference", samples_s)

    assert stats.mean_ms == pytest.approx(43.5)
    assert 1000.0 / stats.mean_ms == pytest.approx(22.99, abs=0.05)  # "23 fps"
    assert stats.p50_ms == pytest.approx(30.0)
    assert stats.p99_ms > 250.0, "the freezes must be visible in p99"
    assert stats.jitter_ratio > 8.0
    assert stats.max_ms == pytest.approx(300.0)


def test_stage_stats_basic_fields():
    stats = StageStats.from_samples("draw", [0.010, 0.020, 0.030])
    assert stats.count == 3
    assert stats.min_ms == pytest.approx(10.0) and stats.max_ms == pytest.approx(30.0)
    assert stats.mean_ms == pytest.approx(20.0)
    assert stats.p50_ms == pytest.approx(20.0)
    assert stats.std_ms == pytest.approx(np.std([10, 20, 30]))
    assert stats.fps == pytest.approx(50.0)
    assert stats.to_dict()["name"] == "draw"


def test_stage_stats_on_empty_samples():
    stats = StageStats.from_samples("idle", [])
    assert stats.count == 0 and np.isnan(stats.p50_ms)
    assert stats.fps == 0.0 and stats.jitter_ratio == 0.0


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


def test_profiler_collects_per_stage_and_orders_the_table():
    profiler = Profiler()
    for i in range(10):
        profiler.record("inference", 0.010 + i * 0.001)
        profiler.record("draw", 0.001)
        profiler.tick()

    assert profiler.stages()[:2] == ["inference", "draw"]
    assert profiler.stats("inference").count == 10
    assert profiler.stats("inference").p50_ms == pytest.approx(14.5)
    assert profiler.frames == 10
    assert "inference" in profiler.format_table()
    assert profiler.to_dict()["stages"]["draw"]["count"] == 10


def test_profiler_window_keeps_only_recent_samples():
    """Old good frames must not mask a recent collapse."""
    profiler = Profiler(window=50)
    for _ in range(100):
        profiler.record("inference", 0.010)
    for _ in range(50):
        profiler.record("inference", 0.100)

    assert profiler.sample_count("inference") == 50
    assert profiler.stats("inference").p50_ms == pytest.approx(100.0)


def test_profiler_discards_warmup_samples():
    profiler = Profiler(warmup_frames=3)
    for value in (5.0, 5.0, 5.0, 0.010, 0.010):
        profiler.record("inference", value)

    assert profiler.sample_count("inference") == 2
    assert profiler.stats("inference").max_ms == pytest.approx(10.0)


def test_profiler_span_times_a_block():
    profiler = Profiler()
    with profiler.span("track") as timer:
        sum(range(20000))
    assert timer.elapsed > 0
    assert profiler.sample_count("track") == 1


def test_profiler_reset():
    profiler = Profiler()
    profiler.record("draw", 0.01)
    profiler.tick()
    profiler.reset()
    assert profiler.frames == 0 and profiler.sample_count("draw") == 0


def test_rolling_rate_uses_injected_timestamps():
    rate = RollingRate(window_s=1.0)
    for i in range(11):
        rate.tick(now=i * 0.05)  # 20 fps
    assert rate.value == pytest.approx(20.0, rel=0.05)

    rate.reset()
    assert rate.value == 0.0


def test_rolling_rate_forgets_old_samples():
    rate = RollingRate(window_s=0.5)
    for i in range(10):
        rate.tick(now=i * 0.01)   # a fast burst
    rate.tick(now=5.0)            # then a long stall
    assert rate.value < 5.0, "a stall must show up immediately"


# ---------------------------------------------------------------------------
# benchmark()
# ---------------------------------------------------------------------------


def test_benchmark_reports_stages_and_discards_warmup():
    detector = MockBackend(num_objects=3)
    result = benchmark(detector, num_frames=25, warmup=4, frame_shape=(180, 320, 3))

    assert isinstance(result, BenchmarkResult)
    assert result.frames == 25
    assert result.warmup_frames == 4
    assert result.detections_per_frame == pytest.approx(3.0)
    assert detector.call_count == 29, "warmup runs happen, then are not timed"
    assert result.stats["inference"].count == 25
    assert {"preprocess", "inference", "postprocess", "total"} <= set(result.stats)
    assert result.stats["inference"].p99_ms >= result.stats["inference"].p50_ms
    assert result.fps > 0


def test_benchmark_over_supplied_frames():
    frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(7)]
    result = benchmark(MockBackend(num_objects=1), frames=frames, warmup=1)

    assert result.frames == 7
    assert tuple(result.frame_shape) == (120, 160, 3)
    assert "backend" in result.to_dict()


def test_benchmark_table_and_json_render():
    result = benchmark(MockBackend(), num_frames=10, warmup=2, frame_shape=(120, 160, 3))
    table = result.format_table()

    assert "p99" in table and "discarded" in table
    assert "mock" in table
    payload = result.to_json()
    assert '"backend"' in payload and '"stages"' in payload


def test_benchmark_rejects_an_empty_frame_list():
    with pytest.raises(ValueError):
        benchmark(MockBackend(), frames=[])
