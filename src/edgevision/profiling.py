"""Per-stage timing with percentiles, and a backend benchmark harness.

Why percentiles and not mean FPS
--------------------------------
"22 FPS average" is close to useless for a real-time system. Average frame
time hides the tail, and the tail is what a human sees. Consider 100 frames
where 95 take 30 ms and 5 take 300 ms: the mean is 43.5 ms, which reports a
perfectly respectable 23 FPS — while the viewer sees five visible freezes,
and a control loop consuming those detections sees five 300 ms holes.

So this module reports p50 (the typical frame), p90 (how bad a common bad
frame is) and p99 (the freezes). On edge hardware the tail has specific,
diagnosable causes, and the shape tells you which one:

* p99 far above p50, periodically   -> thermal throttling or a DVFS clock step
* p99 above p50 only at the start   -> warmup not discarded
* p90 and p99 both high             -> you are genuinely compute-bound
* a high *capture* p99 only         -> USB bandwidth or a camera auto-exposure
  step, not the model

Timings are collected per stage (capture / preprocess / inference / track /
draw / sink) because "the pipeline is slow" is not actionable and "annotation
costs 18 ms because you are drawing text with a scaled font" is.
"""

from __future__ import annotations

import json
import logging
import math
import platform
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

__all__ = [
    "percentile",
    "StageStats",
    "StageTimer",
    "Profiler",
    "RollingRate",
    "BenchmarkResult",
    "benchmark",
]


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of ``values``.

    Implemented here rather than deferred to numpy so the semantics are
    pinned and testable: sort ascending, take the position
    ``rank = q/100 * (n - 1)``, and linearly interpolate between the two
    neighbouring samples. This is numpy's default ``method="linear"``.

    With ``[10, 20, 30, 40]`` and ``q=50`` the rank is 1.5, so the answer is
    25.0 — halfway between the two middle samples, not "the second one".
    """
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    data = sorted(float(v) for v in values)
    n = len(data)
    if n == 0:
        return float("nan")
    if n == 1:
        return data[0]

    rank = (q / 100.0) * (n - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return data[lower]
    weight = rank - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


@dataclass(frozen=True)
class StageStats:
    """Summary of one stage's frame times, in milliseconds."""

    name: str
    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    total_ms: float

    @property
    def fps(self) -> float:
        """Throughput implied by the *median* frame time, not the mean."""
        return 1000.0 / self.p50_ms if self.p50_ms > 0 else 0.0

    @property
    def jitter_ratio(self) -> float:
        """p99 / p50. Above ~2 means visible stutter; investigate the tail."""
        return self.p99_ms / self.p50_ms if self.p50_ms > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["fps"] = round(self.fps, 2)
        payload["jitter_ratio"] = round(self.jitter_ratio, 2)
        return payload

    @classmethod
    def from_samples(cls, name: str, samples_s: Sequence[float]) -> "StageStats":
        """Build stats from a sequence of durations in *seconds*."""
        ms = [float(s) * 1000.0 for s in samples_s]
        if not ms:
            return cls(name, 0, *(float("nan"),) * 7, 0.0)
        mean = sum(ms) / len(ms)
        variance = sum((v - mean) ** 2 for v in ms) / len(ms)
        return cls(
            name=name,
            count=len(ms),
            mean_ms=mean,
            p50_ms=percentile(ms, 50),
            p90_ms=percentile(ms, 90),
            p99_ms=percentile(ms, 99),
            min_ms=min(ms),
            max_ms=max(ms),
            std_ms=math.sqrt(variance),
            total_ms=sum(ms),
        )


class StageTimer:
    """Context manager that records one stage duration into a profiler."""

    __slots__ = ("_profiler", "_stage", "_start", "elapsed")

    def __init__(self, profiler: "Profiler", stage: str) -> None:
        self._profiler = profiler
        self._stage = stage
        self._start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        self._profiler.record(self._stage, self.elapsed)


class Profiler:
    """Rolling per-stage timing collector.

    A bounded window (default 300 frames, i.e. ~10 s at 30 FPS) is used on
    purpose: a long-running pipeline should report *recent* behaviour, so that
    a thermal throttle event shows up in the numbers within seconds instead of
    being averaged away by the good minutes that preceded it.
    """

    #: The order stages are printed in, when they are present.
    STAGE_ORDER = (
        "capture", "preprocess", "inference", "postprocess",
        "track", "draw", "sink", "total",
    )

    def __init__(self, window: int = 300, warmup_frames: int = 0) -> None:
        self.window = int(window)
        self.warmup_frames = int(warmup_frames)
        self._samples: Dict[str, Deque[float]] = {}
        self._counts: Dict[str, int] = {}
        self.frames = 0
        self.started_at = time.perf_counter()

    def reset(self) -> None:
        self._samples.clear()
        self._counts.clear()
        self.frames = 0
        self.started_at = time.perf_counter()

    def record(self, stage: str, seconds: float) -> None:
        """Record one duration in seconds for ``stage``."""
        if self._counts.get(stage, 0) < self.warmup_frames:
            # Discarded on purpose; see Detector.warmup for why the first
            # inferences are meaningless.
            self._counts[stage] = self._counts.get(stage, 0) + 1
            return
        self._counts[stage] = self._counts.get(stage, 0) + 1
        bucket = self._samples.get(stage)
        if bucket is None:
            bucket = deque(maxlen=self.window)
            self._samples[stage] = bucket
        bucket.append(float(seconds))

    @contextmanager
    def span(self, stage: str) -> Iterator[StageTimer]:
        """``with profiler.span("inference"): ...``"""
        timer = StageTimer(self, stage)
        timer.__enter__()
        try:
            yield timer
        finally:
            timer.__exit__(None, None, None)

    def tick(self) -> None:
        """Count one completed frame."""
        self.frames += 1

    def stages(self) -> List[str]:
        known = [s for s in self.STAGE_ORDER if s in self._samples]
        extra = sorted(s for s in self._samples if s not in self.STAGE_ORDER)
        return known + extra

    def stats(self, stage: str) -> StageStats:
        return StageStats.from_samples(stage, self._samples.get(stage, ()))

    def summary(self) -> Dict[str, StageStats]:
        return {stage: self.stats(stage) for stage in self.stages()}

    @property
    def elapsed(self) -> float:
        return max(1e-9, time.perf_counter() - self.started_at)

    @property
    def fps(self) -> float:
        """End-to-end throughput measured over the whole run."""
        return self.frames / self.elapsed

    def sample_count(self, stage: str) -> int:
        return len(self._samples.get(stage, ()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames": self.frames,
            "elapsed_s": round(self.elapsed, 3),
            "fps": round(self.fps, 2),
            "stages": {k: v.to_dict() for k, v in self.summary().items()},
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def format_table(self) -> str:
        """Fixed-width per-stage table, milliseconds."""
        summary = self.summary()
        if not summary:
            return "no timing samples collected"
        name_w = max(6, max(len(k) for k in summary))
        header = (
            f"{'stage'.ljust(name_w)}  {'n':>6}  {'mean':>8}  {'p50':>8}  "
            f"{'p90':>8}  {'p99':>8}  {'max':>8}"
        )
        lines = [header, "-" * len(header)]
        for name, stats in summary.items():
            lines.append(
                f"{name.ljust(name_w)}  {stats.count:>6}  {stats.mean_ms:>8.2f}  "
                f"{stats.p50_ms:>8.2f}  {stats.p90_ms:>8.2f}  {stats.p99_ms:>8.2f}  "
                f"{stats.max_ms:>8.2f}"
            )
        lines.append("")
        lines.append(f"frames={self.frames}  wall={self.elapsed:.2f}s  fps={self.fps:.2f}")
        return "\n".join(lines)


class RollingRate:
    """Frames-per-second over a sliding time window.

    Uses arrival timestamps rather than a decayed average so the number
    responds immediately when the pipeline stalls, instead of drifting down
    over the next ten seconds while you stare at it wondering if it is broken.
    """

    def __init__(self, window_s: float = 2.0, max_samples: int = 600) -> None:
        self.window_s = float(window_s)
        self._times: Deque[float] = deque(maxlen=int(max_samples))

    def tick(self, now: Optional[float] = None) -> float:
        stamp = time.perf_counter() if now is None else float(now)
        self._times.append(stamp)
        cutoff = stamp - self.window_s
        while len(self._times) > 1 and self._times[0] < cutoff:
            self._times.popleft()
        return self.value

    @property
    def value(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0:
            return 0.0
        return (len(self._times) - 1) / span

    def reset(self) -> None:
        self._times.clear()


@dataclass
class BenchmarkResult:
    """Outcome of :func:`benchmark`."""

    backend: str
    model: Optional[str]
    input_size: Sequence[int]
    frame_shape: Sequence[int]
    frames: int
    warmup_frames: int
    warmup_ms: float
    stats: Dict[str, StageStats]
    detections_per_frame: float
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.machine()}")
    notes: List[str] = field(default_factory=list)

    @property
    def inference(self) -> StageStats:
        return self.stats["inference"]

    @property
    def fps(self) -> float:
        return self.inference.fps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "input_size": list(self.input_size),
            "frame_shape": list(self.frame_shape),
            "frames": self.frames,
            "warmup_frames": self.warmup_frames,
            "warmup_ms": round(self.warmup_ms, 3),
            "detections_per_frame": round(self.detections_per_frame, 3),
            "platform": self.platform,
            "fps_from_p50": round(self.fps, 2),
            "stages": {k: v.to_dict() for k, v in self.stats.items()},
            "notes": list(self.notes),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def format_table(self) -> str:
        """A single table you can paste into a benchmark log."""
        lines = [
            f"backend        : {self.backend}",
            f"model          : {self.model}",
            f"input size     : {tuple(self.input_size)}",
            f"frame size     : {tuple(self.frame_shape[1::-1])}",
            f"platform       : {self.platform}",
            f"frames timed   : {self.frames} (after {self.warmup_frames} warmup, discarded)",
            f"first inference: {self.warmup_ms:.1f} ms  <- discarded, not representative",
            f"detections/frm : {self.detections_per_frame:.2f}",
            "",
        ]
        name_w = max(9, max((len(k) for k in self.stats), default=9))
        header = (
            f"{'stage'.ljust(name_w)}  {'mean':>8}  {'p50':>8}  {'p90':>8}  "
            f"{'p99':>8}  {'max':>8}  {'fps@p50':>8}"
        )
        lines += [header, "-" * len(header)]
        for name, stats in self.stats.items():
            lines.append(
                f"{name.ljust(name_w)}  {stats.mean_ms:>8.2f}  {stats.p50_ms:>8.2f}  "
                f"{stats.p90_ms:>8.2f}  {stats.p99_ms:>8.2f}  {stats.max_ms:>8.2f}  "
                f"{stats.fps:>8.1f}"
            )
        jitter = self.inference.jitter_ratio
        lines.append("")
        lines.append(f"p99/p50 jitter ratio: {jitter:.2f}")
        if jitter > 2.0:
            lines.append(
                "  high tail: check for thermal throttling (tegrastats / vcgencmd), "
                "another process on the GPU, or warmup leaking into the measurement"
            )
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)


def benchmark(
    detector: Any,
    frames: Optional[Iterable[np.ndarray]] = None,
    num_frames: int = 100,
    frame_shape: Sequence[int] = (720, 1280, 3),
    warmup: int = 5,
    frame_factory: Optional[Callable[[int], np.ndarray]] = None,
    profile_stages: bool = True,
    notes: Optional[Sequence[str]] = None,
) -> BenchmarkResult:
    """Time a backend over ``num_frames`` and return a report.

    Warmup runs are executed *and thrown away*. The first inference on any
    accelerator pays for memory allocation, kernel autotuning and, on
    TensorRT, occasionally engine deserialisation work deferred until first
    use. Including it inflates the mean and destroys the p99. If your
    benchmark's first frame is 40x the rest, that is not a spike to explain,
    it is a measurement you should not have taken.

    When ``profile_stages`` is set, preprocessing and postprocessing are timed
    separately from the forward pass. That split matters: on a Pi it is common
    for letterboxing plus NMS to cost as much as the network itself, and no
    amount of quantisation will fix that.
    """
    profiler = Profiler(window=max(16, num_frames * 2))
    detector.ensure_loaded()

    if frame_factory is None:
        rng = np.random.default_rng(1234)
        base = rng.integers(0, 255, size=tuple(frame_shape), dtype=np.uint8)

        def frame_factory(index: int) -> np.ndarray:  # type: ignore[misc]
            # Reuse one buffer with a cheap per-frame perturbation: generating
            # fresh random frames would time numpy, not the detector.
            return base

    frame_list: Optional[List[np.ndarray]] = None
    if frames is not None:
        frame_list = list(frames)
        if not frame_list:
            raise ValueError("frames iterable was empty")
        num_frames = len(frame_list)
        frame_shape = frame_list[0].shape

    def get_frame(index: int) -> np.ndarray:
        if frame_list is not None:
            return frame_list[index % len(frame_list)]
        return frame_factory(index)

    warmup_ms = 0.0
    for i in range(max(0, int(warmup))):
        start = time.perf_counter()
        detector.infer(get_frame(i))
        if i == 0:
            warmup_ms = (time.perf_counter() - start) * 1000.0

    total_detections = 0
    for i in range(int(num_frames)):
        frame = get_frame(i)
        start_total = time.perf_counter()
        if profile_stages and hasattr(detector, "_preprocess"):
            with profiler.span("preprocess"):
                tensor, params = detector._preprocess(frame)
            with profiler.span("inference"):
                raw = detector._forward(tensor, params)
            with profiler.span("postprocess"):
                detections = detector._postprocess(raw, params)
        else:
            with profiler.span("inference"):
                detections = detector.infer(frame)
        profiler.record("total", time.perf_counter() - start_total)
        profiler.tick()
        total_detections += len(detections)

    stats = {stage: profiler.stats(stage) for stage in profiler.stages()}
    if "inference" not in stats:  # pragma: no cover - defensive
        stats["inference"] = profiler.stats("inference")

    return BenchmarkResult(
        backend=getattr(detector, "name", type(detector).__name__),
        model=getattr(detector, "model_path", None),
        input_size=getattr(detector, "input_size", (0, 0)),
        frame_shape=tuple(frame_shape),
        frames=int(num_frames),
        warmup_frames=int(warmup),
        warmup_ms=warmup_ms,
        stats=stats,
        detections_per_frame=total_detections / max(1, int(num_frames)),
        notes=list(notes or []),
    )
