"""Pipeline orchestration, and the technique that makes edge detection usable.

capture -> preprocess -> infer -> track -> annotate -> sink

Adaptive frame skipping
-----------------------
This is the centrepiece, so it is worth being blunt about the arithmetic.

Take a Jetson Nano running a 640x640 detector at roughly 100 ms per frame,
and a camera producing a frame every 33 ms. If you detect on every frame you
get 10 FPS of output, and — much worse — the frames you are detecting on are
already 100+ ms old by the time you have an answer, with latency growing
until the capture buffer saturates.

Now detect on every 4th frame instead, and *track* on the other three. The
tracker's constant-velocity model moves each box forward using the velocity
it has already estimated. Cost per non-detection frame is microseconds. The
output is a smooth 30 FPS stream of boxes, the newest detection is never more
than ~130 ms old, and the boxes on skipped frames are not stale — they are
extrapolated, which for anything moving smoothly (people, vehicles, boats) is
a very good approximation over 30-100 ms.

What you give up is honest: an object that *appears* between detection frames
is not seen until the next detection, so worst-case detection latency is
``interval x frame_period``. For a target moving erratically at short range,
that matters and you should lower the interval or the input resolution
instead. For everything else it is the single highest-leverage change you can
make, and it costs nothing but this class.

``AdaptiveFrameSkipper`` picks the interval at runtime from measured timings
rather than a hard-coded constant, because inference time is not constant: it
changes with thermal state, with what else is running on the GPU, and with
resolution. A fixed ``--skip 3`` tuned on a cold board is wrong ten minutes
into a flight.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .capture import FrameGrabber
from .profiling import Profiler, RollingRate
from .sinks import Sink, SinkGroup
from .tracker import SortTracker
from .types import Detection, Track

try:  # pragma: no cover - environment dependent
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

__all__ = [
    "AdaptiveFrameSkipper",
    "ThermalMonitor",
    "ThroughputWatchdog",
    "FrameResult",
    "Pipeline",
    "annotate",
    "track_color",
]


# ---------------------------------------------------------------------------
# Adaptive frame skipping
# ---------------------------------------------------------------------------


class AdaptiveFrameSkipper:
    """Choose how many frames to skip between detections, from measured time.

    The model
    ---------
    With a detection interval of ``N``, the average cost of one *output* frame
    is::

        cost(N) = overhead + inference / N

    where ``overhead`` is everything you pay on every frame (tracking,
    annotation, encoding, sinks) and ``inference`` is the detector cost you
    pay once per ``N`` frames. To sustain ``target_fps`` you need
    ``cost(N) <= 1 / target_fps``, which rearranges to::

        N >= inference / (1/target_fps - overhead)

    Take the ceiling and clamp. If ``overhead`` alone already exceeds the
    budget, no amount of skipping helps — the fix is a smaller frame, a
    cheaper overlay, or fewer sinks — and :attr:`budget_exceeded` says so
    instead of silently pinning to ``max_interval``.

    Measurements are smoothed with an EMA so one slow frame (a page fault, a
    log flush, another process waking up) does not make the interval jump. The
    EMA is what makes this converge instead of oscillate.

    Example
    -------
    >>> skipper = AdaptiveFrameSkipper(target_fps=25.0)
    >>> for _ in range(40):
    ...     _ = skipper.update(inference_s=0.100, overhead_s=0.005)
    >>> skipper.interval
    3
    """

    def __init__(
        self,
        target_fps: float = 30.0,
        min_interval: int = 1,
        max_interval: int = 12,
        alpha: float = 0.2,
        fixed_interval: int = 0,
        warmup_updates: int = 3,
    ) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if min_interval < 1:
            raise ValueError("min_interval must be >= 1")
        if max_interval < min_interval:
            raise ValueError("max_interval must be >= min_interval")

        self.target_fps = float(target_fps)
        self.min_interval = int(min_interval)
        self.max_interval = int(max_interval)
        self.alpha = float(alpha)
        self.fixed_interval = int(fixed_interval)
        self.warmup_updates = int(warmup_updates)

        self.interval = max(self.min_interval, int(fixed_interval) or self.min_interval)
        self.inference_ema: Optional[float] = None
        self.overhead_ema: Optional[float] = None
        self.updates = 0
        self.budget_exceeded = False

    # -- state -------------------------------------------------------------

    @property
    def is_fixed(self) -> bool:
        return self.fixed_interval > 0

    @property
    def budget_s(self) -> float:
        """Time available per output frame to hit the target rate."""
        return 1.0 / self.target_fps

    @property
    def projected_fps(self) -> float:
        """Output FPS predicted by the cost model at the current interval."""
        cost = self.projected_frame_cost(self.interval)
        return 1.0 / cost if cost > 0 else 0.0

    def projected_frame_cost(self, interval: int) -> float:
        inference = self.inference_ema or 0.0
        overhead = self.overhead_ema or 0.0
        return overhead + inference / max(1, int(interval))

    def reset(self) -> None:
        self.inference_ema = None
        self.overhead_ema = None
        self.updates = 0
        self.budget_exceeded = False
        self.interval = max(self.min_interval, self.fixed_interval or self.min_interval)

    # -- update ------------------------------------------------------------

    def _ema(self, previous: Optional[float], sample: float) -> float:
        if previous is None:
            return float(sample)
        return (1.0 - self.alpha) * previous + self.alpha * float(sample)

    def update(
        self,
        inference_s: Optional[float] = None,
        overhead_s: Optional[float] = None,
    ) -> int:
        """Feed one frame's measurements and return the interval to use next.

        ``inference_s`` should be ``None`` on frames where the detector did
        not run, so a skipped frame does not pull the estimate toward zero.
        """
        if inference_s is not None and inference_s > 0:
            self.inference_ema = self._ema(self.inference_ema, inference_s)
        if overhead_s is not None and overhead_s >= 0:
            self.overhead_ema = self._ema(self.overhead_ema, overhead_s)
        self.updates += 1

        if self.is_fixed:
            self.interval = max(self.min_interval, min(self.max_interval, self.fixed_interval))
            return self.interval
        if self.inference_ema is None or self.updates < self.warmup_updates:
            # Not enough evidence yet. Guessing early makes the first second of
            # a run oscillate, which is exactly when a user is watching.
            return self.interval

        headroom = self.budget_s - (self.overhead_ema or 0.0)
        if headroom <= 0:
            self.budget_exceeded = True
            self.interval = self.max_interval
            return self.interval

        self.budget_exceeded = False
        required = self.inference_ema / headroom
        # Ceiling, with a small epsilon so floating point does not push an
        # exactly-affordable interval up by one.
        self.interval = int(
            max(self.min_interval, min(self.max_interval, math.ceil(required - 1e-9)))
        )
        return self.interval

    def should_detect(self, frame_index: int) -> bool:
        """True if the detector should run on this frame index."""
        return frame_index % max(1, self.interval) == 0

    def explain(self) -> str:
        """One line for the log or the HUD."""
        if self.inference_ema is None:
            return f"skip=1 (no inference timing yet, target {self.target_fps:.0f} fps)"
        detail = (
            f"skip={self.interval} "
            f"(infer {self.inference_ema * 1e3:.1f} ms, "
            f"overhead {(self.overhead_ema or 0.0) * 1e3:.1f} ms, "
            f"target {self.target_fps:.0f} fps, "
            f"projected {self.projected_fps:.1f} fps)"
        )
        if self.budget_exceeded:
            detail += " [per-frame overhead alone exceeds the budget]"
        return detail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interval": self.interval,
            "target_fps": self.target_fps,
            "inference_ms": None if self.inference_ema is None else round(self.inference_ema * 1e3, 2),
            "overhead_ms": None if self.overhead_ema is None else round(self.overhead_ema * 1e3, 2),
            "projected_fps": round(self.projected_fps, 2),
            "budget_exceeded": self.budget_exceeded,
            "fixed": self.is_fixed,
        }


# ---------------------------------------------------------------------------
# Thermal / throughput awareness
# ---------------------------------------------------------------------------


class ThermalMonitor:
    """Read SoC temperatures from sysfs, where available.

    Both Jetson and Raspberry Pi expose thermal zones at
    ``/sys/class/thermal/thermal_zone*/temp`` in millidegrees Celsius. That is
    the portable path, so it is the one used here; ``tegrastats`` and
    ``vcgencmd`` give more detail but are platform-specific external
    processes, and shelling out 30 times a second is not free.

    This is advisory. It does not throttle anything itself — the kernel and
    the firmware already do that. Its job is to make the *cause* visible, so
    that "FPS collapsed after 30 seconds" gets diagnosed as a heatsink problem
    in one minute instead of being blamed on the model for a day.
    """

    SYSFS_ROOT = "/sys/class/thermal"

    def __init__(self, warn_celsius: float = 80.0, root: Optional[str] = None) -> None:
        self.warn_celsius = float(warn_celsius)
        self.root = root or self.SYSFS_ROOT
        self._warned = False

    def available(self) -> bool:
        return os.path.isdir(self.root)

    def read(self) -> Dict[str, float]:
        """Zone name -> degrees Celsius. Empty dict if sysfs is unavailable."""
        readings: Dict[str, float] = {}
        if not self.available():
            return readings
        try:
            zones = sorted(z for z in os.listdir(self.root) if z.startswith("thermal_zone"))
        except OSError:  # pragma: no cover - permissions
            return readings
        for zone in zones:
            try:
                with open(os.path.join(self.root, zone, "temp"), "r") as handle:
                    millidegrees = float(handle.read().strip())
                name = zone
                type_path = os.path.join(self.root, zone, "type")
                if os.path.exists(type_path):
                    with open(type_path, "r") as handle:
                        name = handle.read().strip() or zone
                readings[name] = millidegrees / 1000.0
            except (OSError, ValueError):  # pragma: no cover - flaky sysfs
                continue
        return readings

    def hottest(self) -> Optional[Tuple[str, float]]:
        readings = self.read()
        if not readings:
            return None
        name = max(readings, key=lambda k: readings[k])
        return name, readings[name]

    def check(self) -> Optional[str]:
        """Return a warning string once when a zone crosses the threshold."""
        hottest = self.hottest()
        if hottest is None:
            return None
        name, celsius = hottest
        if celsius >= self.warn_celsius and not self._warned:
            self._warned = True
            return (
                f"thermal zone {name} at {celsius:.1f} C (>= {self.warn_celsius:.0f} C). "
                "Expect clocks to drop and FPS with them. Check airflow, then "
                "'sudo tegrastats' on Jetson or 'vcgencmd get_throttled' on a Pi."
            )
        if celsius < self.warn_celsius - 5.0:
            self._warned = False
        return None


class ThroughputWatchdog:
    """Detect a sustained frame-rate collapse relative to an early baseline.

    A steady decline over tens of seconds with no change in scene complexity
    is the signature of thermal throttling. A sudden step is more likely
    another process, a camera exposure change, or a network stall on RTSP.
    Reporting *when* it happened and by how much narrows it down immediately.
    """

    def __init__(
        self,
        baseline_frames: int = 60,
        drop_ratio: float = 0.75,
        patience: int = 30,
    ) -> None:
        self.baseline_frames = int(baseline_frames)
        self.drop_ratio = float(drop_ratio)
        self.patience = int(patience)
        self.baseline_fps: Optional[float] = None
        self._samples: List[float] = []
        self._below = 0
        self.alerts = 0

    def update(self, fps: float) -> Optional[str]:
        """Feed the current FPS; returns a message when a collapse is confirmed."""
        if fps <= 0:
            return None
        if self.baseline_fps is None:
            self._samples.append(float(fps))
            if len(self._samples) >= self.baseline_frames:
                # Median, not mean: the first frames include warmup outliers.
                ordered = sorted(self._samples)
                self.baseline_fps = ordered[len(ordered) // 2]
                LOGGER.info("throughput baseline established: %.1f fps", self.baseline_fps)
            return None

        if fps >= self.baseline_fps * self.drop_ratio:
            self._below = 0
            return None

        self._below += 1
        if self._below < self.patience:
            return None
        self._below = 0
        self.alerts += 1
        return (
            f"throughput dropped to {fps:.1f} fps from a baseline of "
            f"{self.baseline_fps:.1f} fps and stayed there. Most likely thermal "
            "throttling. Check temperature and clock state before touching the model."
        )

    def reset(self) -> None:
        self.baseline_fps = None
        self._samples = []
        self._below = 0


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (56, 176, 0), (255, 179, 0), (0, 143, 255), (214, 0, 120),
    (0, 209, 178), (255, 87, 34), (124, 77, 255), (0, 188, 212),
    (205, 220, 57), (233, 30, 99), (63, 81, 181), (255, 152, 0),
)


def track_color(track_id: int) -> Tuple[int, int, int]:
    """Stable BGR colour for a track ID, so an object keeps its colour."""
    return _PALETTE[int(track_id) % len(_PALETTE)]


def _draw_rect(
    image: np.ndarray,
    box: Sequence[float],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a hollow rectangle with numpy slicing (no cv2 needed)."""
    height, width = image.shape[:2]
    x1 = int(np.clip(round(box[0]), 0, width - 1))
    y1 = int(np.clip(round(box[1]), 0, height - 1))
    x2 = int(np.clip(round(box[2]), 0, width - 1))
    y2 = int(np.clip(round(box[3]), 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        return
    t = max(1, int(thickness))
    image[y1 : min(y1 + t, height), x1 : x2 + 1] = color
    image[max(y2 - t + 1, 0) : y2 + 1, x1 : x2 + 1] = color
    image[y1 : y2 + 1, x1 : min(x1 + t, width)] = color
    image[y1 : y2 + 1, max(x2 - t + 1, 0) : x2 + 1] = color


def _draw_label(
    image: np.ndarray,
    text: str,
    x: float,
    y: float,
    color: Tuple[int, int, int],
    scale: float = 0.5,
) -> None:
    """Draw a label. Falls back to a coloured tab when cv2 is unavailable."""
    height, width = image.shape[:2]
    x = int(np.clip(x, 0, width - 1))
    y = int(np.clip(y, 0, height - 1))
    if CV2_AVAILABLE:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
        top = max(0, y - th - baseline - 2)
        cv2.rectangle(image, (x, top), (min(x + tw + 4, width - 1), y), color, -1)
        cv2.putText(
            image, text, (x + 2, y - baseline - 1), font, scale, (16, 16, 16), 1,
            cv2.LINE_AA,
        )
        return
    tab_w = min(width - x, 8 * max(1, len(text)))
    top = max(0, y - 14)
    image[top:y, x : x + tab_w] = color


def annotate(
    frame: np.ndarray,
    tracks: Sequence[Track] = (),
    detections: Sequence[Detection] = (),
    overlay: Sequence[str] = (),
    copy: bool = True,
    thickness: int = 2,
    show_detections: bool = False,
) -> np.ndarray:
    """Draw tracks (and optionally raw detections) onto a frame.

    Drawing is not free. On a Pi 4 at 1080p, a dozen boxes with anti-aliased
    text can cost several milliseconds — enough to change your skip interval.
    That is why the pipeline times this stage separately and why ``copy`` can
    be turned off when the caller owns the buffer.
    """
    canvas = frame.copy() if copy else frame

    if show_detections:
        for det in detections:
            _draw_rect(canvas, det.as_xyxy(), (128, 128, 128), max(1, thickness - 1))

    for track in tracks:
        color = track_color(track.track_id)
        _draw_rect(canvas, track.as_xyxy(), color, thickness)
        name = track.class_name or str(track.class_id)
        suffix = "" if track.time_since_update == 0 else f" ~{track.time_since_update}"
        _draw_label(
            canvas, f"#{track.track_id} {name} {track.score:.2f}{suffix}",
            track.x1, track.y1, color,
        )

    for row, text in enumerate(overlay):
        _draw_label(canvas, text, 6, 20 + row * 20, (32, 32, 32))
    return canvas


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class FrameResult:
    """Everything one pipeline iteration produced."""

    index: int
    frame: Optional[np.ndarray]
    annotated: Optional[np.ndarray]
    detections: List[Detection]
    tracks: List[Track]
    ran_inference: bool
    skip_interval: int
    timings: Dict[str, float] = field(default_factory=dict)
    fps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.index,
            "ran_inference": self.ran_inference,
            "skip_interval": self.skip_interval,
            "fps": round(self.fps, 2),
            "tracks": [t.to_dict() for t in self.tracks],
            "timings_ms": {k: round(v * 1e3, 3) for k, v in self.timings.items()},
        }


class Pipeline:
    """capture -> infer -> track -> annotate -> sink, with adaptive skipping.

    Example
    -------
    >>> import numpy as np
    >>> from edgevision.backends import MockBackend
    >>> pipeline = Pipeline(MockBackend(), target_fps=30, annotate_frames=False)
    >>> result = pipeline.process(np.zeros((240, 320, 3), dtype=np.uint8))
    >>> result.ran_inference
    True

    Parameters
    ----------
    detector:
        Anything with ``infer(frame) -> list[Detection]``, including
        :class:`~edgevision.roi.TiledInference`.
    tracker:
        Defaults to :class:`~edgevision.tracker.SortTracker`. Pass ``None``
        explicitly via ``track=False`` to disable tracking entirely (then
        skipping is disabled too, because there is nothing to interpolate
        with).
    target_fps:
        The output rate the adaptive skipper aims at.
    skip:
        Fixed detection interval. ``0`` (default) means adaptive.
    """

    def __init__(
        self,
        detector: Any,
        tracker: Optional[Any] = None,
        sinks: Optional[Iterable[Sink]] = None,
        target_fps: float = 30.0,
        skip: int = 0,
        min_interval: int = 1,
        max_interval: int = 12,
        annotate_frames: bool = True,
        show_detections: bool = False,
        profiler: Optional[Profiler] = None,
        thermal: bool = True,
        thermal_check_every: int = 150,
        track: bool = True,
        hud: bool = True,
    ) -> None:
        self.detector = detector
        self.tracker = tracker if tracker is not None else (SortTracker() if track else None)
        self.sinks = SinkGroup(sinks or [])
        self.skipper = AdaptiveFrameSkipper(
            target_fps=target_fps,
            min_interval=min_interval,
            max_interval=max_interval,
            fixed_interval=skip,
        )
        self.annotate_frames = bool(annotate_frames)
        self.show_detections = bool(show_detections)
        self.profiler = profiler or Profiler()
        self.rate = RollingRate(window_s=2.0)
        self.hud = bool(hud)

        self.thermal = ThermalMonitor() if thermal else None
        self.thermal_check_every = max(1, int(thermal_check_every))
        self.watchdog = ThroughputWatchdog()

        self.frame_index = 0
        self.inference_count = 0
        self.last_detections: List[Detection] = []
        self.last_tracks: List[Track] = []
        self.warnings: List[str] = []

    # -- single frame ------------------------------------------------------

    def process(self, frame: np.ndarray) -> FrameResult:
        """Run one frame through the pipeline. Safe to call in a test loop."""
        started = time.perf_counter()
        timings: Dict[str, float] = {}
        index = self.frame_index
        ran_inference = self.skipper.should_detect(index)

        detections: List[Detection] = []
        inference_s: Optional[float] = None
        if ran_inference:
            t0 = time.perf_counter()
            detections = list(self.detector.infer(frame))
            inference_s = time.perf_counter() - t0
            timings["inference"] = inference_s
            self.profiler.record("inference", inference_s)
            self.inference_count += 1
            self.last_detections = detections

        t0 = time.perf_counter()
        if self.tracker is None:
            tracks = []
        elif ran_inference:
            tracks = self.tracker.update(detections)
        else:
            # The load-bearing line: propagate boxes with the motion model
            # instead of re-running the detector.
            tracks = self.tracker.predict()
        track_s = time.perf_counter() - t0
        timings["track"] = track_s
        self.profiler.record("track", track_s)
        self.last_tracks = tracks

        annotated: Optional[np.ndarray] = None
        draw_s = 0.0
        if self.annotate_frames:
            t0 = time.perf_counter()
            annotated = annotate(
                frame,
                tracks,
                detections,
                overlay=self._hud_lines() if self.hud else (),
                show_detections=self.show_detections,
            )
            draw_s = time.perf_counter() - t0
            timings["draw"] = draw_s
            self.profiler.record("draw", draw_s)

        t0 = time.perf_counter()
        meta = {
            "frame": index,
            "ran_inference": ran_inference,
            "skip_interval": self.skipper.interval,
        }
        self.sinks.write(annotated if annotated is not None else frame, tracks, detections, meta)
        sink_s = time.perf_counter() - t0
        timings["sink"] = sink_s
        self.profiler.record("sink", sink_s)

        total_s = time.perf_counter() - started
        timings["total"] = total_s
        self.profiler.record("total", total_s)
        self.profiler.tick()
        fps = self.rate.tick()

        # Overhead is everything paid on *every* frame; inference is the part
        # amortised across the skip interval.
        self.skipper.update(inference_s=inference_s, overhead_s=track_s + draw_s + sink_s)

        self.frame_index += 1
        self._periodic_checks(fps)

        return FrameResult(
            index=index,
            frame=frame,
            annotated=annotated,
            detections=detections,
            tracks=tracks,
            ran_inference=ran_inference,
            skip_interval=self.skipper.interval,
            timings=timings,
            fps=fps,
        )

    # -- run loop ----------------------------------------------------------

    def run(
        self,
        source: Any,
        max_frames: int = 0,
        max_seconds: float = 0.0,
        on_frame: Optional[Callable[[FrameResult], bool]] = None,
        read_timeout: float = 2.0,
    ) -> Profiler:
        """Drive the pipeline from a source until it ends or a limit is hit.

        ``source`` may be a camera index, GStreamer pipeline, RTSP URL, file
        path, an existing :class:`~edgevision.capture.FrameGrabber`, or any
        iterable of frames (used by tests and offline runs).

        ``on_frame`` is called with each :class:`FrameResult`; returning
        ``False`` stops the loop.
        """
        if hasattr(source, "read_with_meta"):
            return self._run_grabber(source, max_frames, max_seconds, on_frame, read_timeout)
        if isinstance(source, (str, int)) or hasattr(source, "__fspath__"):
            grabber = FrameGrabber(source)
            grabber.start()
            try:
                return self._run_grabber(
                    grabber, max_frames, max_seconds, on_frame, read_timeout
                )
            finally:
                grabber.stop()
        return self._run_iterable(source, max_frames, max_seconds, on_frame)

    def _run_iterable(self, frames, max_frames, max_seconds, on_frame) -> Profiler:
        deadline = time.monotonic() + max_seconds if max_seconds else None
        processed = 0
        for frame in frames:
            result = self.process(frame)
            processed += 1
            if on_frame is not None and on_frame(result) is False:
                break
            if max_frames and processed >= max_frames:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
        return self.profiler

    def _run_grabber(self, grabber, max_frames, max_seconds, on_frame, read_timeout) -> Profiler:
        deadline = time.monotonic() + max_seconds if max_seconds else None
        processed = 0
        while True:
            t0 = time.perf_counter()
            item = grabber.read_with_meta(timeout=read_timeout)
            self.profiler.record("capture", time.perf_counter() - t0)
            if item is None:
                if grabber.is_finished:
                    break
                LOGGER.warning("no frame within %.1fs; source may have stalled", read_timeout)
                continue

            result = self.process(item[0])
            processed += 1
            if on_frame is not None and on_frame(result) is False:
                break
            if max_frames and processed >= max_frames:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
        return self.profiler

    # -- reporting ---------------------------------------------------------

    def _hud_lines(self) -> List[str]:
        return [
            f"{self.rate.value:5.1f} fps  skip={self.skipper.interval}",
            f"tracks={len(self.last_tracks)}  det={len(self.last_detections)}",
        ]

    def _periodic_checks(self, fps: float) -> None:
        message = self.watchdog.update(fps)
        if message:
            LOGGER.warning(message)
            self.warnings.append(message)
        if self.thermal is not None and self.frame_index % self.thermal_check_every == 0:
            warning = self.thermal.check()
            if warning:
                LOGGER.warning(warning)
                self.warnings.append(warning)

    @property
    def inference_ratio(self) -> float:
        """Fraction of frames the detector actually ran on."""
        if self.frame_index == 0:
            return 0.0
        return self.inference_count / self.frame_index

    def stats(self) -> Dict[str, Any]:
        payload = self.profiler.to_dict()
        payload.update(
            {
                "detector": getattr(self.detector, "name", type(self.detector).__name__),
                "inference_frames": self.inference_count,
                "inference_ratio": round(self.inference_ratio, 4),
                "skipper": self.skipper.to_dict(),
                "warnings": list(self.warnings),
            }
        )
        if self.thermal is not None:
            hottest = self.thermal.hottest()
            if hottest:
                payload["thermal"] = {"zone": hottest[0], "celsius": round(hottest[1], 1)}
        return payload

    def format_report(self) -> str:
        lines = [self.profiler.format_table(), "", self.skipper.explain()]
        lines.append(
            f"detector ran on {self.inference_count}/{self.frame_index} frames "
            f"({self.inference_ratio * 100:.0f}%)"
        )
        if self.thermal is not None:
            hottest = self.thermal.hottest()
            if hottest:
                lines.append(f"hottest thermal zone: {hottest[0]} {hottest[1]:.1f} C")
        for warning in self.warnings:
            lines.append(f"WARNING: {warning}")
        return "\n".join(lines)

    def close(self) -> None:
        self.sinks.close()
        close = getattr(self.detector, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
