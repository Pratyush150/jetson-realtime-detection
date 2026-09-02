"""Threaded capture with a one-deep latest-frame buffer.

The problem this solves
-----------------------
``cv2.VideoCapture.read()`` is a *blocking pull from a driver-side queue*. If
your inference loop takes 120 ms and the camera pushes a frame every 33 ms,
the driver buffers the frames you did not consume. Each ``read()`` then hands
you the *oldest* queued frame, not the newest one. Latency does not stabilise
— it grows without bound until the buffer caps out. On a Jetson with a USB
camera and a 4-deep V4L2 buffer plus an RTSP jitter buffer, "my detections lag
about three seconds behind reality" is exactly this bug, and no amount of
model optimisation fixes it. The pipeline is keeping up on *throughput* and
failing catastrophically on *latency*.

The fix is to decouple the reader from the consumer and make the shared slot
hold exactly one frame:

* A dedicated thread calls ``read()`` in a tight loop, so the driver queue is
  always drained and never backs up.
* That thread writes into a **1-deep slot**. If the consumer has not taken the
  previous frame yet, the new frame *overwrites* it and a drop counter is
  incremented.

The tradeoff, stated honestly: you throw frames away. If your job is to record
every frame (offline analysis, dataset capture) this is the wrong structure and
you want a real queue plus back-pressure. If your job is real-time — track a
target, close a control loop, trigger an event — the newest frame is the only
one with any value, and an old frame is worse than no frame. This module is
built for the second case, and the drop counters make the cost visible instead
of hiding it.

For a *video file* source that logic is inverted: dropping frames changes the
result, and there is no "stale" frame because the file is not live. So file
sources default to ``drop_stale=False`` and the reader blocks until the
consumer catches up.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Tuple

import numpy as np

try:  # pragma: no cover - environment dependent
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

__all__ = [
    "SourceType",
    "SourceSpec",
    "parse_source",
    "LatestFrameBuffer",
    "CaptureStats",
    "FrameGrabber",
    "csi_pipeline",
    "rtsp_pipeline",
]


class SourceType(str, Enum):
    """How a source string should be opened."""

    DEVICE = "device"
    GSTREAMER = "gstreamer"
    RTSP = "rtsp"
    FILE = "file"


@dataclass(frozen=True)
class SourceSpec:
    """Parsed capture source."""

    type: SourceType
    value: Any
    raw: str

    @property
    def is_live(self) -> bool:
        """Live sources produce frames on their own clock and can go stale."""
        return self.type is not SourceType.FILE


_GST_HINTS = ("appsink", "nvarguscamerasrc", "v4l2src", "rtspsrc", "filesrc", " ! ")
_URL_RE = re.compile(r"^(rtsp|rtmp|http|https|udp|tcp)://", re.IGNORECASE)


def parse_source(source: Any) -> SourceSpec:
    """Classify a ``--source`` argument.

    ``0``/``"0"``          -> USB camera index (V4L2)
    ``"...! appsink"``     -> GStreamer pipeline string (CSI via nvarguscamerasrc)
    ``"rtsp://..."``       -> network stream, gets reconnect handling
    anything else          -> file path
    """
    if isinstance(source, int):
        return SourceSpec(SourceType.DEVICE, int(source), str(source))

    text = str(source).strip()
    if text.isdigit():
        return SourceSpec(SourceType.DEVICE, int(text), text)
    if any(hint in text for hint in _GST_HINTS):
        return SourceSpec(SourceType.GSTREAMER, text, text)
    if _URL_RE.match(text):
        return SourceSpec(SourceType.RTSP, text, text)
    return SourceSpec(SourceType.FILE, text, text)


def csi_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1920,
    capture_height: int = 1080,
    display_width: int = 1280,
    display_height: int = 720,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    """Build an ``nvarguscamerasrc`` pipeline for a Jetson CSI camera.

    The scaling and the BGR conversion are done by ``nvvidconv`` on the VIC,
    not by the CPU. Doing the same resize with ``cv2.resize`` after the fact
    costs a full-frame CPU copy per frame, which on a Nano is several
    milliseconds you cannot spare.
    """
    return (
        f"nvarguscamerasrc sensor-id={int(sensor_id)} ! "
        f"video/x-raw(memory:NVMM), width=(int){int(capture_width)}, "
        f"height=(int){int(capture_height)}, framerate=(fraction){int(framerate)}/1 ! "
        f"nvvidconv flip-method={int(flip_method)} ! "
        f"video/x-raw, width=(int){int(display_width)}, "
        f"height=(int){int(display_height)}, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def rtsp_pipeline(url: str, latency_ms: int = 0, hardware_decode: bool = True) -> str:
    """Build a low-latency RTSP pipeline.

    ``latency=0`` on ``rtspsrc`` disables the jitter buffer. That trades a
    little robustness on a lossy link for a large latency win, which is the
    right trade for control loops. ``nvv4l2decoder`` keeps H.264 decode off
    the CPU on Jetson; without it a 1080p30 stream can eat an entire core.
    """
    decoder = "nvv4l2decoder" if hardware_decode else "avdec_h264"
    convert = "nvvidconv" if hardware_decode else "videoconvert"
    return (
        f"rtspsrc location={url} latency={int(latency_ms)} ! "
        f"rtph264depay ! h264parse ! {decoder} ! {convert} ! "
        "video/x-raw, format=(string)BGRx ! videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


class LatestFrameBuffer:
    """A thread-safe slot that holds at most one frame.

    ``put`` never blocks and never grows. If the slot is occupied, the old
    frame is discarded and :attr:`dropped` is incremented — that counter is
    the honest measure of how far behind the consumer is running.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._item: Optional[Tuple[np.ndarray, float, int]] = None
        self._closed = False
        self.received = 0
        self.dropped = 0
        self.consumed = 0

    def put(self, frame: np.ndarray, timestamp: Optional[float] = None) -> bool:
        """Store ``frame`` as the latest. Returns ``True`` if a frame was dropped."""
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        with self._cond:
            dropped = self._item is not None
            if dropped:
                self.dropped += 1
            self.received += 1
            self._item = (frame, stamp, self.received)
            self._cond.notify()
        return dropped

    def get(self, timeout: Optional[float] = None) -> Optional[Tuple[np.ndarray, float, int]]:
        """Take the latest frame, blocking up to ``timeout`` seconds.

        Returns ``(frame, capture_timestamp, sequence_number)`` or ``None``.
        The sequence number is the *source* frame index, so a consumer can see
        gaps: jumping from 100 to 104 means four frames were dropped.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._item is None and not self._closed:
                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(remaining)
            if self._item is None:
                return None
            item = self._item
            self._item = None
            self.consumed += 1
            return item

    def peek(self) -> Optional[Tuple[np.ndarray, float, int]]:
        """Read the latest frame without consuming it."""
        with self._lock:
            return self._item

    def clear(self) -> None:
        with self._lock:
            self._item = None

    def close(self) -> None:
        """Wake up any blocked consumer so a shutdown cannot deadlock."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def drop_rate(self) -> float:
        if self.received == 0:
            return 0.0
        return self.dropped / self.received


@dataclass
class CaptureStats:
    """Counters describing how the capture thread is behaving."""

    frames_read: int = 0
    frames_dropped: int = 0
    frames_delivered: int = 0
    read_failures: int = 0
    reconnects: int = 0
    started_at: float = field(default_factory=time.monotonic)
    last_frame_at: float = 0.0

    @property
    def elapsed(self) -> float:
        return max(1e-9, time.monotonic() - self.started_at)

    @property
    def source_fps(self) -> float:
        """Frames the *camera* produced per second."""
        return self.frames_read / self.elapsed

    @property
    def delivered_fps(self) -> float:
        """Frames the *consumer* actually processed per second."""
        return self.frames_delivered / self.elapsed

    @property
    def drop_rate(self) -> float:
        if self.frames_read == 0:
            return 0.0
        return self.frames_dropped / self.frames_read

    def to_dict(self) -> dict:
        return {
            "frames_read": self.frames_read,
            "frames_dropped": self.frames_dropped,
            "frames_delivered": self.frames_delivered,
            "read_failures": self.read_failures,
            "reconnects": self.reconnects,
            "source_fps": round(self.source_fps, 2),
            "delivered_fps": round(self.delivered_fps, 2),
            "drop_rate": round(self.drop_rate, 4),
        }


def _default_capture_factory(spec: SourceSpec, width: int, height: int, fps: int) -> Any:
    """Open a source with OpenCV, choosing the right backend for its type."""
    if not CV2_AVAILABLE:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "OpenCV is required to open a real capture source. "
            "Install opencv-python, or pass capture_factory= for testing."
        )
    if spec.type is SourceType.GSTREAMER:
        cap = cv2.VideoCapture(spec.value, cv2.CAP_GSTREAMER)
    elif spec.type is SourceType.DEVICE:
        cap = cv2.VideoCapture(spec.value, cv2.CAP_V4L2)
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        # MJPG keeps the USB bus from becoming the bottleneck at 1080p; raw
        # YUYV at 1080p30 needs ~62 MB/s and simply will not enumerate on many
        # USB 2.0 hubs, so the camera silently falls back to 5 fps.
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:  # pragma: no cover - some builds lack the property
            LOGGER.debug("could not request MJPG fourcc")
    else:
        cap = cv2.VideoCapture(spec.value)

    # Ask the driver for the shallowest buffer it will give us. This is a hint,
    # not a guarantee (many V4L2 and FFmpeg backends ignore it) which is
    # precisely why the reader thread below exists as well.
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:  # pragma: no cover
        pass
    return cap


class FrameGrabber:
    """Threaded frame reader with a latest-frame slot and reconnect.

    Example
    -------
    >>> grabber = FrameGrabber(0)            # doctest: +SKIP
    >>> grabber.start()                      # doctest: +SKIP
    >>> frame = grabber.read(timeout=1.0)    # doctest: +SKIP
    >>> grabber.stop()                       # doctest: +SKIP

    Parameters
    ----------
    source:
        Camera index, GStreamer pipeline string, RTSP URL or file path.
    capture_factory:
        Callable ``(SourceSpec, width, height, fps) -> capture`` where the
        capture object provides ``read() -> (ok, frame)``, ``isOpened()`` and
        ``release()``. Defaults to OpenCV. Injecting it is how the tests run
        the full threading and reconnect logic with no camera and no cv2.
    drop_stale:
        Overwrite un-consumed frames (default for live sources). ``None``
        picks the right behaviour from the source type.
    reconnect:
        Re-open the source after read failures. Cameras on a vehicle *do*
        drop off the bus — a brown-out on the 5 V rail re-enumerates the USB
        device and every subsequent ``read()`` returns ``False`` forever
        unless something re-opens it.
    """

    def __init__(
        self,
        source: Any,
        width: int = 0,
        height: int = 0,
        fps: int = 0,
        drop_stale: Optional[bool] = None,
        reconnect: bool = True,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 10.0,
        max_reconnect_attempts: int = 0,
        read_failure_threshold: int = 30,
        capture_factory: Optional[Callable[..., Any]] = None,
        name: str = "edgevision-capture",
    ) -> None:
        self.spec = parse_source(source)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.drop_stale = self.spec.is_live if drop_stale is None else bool(drop_stale)
        self.reconnect = bool(reconnect)
        self.reconnect_delay = float(reconnect_delay)
        self.max_reconnect_delay = float(max_reconnect_delay)
        self.max_reconnect_attempts = int(max_reconnect_attempts)
        self.read_failure_threshold = int(read_failure_threshold)
        self._factory = capture_factory or _default_capture_factory
        self._name = name

        self.buffer = LatestFrameBuffer()
        self.stats = CaptureStats()
        self._capture: Any = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._eof = threading.Event()
        self._slot_free = threading.Event()
        self._slot_free.set()
        self._error: Optional[BaseException] = None

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> Any:
        """Open (or re-open) the underlying capture object."""
        self._capture = self._factory(self.spec, self.width, self.height, self.fps)
        is_open = getattr(self._capture, "isOpened", lambda: True)()
        if not is_open:
            raise RuntimeError(f"failed to open source {self.spec.raw!r}")
        LOGGER.info("opened %s source %r", self.spec.type.value, self.spec.raw)
        return self._capture

    def start(self) -> "FrameGrabber":
        """Open the source and start the reader thread."""
        if self._thread is not None:
            raise RuntimeError("grabber already started")
        self.open()
        self._stop.clear()
        self._eof.clear()
        self.stats = CaptureStats()
        self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the reader thread and release the source. Idempotent."""
        self._stop.set()
        self._slot_free.set()
        self.buffer.close()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._release()

    def _release(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            try:
                capture.release()
            except Exception:  # pragma: no cover - defensive
                LOGGER.debug("capture release raised", exc_info=True)

    def __enter__(self) -> "FrameGrabber":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    # -- consumer API ------------------------------------------------------

    def read(self, timeout: Optional[float] = 1.0) -> Optional[np.ndarray]:
        """Return the newest frame, or ``None`` on timeout / end of stream."""
        item = self.read_with_meta(timeout=timeout)
        return None if item is None else item[0]

    def read_with_meta(
        self, timeout: Optional[float] = 1.0
    ) -> Optional[Tuple[np.ndarray, float, int]]:
        """Like :meth:`read` but also returns capture time and sequence number."""
        item = self.buffer.get(timeout=timeout)
        self._slot_free.set()
        if item is None:
            return None
        self.stats.frames_delivered += 1
        return item

    def __iter__(self):
        while True:
            item = self.read_with_meta(timeout=1.0)
            if item is None:
                if self.is_finished:
                    return
                continue
            yield item[0]

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def is_finished(self) -> bool:
        """True once a file source hit EOF or the thread stopped for good."""
        return self._eof.is_set() or (self._stop.is_set() and not self.is_running)

    @property
    def error(self) -> Optional[BaseException]:
        """The exception that killed the reader thread, if any."""
        return self._error

    def latency(self) -> float:
        """Age of the frame currently sitting in the slot, in seconds."""
        item = self.buffer.peek()
        if item is None:
            return 0.0
        return max(0.0, time.monotonic() - item[1])

    # -- reader thread -----------------------------------------------------

    def _loop(self) -> None:
        consecutive_failures = 0
        attempts = 0
        delay = self.reconnect_delay

        while not self._stop.is_set():
            if self._capture is None:
                try:
                    self.open()
                except Exception as exc:
                    if not self._should_retry(attempts):
                        self._error = exc
                        LOGGER.error("giving up on %r: %s", self.spec.raw, exc)
                        break
                    attempts += 1
                    self.stats.reconnects += 1
                    LOGGER.warning(
                        "reopen of %r failed (%s); retrying in %.1fs",
                        self.spec.raw,
                        exc,
                        delay,
                    )
                    if self._stop.wait(delay):
                        break
                    delay = min(delay * 2.0, self.max_reconnect_delay)
                    continue
                attempts = 0
                delay = self.reconnect_delay
                consecutive_failures = 0

            if not self.drop_stale:
                # File playback: block until the consumer has taken the last
                # frame, so nothing is silently skipped.
                if not self._slot_free.wait(timeout=0.1):
                    continue
                self._slot_free.clear()

            try:
                ok, frame = self._capture.read()
            except Exception as exc:  # pragma: no cover - driver level failure
                LOGGER.warning("read() raised: %s", exc)
                ok, frame = False, None

            if not ok or frame is None:
                consecutive_failures += 1
                self.stats.read_failures += 1
                self._slot_free.set()
                if self.spec.type is SourceType.FILE:
                    LOGGER.info("end of file %r", self.spec.raw)
                    self._eof.set()
                    break
                if consecutive_failures >= self.read_failure_threshold:
                    if not self.reconnect or not self._should_retry(attempts):
                        LOGGER.error("source %r is gone; stopping", self.spec.raw)
                        self._eof.set()
                        break
                    LOGGER.warning(
                        "%d consecutive read failures on %r; reconnecting",
                        consecutive_failures,
                        self.spec.raw,
                    )
                    self._release()
                    self.stats.reconnects += 1
                    attempts += 1
                    consecutive_failures = 0
                    if self._stop.wait(delay):
                        break
                    delay = min(delay * 2.0, self.max_reconnect_delay)
                continue

            consecutive_failures = 0
            self.stats.frames_read += 1
            self.stats.last_frame_at = time.monotonic()
            if self.buffer.put(frame):
                self.stats.frames_dropped += 1

        self.buffer.close()

    def _should_retry(self, attempts: int) -> bool:
        if not self.reconnect:
            return False
        if self.max_reconnect_attempts <= 0:
            return True
        return attempts < self.max_reconnect_attempts
