"""Output sinks: video file, event snapshots, JSON-lines log, MJPEG preview.

A headless Jetson bolted to a frame has no display. The three things you
actually need from it are: a recording you can review later, a machine-
readable log you can grep and plot, and a way to *look at it right now* from
a laptop without installing anything.

The MJPEG server covers the third. It is deliberately stdlib-only
(``http.server``): no Flask, no aiohttp, no extra wheel to cross-compile onto
an ARM board. MJPEG is not efficient — every frame is a full JPEG, so a 720p
stream at 15 FPS is a few megabits — but it has one property nothing else
has: it works in any browser with a plain ``<img>`` tag, no WebRTC signalling
and no player plugin. For "is the camera pointed at the right thing", that is
the correct trade. For anything sustained, record to file and copy it off.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .types import Detection, Track

try:  # pragma: no cover - environment dependent
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

__all__ = [
    "Sink",
    "SinkGroup",
    "VideoWriterSink",
    "SnapshotSink",
    "JsonLinesSink",
    "MjpegPreviewServer",
    "encode_jpeg",
    "encode_ppm",
    "multipart_chunk",
    "jpeg_encoder_available",
]

BOUNDARY = "edgevisionframe"


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def jpeg_encoder_available() -> bool:
    """True if some JPEG encoder (OpenCV or Pillow) is importable."""
    if CV2_AVAILABLE:
        return True
    try:  # pragma: no cover - depends on environment
        import PIL.Image  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode a BGR frame as JPEG bytes.

    Quality is a real lever on an edge board: at 720p, quality 95 is roughly
    3x the bytes of quality 75 for a preview nobody is measuring. Default 80.
    """
    if CV2_AVAILABLE:
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not ok:  # pragma: no cover - cv2 failure is not reproducible offline
            raise RuntimeError("cv2.imencode failed")
        return buffer.tobytes()

    try:  # pragma: no cover - depends on environment
        import io

        from PIL import Image  # type: ignore

        rgb = frame[:, :, ::-1] if frame.ndim == 3 else frame
        buffer = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb)).save(
            buffer, format="JPEG", quality=int(quality)
        )
        return buffer.getvalue()
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "JPEG encoding needs OpenCV or Pillow. Install one, or use "
            "encode_ppm() for a dependency-free (much larger) fallback."
        ) from exc


def encode_ppm(frame: np.ndarray) -> bytes:
    """Encode a BGR frame as a binary PPM (P6). No dependencies, no compression.

    Exists so snapshot capture still works on a stripped image with neither
    OpenCV nor Pillow. A 720p PPM is ~2.7 MB, so this is a fallback, not a
    plan; the file it writes gets a ``.ppm`` extension so nothing downstream
    is misled about what it is.
    """
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    rgb = np.ascontiguousarray(array[:, :, ::-1].astype(np.uint8))
    height, width = rgb.shape[:2]
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return header + rgb.tobytes()


def multipart_chunk(payload: bytes, boundary: str = BOUNDARY,
                    content_type: str = "image/jpeg") -> bytes:
    """One ``multipart/x-mixed-replace`` part, framed exactly as browsers expect.

    The details that matter and are easy to get wrong: the boundary line is
    prefixed with ``--``, ``Content-Length`` must be present or some browsers
    buffer forever waiting for the next boundary, and the part ends with a
    bare CRLF before the next ``--boundary``.
    """
    head = (
        f"--{boundary}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(payload)}\r\n\r\n"
    ).encode("ascii")
    return head + payload + b"\r\n"


# ---------------------------------------------------------------------------
# Sink interface
# ---------------------------------------------------------------------------


class Sink(abc.ABC):
    """Consumes annotated frames and/or detection results."""

    @abc.abstractmethod
    def write(
        self,
        frame: Optional[np.ndarray],
        tracks: Sequence[Track] = (),
        detections: Sequence[Detection] = (),
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle one frame's worth of output."""

    def close(self) -> None:
        """Flush and release resources. Must be idempotent."""

    def __enter__(self) -> "Sink":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class SinkGroup(Sink):
    """Fan one frame out to several sinks, isolating failures.

    A sink that raises must not kill the pipeline. If the SD card fills up
    mid-flight, the video writer failing is not a reason to stop detecting;
    the error is logged once per sink and that sink is dropped.
    """

    def __init__(self, sinks: Iterable[Sink] = ()) -> None:
        self.sinks: List[Sink] = list(sinks)
        self.failed: List[Tuple[Sink, str]] = []

    def add(self, sink: Sink) -> "SinkGroup":
        self.sinks.append(sink)
        return self

    def write(self, frame, tracks=(), detections=(), meta=None) -> None:
        survivors: List[Sink] = []
        for sink in self.sinks:
            try:
                sink.write(frame, tracks, detections, meta)
                survivors.append(sink)
            except Exception as exc:
                LOGGER.error("sink %s failed and was dropped: %s", type(sink).__name__, exc)
                self.failed.append((sink, str(exc)))
        self.sinks = survivors

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:  # pragma: no cover - defensive
                LOGGER.debug("sink close raised", exc_info=True)
        self.sinks = []


# ---------------------------------------------------------------------------
# Concrete sinks
# ---------------------------------------------------------------------------


class VideoWriterSink(Sink):
    """Write annotated frames to a video file via OpenCV.

    The writer is opened lazily on the first frame because the frame size is
    not known until then — and guessing it wrong produces a file that plays
    as a green smear, which is a painful thing to discover after a flight.

    ``fps`` is metadata only: it sets playback speed, it does not control
    capture. If you record a stream that actually ran at 12 FPS with
    ``fps=30``, playback is 2.5x fast. Pass the measured rate.
    """

    def __init__(
        self,
        path: str,
        fps: float = 30.0,
        fourcc: str = "mp4v",
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        if not CV2_AVAILABLE:  # pragma: no cover - environment dependent
            raise RuntimeError("VideoWriterSink requires OpenCV (opencv-python)")
        self.path = path
        self.fps = float(fps)
        self.fourcc = fourcc
        self.frame_size = frame_size
        self._writer: Any = None
        self.frames_written = 0

    def _ensure_writer(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            return
        height, width = frame.shape[:2]
        self.frame_size = self.frame_size or (width, height)
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        self._writer = cv2.VideoWriter(
            self.path,
            cv2.VideoWriter_fourcc(*self.fourcc),
            self.fps,
            self.frame_size,
        )
        if not self._writer.isOpened():  # pragma: no cover - codec dependent
            raise RuntimeError(
                f"could not open {self.path} with fourcc {self.fourcc!r}; "
                "on Jetson try 'avc1' or install a GStreamer-backed OpenCV"
            )
        LOGGER.info("recording to %s at %.1f fps", self.path, self.fps)

    def write(self, frame, tracks=(), detections=(), meta=None) -> None:
        if frame is None:
            return
        self._ensure_writer(frame)
        height, width = frame.shape[:2]
        if (width, height) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.release()
            LOGGER.info("wrote %d frames to %s", self.frames_written, self.path)


class SnapshotSink(Sink):
    """Save a still image when an event fires.

    Rate limiting is the whole point. Without ``min_interval_s`` a person
    standing in frame produces 30 files per second and fills the card in
    minutes. The default triggers on any confirmed track and writes at most
    one image per second.
    """

    def __init__(
        self,
        directory: str,
        trigger: Optional[Callable[[Sequence[Track], Sequence[Detection]], bool]] = None,
        min_interval_s: float = 1.0,
        quality: int = 85,
        prefix: str = "event",
        max_files: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.directory = directory
        self.trigger = trigger or (lambda tracks, dets: bool(tracks))
        self.min_interval_s = float(min_interval_s)
        self.quality = int(quality)
        self.prefix = prefix
        self.max_files = int(max_files)
        self._clock = clock
        self._last_write = -float("inf")
        self.written: List[str] = []
        os.makedirs(self.directory, exist_ok=True)

    def write(self, frame, tracks=(), detections=(), meta=None) -> None:
        if frame is None:
            return
        if not self.trigger(tracks, detections):
            return
        now = self._clock()
        if now - self._last_write < self.min_interval_s:
            return
        if self.max_files and len(self.written) >= self.max_files:
            return

        if jpeg_encoder_available():
            payload, suffix = encode_jpeg(frame, self.quality), ".jpg"
        else:  # pragma: no cover - depends on environment
            payload, suffix = encode_ppm(frame), ".ppm"

        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{self.prefix}_{stamp}_{len(self.written):05d}{suffix}"
        path = os.path.join(self.directory, name)
        with open(path, "wb") as handle:
            handle.write(payload)
        self._last_write = now
        self.written.append(path)
        LOGGER.info("snapshot %s (%d tracks)", path, len(tracks))

    def close(self) -> None:
        LOGGER.info("snapshot sink wrote %d files", len(self.written))


class JsonLinesSink(Sink):
    """Append one JSON object per frame to a ``.jsonl`` file.

    JSON Lines rather than one big JSON array so the file is valid and
    readable at every instant, including after a power cut mid-write. You can
    ``tail -f`` it, ``grep`` it, and load it with ``pandas.read_json(...,
    lines=True)`` without ever holding the whole run in memory.

    Frames with nothing in them are skipped by default; on a 30 FPS stream
    that is usually 90% of the file.
    """

    def __init__(
        self,
        path: str,
        include_empty: bool = False,
        flush_every: int = 30,
        include_detections: bool = False,
    ) -> None:
        self.path = path
        self.include_empty = bool(include_empty)
        self.flush_every = int(flush_every)
        self.include_detections = bool(include_detections)
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self._handle = open(path, "a", encoding="utf-8")
        self._since_flush = 0
        self.records_written = 0

    def write(self, frame, tracks=(), detections=(), meta=None) -> None:
        if not tracks and not detections and not self.include_empty:
            return
        record: Dict[str, Any] = {
            "t": round(time.time(), 3),
            "tracks": [t.to_dict() for t in tracks],
        }
        if self.include_detections:
            record["detections"] = [d.to_dict() for d in detections]
        if meta:
            record.update(meta)
        self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.records_written += 1
        self._since_flush += 1
        if self.flush_every and self._since_flush >= self.flush_every:
            self._handle.flush()
            self._since_flush = 0

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None and not handle.closed:
            handle.flush()
            handle.close()


# ---------------------------------------------------------------------------
# MJPEG preview server
# ---------------------------------------------------------------------------


class _FrameBroker:
    """Latest-frame broadcast slot for N HTTP clients.

    Unlike the capture buffer this does *not* consume: every connected client
    should see the newest frame, and a slow client must never hold up the
    pipeline. Clients that fall behind simply skip frames, which is the right
    behaviour for a preview.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._payload: Optional[bytes] = None
        self._version = 0
        self._closed = False

    def publish(self, payload: bytes) -> None:
        with self._condition:
            self._payload = payload
            self._version += 1
            self._condition.notify_all()

    def wait(self, last_version: int, timeout: float = 5.0) -> Tuple[Optional[bytes], int]:
        with self._condition:
            if self._version == last_version and not self._closed:
                self._condition.wait(timeout)
            if self._closed:
                return None, self._version
            return self._payload, self._version

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def version(self) -> int:
        return self._version


_INDEX_PAGE = """<!doctype html>
<title>edgevision preview</title>
<style>
 body{{margin:0;background:#111;color:#ddd;font:14px system-ui,sans-serif}}
 header{{padding:8px 12px;background:#1c1c1c}}
 img{{display:block;max-width:100%;height:auto}}
</style>
<header>edgevision preview - {host}</header>
<img src="/stream.mjpg" alt="live preview">
"""


class _PreviewHandler(BaseHTTPRequestHandler):
    """Serves an index page, an MJPEG stream and a single-shot JPEG."""

    protocol_version = "HTTP/1.1"
    server_version = "edgevision/1.0"
    broker: _FrameBroker = None  # type: ignore[assignment]
    boundary: str = BOUNDARY

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        LOGGER.debug("mjpeg %s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_index()
        elif path in ("/stream.mjpg", "/stream"):
            self._send_stream()
        elif path in ("/snapshot.jpg", "/snapshot"):
            self._send_snapshot()
        elif path == "/healthz":
            self._send_bytes(b"ok\n", "text/plain")
        else:
            self.send_error(404)

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_index(self) -> None:
        page = _INDEX_PAGE.format(host=self.headers.get("Host", "")).encode("utf-8")
        self._send_bytes(page, "text/html; charset=utf-8")

    def _send_snapshot(self) -> None:
        payload, _ = self.broker.wait(-1, timeout=2.0)
        if payload is None:
            self.send_error(503, "no frame yet")
            return
        self._send_bytes(payload, "image/jpeg")

    def _send_stream(self) -> None:
        self.send_response(200)
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={self.boundary}"
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()

        version = -1
        try:
            while True:
                payload, version = self.broker.wait(version, timeout=5.0)
                if payload is None:
                    break
                self.wfile.write(multipart_chunk(payload, self.boundary))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The browser tab was closed. Entirely normal; not an error.
            LOGGER.debug("preview client disconnected")


class MjpegPreviewServer(Sink):
    """Serve the annotated stream over HTTP so a laptop can watch a headless board.

    Example
    -------
    >>> server = MjpegPreviewServer(port=0).start()   # port 0 = pick a free one
    >>> url = server.url
    >>> server.close()

    Encoding happens on the pipeline thread by design: it is a real cost
    (a few ms per 720p frame) and hiding it in a worker would make the
    profiler under-report where time is going. Use ``every_n`` to publish
    only every Nth frame if the preview is not worth that cost.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8090,
        quality: int = 70,
        every_n: int = 1,
        encoder: Optional[Callable[[np.ndarray], bytes]] = None,
        boundary: str = BOUNDARY,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.quality = int(quality)
        self.every_n = max(1, int(every_n))
        self.boundary = boundary
        self._encoder = encoder or (lambda frame: encode_jpeg(frame, self.quality))
        self.broker = _FrameBroker()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._frame_index = 0
        self.frames_published = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "MjpegPreviewServer":
        """Bind and serve in a daemon thread. ``port=0`` picks a free port."""
        if self._server is not None:
            return self

        broker = self.broker
        boundary = self.boundary

        class Handler(_PreviewHandler):
            pass

        Handler.broker = broker
        Handler.boundary = boundary

        ThreadingHTTPServer.allow_reuse_address = True
        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="edgevision-mjpeg",
            daemon=True,
        )
        self._thread.start()
        LOGGER.info("MJPEG preview on %s", self.url)
        return self

    @property
    def url(self) -> str:
        host = self.host
        if host in ("0.0.0.0", ""):
            host = socket.gethostname()
        return f"http://{host}:{self.port}/"

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- sink API ----------------------------------------------------------

    def write(self, frame, tracks=(), detections=(), meta=None) -> None:
        if frame is None:
            return
        self._frame_index += 1
        if self._frame_index % self.every_n:
            return
        self.publish(frame)

    def publish(self, frame: np.ndarray) -> None:
        """Encode and hand a frame to every connected client."""
        self.broker.publish(self._encoder(frame))
        self.frames_published += 1

    def close(self) -> None:
        self.broker.close()
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
