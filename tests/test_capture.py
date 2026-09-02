"""Capture: source parsing, the 1-deep latest-frame buffer, and reconnect.

None of these tests need a camera or OpenCV. ``FrameGrabber`` takes a
``capture_factory``, so the real threading, drop-counting and reconnect logic
runs against a fake device.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from edgevision.capture import (
    CaptureStats,
    FrameGrabber,
    LatestFrameBuffer,
    SourceType,
    csi_pipeline,
    parse_source,
    rtsp_pipeline,
)


def frame(value: int, size=(4, 4)) -> np.ndarray:
    return np.full((*size, 3), value % 256, dtype=np.uint8)


class FakeCapture:
    """Minimal stand-in for ``cv2.VideoCapture``.

    ``fail_after`` makes ``read()`` start returning ``(False, None)``, which is
    exactly what a USB camera does after it re-enumerates on a brown-out.
    """

    def __init__(self, total=None, fail_after=None, delay=0.0, opened=True):
        self.total = total
        self.fail_after = fail_after
        self.delay = delay
        self.opened = opened
        self.reads = 0
        self.released = False

    def isOpened(self):  # noqa: N802 - mirrors the cv2 API
        return self.opened

    def read(self):
        if self.delay:
            time.sleep(self.delay)
        if self.total is not None and self.reads >= self.total:
            return False, None
        if self.fail_after is not None and self.reads >= self.fail_after:
            self.reads += 1
            return False, None
        self.reads += 1
        return True, frame(self.reads)

    def release(self):
        self.released = True


# ---------------------------------------------------------------------------
# Source parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        (0, SourceType.DEVICE),
        ("0", SourceType.DEVICE),
        ("2", SourceType.DEVICE),
        ("rtsp://10.0.0.5:554/live", SourceType.RTSP),
        ("http://cam.local/stream", SourceType.RTSP),
        ("/data/flight.mp4", SourceType.FILE),
        ("clip.avi", SourceType.FILE),
    ],
)
def test_parse_source_classification(source, expected):
    assert parse_source(source).type is expected


def test_gstreamer_pipelines_are_detected_and_live():
    csi = parse_source(csi_pipeline(sensor_id=0))
    rtsp = parse_source(rtsp_pipeline("rtsp://cam/live"))
    assert csi.type is SourceType.GSTREAMER and csi.is_live
    assert rtsp.type is SourceType.GSTREAMER
    assert "nvarguscamerasrc" in csi.value
    # A 1-deep sink is the GStreamer-side half of the same anti-staleness idea.
    assert "drop=true" in csi.value and "max-buffers=1" in csi.value
    assert "latency=0" in rtsp.value


def test_file_sources_are_not_live():
    assert parse_source("clip.mp4").is_live is False
    assert parse_source(0).is_live is True


# ---------------------------------------------------------------------------
# The latest-frame buffer
# ---------------------------------------------------------------------------


def test_buffer_drops_stale_frames_instead_of_queueing_them():
    """The central claim of capture.py, asserted directly."""
    buffer = LatestFrameBuffer()
    for i in range(1, 6):
        buffer.put(frame(i))

    got = buffer.get(timeout=0.1)
    assert got is not None
    payload, _, sequence = got

    assert payload[0, 0, 0] == 5, "consumer must receive the NEWEST frame"
    assert buffer.received == 5
    assert buffer.dropped == 4, "the four older frames must be counted as dropped"
    assert buffer.drop_rate == pytest.approx(0.8)
    assert sequence == 5, "sequence number exposes the gap to the consumer"

    # The slot is empty after a get: no backlog can accumulate.
    assert buffer.get(timeout=0.01) is None


def test_buffer_put_reports_whether_it_overwrote():
    buffer = LatestFrameBuffer()
    assert buffer.put(frame(1)) is False
    assert buffer.put(frame(2)) is True
    buffer.get(timeout=0.01)
    assert buffer.put(frame(3)) is False


def test_buffer_peek_does_not_consume():
    buffer = LatestFrameBuffer()
    buffer.put(frame(9))
    assert buffer.peek()[0][0, 0, 0] == 9
    assert buffer.consumed == 0
    assert buffer.get(timeout=0.01) is not None
    assert buffer.consumed == 1


def test_buffer_get_blocks_then_returns_the_frame():
    buffer = LatestFrameBuffer()

    def producer():
        time.sleep(0.05)
        buffer.put(frame(42))

    threading.Thread(target=producer, daemon=True).start()
    got = buffer.get(timeout=2.0)
    assert got is not None and got[0][0, 0, 0] == 42


def test_buffer_close_unblocks_a_waiting_consumer():
    buffer = LatestFrameBuffer()
    threading.Timer(0.05, buffer.close).start()
    started = time.monotonic()
    assert buffer.get(timeout=5.0) is None
    assert time.monotonic() - started < 2.0, "close() must not wait for the timeout"


# ---------------------------------------------------------------------------
# The grabber
# ---------------------------------------------------------------------------


def test_grabber_delivers_frames_and_counts_drops():
    capture = FakeCapture(delay=0.001)
    grabber = FrameGrabber(
        0, capture_factory=lambda *a: capture, reconnect=False
    ).start()
    try:
        assert grabber.read(timeout=2.0) is not None
        time.sleep(0.15)  # let the reader run ahead of this (idle) consumer
        assert grabber.read(timeout=2.0) is not None
    finally:
        grabber.stop()

    assert grabber.stats.frames_read > grabber.stats.frames_delivered
    assert grabber.stats.frames_dropped > 0, "a slow consumer must show drops"
    assert 0.0 < grabber.stats.drop_rate <= 1.0
    assert capture.released is True
    assert "drop_rate" in grabber.stats.to_dict()


def test_grabber_never_returns_a_stale_frame():
    """After a long consumer stall the next frame must be a recent one."""
    capture = FakeCapture(delay=0.002)
    grabber = FrameGrabber(0, capture_factory=lambda *a: capture, reconnect=False).start()
    try:
        first = grabber.read_with_meta(timeout=2.0)
        time.sleep(0.2)
        second = grabber.read_with_meta(timeout=2.0)
    finally:
        grabber.stop()

    assert first is not None and second is not None
    # The sequence number jumped by far more than one: frames were dropped,
    # not queued, so the delivered frame is the newest one.
    assert second[2] - first[2] > 5
    # And it was captured a moment ago, not 200 ms ago.
    assert time.monotonic() - second[1] < 0.15


def test_file_source_blocks_instead_of_dropping():
    """Dropping frames from a file would silently change the result."""
    capture = FakeCapture(total=6)
    grabber = FrameGrabber(
        "clip.mp4", capture_factory=lambda *a: capture, reconnect=False
    ).start()
    try:
        collected = []
        while True:
            item = grabber.read(timeout=1.0)
            if item is None:
                break
            collected.append(int(item[0, 0, 0]))
    finally:
        grabber.stop()

    assert collected == [1, 2, 3, 4, 5, 6], "no frame may be skipped from a file"
    assert grabber.stats.frames_dropped == 0
    assert grabber.is_finished


def test_grabber_reconnects_after_the_camera_disappears():
    """A USB camera that re-enumerates must not end the run."""
    created = []

    def factory(*args):
        capture = FakeCapture(fail_after=3 if not created else None)
        created.append(capture)
        return capture

    grabber = FrameGrabber(
        0,
        capture_factory=factory,
        reconnect=True,
        reconnect_delay=0.01,
        read_failure_threshold=3,
    ).start()
    try:
        deadline = time.monotonic() + 5.0
        while len(created) < 2 and time.monotonic() < deadline:
            grabber.read(timeout=0.05)
        assert len(created) >= 2, "a new capture object must have been opened"

        # Frames must flow again from the replacement device.
        grabber.buffer.clear()
        assert grabber.read(timeout=2.0) is not None
    finally:
        grabber.stop()

    assert grabber.stats.reconnects >= 1
    assert created[0].released is True, "the dead device must be released"
    assert created[1].reads > 0, "the replacement device must actually be read"


def test_grabber_gives_up_when_reconnect_is_disabled():
    grabber = FrameGrabber(
        0,
        capture_factory=lambda *a: FakeCapture(fail_after=0),
        reconnect=False,
        read_failure_threshold=2,
    ).start()
    try:
        assert grabber.read(timeout=1.0) is None
        deadline = time.monotonic() + 2.0
        while grabber.is_running and time.monotonic() < deadline:
            time.sleep(0.01)
        assert grabber.is_finished
    finally:
        grabber.stop()

    assert grabber.stats.read_failures >= 2


def test_grabber_raises_when_the_source_cannot_be_opened():
    grabber = FrameGrabber(0, capture_factory=lambda *a: FakeCapture(opened=False))
    with pytest.raises(RuntimeError, match="failed to open"):
        grabber.start()


def test_capture_stats_rates():
    stats = CaptureStats(frames_read=100, frames_dropped=25, frames_delivered=75)
    assert stats.drop_rate == pytest.approx(0.25)
    assert stats.source_fps > 0
    assert stats.to_dict()["frames_delivered"] == 75
