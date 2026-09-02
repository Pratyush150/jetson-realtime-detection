"""Sinks: JSON-lines log, event snapshots, sink isolation, MJPEG framing."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from edgevision.sinks import (
    BOUNDARY,
    JsonLinesSink,
    MjpegPreviewServer,
    Sink,
    SinkGroup,
    SnapshotSink,
    encode_ppm,
    jpeg_encoder_available,
    multipart_chunk,
)
from edgevision.types import Detection, Track, TrackState


def blank(height=32, width=48):
    return np.zeros((height, width, 3), dtype=np.uint8)


def sample_track(track_id=1):
    return Track(track_id, 10, 20, 40, 60, 0.87, 0, "person", TrackState.CONFIRMED, hits=5)


class Clock:
    """Injectable monotonic clock so the snapshot cooldown is deterministic."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


# ---------------------------------------------------------------------------
# JSON lines
# ---------------------------------------------------------------------------


def test_jsonl_sink_writes_one_valid_object_per_frame(tmp_path):
    path = tmp_path / "log" / "detections.jsonl"
    with JsonLinesSink(str(path), flush_every=1) as sink:
        sink.write(blank(), [sample_track(1)], [], {"frame": 0})
        sink.write(blank(), [sample_track(1), sample_track(2)], [], {"frame": 1})

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]

    assert records[0]["frame"] == 0
    assert len(records[1]["tracks"]) == 2
    assert records[0]["tracks"][0]["track_id"] == 1
    assert records[0]["tracks"][0]["class_name"] == "person"
    assert records[0]["tracks"][0]["bbox"] == [10.0, 20.0, 40.0, 60.0]
    assert records[0]["tracks"][0]["state"] == "confirmed"


def test_jsonl_sink_skips_empty_frames_by_default(tmp_path):
    path = tmp_path / "d.jsonl"
    sink = JsonLinesSink(str(path), flush_every=1)
    sink.write(blank(), [], [])
    sink.write(blank(), [sample_track()], [])
    sink.close()

    assert len(path.read_text().strip().splitlines()) == 1
    assert sink.records_written == 1


def test_jsonl_sink_can_record_empty_frames_and_raw_detections(tmp_path):
    path = tmp_path / "d.jsonl"
    sink = JsonLinesSink(str(path), include_empty=True, include_detections=True, flush_every=1)
    sink.write(blank(), [], [Detection(1, 2, 3, 4, 0.5, 0, "person")])
    sink.close()

    record = json.loads(path.read_text().strip())
    assert record["tracks"] == []
    assert record["detections"][0]["score"] == 0.5


def test_jsonl_sink_appends_across_sessions(tmp_path):
    path = tmp_path / "d.jsonl"
    for _ in range(2):
        sink = JsonLinesSink(str(path), flush_every=1)
        sink.write(blank(), [sample_track()], [])
        sink.close()
    assert len(path.read_text().strip().splitlines()) == 2


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def test_snapshot_sink_rate_limits_writes(tmp_path):
    """Without a cooldown a person standing still fills the card in minutes."""
    clock = Clock()
    sink = SnapshotSink(str(tmp_path), min_interval_s=1.0, clock=clock)

    for _ in range(30):
        clock.now += 1.0 / 30.0
        sink.write(blank(), [sample_track()], [])

    assert len(sink.written) == 1, "30 triggered frames in 1 s must produce 1 file"

    clock.now += 2.0
    sink.write(blank(), [sample_track()], [])
    assert len(sink.written) == 2

    for path in sink.written:
        assert path.endswith((".jpg", ".ppm"))
        with open(path, "rb") as handle:
            assert len(handle.read()) > 0


def test_snapshot_sink_only_fires_on_the_trigger(tmp_path):
    sink = SnapshotSink(
        str(tmp_path),
        trigger=lambda tracks, dets: any(t.class_name == "car" for t in tracks),
        min_interval_s=0.0,
        clock=Clock(),
    )
    sink.write(blank(), [sample_track()], [])
    assert sink.written == []

    car = Track(9, 0, 0, 10, 10, 0.9, 2, "car", TrackState.CONFIRMED)
    sink.write(blank(), [car], [])
    assert len(sink.written) == 1


def test_snapshot_sink_respects_max_files(tmp_path):
    sink = SnapshotSink(str(tmp_path), min_interval_s=0.0, max_files=2, clock=Clock())
    for _ in range(5):
        sink.write(blank(), [sample_track()], [])
    assert len(sink.written) == 2


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def test_ppm_fallback_encodes_a_valid_header():
    payload = encode_ppm(blank(4, 6))
    assert payload.startswith(b"P6\n6 4\n255\n")
    assert len(payload) == len(b"P6\n6 4\n255\n") + 4 * 6 * 3


def test_ppm_writes_rgb_order():
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)  # BGR blue
    assert encode_ppm(frame).endswith(bytes([0, 0, 255]))


def test_multipart_chunk_framing():
    chunk = multipart_chunk(b"JPEGBYTES", "B")
    assert chunk.startswith(b"--B\r\n")
    assert b"Content-Type: image/jpeg\r\n" in chunk
    assert b"Content-Length: 9\r\n\r\n" in chunk
    assert chunk.endswith(b"JPEGBYTES\r\n")


def test_jpeg_encoder_probe_returns_a_bool():
    assert isinstance(jpeg_encoder_available(), bool)


# ---------------------------------------------------------------------------
# Sink group
# ---------------------------------------------------------------------------


class Counter(Sink):
    def __init__(self):
        self.count = 0

    def write(self, frame, tracks=(), detections=(), meta=None):
        self.count += 1


class Broken(Sink):
    def write(self, frame, tracks=(), detections=(), meta=None):
        raise IOError("no space left on device")


def test_sink_group_isolates_a_failing_sink():
    good, bad = Counter(), Broken()
    group = SinkGroup([bad, good])
    for _ in range(3):
        group.write(blank())

    assert good.count == 3
    assert group.sinks == [good]
    assert len(group.failed) == 1 and "no space left" in group.failed[0][1]

    group.close()
    assert group.sinks == []


# ---------------------------------------------------------------------------
# MJPEG preview server (loopback only, no external network)
# ---------------------------------------------------------------------------


@pytest.fixture
def preview():
    server = MjpegPreviewServer(
        host="127.0.0.1", port=0, encoder=lambda frame: b"FAKEJPEGDATA"
    ).start()
    try:
        yield server
    finally:
        server.close()


def test_preview_server_binds_and_reports_a_url(preview):
    assert preview.port > 0
    assert preview.is_running
    assert preview.url.endswith(f":{preview.port}/")


def test_preview_server_serves_a_health_check(preview):
    with urllib.request.urlopen(f"http://127.0.0.1:{preview.port}/healthz", timeout=5) as r:
        assert r.status == 200
        assert r.read() == b"ok\n"


def test_preview_server_serves_an_index_page(preview):
    with urllib.request.urlopen(f"http://127.0.0.1:{preview.port}/", timeout=5) as r:
        body = r.read().decode()
    assert "/stream.mjpg" in body


def test_preview_server_streams_multipart_frames(preview):
    preview.write(blank(), [sample_track()])
    assert preview.frames_published == 1

    with urllib.request.urlopen(f"http://127.0.0.1:{preview.port}/snapshot.jpg", timeout=5) as r:
        assert r.headers["Content-Type"] == "image/jpeg"
        assert r.read() == b"FAKEJPEGDATA"

    with urllib.request.urlopen(f"http://127.0.0.1:{preview.port}/stream.mjpg", timeout=5) as r:
        assert BOUNDARY in r.headers["Content-Type"]
        head = r.read(len(multipart_chunk(b"FAKEJPEGDATA")))
    assert head == multipart_chunk(b"FAKEJPEGDATA")


def test_preview_server_publishes_only_every_nth_frame():
    server = MjpegPreviewServer(host="127.0.0.1", port=0, every_n=3, encoder=lambda f: b"X")
    for _ in range(9):
        server.write(blank())
    assert server.frames_published == 3
    server.close()


def test_preview_server_returns_404_for_unknown_paths(preview):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(f"http://127.0.0.1:{preview.port}/nope", timeout=5)
    assert excinfo.value.code == 404
