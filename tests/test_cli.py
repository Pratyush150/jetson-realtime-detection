"""CLI argument handling and offline end-to-end runs through the entry points."""

from __future__ import annotations

import json

import numpy as np
import pytest

from edgevision.cli import (
    build_bench_parser,
    build_run_parser,
    main_bench,
    main_run,
    parse_classes,
)


def test_run_parser_defaults():
    args = build_run_parser().parse_args([])
    assert args.source == "0"
    assert args.backend == "auto"
    assert args.skip == 0, "0 means adaptive, not 'no skipping'"
    assert args.target_fps == 30.0
    assert args.tracker == "sort"
    assert args.jsonl is None


def test_run_parser_accepts_the_documented_flags():
    args = build_run_parser().parse_args(
        [
            "--backend", "tensorrt",
            "--source", "rtsp://cam/live",
            "--skip", "4",
            "--json", "out.jsonl",
            "--model", "yolov8n.engine",
            "--classes", "0,2",
            "--preview",
            "--max-frames", "50",
        ]
    )
    assert args.backend == "tensorrt"
    assert args.source == "rtsp://cam/live"
    assert args.skip == 4
    assert args.jsonl == "out.jsonl"
    assert args.preview is True
    assert args.max_frames == 50


def test_run_parser_accepts_a_gstreamer_pipeline_as_source():
    pipeline = "nvarguscamerasrc ! video/x-raw ! appsink"
    args = build_run_parser().parse_args(["--source", pipeline])
    assert args.source == pipeline


def test_bench_parser_defaults_and_flags():
    parser = build_bench_parser()
    args = parser.parse_args([])
    assert args.frames == 100 and args.warmup == 10
    assert args.resolution == "1280x720"

    args = parser.parse_args(["--backend", "onnxruntime", "--frames", "20", "--json"])
    assert args.as_json is True and args.frames == 20


def test_parse_classes():
    assert parse_classes("0,2,7") == [0, 2, 7]
    assert parse_classes(" 1 , 3 ") == [1, 3]
    assert parse_classes("") is None
    assert parse_classes(None) is None


def test_list_backends_exits_cleanly(capsys):
    assert main_run(["--list-backends"]) == 0
    assert "backend" in capsys.readouterr().out

    assert main_bench(["--list-backends"]) == 0
    assert "reason" in capsys.readouterr().out


def test_bench_cli_runs_offline_and_emits_json(capsys):
    code = main_bench(
        ["--backend", "mock", "--frames", "8", "--warmup", "2",
         "--resolution", "320x240", "--json"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["backend"] == "mock"
    assert payload["frames"] == 8
    assert payload["warmup_frames"] == 2
    assert payload["stages"]["inference"]["count"] == 8
    assert payload["frame_shape"] == [240, 320, 3]


def test_bench_cli_prints_a_table(capsys):
    assert main_bench(["--backend", "mock", "--frames", "5", "--warmup", "1",
                       "--resolution", "160x120"]) == 0
    out = capsys.readouterr().out
    assert "p99" in out and "discarded" in out


def test_bench_cli_rejects_a_bad_resolution():
    with pytest.raises(SystemExit):
        main_bench(["--backend", "mock", "--resolution", "big"])


def test_run_cli_end_to_end_over_a_video_file_stub(tmp_path, monkeypatch, capsys):
    """Drive main_run with a fake capture device so no camera is needed."""
    from edgevision import capture as capture_module

    class FakeCapture:
        def __init__(self, total=12):
            self.total = total
            self.reads = 0

        def isOpened(self):  # noqa: N802
            return True

        def read(self):
            if self.reads >= self.total:
                return False, None
            self.reads += 1
            return True, np.zeros((120, 160, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(
        capture_module, "_default_capture_factory", lambda *a, **k: FakeCapture()
    )

    log = tmp_path / "det.jsonl"
    code = main_run(
        ["--backend", "mock", "--source", str(tmp_path / "clip.mp4"),
         "--skip", "2", "--json", str(log), "--no-annotate", "--stats-json"]
    )
    assert code == 0

    stats = json.loads(capsys.readouterr().out)
    assert stats["detector"] == "mock"
    assert stats["frames"] == 12
    assert stats["inference_frames"] == 6, "skip=2 must halve the detector runs"
    assert stats["skipper"]["interval"] == 2

    records = [json.loads(line) for line in log.read_text().strip().splitlines()]
    assert records, "the JSON-lines log must have content"
    assert all("tracks" in r for r in records)
