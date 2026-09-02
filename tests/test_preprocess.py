"""Letterbox / un-letterbox coordinate round-tripping.

This is the test that matters most in the whole preprocessing path: if the
inverse mapping is wrong, every box is offset by the padding and the bug is
usually misdiagnosed as a model or tracker problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgevision.preprocess import (
    LetterboxParams,
    letterbox,
    letterbox_boxes,
    resize,
    to_nchw,
    unletterbox_boxes,
)


def make_frame(height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "src_h,src_w,dst",
    [
        (1080, 1920, (640, 640)),
        (480, 640, (640, 640)),
        (720, 1280, (416, 416)),
        (300, 300, (640, 640)),
        (1000, 200, (320, 320)),
        (240, 320, (512, 288)),
    ],
)
def test_letterbox_output_shape_and_scale(src_h, src_w, dst):
    frame = make_frame(src_h, src_w)
    padded, params = letterbox(frame, dst)

    assert padded.shape == (dst[1], dst[0], 3)
    assert params.src_w == src_w and params.src_h == src_h
    # The scale must be the *smaller* of the two ratios, i.e. aspect preserved.
    assert params.scale == pytest.approx(min(dst[0] / src_w, dst[1] / src_h))
    # Padding is only ever added on one axis.
    assert params.pad_x == 0 or params.pad_y == 0


@pytest.mark.parametrize(
    "src_h,src_w,dst",
    [
        (1080, 1920, (640, 640)),
        (480, 640, (640, 640)),
        (720, 1280, (416, 416)),
        (1000, 200, (320, 320)),
        (240, 320, (512, 288)),
    ],
)
def test_letterbox_unletterbox_round_trips_coordinates(src_h, src_w, dst):
    """Forward-map boxes into network space, map back, expect the original."""
    frame = make_frame(src_h, src_w)
    _, params = letterbox(frame, dst)

    rng = np.random.default_rng(11)
    x1 = rng.uniform(0, src_w * 0.5, size=25)
    y1 = rng.uniform(0, src_h * 0.5, size=25)
    boxes = np.stack(
        [x1, y1, x1 + rng.uniform(1, src_w * 0.4, 25), y1 + rng.uniform(1, src_h * 0.4, 25)],
        axis=1,
    ).astype(np.float32)
    boxes[:, 2] = np.minimum(boxes[:, 2], src_w)
    boxes[:, 3] = np.minimum(boxes[:, 3], src_h)

    mapped = letterbox_boxes(boxes, params)
    recovered = unletterbox_boxes(mapped, params)

    assert np.allclose(recovered, boxes, atol=1e-2), "letterbox inverse is not exact"


def test_letterboxed_boxes_stay_inside_the_network_input():
    frame = make_frame(1080, 1920)
    _, params = letterbox(frame, (640, 640))
    corners = np.array([[0, 0, 1920, 1080]], dtype=np.float32)
    mapped = letterbox_boxes(corners, params)

    assert mapped[0, 0] >= -1e-6 and mapped[0, 1] >= -1e-6
    assert mapped[0, 2] <= params.dst_w + 1e-6
    assert mapped[0, 3] <= params.dst_h + 1e-6


def test_unletterbox_clips_to_the_source_frame():
    """A box predicted inside the grey padding must not escape the frame."""
    frame = make_frame(1080, 1920)
    _, params = letterbox(frame, (640, 640))
    # y = 5 is inside the top padding band (pad_y = 140).
    boxes = np.array([[-50.0, 5.0, 700.0, 20.0]], dtype=np.float32)
    clipped = unletterbox_boxes(boxes, params, clip=True)

    assert clipped[0, 0] >= 0.0
    assert clipped[0, 1] >= 0.0
    assert clipped[0, 2] <= params.src_w
    assert clipped[0, 3] <= params.src_h

    unclipped = unletterbox_boxes(boxes, params, clip=False)
    assert unclipped[0, 0] < 0.0, "clip=False must not clamp"


def test_scaleup_false_never_upscales():
    frame = make_frame(240, 320)
    _, params = letterbox(frame, (640, 640), scaleup=False)
    assert params.scale == pytest.approx(1.0)

    _, upscaled = letterbox(frame, (640, 640), scaleup=True)
    assert upscaled.scale > 1.0


def test_non_centred_letterbox_pads_bottom_right_only():
    frame = make_frame(360, 640)
    padded, params = letterbox(frame, (640, 640), center=False)
    assert params.pad_x == 0.0 and params.pad_y == 0.0
    assert padded.shape == (640, 640, 3)


def test_resize_preserves_dtype_and_shape():
    frame = make_frame(100, 200)
    out = resize(frame, (50, 25))
    assert out.shape == (25, 50, 3)
    assert out.dtype == np.uint8


def test_to_nchw_layout_and_range():
    frame = np.full((8, 6, 3), 255, dtype=np.uint8)
    tensor = to_nchw(frame)
    assert tensor.shape == (1, 3, 8, 6)
    assert tensor.dtype == np.float32
    assert tensor.max() == pytest.approx(1.0)


def test_letterbox_rejects_empty_frames():
    with pytest.raises(ValueError):
        letterbox(np.zeros((0, 0, 3), dtype=np.uint8), (64, 64))


def test_unletterbox_rejects_zero_scale():
    params = LetterboxParams(0.0, 0.0, 0.0, 10, 10, 10, 10)
    with pytest.raises(ValueError):
        unletterbox_boxes(np.zeros((1, 4), np.float32), params)
