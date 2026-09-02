"""Letterbox preprocessing and its exact inverse.

Why this file exists at all
---------------------------
Every "YOLO demo script" quietly resizes the frame to 640x640 with a plain
``cv2.resize``. That distorts the aspect ratio, which costs accuracy on tall
or wide objects, and — worse — it makes the coordinate mapping back to the
original frame ambiguous once you start cropping ROIs or tiling. Letterboxing
(scale by the *smaller* factor, pad the rest) keeps the aspect ratio and gives
an inverse that is exact to floating point.

The inverse is the part people get wrong. If you forget the padding offset,
every box is shifted by a constant number of pixels; on a 1920x1080 source
letterboxed into 640x640 that is 80 px of vertical error, which looks like a
tracker bug and gets debugged for a day.

``cv2`` is used when present because its resize is SIMD/NEON accelerated and
on a Pi the resize is a measurable slice of the frame budget. A pure-numpy
bilinear fallback keeps the module importable (and the tests runnable) on a
machine with no OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly depending on environment
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False

__all__ = [
    "LetterboxParams",
    "letterbox",
    "letterbox_boxes",
    "unletterbox_boxes",
    "resize",
    "to_nchw",
    "CV2_AVAILABLE",
]


@dataclass(frozen=True)
class LetterboxParams:
    """Everything needed to undo a letterbox, and nothing else.

    Attributes
    ----------
    scale:
        Uniform scale factor applied to the source frame.
    pad_x, pad_y:
        Padding added on the *left* and *top* edge, in network-input pixels.
    src_w, src_h:
        Source frame size, kept so the inverse can clip to real frame bounds.
    dst_w, dst_h:
        Network input size.
    """

    scale: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int

    @property
    def pad(self) -> Tuple[float, float]:
        return (self.pad_x, self.pad_y)


def _bilinear_resize(image: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Bilinear resize with cv2's half-pixel convention (no cv2 required)."""
    src_h, src_w = image.shape[:2]
    if (src_w, src_h) == (out_w, out_h):
        return image.copy()

    ys = (np.arange(out_h, dtype=np.float64) + 0.5) * (src_h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float64) + 0.5) * (src_w / out_w) - 0.5
    ys = np.clip(ys, 0.0, src_h - 1.0)
    xs = np.clip(xs, 0.0, src_w - 1.0)

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (ys - y0).astype(np.float32)
    wx = (xs - x0).astype(np.float32)

    src = image.astype(np.float32)
    if src.ndim == 2:
        wy = wy[:, None]
        wx = wx[None, :]
    else:
        wy = wy[:, None, None]
        wx = wx[None, :, None]

    top = src[y0][:, x0] * (1.0 - wx) + src[y0][:, x1] * wx
    bot = src[y1][:, x0] * (1.0 - wx) + src[y1][:, x1] * wx
    out = top * (1.0 - wy) + bot * wy
    return out.astype(image.dtype)


def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize ``image`` to ``(width, height)``, preferring cv2 when available."""
    out_w, out_h = int(size[0]), int(size[1])
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"invalid resize target {size!r}")
    if CV2_AVAILABLE:
        interp = cv2.INTER_LINEAR
        src_h, src_w = image.shape[:2]
        if out_w < src_w or out_h < src_h:
            # Downscaling with INTER_AREA is both sharper and cheaper than
            # INTER_LINEAR; on a Pi 4 this is a real few-percent win because
            # the capture resolution is usually far above the network input.
            interp = cv2.INTER_AREA
        return cv2.resize(image, (out_w, out_h), interpolation=interp)
    return _bilinear_resize(image, out_w, out_h)


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    scaleup: bool = True,
    center: bool = True,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, LetterboxParams]:
    """Resize with unchanged aspect ratio and pad to ``new_shape``.

    Parameters
    ----------
    image:
        HxW or HxWxC source frame.
    new_shape:
        ``(width, height)`` of the network input.
    scaleup:
        If ``False``, never upscale a small frame. Upscaling a 320x240 USB
        stream to 640x640 buys no accuracy and costs a resize; leave it off
        when the source is already smaller than the network input.
    center:
        Pad symmetrically (YOLO default). ``False`` pads bottom/right only,
        which some TensorRT sample pipelines expect.
    stride:
        If set, the padded size is rounded up to a multiple of ``stride``
        instead of being fixed to ``new_shape``. Useful for fully-convolutional
        exports with dynamic shapes.

    Returns
    -------
    (padded_image, params)
        ``params`` is everything :func:`unletterbox_boxes` needs.
    """
    if image.ndim not in (2, 3):
        raise ValueError("letterbox expects a HxW or HxWxC array")

    src_h, src_w = image.shape[:2]
    dst_w, dst_h = int(new_shape[0]), int(new_shape[1])
    if src_h <= 0 or src_w <= 0:
        raise ValueError("cannot letterbox an empty frame")

    scale = min(dst_w / src_w, dst_h / src_h)
    if not scaleup:
        scale = min(scale, 1.0)

    unpad_w = int(round(src_w * scale))
    unpad_h = int(round(src_h * scale))

    if stride:
        dst_w = int(np.ceil(unpad_w / stride) * stride)
        dst_h = int(np.ceil(unpad_h / stride) * stride)

    pad_w = dst_w - unpad_w
    pad_h = dst_h - unpad_h
    if center:
        pad_x = pad_w / 2.0
        pad_y = pad_h / 2.0
    else:
        pad_x = 0.0
        pad_y = 0.0

    # Round the left/top pad the way YOLO does; the remaining pad_w - left
    # and pad_h - top pixels on the other side are already filled by the
    # canvas, so they need no separate variables.
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))

    resized = resize(image, (unpad_w, unpad_h))

    if image.ndim == 3:
        channels = image.shape[2]
        fill = np.array(color[:channels], dtype=image.dtype)
        canvas = np.empty((dst_h, dst_w, channels), dtype=image.dtype)
        canvas[:, :] = fill
        canvas[top : top + unpad_h, left : left + unpad_w, :] = resized
    else:
        canvas = np.full((dst_h, dst_w), color[0], dtype=image.dtype)
        canvas[top : top + unpad_h, left : left + unpad_w] = resized

    params = LetterboxParams(
        scale=float(scale),
        pad_x=float(left),
        pad_y=float(top),
        src_w=int(src_w),
        src_h=int(src_h),
        dst_w=int(dst_w),
        dst_h=int(dst_h),
    )
    return canvas, params


def letterbox_boxes(boxes: np.ndarray, params: LetterboxParams) -> np.ndarray:
    """Map ``xyxy`` boxes from source frame space into letterboxed space."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * params.scale + params.pad_x
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * params.scale + params.pad_y
    return boxes


def unletterbox_boxes(
    boxes: np.ndarray, params: LetterboxParams, clip: bool = True
) -> np.ndarray:
    """Map ``xyxy`` boxes from letterboxed space back to source frame space.

    This is the exact inverse of :func:`letterbox_boxes`. Get it wrong and
    every box is offset by the padding; the symptom looks like a tracker or
    a model bug, but it is arithmetic.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if params.scale <= 0:
        raise ValueError("letterbox scale must be positive")
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - params.pad_x) / params.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - params.pad_y) / params.scale
    if clip:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0.0, params.src_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0.0, params.src_h)
    return boxes


def to_nchw(
    image: np.ndarray,
    scale: float = 1.0 / 255.0,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    rgb: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """HWC uint8 BGR frame -> NCHW float tensor.

    Kept separate from :func:`letterbox` so a backend that wants NHWC uint8
    (Hailo, most INT8 TFLite exports) can skip normalisation entirely instead
    of paying for a float conversion it will immediately quantise away.
    """
    array = image
    if array.ndim == 2:
        array = array[:, :, None]
    if rgb and array.shape[2] == 3:
        array = array[:, :, ::-1]
    array = array.astype(dtype) * float(scale)
    if mean is not None:
        array = array - np.asarray(mean, dtype=dtype)
    if std is not None:
        array = array / np.asarray(std, dtype=dtype)
    array = np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])
    return array.astype(dtype)
