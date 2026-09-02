"""Region-of-interest cropping and tiled inference for small distant objects.

The problem
-----------
Detectors have a floor on object size. A 640x640 YOLO applied to a 1920x1080
frame downsamples by 3x; an object that is 24 px across in the source becomes
8 px at the network input, which is below the stride of the shallowest
detection head. It is not that the model is bad — the object is gone before
the first convolution.

This is the normal case for anything airborne: a person seen from 80 m, a
vehicle at the far end of a field, a boat near the horizon.

Two fixes, both here:

``crop`` + ``remap_detections``
    If you already know where to look (a gimbal is pointed at a target, or a
    previous detection gives you a prior), crop and run at full effective
    resolution. Cheapest possible option — the inference cost is unchanged.

``TiledInference``
    Split the frame into overlapping tiles and run the detector on each. An
    object 24 px across in a 640x640 tile stays 24 px at the network input.
    The cost is linear in tile count: a 2x2 grid is 4x the inference, so on
    an edge board you pair this with frame skipping and run the tiled pass
    only every Nth detection, or only over tiles where the tracker says
    something is happening.

The overlap exists because an object straddling a tile boundary is cut in
half in both tiles. Overlap by more than the largest object you care about,
then merge with class-aware NMS to collapse the duplicates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .postprocess import batched_nms
from .types import Detection

LOGGER = logging.getLogger(__name__)

__all__ = ["Region", "crop", "remap_detections", "tile_regions", "TiledInference"]


@dataclass(frozen=True)
class Region:
    """An axis-aligned crop rectangle in source-frame pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def clipped(self, frame_width: int, frame_height: int) -> "Region":
        """Clamp to the frame, keeping at least a 1 px rectangle."""
        x = int(max(0, min(self.x, frame_width - 1)))
        y = int(max(0, min(self.y, frame_height - 1)))
        w = int(max(1, min(self.width, frame_width - x)))
        h = int(max(1, min(self.height, frame_height - y)))
        return Region(x, y, w, h)

    def expanded(self, margin: float, frame_width: int, frame_height: int) -> "Region":
        """Grow by ``margin`` (a fraction of size) and clip to the frame.

        Used to build a search window around a previous detection: the target
        moved between frames, so cropping exactly to the last box will clip it.
        """
        dx = int(round(self.width * margin))
        dy = int(round(self.height * margin))
        return Region(
            self.x - dx, self.y - dy, self.width + 2 * dx, self.height + 2 * dy
        ).clipped(frame_width, frame_height)

    @classmethod
    def from_xyxy(cls, box: Sequence[float]) -> "Region":
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        return cls(int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)))

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x2, self.y2)


def crop(frame: np.ndarray, region: Region) -> np.ndarray:
    """Crop ``frame`` to ``region``, clipping the region to the frame first.

    Returns a *view* where possible; do not write to it in place unless you
    intend to modify the source frame. Avoiding the copy matters: a full
    1080p copy per ROI per frame is a few milliseconds of pure memory
    bandwidth on a Pi.
    """
    height, width = frame.shape[:2]
    clipped = region.clipped(width, height)
    return frame[clipped.y : clipped.y2, clipped.x : clipped.x2]


def remap_detections(
    detections: Sequence[Detection],
    region: Region,
    scale: Tuple[float, float] = (1.0, 1.0),
    frame_size: Optional[Tuple[int, int]] = None,
) -> List[Detection]:
    """Map detections from crop coordinates back to full-frame coordinates.

    ``scale`` is applied *before* the translation and covers the case where
    the crop was resized before inference (for example, upscaling a small ROI
    to the network input to recover detail). ``frame_size`` is ``(width,
    height)``; when given, results are clipped to the frame.
    """
    sx, sy = float(scale[0]), float(scale[1])
    out: List[Detection] = []
    for det in detections:
        moved = det.scaled(sx, sy).translated(float(region.x), float(region.y))
        if frame_size is not None:
            width, height = frame_size
            moved.x1 = float(np.clip(moved.x1, 0.0, width))
            moved.y1 = float(np.clip(moved.y1, 0.0, height))
            moved.x2 = float(np.clip(moved.x2, 0.0, width))
            moved.y2 = float(np.clip(moved.y2, 0.0, height))
            if moved.area <= 0:
                continue
        out.append(moved)
    return out


def tile_regions(
    frame_width: int,
    frame_height: int,
    tile_size: Tuple[int, int] = (640, 640),
    overlap: float = 0.2,
    include_full_frame: bool = False,
) -> List[Region]:
    """Cover a frame with overlapping tiles.

    Tiles are laid out on a grid with a stride of ``tile * (1 - overlap)``,
    and the last row/column is pulled back so it ends exactly at the frame
    edge rather than hanging off it. That keeps every tile the same size,
    which matters for a fixed-shape TensorRT engine — a ragged edge tile
    would need either padding or a second engine profile.

    ``include_full_frame`` adds one whole-frame pass at the front. Tiles find
    small things; the full frame still finds large things that no single tile
    contains. Running both and merging is the usual compromise.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    tile_w = min(int(tile_size[0]), int(frame_width))
    tile_h = min(int(tile_size[1]), int(frame_height))
    stride_x = max(1, int(round(tile_w * (1.0 - overlap))))
    stride_y = max(1, int(round(tile_h * (1.0 - overlap))))

    xs = list(range(0, max(1, frame_width - tile_w + 1), stride_x))
    ys = list(range(0, max(1, frame_height - tile_h + 1), stride_y))
    if xs[-1] + tile_w < frame_width:
        xs.append(frame_width - tile_w)
    if ys[-1] + tile_h < frame_height:
        ys.append(frame_height - tile_h)

    regions = []
    if include_full_frame:
        regions.append(Region(0, 0, int(frame_width), int(frame_height)))
    regions.extend(Region(int(x), int(y), tile_w, tile_h) for y in ys for x in xs)
    return regions


class TiledInference:
    """Run a detector over overlapping tiles and merge the results.

    Example
    -------
    >>> import numpy as np
    >>> from edgevision.backends import MockBackend
    >>> tiled = TiledInference(MockBackend(num_objects=1), tile_size=(320, 320))
    >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
    >>> isinstance(tiled.infer(frame), list)
    True

    Parameters
    ----------
    detector:
        Any :class:`~edgevision.backends.base.Detector`.
    tile_size:
        Tile size in *source* pixels. Match it to the network input to avoid
        an extra rescale, unless you are deliberately upscaling.
    overlap:
        Fractional overlap between neighbouring tiles.
    merge_iou:
        IoU threshold for the class-aware NMS that collapses duplicates from
        overlapping tiles.
    min_relative_area / max_relative_area:
        Discard detections that fill an implausible fraction of a tile.
        Tiling produces a specific artefact: a large object cut by a tile
        boundary is detected as a tile-sized box in each tile. Rejecting
        detections that occupy nearly the whole tile removes most of them.
    """

    def __init__(
        self,
        detector,
        tile_size: Tuple[int, int] = (640, 640),
        overlap: float = 0.2,
        merge_iou: float = 0.5,
        include_full_frame: bool = False,
        max_relative_area: float = 0.95,
        min_relative_area: float = 0.0,
    ) -> None:
        self.detector = detector
        self.tile_size = (int(tile_size[0]), int(tile_size[1]))
        self.overlap = float(overlap)
        self.merge_iou = float(merge_iou)
        self.include_full_frame = bool(include_full_frame)
        self.max_relative_area = float(max_relative_area)
        self.min_relative_area = float(min_relative_area)
        self.last_tile_count = 0

    @property
    def name(self) -> str:
        return f"tiled({getattr(self.detector, 'name', 'detector')})"

    def regions_for(self, frame: np.ndarray) -> List[Region]:
        height, width = frame.shape[:2]
        return tile_regions(
            width, height, self.tile_size, self.overlap, self.include_full_frame
        )

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Detect over all tiles and merge into full-frame coordinates."""
        height, width = frame.shape[:2]
        regions = self.regions_for(frame)
        self.last_tile_count = len(regions)

        collected: List[Detection] = []
        for region in regions:
            tile = crop(frame, region)
            tile_area = max(1, tile.shape[0] * tile.shape[1])
            detections = self.detector.infer(tile)
            kept = [
                d
                for d in detections
                if self.min_relative_area
                <= (d.area / tile_area)
                <= self.max_relative_area
            ]
            collected.extend(
                remap_detections(kept, region, frame_size=(width, height))
            )

        return self.merge(collected)

    __call__ = infer

    def merge(self, detections: Sequence[Detection]) -> List[Detection]:
        """Class-aware NMS across tiles, keeping the highest-scoring copy."""
        if len(detections) < 2:
            return list(detections)
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
        scores = np.array([d.score for d in detections], dtype=np.float32)
        classes = np.array([d.class_id for d in detections], dtype=np.int64)
        keep = batched_nms(boxes, scores, classes, self.merge_iou)
        return [detections[int(i)] for i in keep]
