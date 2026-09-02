"""ROI cropping, tiling coverage, and box remapping back to the full frame."""

from __future__ import annotations

import numpy as np
import pytest

from edgevision.backends import MockBackend
from edgevision.roi import Region, TiledInference, crop, remap_detections, tile_regions
from edgevision.types import Detection


def test_region_geometry_and_clipping():
    region = Region(10, 20, 100, 50)
    assert region.to_xyxy() == (10, 20, 110, 70)
    assert region.area == 5000

    clipped = Region(-10, -10, 5000, 5000).clipped(640, 480)
    assert clipped.to_xyxy() == (0, 0, 640, 480)


def test_region_expansion_stays_inside_the_frame():
    region = Region.from_xyxy([100, 100, 200, 180])
    grown = region.expanded(0.5, 640, 480)

    assert grown.width > region.width and grown.height > region.height
    assert grown.x >= 0 and grown.y >= 0
    assert grown.x2 <= 640 and grown.y2 <= 480

    edge = Region(0, 0, 40, 40).expanded(1.0, 640, 480)
    assert edge.x == 0 and edge.y == 0


def test_crop_returns_the_requested_pixels():
    frame = np.arange(20 * 30, dtype=np.uint8).reshape(20, 30)
    patch = crop(frame, Region(5, 4, 10, 6))
    assert patch.shape == (6, 10)
    assert patch[0, 0] == frame[4, 5]


def test_crop_clips_an_oversized_region():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop(frame, Region(90, 90, 50, 50)).shape == (10, 10, 3)


def test_remap_detections_moves_boxes_into_full_frame_space():
    detections = [Detection(10, 20, 30, 40, 0.8, 1, "bicycle")]
    remapped = remap_detections(detections, Region(200, 300, 640, 640))

    assert remapped[0].as_xyxy() == pytest.approx([210, 320, 230, 340])
    assert remapped[0].class_name == "bicycle" and remapped[0].score == pytest.approx(0.8)
    # The originals must not be mutated.
    assert detections[0].x1 == 10


def test_remap_detections_applies_scale_before_translation():
    """Covers the case where an ROI was upscaled before inference."""
    detections = [Detection(10, 10, 20, 20, 0.5)]
    remapped = remap_detections(detections, Region(100, 50, 200, 200), scale=(2.0, 4.0))
    assert remapped[0].as_xyxy() == pytest.approx([120, 90, 140, 130])


def test_remap_detections_clips_and_drops_degenerate_boxes():
    detections = [Detection(-500, -500, -400, -400, 0.5), Detection(0, 0, 50, 50, 0.9)]
    remapped = remap_detections(detections, Region(0, 0, 100, 100), frame_size=(100, 100))
    assert len(remapped) == 1 and remapped[0].score == pytest.approx(0.9)


def test_tiles_cover_the_whole_frame_and_stay_the_same_size():
    regions = tile_regions(1920, 1080, tile_size=(640, 640), overlap=0.2)

    assert all(r.width == 640 and r.height == 640 for r in regions), "ragged tiles"
    assert min(r.x for r in regions) == 0 and min(r.y for r in regions) == 0
    assert max(r.x2 for r in regions) == 1920, "right edge must be covered"
    assert max(r.y2 for r in regions) == 1080, "bottom edge must be covered"

    # Every pixel belongs to at least one tile.
    mask = np.zeros((1080, 1920), dtype=bool)
    for r in regions:
        mask[r.y : r.y2, r.x : r.x2] = True
    assert mask.all()


def test_tiles_actually_overlap():
    regions = tile_regions(1280, 720, tile_size=(640, 640), overlap=0.25)
    xs = sorted({r.x for r in regions})
    assert len(xs) >= 2
    assert xs[1] - xs[0] < 640, "stride must be smaller than the tile"


def test_tile_size_larger_than_the_frame_yields_one_tile():
    regions = tile_regions(320, 240, tile_size=(640, 640), overlap=0.2)
    assert len(regions) == 1
    assert regions[0].to_xyxy() == (0, 0, 320, 240)


def test_include_full_frame_prepends_a_whole_frame_pass():
    regions = tile_regions(1280, 720, (640, 640), 0.2, include_full_frame=True)
    assert regions[0].to_xyxy() == (0, 0, 1280, 720)


def test_tile_regions_rejects_a_silly_overlap():
    with pytest.raises(ValueError):
        tile_regions(640, 480, (320, 320), overlap=1.0)


def test_tiled_inference_remaps_every_tile_into_frame_coordinates():
    """MockBackend places an object at a fixed offset inside each tile."""
    detector = MockBackend(num_objects=1, velocity=(0.0, 0.0), origin=(20.0, 30.0),
                           box_size=(40.0, 25.0))
    tiled = TiledInference(detector, tile_size=(320, 320), overlap=0.0, merge_iou=0.5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = tiled.infer(frame)

    assert tiled.last_tile_count == 4, "2x2 grid with no overlap"
    assert len(detections) == 4, "one detection per tile, none merged away"
    corners = sorted((round(d.x1), round(d.y1)) for d in detections)
    assert corners == [(20, 30), (20, 190), (340, 30), (340, 190)]
    # Everything stays inside the frame.
    assert all(0 <= d.x1 and d.x2 <= 640 and 0 <= d.y1 and d.y2 <= 480 for d in detections)


def test_tiled_inference_merges_duplicates_from_overlapping_tiles():
    duplicated = [
        Detection(100, 100, 200, 200, 0.7, 0, "person"),
        Detection(104, 103, 203, 202, 0.9, 0, "person"),  # same object, other tile
        Detection(500, 400, 560, 460, 0.6, 0, "person"),  # different object
    ]
    tiled = TiledInference(MockBackend(), merge_iou=0.5)
    merged = tiled.merge(duplicated)

    assert len(merged) == 2
    assert max(d.score for d in merged) == pytest.approx(0.9)
    assert 0.7 not in [d.score for d in merged], "the weaker duplicate must lose"


def test_tiled_inference_rejects_tile_filling_detections():
    """A large object cut by a tile edge shows up as a tile-sized box."""
    detector = MockBackend(num_objects=1, velocity=(0.0, 0.0), origin=(0.0, 0.0),
                           box_size=(316.0, 316.0))
    permissive = TiledInference(detector, tile_size=(320, 320), overlap=0.0,
                                max_relative_area=1.0)
    detector.reset()
    strict = TiledInference(detector, tile_size=(320, 320), overlap=0.0,
                            max_relative_area=0.5)

    frame = np.zeros((320, 640, 3), dtype=np.uint8)
    detector.reset()
    assert len(permissive.infer(frame)) > 0
    detector.reset()
    assert strict.infer(frame) == []


def test_tiled_inference_reports_a_useful_name():
    assert TiledInference(MockBackend()).name == "tiled(mock)"
