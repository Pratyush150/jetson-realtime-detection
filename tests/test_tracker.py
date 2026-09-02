"""Tracker behaviour: ID stability, ID lifecycle, and assignment correctness."""

from __future__ import annotations

import numpy as np
import pytest

from edgevision.backends import MockBackend
from edgevision.tracker import (
    SCIPY_AVAILABLE,
    IoUTracker,
    KalmanBoxTracker,
    SortTracker,
    associate,
    build_tracker,
    hungarian,
)
from edgevision.types import Detection, TrackState

FRAME = (480, 640)


def moving_detection(step: int, x0: float = 40.0, vx: float = 9.0) -> Detection:
    x = x0 + vx * step
    return Detection(x, 100.0, x + 60.0, 180.0, 0.9, 0, "person")


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


def test_hungarian_matches_a_known_optimum():
    cost = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    rows, cols = hungarian(cost)
    assert cost[rows, cols].sum() == pytest.approx(5.0)
    assert sorted(cols.tolist()) == [0, 1, 2]


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
def test_hungarian_agrees_with_scipy_on_random_rectangular_matrices():
    """The bundled solver must be a drop-in for scipy, not an approximation."""
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(0)
    for _ in range(60):
        rows_n = int(rng.integers(1, 8))
        cols_n = int(rng.integers(1, 8))
        cost = rng.normal(size=(rows_n, cols_n)) * float(rng.integers(1, 10))
        mine = cost[hungarian(cost)].sum()
        theirs = cost[linear_sum_assignment(cost)].sum()
        assert mine == pytest.approx(theirs, abs=1e-9)


def test_hungarian_handles_empty_and_rejects_non_finite():
    rows, cols = hungarian(np.zeros((0, 3)))
    assert len(rows) == 0 and len(cols) == 0
    with pytest.raises(ValueError):
        hungarian(np.array([[np.inf, 1.0]]))


def test_associate_gates_on_iou():
    tracks = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    dets = np.array([[1, 1, 11, 11], [500, 500, 510, 510]], dtype=np.float32)
    matches, unmatched_tracks, unmatched_dets = associate(tracks, dets, 0.3)

    assert matches == [(0, 0)]
    assert unmatched_tracks == [1]
    assert unmatched_dets == [1]


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------


def test_kalman_predicts_forward_along_estimated_velocity():
    kf = KalmanBoxTracker([0, 0, 20, 20], track_id=1)
    for step in range(1, 8):
        kf.predict()
        kf.update([10 * step, 0, 10 * step + 20, 20])

    before = kf.bbox.copy()
    predicted = kf.predict()
    assert predicted[0] > before[0], "constant-velocity model must move right"
    assert kf.velocity[0] > 5.0
    assert abs(kf.velocity[1]) < 2.0


def test_kalman_coast_does_not_count_as_a_missed_observation():
    """Coasting on skipped frames must not age a track toward deletion."""
    kf = KalmanBoxTracker([0, 0, 20, 20], track_id=1)
    kf.update([5, 0, 25, 20])
    assert kf.time_since_update == 0

    kf.coast()
    kf.coast()
    assert kf.time_since_update == 0, "coast() is not a miss"
    assert kf.age == 2

    kf.predict()
    assert kf.time_since_update == 1, "predict() is a miss"


# ---------------------------------------------------------------------------
# SORT
# ---------------------------------------------------------------------------


def test_track_id_is_stable_across_frames_of_a_moving_object():
    tracker = SortTracker(max_age=10, min_hits=3, iou_threshold=0.3)
    seen = []
    for step in range(25):
        tracks = tracker.update([moving_detection(step)])
        if tracks:
            seen.append(tracks[0].track_id)

    assert len(seen) >= 20
    assert len(set(seen)) == 1, f"ID flipped during smooth motion: {sorted(set(seen))}"
    assert tracker.tracks[0].state is TrackState.CONFIRMED


def test_two_objects_keep_distinct_stable_ids():
    detector = MockBackend(num_objects=2, velocity=(7.0, 3.0), spacing=(0.0, 200.0))
    tracker = SortTracker(max_age=10, min_hits=3)
    frame = np.zeros((*FRAME, 3), dtype=np.uint8)

    id_sets = []
    for _ in range(20):
        tracks = tracker.update(detector.infer(frame))
        if len(tracks) == 2:
            id_sets.append(tuple(t.track_id for t in tracks))

    assert len(id_sets) >= 15
    assert len(set(id_sets)) == 1, f"IDs were not stable: {set(id_sets)}"
    assert id_sets[0][0] != id_sets[0][1]


def test_new_id_is_assigned_after_max_age_expiry():
    """The lifecycle test: coast, delete, and never reuse the number."""
    tracker = SortTracker(max_age=5, min_hits=3, iou_threshold=0.3)

    for step in range(10):
        tracker.update([moving_detection(step)])
    original = tracker.report()[0].track_id

    # Object disappears. Within max_age it is coasted, not deleted.
    tracker.update([])
    assert tracker.report()[0].track_id == original
    assert tracker.report()[0].state is TrackState.LOST

    for _ in range(6):
        tracker.update([])
    assert tracker.active_ids == [], "track should be deleted past max_age"

    # It comes back in the same place; it must NOT get the old ID back.
    for step in range(10, 14):
        tracks = tracker.update([moving_detection(step)])
    new_ids = [t.track_id for t in tracks]

    assert new_ids and original not in new_ids
    assert min(new_ids) > original


def test_track_ids_are_never_reused():
    tracker = SortTracker(max_age=1, min_hits=1)
    issued = []
    for cycle in range(6):
        for _ in range(2):
            tracks = tracker.update([moving_detection(0, x0=50.0 + 300 * (cycle % 2))])
        issued.extend(t.track_id for t in tracks)
        for _ in range(4):
            tracker.update([])

    assert len(issued) == len(set(issued)), f"IDs were recycled: {issued}"
    assert issued == sorted(issued), "IDs must be monotonically increasing"


def test_min_hits_suppresses_single_frame_false_positives():
    tracker = SortTracker(max_age=5, min_hits=3)
    # Burn past the startup grace period where tracks confirm immediately.
    for step in range(6):
        tracker.update([moving_detection(step)])
    before = set(tracker.active_ids)

    spurious = Detection(500, 50, 540, 90, 0.4, 1, "bicycle")
    tracks = tracker.update([moving_detection(6), spurious])
    reported = {t.track_id for t in tracks}

    assert reported == before, "a one-frame blip must not be reported"
    # And it is discarded rather than lingering.
    for step in range(7, 10):
        tracker.update([moving_detection(step)])
    assert set(tracker.active_ids) == before


def test_predict_propagates_boxes_between_detections():
    """Frame skipping relies on this: boxes must keep moving with no detector."""
    tracker = SortTracker(max_age=30, min_hits=2)
    for step in range(8):
        tracker.update([moving_detection(step)])
    start_x = tracker.report()[0].x1
    track_id = tracker.report()[0].track_id

    positions = []
    for _ in range(3):
        tracks = tracker.predict()
        positions.append(tracks[0].x1)

    assert all(t == track_id for t in [tracker.report()[0].track_id])
    assert positions == sorted(positions), "coasted boxes must advance monotonically"
    assert positions[-1] > start_x + 10, "boxes did not move during skipped frames"
    assert tracker.report()[0].state is TrackState.CONFIRMED


def test_class_aware_association_refuses_cross_class_matches():
    tracker = SortTracker(max_age=5, min_hits=1, class_aware=True)
    tracker.update([Detection(0, 0, 50, 50, 0.9, 0, "person")])
    tracks = tracker.update([Detection(2, 2, 52, 52, 0.9, 7, "truck")])

    assert len(tracks) == 2, "same box, different class -> a new track"


def test_reset_clears_tracks_but_not_the_id_counter():
    tracker = SortTracker(min_hits=1)
    tracker.update([moving_detection(0)])
    tracker.reset()
    assert tracker.active_ids == []
    tracks = tracker.update([moving_detection(0)])
    assert tracks[0].track_id > 1


# ---------------------------------------------------------------------------
# IoU tracker
# ---------------------------------------------------------------------------


def test_iou_tracker_holds_a_stable_id_and_extrapolates():
    tracker = IoUTracker(max_age=5, min_hits=2, iou_threshold=0.3)
    ids = []
    for step in range(12):
        tracks = tracker.update([moving_detection(step)])
        if tracks:
            ids.append(tracks[0].track_id)

    assert len(set(ids)) == 1

    before = tracker.report()[0].x1
    tracker.predict()
    assert tracker.report()[0].x1 > before


def test_iou_tracker_expires_and_issues_a_new_id():
    tracker = IoUTracker(max_age=3, min_hits=2)
    for step in range(6):
        tracker.update([moving_detection(step)])
    first = tracker.report()[0].track_id

    for _ in range(6):
        tracker.update([])
    assert tracker.active_ids == []

    for step in range(3):
        tracks = tracker.update([moving_detection(step)])
    assert tracks[0].track_id > first


def test_build_tracker_factory():
    assert isinstance(build_tracker("sort"), SortTracker)
    assert isinstance(build_tracker("iou"), IoUTracker)
    with pytest.raises(KeyError):
        build_tracker("deepsort")
