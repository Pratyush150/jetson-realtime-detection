"""Multi-object tracking: SORT-style Kalman + IoU, and a cheap IoU-only fallback.

Why track at all if the detector already gives you boxes?

1. **Identity.** "Three people" is not the same information as "person #7 has
   been in the frame for 4 seconds moving left". Counting, dwell time, and any
   kind of follow-me behaviour need an ID that survives frames.
2. **Frames you did not run the detector on.** This is the load-bearing reason
   on an edge board. If inference costs 100 ms and frames arrive every 33 ms,
   you can only detect on every third or fourth frame. The tracker's motion
   model fills the gap: boxes keep moving, the overlay stays smooth, and the
   consumer of your output never sees a 3 FPS stutter.
3. **Detector flicker.** A detector that drops an object for one frame is
   normal. A tracker with ``max_age`` coasts through it instead of destroying
   and re-creating the ID.

Two trackers are provided:

``SortTracker``
    Constant-velocity Kalman filter per track, IoU cost matrix, optimal
    (Hungarian) assignment. This is what you want by default. The Kalman step
    is a handful of 7x7 matrix operations per track — negligible next to
    inference, even on a Pi.

``IoUTracker``
    No filter, greedy assignment, boxes frozen (or linearly extrapolated)
    between detections. For very weak hardware or very high object counts
    where you want the tracking cost to be provably trivial. It is measurably
    worse under occlusion and fast motion; that is the trade.

``scipy`` is used for the assignment when present, but a self-contained
Hungarian implementation is included and is verified against scipy in the
test suite, so the tracker has no hard dependency beyond numpy.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .postprocess import iou_matrix
from .types import Detection, Track, TrackState

try:  # pragma: no cover - environment dependent
    from scipy.optimize import linear_sum_assignment as _scipy_lsa  # type: ignore

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    _scipy_lsa = None  # type: ignore
    SCIPY_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

__all__ = [
    "hungarian",
    "linear_assignment",
    "associate",
    "KalmanBoxTracker",
    "SortTracker",
    "IoUTracker",
    "SCIPY_AVAILABLE",
]


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


def hungarian(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Optimal rectangular assignment, minimising total cost.

    Self-contained O(n^2 m) shortest-augmenting-path implementation (the
    Jonker-Volgenant / e-maxx formulation) so the tracker works on a board
    where nobody wants to compile scipy. Returns ``(row_indices,
    col_indices)`` exactly like ``scipy.optimize.linear_sum_assignment``.

    Greedy "match the best pair, then the next best" is *not* equivalent: a
    locally optimal pairing can force two later tracks to swap boxes, which
    shows up as two IDs trading places every few frames. Optimal assignment
    is cheap at these matrix sizes and removes that failure mode entirely.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("cost must be a 2-D matrix")
    if matrix.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("cost matrix must be finite")

    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T
    n_rows, n_cols = matrix.shape

    # 1-based bookkeeping arrays, as in the classic formulation.
    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    parent = np.zeros(n_cols + 1, dtype=np.int64)   # parent[j] = row matched to col j
    way = np.zeros(n_cols + 1, dtype=np.int64)      # alternating-path predecessor

    for row in range(1, n_rows + 1):
        parent[0] = row
        col0 = 0
        min_values = np.full(n_cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)

        while True:
            used[col0] = True
            row0 = parent[col0]
            free = ~used[1:]
            if not free.any():
                break
            candidate = matrix[row0 - 1] - u[row0] - v[1:]
            improved = free & (candidate < min_values[1:])
            if improved.any():
                min_values[1:][improved] = candidate[improved]
                way[1:][improved] = col0
            masked = np.where(free, min_values[1:], np.inf)
            col1 = int(np.argmin(masked)) + 1
            delta = float(masked[col1 - 1])

            u[parent[used]] += delta
            v[used] -= delta
            min_values[~used] -= delta

            col0 = col1
            if parent[col0] == 0:
                break

        while col0:
            col1 = way[col0]
            parent[col0] = parent[col1]
            col0 = col1

    rows: List[int] = []
    cols: List[int] = []
    for col in range(1, n_cols + 1):
        if parent[col] != 0:
            rows.append(int(parent[col]) - 1)
            cols.append(col - 1)

    row_ind = np.asarray(rows, dtype=np.int64)
    col_ind = np.asarray(cols, dtype=np.int64)
    if transposed:
        row_ind, col_ind = col_ind, row_ind
    order = np.argsort(row_ind, kind="stable")
    return row_ind[order], col_ind[order]


def linear_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the assignment problem, using scipy when it is installed."""
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if SCIPY_AVAILABLE:  # pragma: no branch - trivial dispatch
        row_ind, col_ind = _scipy_lsa(matrix)
        return np.asarray(row_ind, dtype=np.int64), np.asarray(col_ind, dtype=np.int64)
    return hungarian(matrix)


def associate(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    iou_threshold: float = 0.3,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match tracks to detections by IoU.

    Returns ``(matches, unmatched_track_indices, unmatched_detection_indices)``.

    The gate is applied *after* the optimal assignment, not before: the solver
    minimises total cost over everything, then any pair whose IoU is below the
    threshold is broken apart. Filtering first can force the solver into a
    worse global arrangement.
    """
    track_boxes = np.asarray(track_boxes, dtype=np.float32).reshape(-1, 4)
    detection_boxes = np.asarray(detection_boxes, dtype=np.float32).reshape(-1, 4)

    if len(track_boxes) == 0 or len(detection_boxes) == 0:
        return [], list(range(len(track_boxes))), list(range(len(detection_boxes)))

    ious = iou_matrix(track_boxes, detection_boxes)
    row_ind, col_ind = linear_assignment(-ious)

    matches: List[Tuple[int, int]] = []
    matched_tracks = set()
    matched_dets = set()
    for r, c in zip(row_ind, col_ind):
        if ious[r, c] < iou_threshold:
            continue
        matches.append((int(r), int(c)))
        matched_tracks.add(int(r))
        matched_dets.add(int(c))

    unmatched_tracks = [i for i in range(len(track_boxes)) if i not in matched_tracks]
    unmatched_dets = [j for j in range(len(detection_boxes)) if j not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------


def _bbox_to_z(bbox: Sequence[float]) -> np.ndarray:
    """``[x1,y1,x2,y2]`` -> ``[cx, cy, area, aspect]`` measurement vector."""
    width = max(1e-6, float(bbox[2]) - float(bbox[0]))
    height = max(1e-6, float(bbox[3]) - float(bbox[1]))
    cx = float(bbox[0]) + width / 2.0
    cy = float(bbox[1]) + height / 2.0
    return np.array([cx, cy, width * height, width / height], dtype=np.float64).reshape(4, 1)


def _z_to_bbox(state: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_bbox_to_z` from the first four state elements."""
    area = max(1e-6, float(state[2]))
    aspect = max(1e-6, float(state[3]))
    width = np.sqrt(area * aspect)
    height = area / width
    cx, cy = float(state[0]), float(state[1])
    return np.array(
        [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0],
        dtype=np.float32,
    )


class KalmanBoxTracker:
    """Constant-velocity Kalman filter over ``[cx, cy, area, aspect]``.

    State is ``[cx, cy, s, r, vx, vy, vs]``. Aspect ratio has no velocity term
    on purpose: a rigid object's aspect ratio should not drift, and giving it
    one mostly lets boxes stretch during occlusions.

    Tracking in area/aspect rather than width/height means a target moving
    towards the camera grows smoothly instead of the filter fighting two
    correlated dimensions.
    """

    def __init__(
        self,
        bbox: Sequence[float],
        track_id: int,
        score: float = 0.0,
        class_id: int = 0,
        class_name: str = "",
        process_noise_scale: float = 1.0,
        measurement_noise_scale: float = 1.0,
    ) -> None:
        self.track_id = int(track_id)
        self.score = float(score)
        self.class_id = int(class_id)
        self.class_name = class_name

        self.state = TrackState.TENTATIVE
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0

        # F: constant velocity on cx, cy and area.
        self._F = np.eye(7, dtype=np.float64)
        self._F[0, 4] = self._F[1, 5] = self._F[2, 6] = 1.0
        # H: we observe the box, not its velocity.
        self._H = np.zeros((4, 7), dtype=np.float64)
        self._H[0, 0] = self._H[1, 1] = self._H[2, 2] = self._H[3, 3] = 1.0

        # Area and aspect are noisier measurements than the centre, because a
        # detector's box edges jitter more than its centroid.
        self._R = np.eye(4, dtype=np.float64) * measurement_noise_scale
        self._R[2, 2] *= 10.0
        self._R[3, 3] *= 10.0

        self._P = np.eye(7, dtype=np.float64) * 10.0
        # Velocities are completely unknown at birth: huge initial covariance.
        self._P[4:, 4:] *= 1000.0

        self._Q = np.eye(7, dtype=np.float64) * process_noise_scale
        self._Q[4:, 4:] *= 0.01
        self._Q[6, 6] *= 0.01

        self._x = np.zeros((7, 1), dtype=np.float64)
        self._x[:4] = _bbox_to_z(bbox)

    # -- filter ------------------------------------------------------------

    def predict(self) -> np.ndarray:
        """Advance one time step and return the predicted ``xyxy`` box."""
        # Keep area non-negative: a shrinking track can otherwise predict a
        # negative area and produce a NaN box.
        if self._x[2, 0] + self._x[6, 0] <= 0:
            self._x[6, 0] = 0.0
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 1:
            self.hit_streak = 0
        return _z_to_bbox(self._x[:, 0])

    def update(self, bbox: Sequence[float], score: float = 0.0, class_id: Optional[int] = None,
               class_name: Optional[str] = None) -> None:
        """Correct the state with a new detection."""
        measurement = _bbox_to_z(bbox)
        residual = measurement - self._H @ self._x
        innovation_cov = self._H @ self._P @ self._H.T + self._R
        gain = self._P @ self._H.T @ np.linalg.inv(innovation_cov)
        self._x = self._x + gain @ residual
        identity = np.eye(7, dtype=np.float64)
        self._P = (identity - gain @ self._H) @ self._P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = float(score)
        if class_id is not None:
            self.class_id = int(class_id)
        if class_name is not None:
            self.class_name = class_name

    def coast(self) -> np.ndarray:
        """Advance the motion model for a frame we deliberately did not detect on.

        Identical to :meth:`predict` except that it does not count as a missed
        observation. This distinction is what makes adaptive frame skipping
        safe: skipping frames must not age tracks toward deletion, because we
        chose not to look, and the object did not disappear.
        """
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self.age += 1
        return _z_to_bbox(self._x[:, 0])

    # -- accessors ---------------------------------------------------------

    @property
    def bbox(self) -> np.ndarray:
        return _z_to_bbox(self._x[:, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        return (float(self._x[4, 0]), float(self._x[5, 0]))

    def to_track(self) -> Track:
        x1, y1, x2, y2 = self.bbox
        return Track(
            track_id=self.track_id,
            x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
            score=self.score,
            class_id=self.class_id,
            class_name=self.class_name,
            state=self.state,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            velocity=self.velocity,
        )


# ---------------------------------------------------------------------------
# Trackers
# ---------------------------------------------------------------------------


class _BaseTracker:
    """Shared ID allocation and reporting policy."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3,
                 tentative_max_age: int = 1) -> None:
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.tentative_max_age = int(tentative_max_age)
        self.frame_count = 0
        self._next_id = 1

    def reset(self) -> None:
        """Drop all tracks. IDs keep counting up; they are never recycled."""
        self.frame_count = 0

    def _allocate_id(self) -> int:
        """IDs are monotonic and never reused.

        Recycling an ID after a track dies is how you get a downstream
        consumer confidently reporting that the object which just left frame
        came back, when in fact it is a different object entirely.
        """
        track_id = self._next_id
        self._next_id += 1
        return track_id


class SortTracker(_BaseTracker):
    """SORT: Kalman prediction + IoU association + Hungarian assignment.

    Parameters
    ----------
    max_age:
        Frames a confirmed track may coast without an observation before it is
        deleted. Too low and every brief occlusion costs an ID; too high and
        a stale box lingers over empty background. At 30 FPS, 15-30 is sane.
    min_hits:
        Consecutive detections before a track is reported. Suppresses
        single-frame false positives from spawning visible IDs.
    iou_threshold:
        Minimum IoU for a track/detection pair to be considered the same
        object. Lower it for fast motion or a low detection rate (frame
        skipping makes objects move further between observations).
    class_aware:
        If True, a track will only associate with detections of its own class.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        tentative_max_age: int = 1,
        class_aware: bool = False,
        process_noise_scale: float = 1.0,
        measurement_noise_scale: float = 1.0,
    ) -> None:
        super().__init__(max_age, min_hits, iou_threshold, tentative_max_age)
        self.class_aware = bool(class_aware)
        self.process_noise_scale = float(process_noise_scale)
        self.measurement_noise_scale = float(measurement_noise_scale)
        self.tracks: List[KalmanBoxTracker] = []

    def reset(self) -> None:
        super().reset()
        self.tracks = []

    # -- main API ----------------------------------------------------------

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        """Advance one frame with a fresh set of detections."""
        self.frame_count += 1
        detections = list(detections or [])

        predicted: List[np.ndarray] = []
        alive: List[KalmanBoxTracker] = []
        for track in self.tracks:
            box = track.predict()
            if np.all(np.isfinite(box)):
                predicted.append(box)
                alive.append(track)
            else:  # pragma: no cover - guards against a diverged filter
                LOGGER.debug("dropping track %d with non-finite state", track.track_id)
        self.tracks = alive

        det_boxes = np.array(
            [[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32
        ).reshape(-1, 4)
        track_boxes = np.array(predicted, dtype=np.float32).reshape(-1, 4)

        cost_mask = None
        if self.class_aware and len(self.tracks) and len(detections):
            track_classes = np.array([t.class_id for t in self.tracks])[:, None]
            det_classes = np.array([d.class_id for d in detections])[None, :]
            cost_mask = track_classes == det_classes

        matches, unmatched_tracks, unmatched_dets = self._associate(
            track_boxes, det_boxes, cost_mask
        )

        for track_idx, det_idx in matches:
            det = detections[det_idx]
            self.tracks[track_idx].update(
                (det.x1, det.y1, det.x2, det.y2), det.score, det.class_id, det.class_name
            )

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self.tracks.append(
                KalmanBoxTracker(
                    (det.x1, det.y1, det.x2, det.y2),
                    self._allocate_id(),
                    det.score,
                    det.class_id,
                    det.class_name,
                    self.process_noise_scale,
                    self.measurement_noise_scale,
                )
            )

        self._update_states()
        self._prune()
        return self.report()

    def predict(self) -> List[Track]:
        """Coast every track one frame without running the detector.

        Call this on every frame the pipeline *skipped*. It moves boxes along
        their velocity so the overlay stays smooth at capture rate, without
        aging tracks toward deletion.
        """
        for track in self.tracks:
            track.coast()
        return self.report()

    # -- internals ---------------------------------------------------------

    def _associate(
        self,
        track_boxes: np.ndarray,
        det_boxes: np.ndarray,
        cost_mask: Optional[np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(track_boxes) == 0 or len(det_boxes) == 0:
            return [], list(range(len(track_boxes))), list(range(len(det_boxes)))

        ious = iou_matrix(track_boxes, det_boxes)
        if cost_mask is not None:
            ious = np.where(cost_mask, ious, 0.0)

        row_ind, col_ind = linear_assignment(-ious)
        matches: List[Tuple[int, int]] = []
        matched_tracks, matched_dets = set(), set()
        for r, c in zip(row_ind, col_ind):
            if ious[r, c] < self.iou_threshold:
                continue
            matches.append((int(r), int(c)))
            matched_tracks.add(int(r))
            matched_dets.add(int(c))
        unmatched_tracks = [i for i in range(len(track_boxes)) if i not in matched_tracks]
        unmatched_dets = [j for j in range(len(det_boxes)) if j not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets

    def _update_states(self) -> None:
        for track in self.tracks:
            if track.time_since_update == 0:
                if track.state is TrackState.TENTATIVE:
                    # Confirm on hit count, or immediately during the first few
                    # frames of a run so a stream does not start with a
                    # min_hits-long blind spot.
                    if track.hits >= self.min_hits or self.frame_count <= self.min_hits:
                        track.state = TrackState.CONFIRMED
                else:
                    track.state = TrackState.CONFIRMED
            elif track.state is TrackState.CONFIRMED:
                track.state = TrackState.LOST

    def _prune(self) -> None:
        survivors: List[KalmanBoxTracker] = []
        for track in self.tracks:
            if track.state is TrackState.TENTATIVE:
                limit = self.tentative_max_age
            else:
                limit = self.max_age
            if track.time_since_update > limit:
                track.state = TrackState.REMOVED
                LOGGER.debug(
                    "removing track %d after %d frames without an observation",
                    track.track_id, track.time_since_update,
                )
                continue
            survivors.append(track)
        self.tracks = survivors

    def report(self, include_lost: bool = True) -> List[Track]:
        """Tracks worth showing downstream, ordered by ID."""
        out: List[Track] = []
        for track in self.tracks:
            if track.state is TrackState.TENTATIVE:
                continue
            if not include_lost and track.state is TrackState.LOST:
                continue
            out.append(track.to_track())
        out.sort(key=lambda t: t.track_id)
        return out

    @property
    def active_ids(self) -> List[int]:
        return [t.track_id for t in self.tracks]


class _IoUTrack:
    """A track with no filter: last box plus a smoothed velocity estimate."""

    __slots__ = (
        "track_id", "box", "score", "class_id", "class_name",
        "state", "age", "hits", "time_since_update", "vx", "vy",
    )

    def __init__(self, track_id: int, det: Detection) -> None:
        self.track_id = track_id
        self.box = np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float32)
        self.score = det.score
        self.class_id = det.class_id
        self.class_name = det.class_name
        self.state = TrackState.TENTATIVE
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.vx = 0.0
        self.vy = 0.0

    def update(self, det: Detection, smoothing: float) -> None:
        new_box = np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float32)
        dx = float((new_box[0] + new_box[2]) / 2 - (self.box[0] + self.box[2]) / 2)
        dy = float((new_box[1] + new_box[3]) / 2 - (self.box[1] + self.box[3]) / 2)
        self.vx = smoothing * self.vx + (1.0 - smoothing) * dx
        self.vy = smoothing * self.vy + (1.0 - smoothing) * dy
        self.box = new_box
        self.score = det.score
        self.class_id = det.class_id
        self.class_name = det.class_name
        self.hits += 1
        self.time_since_update = 0

    def extrapolate(self) -> None:
        self.box = self.box + np.array(
            [self.vx, self.vy, self.vx, self.vy], dtype=np.float32
        )

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            x1=float(self.box[0]), y1=float(self.box[1]),
            x2=float(self.box[2]), y2=float(self.box[3]),
            score=self.score, class_id=self.class_id, class_name=self.class_name,
            state=self.state, age=self.age, hits=self.hits,
            time_since_update=self.time_since_update,
            velocity=(self.vx, self.vy),
        )


class IoUTracker(_BaseTracker):
    """Greedy IoU tracker for hardware that cannot spare the cycles.

    No Kalman filter, no optimal assignment: sort candidate pairs by IoU and
    take them greedily. Costs a few microseconds per frame regardless of
    object count, at the price of worse behaviour when two objects cross or
    when a detection is missed during fast motion.

    Use it when the tracker is genuinely on your critical path (hundreds of
    objects, or a Pi Zero class board). Otherwise use :class:`SortTracker`;
    on any board that can run a detector at all, the Kalman step is noise.
    """

    def __init__(
        self,
        max_age: int = 10,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
        tentative_max_age: int = 1,
        velocity_smoothing: float = 0.5,
        extrapolate: bool = True,
    ) -> None:
        super().__init__(max_age, min_hits, iou_threshold, tentative_max_age)
        self.velocity_smoothing = float(velocity_smoothing)
        self.extrapolate_between = bool(extrapolate)
        self.tracks: List[_IoUTrack] = []

    def reset(self) -> None:
        super().reset()
        self.tracks = []

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        self.frame_count += 1
        detections = list(detections or [])

        for track in self.tracks:
            track.age += 1

        matches = self._greedy_match(detections)
        matched_tracks = {t for t, _ in matches}
        matched_dets = {d for _, d in matches}

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx], self.velocity_smoothing)

        for idx, track in enumerate(self.tracks):
            if idx not in matched_tracks:
                track.time_since_update += 1
                if self.extrapolate_between:
                    track.extrapolate()

        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                self.tracks.append(_IoUTrack(self._allocate_id(), det))

        self._update_states()
        self._prune()
        return self.report()

    def predict(self) -> List[Track]:
        """Extrapolate every track one frame without a detector run."""
        for track in self.tracks:
            track.age += 1
            if self.extrapolate_between:
                track.extrapolate()
        return self.report()

    def _greedy_match(self, detections: Sequence[Detection]) -> List[Tuple[int, int]]:
        if not self.tracks or not detections:
            return []
        track_boxes = np.array([t.box for t in self.tracks], dtype=np.float32)
        det_boxes = np.array(
            [[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32
        )
        ious = iou_matrix(track_boxes, det_boxes)

        pairs = [
            (float(ious[i, j]), i, j)
            for i in range(ious.shape[0])
            for j in range(ious.shape[1])
            if ious[i, j] >= self.iou_threshold
        ]
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

        matches: List[Tuple[int, int]] = []
        used_tracks, used_dets = set(), set()
        for _, i, j in pairs:
            if i in used_tracks or j in used_dets:
                continue
            matches.append((i, j))
            used_tracks.add(i)
            used_dets.add(j)
        return matches

    def _update_states(self) -> None:
        for track in self.tracks:
            if track.time_since_update == 0:
                if track.state is TrackState.TENTATIVE:
                    if track.hits >= self.min_hits or self.frame_count <= self.min_hits:
                        track.state = TrackState.CONFIRMED
                else:
                    track.state = TrackState.CONFIRMED
            elif track.state is TrackState.CONFIRMED:
                track.state = TrackState.LOST

    def _prune(self) -> None:
        survivors: List[_IoUTrack] = []
        for track in self.tracks:
            limit = (
                self.tentative_max_age
                if track.state is TrackState.TENTATIVE
                else self.max_age
            )
            if track.time_since_update > limit:
                track.state = TrackState.REMOVED
                continue
            survivors.append(track)
        self.tracks = survivors

    def report(self, include_lost: bool = True) -> List[Track]:
        out = [
            t.to_track()
            for t in self.tracks
            if t.state is not TrackState.TENTATIVE
            and (include_lost or t.state is not TrackState.LOST)
        ]
        out.sort(key=lambda t: t.track_id)
        return out

    @property
    def active_ids(self) -> List[int]:
        return [t.track_id for t in self.tracks]


def build_tracker(kind: str = "sort", **kwargs) -> _BaseTracker:
    """Factory used by the CLI: ``sort`` or ``iou``."""
    kind = (kind or "sort").lower()
    if kind == "sort":
        return SortTracker(**kwargs)
    if kind in ("iou", "simple"):
        return IoUTracker(**kwargs)
    raise KeyError(f"unknown tracker {kind!r}; expected 'sort' or 'iou'")
