"""ROS 2 node: import safety and message shaping without a ROS install.

The message-shaping helpers are plain Python on purpose so the field layout
(centre-based BoundingBox2D, string class_id) is verified everywhere, not just
on a machine with a sourced ROS 2 workspace.
"""

from __future__ import annotations

import pytest

from edgevision import ros2_node
from edgevision.types import Detection, Track, TrackState


def test_module_imports_and_reports_status_regardless_of_ros():
    status = ros2_node.ros2_status()
    assert set(status) == {"rclpy", "vision_msgs", "cv_bridge"}
    assert all(isinstance(v, bool) for v in status.values())
    assert isinstance(ros2_node.RCLPY_AVAILABLE, bool)
    assert isinstance(ros2_node.VISION_MSGS_AVAILABLE, bool)


def test_detection_to_dict_uses_centre_and_size_not_corners():
    """BoundingBox2D is centre-based; publishing x1,y1 shifts every box."""
    detection = Detection(100, 200, 140, 260, 0.75, 0, "person")
    shaped = ros2_node.detection_to_dict(detection)

    assert shaped["bbox"]["center"]["x"] == pytest.approx(120.0)
    assert shaped["bbox"]["center"]["y"] == pytest.approx(230.0)
    assert shaped["bbox"]["size_x"] == pytest.approx(40.0)
    assert shaped["bbox"]["size_y"] == pytest.approx(60.0)
    assert shaped["bbox"]["center"]["theta"] == 0.0
    assert "id" not in shaped


def test_hypothesis_class_id_is_a_string():
    """vision_msgs uses a string class_id; an int fails only at serialisation."""
    shaped = ros2_node.detection_to_dict(Detection(0, 0, 10, 10, 0.5, 7, ""))
    result = shaped["results"][0]
    assert isinstance(result["class_id"], str)
    assert result["class_id"] == "7", "falls back to the numeric id when unlabelled"
    assert result["score"] == pytest.approx(0.5)

    labelled = ros2_node.detection_to_dict(Detection(0, 0, 10, 10, 0.5, 2, "car"))
    assert labelled["results"][0]["class_id"] == "car"


def test_tracks_carry_their_id_into_the_message():
    track = Track(42, 10, 10, 30, 50, 0.9, 0, "person", TrackState.CONFIRMED)
    array = ros2_node.detections_to_dicts([track], frame_id="camera_optical")

    assert array["header"]["frame_id"] == "camera_optical"
    assert array["header"]["stamp"] == {"sec": 0, "nanosec": 0}
    assert len(array["detections"]) == 1
    assert array["detections"][0]["id"] == "42"


def test_detections_and_tracks_can_be_mixed():
    items = [Detection(0, 0, 10, 10, 0.4), Track(3, 20, 20, 40, 40, 0.6)]
    array = ros2_node.detections_to_dicts(items, stamp=(12, 500))

    assert array["header"]["stamp"] == {"sec": 12, "nanosec": 500}
    assert "id" not in array["detections"][0]
    assert array["detections"][1]["id"] == "3"


def test_empty_input_produces_an_empty_array():
    array = ros2_node.detections_to_dicts([])
    assert array["detections"] == []
    assert array["header"]["frame_id"] == "camera"


@pytest.mark.skipif(
    ros2_node.RCLPY_AVAILABLE and ros2_node.VISION_MSGS_AVAILABLE,
    reason="ROS 2 with vision_msgs is available here",
)
def test_node_construction_fails_with_a_useful_message():
    with pytest.raises(RuntimeError) as excinfo:
        ros2_node.EdgeVisionNode()
    message = str(excinfo.value)
    assert "rclpy" in message or "vision_msgs" in message
