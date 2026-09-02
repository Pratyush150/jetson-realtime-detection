"""ROS 2 node publishing ``vision_msgs/Detection2DArray`` and an annotated image.

Import safety
-------------
This module imports cleanly with no ROS installed. That is not politeness, it
is a practical requirement: you want the same package on your laptop (where
you run the tests), on a Pi with a bare Python (where you run the detector
standalone), and on the robot (where ROS exists). Three separate import
guards are used, because in the field they fail independently — a ROS 2
install with ``rclpy`` but without ``vision_msgs`` is entirely normal, since
``vision_msgs`` is a separate apt package that people forget.

The message-shaping logic lives in :func:`detection_to_dict` and
:func:`detections_to_dicts`, which are plain Python. They are what the unit
tests exercise, so the field layout (centre-based ``BoundingBox2D``, not
corner-based) is verified without a ROS install anywhere near CI.

Coordinate convention, since it is the usual source of bugs: ROS
``BoundingBox2D`` is centre + size, while everything in this package is
corner-based ``xyxy``. Publishing ``x1, y1`` into ``center`` puts every box
up and to the left by half its size, which looks like a calibration error.

Packaging snippets (``package.xml``, ``setup.py``) are in ``docs/ROS2.md``
rather than in this repo's build, so that a plain ``pip install`` of this
package does not pretend to be an ament package it is not.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._compat import module_available
from .types import Detection, Track

LOGGER = logging.getLogger(__name__)

__all__ = [
    "RCLPY_AVAILABLE",
    "VISION_MSGS_AVAILABLE",
    "CV_BRIDGE_AVAILABLE",
    "ros2_status",
    "detection_to_dict",
    "detections_to_dicts",
    "EdgeVisionNode",
    "main",
]

try:  # pragma: no cover - environment dependent
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore

    RCLPY_AVAILABLE = True
except Exception:  # pragma: no cover
    rclpy = None  # type: ignore
    Node = object  # type: ignore
    RCLPY_AVAILABLE = False

# vision_msgs and cv_bridge are probed rather than imported. Both pull in
# compiled extensions (cv_bridge links OpenCV and Boost), and importing them
# eagerly means a mismatched numpy ABI on the robot spews errors the moment
# anything imports this package - even code paths that never touch ROS.
VISION_MSGS_AVAILABLE = module_available("vision_msgs")
CV_BRIDGE_AVAILABLE = module_available("cv_bridge")


def ros2_status() -> Dict[str, bool]:
    """What is importable right now. Useful in a startup log."""
    return {
        "rclpy": RCLPY_AVAILABLE,
        "vision_msgs": VISION_MSGS_AVAILABLE,
        "cv_bridge": CV_BRIDGE_AVAILABLE,
    }


# ---------------------------------------------------------------------------
# Message shaping (pure Python, testable without ROS)
# ---------------------------------------------------------------------------


def detection_to_dict(
    detection: Detection,
    track_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Shape one detection the way ``vision_msgs/Detection2D`` expects.

    ``bbox`` is centre + size, matching ``BoundingBox2D``. ``id`` carries the
    class label as a string, because ``ObjectHypothesis.class_id`` is a string
    in ``vision_msgs`` from Humble onward — passing an int there is a silent
    type error in Python that only surfaces at serialisation.
    """
    cx, cy = detection.center
    payload: Dict[str, Any] = {
        "bbox": {
            "center": {"x": float(cx), "y": float(cy), "theta": 0.0},
            "size_x": float(detection.width),
            "size_y": float(detection.height),
        },
        "results": [
            {
                "class_id": detection.class_name or str(detection.class_id),
                "score": float(detection.score),
            }
        ],
    }
    if track_id is not None:
        payload["id"] = str(track_id)
    return payload


def detections_to_dicts(
    items: Sequence[Any],
    frame_id: str = "camera",
    stamp: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Shape a ``Detection2DArray``. Accepts ``Detection`` or ``Track`` objects."""
    detections: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, Track):
            detections.append(detection_to_dict(item.as_detection(), item.track_id))
        else:
            detections.append(detection_to_dict(item))
    sec, nanosec = stamp if stamp is not None else (0, 0)
    return {
        "header": {
            "frame_id": frame_id,
            "stamp": {"sec": int(sec), "nanosec": int(nanosec)},
        },
        "detections": detections,
    }


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------


class EdgeVisionNode(Node):  # type: ignore[misc]
    """Publishes detections and an annotated image from an edgevision pipeline.

    Topics
    ------
    ``~/detections`` : ``vision_msgs/Detection2DArray``
    ``~/image_annotated`` : ``sensor_msgs/Image`` (only if ``cv_bridge`` exists)

    Parameters
    ----------
    source, backend, model, target_fps, skip, frame_id, publish_image

    A note on QoS: detections are published ``RELIABLE`` with depth 1 and the
    image ``BEST_EFFORT``. Losing an annotated frame over a flaky link is
    fine; silently dropping the detection that a behaviour tree is waiting on
    is not. Publishing images at all over Wi-Fi is usually a mistake — prefer
    the MJPEG preview in :mod:`edgevision.sinks`, which is one JPEG per frame
    instead of a raw ``sensor_msgs/Image``.
    """

    def __init__(self, **parameters: Any) -> None:  # pragma: no cover - needs ROS
        if not RCLPY_AVAILABLE:
            raise RuntimeError(
                "rclpy is not available. Source a ROS 2 setup.bash, or use "
                "edgevision.pipeline.Pipeline directly without ROS."
            )
        if not VISION_MSGS_AVAILABLE:
            raise RuntimeError(
                "vision_msgs is not available. Install it "
                "(sudo apt install ros-$ROS_DISTRO-vision-msgs); it is a separate "
                "package from the ROS 2 base install."
            )
        super().__init__("edgevision")

        from rclpy.qos import QoSProfile, ReliabilityPolicy  # type: ignore
        from sensor_msgs.msg import Image  # type: ignore
        from vision_msgs.msg import Detection2DArray  # type: ignore

        self.declare_parameter("source", str(parameters.get("source", "0")))
        self.declare_parameter("backend", str(parameters.get("backend", "auto")))
        self.declare_parameter("model", str(parameters.get("model", "")))
        self.declare_parameter("target_fps", float(parameters.get("target_fps", 30.0)))
        self.declare_parameter("skip", int(parameters.get("skip", 0)))
        self.declare_parameter("frame_id", str(parameters.get("frame_id", "camera")))
        self.declare_parameter("publish_image", bool(parameters.get("publish_image", False)))

        self.frame_id = self.get_parameter("frame_id").value
        self.publish_image = bool(self.get_parameter("publish_image").value) and CV_BRIDGE_AVAILABLE
        if self.get_parameter("publish_image").value and not CV_BRIDGE_AVAILABLE:
            self.get_logger().warn("cv_bridge missing; annotated image publishing disabled")

        reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        best_effort = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.detection_pub = self.create_publisher(Detection2DArray, "~/detections", reliable)
        self.image_pub = (
            self.create_publisher(Image, "~/image_annotated", best_effort)
            if self.publish_image
            else None
        )
        if self.publish_image:
            from cv_bridge import CvBridge  # type: ignore

            self.bridge = CvBridge()
        else:
            self.bridge = None

        from .backends import create_backend
        from .capture import FrameGrabber
        from .pipeline import Pipeline

        model = self.get_parameter("model").value or None
        detector = create_backend(self.get_parameter("backend").value, model_path=model)
        self.pipeline = Pipeline(
            detector,
            target_fps=float(self.get_parameter("target_fps").value),
            skip=int(self.get_parameter("skip").value),
            annotate_frames=self.publish_image,
        )
        self.grabber = FrameGrabber(self.get_parameter("source").value).start()

        # A short timer rather than a blocking loop, so rclpy keeps servicing
        # parameter and lifecycle callbacks.
        self.timer = self.create_timer(0.001, self._tick)
        self.get_logger().info(f"edgevision node running with {detector.name}")

    def _tick(self) -> None:  # pragma: no cover - needs ROS
        item = self.grabber.read_with_meta(timeout=0.05)
        if item is None:
            return
        result = self.pipeline.process(item[0])
        self.detection_pub.publish(self.build_detection_array(result.tracks))
        if self.image_pub is not None and result.annotated is not None:
            message = self.bridge.cv2_to_imgmsg(result.annotated, encoding="bgr8")
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self.frame_id
            self.image_pub.publish(message)

    def build_detection_array(self, tracks: Sequence[Track]) -> Any:  # pragma: no cover
        """Convert tracks into a populated ``Detection2DArray``."""
        from vision_msgs.msg import (  # type: ignore
            BoundingBox2D,
            Detection2D,
            Detection2DArray,
            ObjectHypothesisWithPose,
        )

        message = Detection2DArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        for track in tracks:
            shaped = detection_to_dict(track.as_detection(), track.track_id)
            detection = Detection2D()
            detection.header = message.header
            detection.id = shaped["id"]
            box = BoundingBox2D()
            box.center.position.x = shaped["bbox"]["center"]["x"]
            box.center.position.y = shaped["bbox"]["center"]["y"]
            box.center.theta = 0.0
            box.size_x = shaped["bbox"]["size_x"]
            box.size_y = shaped["bbox"]["size_y"]
            detection.bbox = box
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = shaped["results"][0]["class_id"]
            hypothesis.hypothesis.score = shaped["results"][0]["score"]
            detection.results = [hypothesis]
            message.detections.append(detection)
        return message

    def destroy_node(self) -> bool:  # pragma: no cover - needs ROS
        try:
            self.grabber.stop()
            self.pipeline.close()
        finally:
            return super().destroy_node()


def main(args: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - needs ROS
    """Console entry point: ``ros2 run edgevision detection_node``."""
    if not RCLPY_AVAILABLE:
        status = ", ".join(f"{k}={v}" for k, v in ros2_status().items())
        raise SystemExit(f"ROS 2 is not available in this environment ({status})")
    rclpy.init(args=list(args) if args is not None else None)
    node = EdgeVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0
