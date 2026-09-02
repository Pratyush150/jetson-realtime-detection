# Running edgevision as a ROS 2 node

`edgevision/ros2_node.py` publishes `vision_msgs/Detection2DArray` and,
optionally, an annotated `sensor_msgs/Image`. It imports cleanly with no ROS
installed, so the same package works on your laptop, on a bare Pi, and on the
robot.

## Dependencies

```bash
sudo apt install ros-$ROS_DISTRO-vision-msgs ros-$ROS_DISTRO-cv-bridge
```

`vision_msgs` is a separate package from the ROS 2 base install and is the one
people forget. `edgevision.ros2_node.ros2_status()` tells you what is actually
importable:

```python
from edgevision.ros2_node import ros2_status
print(ros2_status())   # {'rclpy': True, 'vision_msgs': False, 'cv_bridge': True}
```

## Topics

| Topic | Type | QoS |
|---|---|---|
| `~/detections` | `vision_msgs/Detection2DArray` | reliable, depth 1 |
| `~/image_annotated` | `sensor_msgs/Image` | best effort, depth 1 |

Detections are **reliable**: silently dropping the detection a behaviour tree
is waiting on is not acceptable. The annotated image is **best effort**: losing
a preview frame is fine, and reliable delivery of images over a flaky link
causes head-of-line blocking that delays everything else.

Publishing raw `sensor_msgs/Image` over Wi-Fi is usually a mistake — a 720p
BGR frame is 2.7 MB. Prefer the MJPEG preview server in `edgevision.sinks`
(one JPEG per frame, viewable in any browser) and leave `publish_image` off.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `source` | `"0"` | camera index, RTSP URL, GStreamer pipeline or file |
| `backend` | `"auto"` | `tensorrt` / `hailo` / `onnxruntime` / `ultralytics` / `mock` |
| `model` | `""` | path to the engine / ONNX / weights |
| `target_fps` | `30.0` | output rate the adaptive skipper aims at |
| `skip` | `0` | fixed detection interval; 0 = adaptive |
| `frame_id` | `"camera"` | header frame id |
| `publish_image` | `false` | publish the annotated image as well |

## Coordinate convention

`vision_msgs/BoundingBox2D` is **centre + size**. Everything inside edgevision
is corner-based `xyxy`. Publishing `x1, y1` into `center` shifts every box up
and to the left by half its size, which looks like a camera calibration error
and gets debugged as one. `detection_to_dict()` does the conversion, and
`tests/test_ros2_node.py` asserts it.

`ObjectHypothesis.class_id` is a **string** in Humble and later. Assigning an
int is a silent type error in Python that only surfaces at serialisation time.

## Packaging it as an ament package

This repo is a plain Python package, not an ament one, so that `pip install`
works normally. To run it under `ros2 run`, create a small ament wrapper
package that depends on it. Below are the two files you need.

`package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>edgevision_ros</name>
  <version>0.1.0</version>
  <description>Real-time detection and tracking for edge boards, as a ROS 2 node.</description>
  <maintainer email="you@example.com">Pratyush Vatsa</maintainer>
  <license>MIT</license>

  <exec_depend>rclpy</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>vision_msgs</exec_depend>
  <exec_depend>cv_bridge</exec_depend>
  <exec_depend>python3-numpy</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

`setup.py`:

```python
from setuptools import setup

package_name = "edgevision_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/detection.launch.py"]),
    ],
    install_requires=["setuptools", "edgevision"],
    zip_safe=True,
    maintainer="Pratyush Vatsa",
    license="MIT",
    entry_points={
        "console_scripts": [
            "detection_node = edgevision.ros2_node:main",
        ],
    },
)
```

`launch/detection.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="0"),
        DeclareLaunchArgument("model", default_value=""),
        DeclareLaunchArgument("backend", default_value="auto"),
        DeclareLaunchArgument("target_fps", default_value="30.0"),
        Node(
            package="edgevision_ros",
            executable="detection_node",
            name="edgevision",
            output="screen",
            parameters=[{
                "source": LaunchConfiguration("source"),
                "model": LaunchConfiguration("model"),
                "backend": LaunchConfiguration("backend"),
                "target_fps": LaunchConfiguration("target_fps"),
                "publish_image": False,
            }],
        ),
    ])
```

Then:

```bash
colcon build --packages-select edgevision_ros
source install/setup.bash
ros2 launch edgevision_ros detection.launch.py source:=0 model:=yolov8n.engine
ros2 topic echo /edgevision/detections
```

## A note on the pytest plugins

ROS 2 (Humble and older) ships pytest plugins (`launch_testing`, `launch_ros`)
that are incompatible with pytest >= 8 and are auto-loaded from the ambient
environment. On a machine with ROS sourced, they break collection before a
single test runs. This repo's `pyproject.toml` disables them by name:

```toml
addopts = "-p no:launch_testing -p no:launch_ros"
```

That is harmless when ROS is absent.
