# jetson-realtime-detection

Real-time object detection and tracking for constrained edge boards — Jetson
Nano / Orin Nano / Xavier NX, Raspberry Pi 4 / 5, Hailo-8.

This is not a wrapper around `model.predict()`. The hard part of edge
perception is not calling a detector; it is everything around it: not falling
seconds behind the camera, not stalling when the board heats up, and getting a
useful output rate out of hardware that cannot run the model on every frame.

## The problem

You put YOLO on a Jetson Nano. It reports 10 FPS. Then:

- **Detections lag reality by seconds.** `cv2.VideoCapture.read()` pulls from a
  driver-side queue. If inference takes 120 ms and frames arrive every 33 ms,
  the frames you did not consume are buffered, and every `read()` hands you the
  *oldest* one. Latency grows without bound. No model optimisation fixes this —
  the pipeline is keeping up on throughput and failing catastrophically on
  latency.
- **10 FPS is unusable** for anything that has to react, even though the model
  is running as fast as the silicon allows.
- **After thirty seconds it drops to 6 FPS** and nobody knows why.
- **The camera drops off the bus** on a voltage dip and every subsequent read
  returns `False`, forever.
- **Distant objects are invisible**, because a 24 px target in a 1080p frame is
  8 px after downsampling to 640x640 — smaller than the network's stride.

Every one of those is a solved problem. This repo is the solutions, wired
together and tested.

## What it does about them

| Problem | Approach |
|---|---|
| Stale frames | Reader thread drains the driver queue into a **1-deep latest-frame slot**; older frames are overwritten and counted, never queued |
| Inference slower than capture | **Adaptive frame skipping**: detect every Nth frame, propagate boxes with the tracker in between, N chosen at runtime from measured timings |
| "Where is my time going?" | Per-stage p50/p90/p99 timing — capture, preprocess, inference, track, draw, sink |
| FPS collapse after 30 s | Throughput watchdog + thermal zone reporting, with the diagnostic commands documented |
| Camera disconnect | Reconnect with exponential backoff and drop/failure counters |
| Small distant objects | Tiled inference with box remapping and cross-tile NMS |
| Different hardware per board | Backend registry with runtime capability probing that **logs why** it chose what it chose |
| Headless board | Stdlib-only MJPEG server — view it from a laptop browser, no extra dependencies |

## Quickstart

```bash
git clone https://github.com/Pratyush150/jetson-realtime-detection
cd jetson-realtime-detection
pip install -r requirements.txt      # numpy only
python3 -m pytest -q                 # 187 tests, no hardware, no network
python3 examples/01_quickstart.py    # full pipeline on synthetic frames
```

Then on real hardware:

```bash
pip install opencv-python onnxruntime      # or your platform's runtime

# What can this board actually run, and why?
python3 examples/06_probe_backends.py yolov8n.onnx

# Benchmark the backend in isolation (warmup discarded automatically)
tools/edgevision-bench --backend auto --model yolov8n.onnx --frames 200

# Run the full pipeline with an MJPEG preview
tools/edgevision-run --source 0 --backend auto --model yolov8n.onnx --preview
```

`--skip 0` (the default) adapts the detection interval at runtime. Pass
`--skip 3` to pin it.

### As a library

```python
from edgevision import FrameGrabber, Pipeline, SortTracker, create_backend
from edgevision.sinks import JsonLinesSink, MjpegPreviewServer

detector = create_backend("auto", model_path="yolov8n.engine")
detector.warmup(5)                       # never time the first inference

pipeline = Pipeline(
    detector,
    tracker=SortTracker(max_age=30, min_hits=3),
    sinks=[JsonLinesSink("out.jsonl"), MjpegPreviewServer(port=8090).start()],
    target_fps=30.0,
    skip=0,                              # 0 = adaptive
)

grabber = FrameGrabber("rtsp://10.0.0.5:554/live", reconnect=True).start()
pipeline.run(grabber, max_seconds=60)
print(pipeline.format_report())
```

## Architecture

```
                    capture thread                     pipeline thread
  ┌──────────┐   ┌──────────────────┐   ┌──────────┐
  │  camera  │──►│  FrameGrabber    │──►│ 1-deep   │──► newest frame only
  │ CSI/USB  │   │  read() in a     │   │ latest-  │    (older ones are
  │ RTSP/file│   │  tight loop      │   │ frame    │     dropped + counted)
  └──────────┘   └──────────────────┘   │ buffer   │
                  reconnect + backoff   └────┬─────┘
                                             │
        ┌────────────────────────────────────┘
        ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │  Pipeline.process(frame)                                          │
  │                                                                   │
  │   every Nth frame          all other frames                       │
  │   ─────────────────        ────────────────                       │
  │   letterbox                                                       │
  │      ▼                                                            │
  │   Detector.infer  ─┐                                              │
  │      ▼             │                                              │
  │   decode + NMS     │                                              │
  │      ▼             │                                              │
  │   un-letterbox     │                                              │
  │      ▼             ▼                                              │
  │   tracker.update(dets)     tracker.predict()   ◄── constant-      │
  │      │                          │                  velocity       │
  │      └──────────┬───────────────┘                  Kalman coast   │
  │                 ▼                                                 │
  │              annotate ──► sinks                                   │
  │                 │                                                 │
  │      AdaptiveFrameSkipper.update(inference_ms, overhead_ms) ──► N  │
  └───────────────────────────────────────────────────────────────────┘
        │                │              │              │
        ▼                ▼              ▼              ▼
   video file      JSON lines      snapshots     MJPEG preview
                                                (any browser)

  Backend registry (probed at runtime, best available wins, choice logged)
  ┌──────────┬────────┬─────────────┬─────────────┬──────┐
  │ TensorRT │ Hailo  │ ONNXRuntime │ Ultralytics │ Mock │
  │   (80)   │  (70)  │    (40+)    │    (20)     │(-100)│
  └──────────┴────────┴─────────────┴─────────────┴──────┘
```

## The centrepiece: adaptive frame skipping

With a detection interval of `N`, the average cost of one output frame is:

```
cost(N) = overhead + inference / N
```

where `overhead` is what you pay every frame (tracking, annotation, sinks) and
`inference` is paid once per `N` frames. To sustain a target rate:

```
N >= inference / (1/target_fps - overhead)
```

`AdaptiveFrameSkipper` measures both terms with an EMA and solves this every
frame. Concretely (from `examples/02_adaptive_skip.py`, targeting 30 FPS):

| Case | inference | overhead | skip | naive FPS | with skipping |
|---|---|---|---|---|---|
| fast engine | 12 ms | 3 ms | 1 | 66.7 | 66.7 |
| mid-range engine | 45 ms | 4 ms | 2 | 20.4 | 37.7 |
| heavy model, small board | 100 ms | 5 ms | 4 | 9.5 | 33.3 |

Skipping does not make inference faster. It amortises it, and the tracker's
motion model carries the boxes in between — so the *output* keeps up with the
camera and stays close to real time.

**What you give up, stated honestly:** an object that first *appears* between
detection frames is not seen until the next detection, so worst-case detection
latency is `N x frame_period`. For erratic close-range motion, lower the
interval or the input resolution instead. For people, vehicles and boats it is
the highest-leverage change available.

## Modules

| Module | What it is for |
|---|---|
| `capture.py` | Threaded grabber, 1-deep latest-frame buffer, USB / CSI / RTSP / file, reconnect, drop counters |
| `backends/` | `Detector` ABC + TensorRT, Hailo, ONNX Runtime, Ultralytics, Mock; registry with capability probing |
| `preprocess.py` | Letterbox and its exact inverse (pure numpy fallback, cv2 when available) |
| `postprocess.py` | YOLOv5/v8 head decoding, IoU, class-aware NMS |
| `tracker.py` | SORT-style Kalman + Hungarian tracker, plus a cheap IoU-only tracker; Hungarian implemented from scratch |
| `pipeline.py` | Orchestration, adaptive frame skipping, per-stage timing, thermal/throughput watchdogs, annotation |
| `profiling.py` | p50/p90/p99 timing and a `benchmark()` harness that discards warmup |
| `roi.py` | ROI cropping and tiled inference with box remapping, for small distant objects |
| `sinks.py` | Video writer, event snapshots, JSON-lines log, stdlib MJPEG preview server |
| `ros2_node.py` | `vision_msgs/Detection2DArray` publisher, guarded so it imports without ROS |

## What makes this different from a YOLO demo script

A demo script is a `while True:` loop around `cap.read()` and
`model.predict()`. That is roughly 15 lines and it is genuinely all you need
on a desktop with a discrete GPU. On an edge board it fails in specific ways,
and this repo is the difference:

1. **It does not process stale frames.** A demo script's latency grows without
   bound because the driver queues what you did not consume. Here a reader
   thread drains the queue into a one-deep slot; you always get the newest
   frame, and the drop counter shows what that cost.
2. **It gets more than `1/inference_time` FPS out.** Adaptive frame skipping
   plus a real motion model turns a 10 FPS detector into a 30 FPS output
   stream. This is the technique that makes edge detection usable, and it is
   the centrepiece here rather than an afterthought.
3. **It tells you where the time goes.** Per-stage p50/p90/p99, not mean FPS.
   Mean hides the tail, and the tail is what a human sees as stutter and what a
   control loop sees as a hole. On a Pi it is common for letterboxing plus
   annotation to cost as much as the network; no amount of quantisation fixes
   that, and only a per-stage breakdown tells you.
4. **The coordinate math is right and tested.** Letterbox with an exact
   inverse, class-aware NMS, and a round-trip test. Getting the padding offset
   wrong shifts every box by a constant and gets misdiagnosed as a model bug.
5. **Tracking is real tracking.** A constant-velocity Kalman filter with
   optimal (Hungarian) assignment, track states, `max_age`, `min_hits`, and IDs
   that are never recycled — with tests that prove ID stability across motion
   and a *new* ID after expiry. Greedy IoU matching lets two crossing objects
   swap identities; optimal assignment does not.
6. **It survives the field.** Camera reconnect with backoff, sink failures
   isolated so a full SD card does not stop perception, thermal and throughput
   watchdogs that name the likely cause.
7. **It picks a backend and says why.** A silent fallback to CPU is how a robot
   ends up at 4 FPS while everyone insists the GPU path is on.
8. **It imports and tests with nothing installed.** torch, ultralytics, cv2,
   onnxruntime, tensorrt, hailo_platform and rclpy are all optional and all
   guarded. 187 tests, offline, no hardware.

## Honest limitations

- **No published benchmark numbers.** [`benchmarks/README.md`](benchmarks/README.md)
  is a template with `—` in every cell and instructions for filling it in on
  real hardware. Invented edge FPS numbers are worse than none, because they
  depend on JetPack version, power mode, cooling and thermal state — things a
  table never captures.
- **The tracker has no appearance model.** It is SORT, not DeepSORT. Two
  similar objects that cross while occluded can swap IDs. Adding a re-ID
  embedding costs another network per frame, which on a Nano is usually not
  affordable — that trade is deliberate.
- **The Kalman model is constant velocity.** Good for people, vehicles, boats.
  Poor for erratic, accelerating targets at close range.
- **Frame skipping delays first detection** of a newly appeared object by up to
  `N` frames. Adaptive skipping bounds `N` but cannot remove this.
- **No batching.** Single-stream, batch size 1. Batching helps throughput on a
  discrete GPU and hurts latency, which is the wrong trade here — but if you
  are running eight cameras on one Orin, this is not the right structure.
- **Tiled inference multiplies inference cost** linearly in tile count. It buys
  small-object recall and nothing else.
- **TensorRT engines are not portable.** Built for a specific GPU architecture
  and TensorRT version. Build on the target board.
- **The MJPEG preview is not efficient.** One full JPEG per frame. It is for
  "is the camera pointed at the right thing", not for sustained streaming.
- **Real backends are code-complete but hardware-untestable here.** The
  TensorRT, Hailo, ONNX Runtime and Ultralytics paths are written against
  their real APIs; the automated tests exercise the shared pre/post-processing
  tail, the registry, and the pure decoding functions, since CI has no Jetson,
  no Hailo device and no model weights.

## Documentation

- [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) — the real playbook: FP16 vs
  INT8 and when INT8 hurts, warmup, resolution as the biggest lever,
  `nvpmodel`/`jetson_clocks`, CSI vs USB, hardware decode, zero-copy pitfalls,
  DLA vs GPU, and how to diagnose "my FPS collapsed after 30 seconds".
- [`docs/ROS2.md`](docs/ROS2.md) — topics, QoS reasoning, coordinate
  conventions, and the `package.xml` / `setup.py` / launch file for wrapping
  this as an ament package.
- [`benchmarks/README.md`](benchmarks/README.md) — measurement templates.

## Related repositories

- [px4-mavlink-companion](https://github.com/Pratyush150/px4-mavlink-companion) — MAVLink bridge and offboard control between a flight controller and a companion computer
- [drone-control-toolkit](https://github.com/Pratyush150/drone-control-toolkit) — PID/LQR/EKF control and estimation with a simulation harness
- [ros2-drone-bringup](https://github.com/Pratyush150/ros2-drone-bringup) — ROS 2 bringup for a PX4 drone
- [flight-log-analyzer](https://github.com/Pratyush150/flight-log-analyzer) — PX4 ULog / ArduPilot log analysis

## License

MIT. See [LICENSE](LICENSE).
