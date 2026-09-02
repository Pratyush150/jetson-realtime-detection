# Benchmarks

> **All numbers in these tables are placeholders (`—`).**
> They are to be filled in from measured runs on real hardware. Nothing here
> is estimated, extrapolated from a datasheet, or copied from someone else's
> blog post. An unmeasured cell stays `—`.

Publishing invented FPS numbers for edge hardware is worse than publishing
none, because the numbers are extremely sensitive to things a table never
captures: JetPack version, power mode, whether `jetson_clocks` was on, the
thermal state at minute five, the camera path, and what else was on the GPU.

## How to fill these in

```bash
# 1. Confirm which backend the board can actually use.
python3 examples/06_probe_backends.py /path/to/model

# 2. Measure the backend in isolation, on synthetic frames.
#    Warmup is discarded automatically.
edgevision-bench --backend tensorrt --model yolov8n.engine \
                 --imgsz 640 --resolution 1280x720 \
                 --frames 300 --warmup 20 --json > result.json

# 3. Measure the full pipeline on the real camera, including capture and draw.
edgevision-run --source 0 --backend tensorrt --model yolov8n.engine \
               --duration 120 --stats-json > pipeline.json
```

Record, for every row:

- board and JetPack / Raspberry Pi OS version
- power mode (`nvpmodel -q`) and whether `jetson_clocks` was applied
- ambient temperature and whether a fan was fitted
- the exact model file and its precision (FP32 / FP16 / INT8)

Run for **at least two minutes**. A 10-second benchmark on a cold board
measures the best case you will never see again; see
[`docs/OPTIMIZATION.md`](../docs/OPTIMIZATION.md#my-fps-collapsed-after-30-seconds).

## Backend inference only

Frames are synthetic; this isolates the detector from capture and drawing.
`p99` is the 99th-percentile end-to-end inference latency in milliseconds.

| Platform | Model | Precision | Input | Backend | FPS (from p50) | p50 (ms) | p90 (ms) | p99 (ms) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Jetson Nano (4 GB) | — | — | 640x640 | TensorRT | — | — | — | — | — |
| Jetson Nano (4 GB) | — | — | 416x416 | TensorRT | — | — | — | — | — |
| Jetson Nano (4 GB) | — | — | 640x640 | ONNX Runtime (CUDA) | — | — | — | — | — |
| Jetson Orin Nano (8 GB) | — | — | 640x640 | TensorRT | — | — | — | — | — |
| Jetson Orin Nano (8 GB) | — | — | 640x640 | TensorRT (DLA) | — | — | — | — | — |
| Jetson Xavier NX | — | — | 640x640 | TensorRT | — | — | — | — | — |
| Jetson Xavier NX | — | — | 640x640 | TensorRT (DLA) | — | — | — | — | — |
| Raspberry Pi 4 (4 GB) | — | — | 320x320 | ONNX Runtime (CPU) | — | — | — | — | — |
| Raspberry Pi 5 (8 GB) | — | — | 640x640 | ONNX Runtime (CPU) | — | — | — | — | — |
| Raspberry Pi 5 + Hailo-8L | — | INT8 | 640x640 | HailoRT | — | — | — | — | — |
| Raspberry Pi 5 + Hailo-8 | — | INT8 | 640x640 | HailoRT | — | — | — | — | — |

## Full pipeline

End to end: capture -> preprocess -> infer -> track -> annotate -> sink.
`Skip` is the detection interval the adaptive skipper settled on for the
stated target. `Output FPS` is what a consumer of the stream actually sees.

| Platform | Camera | Capture res | Model | Backend | Target FPS | Skip | Output FPS | Inference p99 (ms) | End-to-end p99 (ms) | Drop rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Jetson Nano | CSI (IMX219) | 1280x720 | — | TensorRT | 30 | — | — | — | — | — |
| Jetson Nano | USB (MJPG) | 1280x720 | — | TensorRT | 30 | — | — | — | — | — |
| Jetson Orin Nano | CSI | 1920x1080 | — | TensorRT | 30 | — | — | — | — | — |
| Jetson Xavier NX | RTSP (H.264) | 1920x1080 | — | TensorRT | 30 | — | — | — | — | — |
| Raspberry Pi 4 | USB (MJPG) | 640x480 | — | ONNX Runtime | 15 | — | — | — | — | — |
| Raspberry Pi 5 | CSI | 1280x720 | — | ONNX Runtime | 30 | — | — | — | — | — |
| Raspberry Pi 5 + Hailo-8L | CSI | 1280x720 | — | HailoRT | 30 | — | — | — | — | — |

## Per-stage breakdown

Fill one of these per configuration you care about. It is the table that tells
you what to fix. If `draw` is comparable to `inference`, quantising the model
is a waste of a weekend.

| Platform | Stage | mean (ms) | p50 (ms) | p90 (ms) | p99 (ms) | Notes |
|---|---|---|---|---|---|---|
| — | capture | — | — | — | — | — |
| — | preprocess | — | — | — | — | — |
| — | inference | — | — | — | — | — |
| — | postprocess | — | — | — | — | — |
| — | track | — | — | — | — | — |
| — | draw | — | — | — | — | — |
| — | sink | — | — | — | — | — |

## Thermal sustain

The number people forget to measure, and the one that decides whether a
system works in the field.

| Platform | Cooling | FPS @ 10 s | FPS @ 60 s | FPS @ 300 s | Peak zone temp (C) | Throttled? |
|---|---|---|---|---|---|---|
| Jetson Nano | passive heatsink | — | — | — | — | — |
| Jetson Nano | heatsink + fan | — | — | — | — | — |
| Jetson Orin Nano | stock | — | — | — | — | — |
| Raspberry Pi 5 | active cooler | — | — | — | — | — |
| Raspberry Pi 5 | bare board | — | — | — | — | — |

## Reporting template

```
board          :
os / jetpack   :
power mode     :  (nvpmodel -q)
jetson_clocks  :  yes / no
cooling        :
ambient        :  C
camera         :
model file     :
precision      :  fp32 / fp16 / int8
input size     :
run length     :  s
```
