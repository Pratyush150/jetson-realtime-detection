# Making detection fast on an edge board

The playbook, roughly in order of leverage. Measure between every step —
`edgevision-bench` and the per-stage table from `edgevision-run` exist so you
are never guessing which change helped.

---

## 0. Measure the right thing first

Before touching the model, get a per-stage breakdown:

```bash
edgevision-run --source 0 --backend auto --model yolov8n.onnx --duration 60 --stats-json
```

Look at `p50` and `p99` per stage, not at mean FPS.

A very common outcome on a Pi is that `preprocess` + `draw` together cost as
much as `inference`. Quantising the model in that situation buys you almost
nothing, and you will have spent a day on it. The second most common outcome
is that `capture` p99 is enormous, which is a camera or a USB problem, not a
model problem.

**Mean FPS hides the tail.** 95 frames at 30 ms and 5 frames at 300 ms
averages to 43.5 ms — "23 FPS", which sounds fine — while a viewer sees five
freezes and a control loop sees five 300 ms holes. Always read p99.

---

## 1. Input resolution is the biggest single lever

Inference cost scales roughly with pixel count. Going from 640x640 to 416x416
is 0.42x the pixels, and in practice something close to that in time. From
640 to 320 is 0.25x.

What you lose is small-object recall, and only that. If the objects you care
about are large in frame (a person at 5 m, a vehicle at 20 m), dropping to
416 often costs nothing you can measure and nearly halves your latency. If
they are small, do not drop the resolution — use tiling
(`edgevision.roi.TiledInference`) so distant objects stay large *relative to
the network input*, and pay for it with frame skipping instead.

Rule of thumb: a detector needs roughly 16 px of object to fire reliably. At
640x640 on a 1920x1080 frame you are downsampling 3x, so an object smaller
than about 48 px in the source is already marginal.

Try, in order: 640 -> 512 -> 416 -> 320, checking recall on your own footage
at each step. Not COCO mAP — your footage.

---

## 2. FP16 almost always, INT8 sometimes

**FP16** on any Jetson (Nano onwards) is close to free accuracy-wise and
typically 1.5-2x faster than FP32. Turn it on and stop thinking about it:

```bash
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
```

FP16 has ~10 bits of mantissa and a much smaller exponent range than FP32.
Detection networks are robust to that. The failure mode when it does bite is
overflow to `inf` in an unnormalised layer, which shows up as *everything*
breaking, not as a subtle accuracy drop — so it is easy to spot.

**INT8** is a different proposition. It is another ~2x, and it requires
calibration: you run a few hundred representative images through the network
to learn per-tensor scale factors.

When INT8 hurts, and it does:

- **Calibration set does not match deployment.** Calibrate on daytime images,
  fly at dusk, and the activation ranges are wrong. Accuracy falls off a
  cliff, not gracefully. Calibrate on *your* footage, covering your lighting.
- **Small objects go first.** Quantisation error is roughly constant in
  absolute terms, so it eats a larger fraction of the weak activations that
  small, low-contrast objects produce. If your job is spotting distant
  targets, validate small-object recall specifically before and after.
- **Low-confidence detections vanish.** Scores compress toward the middle. A
  confidence threshold tuned on FP16 is usually wrong for INT8 — retune it.
- **Some layers must stay in higher precision.** Detection heads and the
  final concat are frequent offenders. TensorRT's mixed-precision fallback
  usually handles this; if accuracy is bad, force the head to FP16.

On Hailo, INT8 is not optional — the network is quantised at compile time by
the Dataflow Compiler. If accuracy is poor there, the fix is a better
calibration set and recompiling, not a runtime flag.

**Always validate accuracy after quantising, on your own data.** A 2x speedup
that misses the object you built the system for is not a speedup.

---

## 3. Warmup, and why you must discard it

The first inference is not representative. It pays for:

- CUDA context creation and device memory allocation
- kernel autotuning / cuDNN algorithm selection
- lazy engine deserialisation work
- first-touch page faults on host buffers

It is routinely 10-100x the steady-state time. Two consequences:

1. **Never include it in a benchmark.** It destroys the mean and annihilates
   p99. `edgevision.profiling.benchmark()` runs warmup frames and throws the
   timings away; `Detector.warmup()` does the same for a live pipeline.
2. **Always run it before going live.** Otherwise your first second of frames
   is dropped, which on a moving vehicle is the second you cared about.

```python
detector = create_backend("auto", model_path="yolov8n.engine")
detector.warmup(iterations=5)   # then start the pipeline
```

If your benchmark's first frame is 40x the rest, that is not a spike to
explain — it is a measurement you should not have taken.

---

## 4. Adaptive frame skipping

Covered in depth in `edgevision/pipeline.py`, summarised here because it is
usually the largest end-to-end win after resolution.

With detection interval `N`:

```
cost_per_output_frame = overhead + inference / N
```

Detect on every 4th frame, track on the other three, and 100 ms inference
stops being a 10 FPS pipeline and becomes a 30 FPS pipeline whose newest
detection is at most ~130 ms old.

What you give up: an object that *appears* between detection frames is not
seen until the next one, so worst-case detection latency is
`N x frame_period`. For erratic close-range motion that matters. For people,
vehicles and boats it does not.

Let it adapt (`--skip 0`, the default) rather than hard-coding a constant. A
`--skip 3` tuned on a cold board is wrong ten minutes into a flight, because
inference time changes with thermal state.

---

## 5. Set the power mode and pin the clocks

Jetson boards boot into a conservative power mode and use dynamic frequency
scaling. Both cost you a lot of measurable performance.

```bash
sudo nvpmodel -q                 # what mode am I in?
sudo nvpmodel -m 0               # max-performance mode (mode ids are board-specific)
sudo jetson_clocks               # pin CPU/GPU/EMC clocks to maximum
sudo jetson_clocks --show        # confirm
```

`nvpmodel -m 0` raises the power cap and enables all cores. `jetson_clocks`
stops the governor from ramping clocks down between frames — which it will do
during your inter-frame gaps, so your *next* frame starts on a cold clock.
Together these are commonly a 20-40% difference on a Nano, and they cost
nothing but power and heat.

Both are runtime settings and do not survive a reboot. Put them in a systemd
unit or your startup script, or you will benchmark a tuned board and deploy an
untuned one.

On a Raspberry Pi the equivalent concerns are the power supply (an underpowered
USB-C supply triggers undervoltage throttling) and cooling. `vcgencmd
measure_clock arm` tells you the actual core clock.

---

## 6. Camera path: CSI beats USB, and it is not close

A **CSI camera** on Jetson goes through `nvarguscamerasrc`: the ISP does
debayering, the VIC does scaling and colour conversion, and the frame lands in
NVMM memory. CPU cost is close to zero.

A **USB camera** costs you real CPU on every frame:

- USB transfer and V4L2 buffer handling
- MJPEG decode (if MJPG) — this is the expensive part, and it is on the CPU
- colour conversion to BGR

At 1080p30 that is easily a full core on a Nano, competing directly with the
Python side of your pipeline. On a 4-core board that is 25% of your compute
spent moving pixels.

Practical notes:

- **Force MJPG on USB cameras.** Raw YUYV at 1080p30 needs ~62 MB/s, which
  many USB 2.0 buses will not sustain; the camera silently negotiates down to
  5 FPS or fails to enumerate. `edgevision.capture` requests MJPG by default.
- **Two USB cameras on one hub usually will not both run at full rate.**
  Bandwidth is shared per host controller, not per port.
- **A brown-out re-enumerates the device.** Every subsequent `read()` returns
  `False` forever unless something reopens it. `FrameGrabber(reconnect=True)`
  handles this; without it, one voltage dip ends your flight's perception.

---

## 7. Hardware decode for RTSP and files

Decoding H.264 in software costs a lot of CPU: 1080p30 can be most of a core.
On Jetson, route it through the hardware decoder:

```
rtspsrc location=... latency=0 ! rtph264depay ! h264parse !
nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx !
videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1
```

`edgevision.capture.rtsp_pipeline()` builds this. Three details matter:

- `nvv4l2decoder` is the hardware decoder. `avdec_h264` is software.
- `latency=0` disables `rtspsrc`'s jitter buffer. On a clean link that is a
  large latency win; on a lossy link you will see more artefacts. For control
  loops, take the latency.
- `drop=true max-buffers=1` on the appsink is the GStreamer-side half of the
  same idea as the latest-frame buffer: never let frames queue.

It matters much less if your source is a 480p webcam. It matters enormously
for 1080p RTSP.

---

## 8. Zero-copy: worth understanding, easy to get wrong

The dream is: camera writes to memory, GPU reads the same memory, no copies.
The reality on Jetson:

- Jetson has **physically unified memory**, so a copy is memory bandwidth, not
  a PCIe transfer. It is cheaper than on a discrete GPU, but not free —
  bandwidth is the scarce resource on a Nano.
- `nvarguscamerasrc` produces frames in **NVMM** (hardware-accessible) memory.
  The moment you convert to `video/x-raw` for `appsink` so OpenCV can hand you
  a numpy array, you have copied out of NVMM. Reading a frame into Python at
  all costs you the zero-copy path.
- Genuine zero-copy needs the whole chain (capture, preprocess, inference) in
  a single framework — DeepStream, or your own CUDA/VPI code. That is a
  significant rewrite, and it is the right call only after resolution,
  precision and frame skipping are exhausted.
- **Do not allocate per frame.** This is the version of "zero-copy" that
  actually pays off in Python: allocate host and device buffers once at load
  time and reuse them (`TensorRTBackend` does). Allocating a pinned buffer per
  frame can eat most of the speedup you just bought.
- **Watch out for accidental copies.** `frame.copy()` in an annotation path,
  `np.ascontiguousarray` on an already-contiguous array, a `cv2.cvtColor` you
  do not need. At 1080p each of these is a few MB of memory traffic per frame.

---

## 9. DLA on Xavier and Orin

Xavier and Orin have **NVDLA** cores: fixed-function deep learning
accelerators, separate from the GPU.

- **They are not faster than the GPU.** Per-layer, DLA is generally slower.
- **They are more power efficient**, and — the real reason to care — they run
  *concurrently* with the GPU. Put the detector on DLA and the GPU is free for
  the rest of your stack, or run two networks at once.
- **Layer support is limited.** Unsupported layers fall back to the GPU,
  which means round trips between DLA and GPU memory. A model with a few
  scattered unsupported layers can end up *slower* than pure GPU. Check the
  build log for fallbacks.
- **Build for it explicitly:**

  ```bash
  trtexec --onnx=model.onnx --saveEngine=model_dla.engine \
          --fp16 --useDLACore=0 --allowGPUFallback
  ```

  DLA is FP16/INT8 only — there is no FP32 path.

Rule: if the detector is your only GPU workload, use the GPU. If you are also
running SLAM, stereo, or a second network, moving detection to DLA is often
the cheapest way to stop them fighting.

---

## 10. "My FPS collapsed after 30 seconds"

This is thermal throttling until proven otherwise. The signature is
distinctive: performance is fine at first, degrades over tens of seconds,
stabilises at a lower level, and recovers if you let the board cool.

**Jetson:**

```bash
sudo tegrastats --interval 1000
# watch CPU@..C GPU@..C, and the CPU/GPU/EMC frequency columns

cat /sys/devices/virtual/thermal/thermal_zone*/type
cat /sys/devices/virtual/thermal/thermal_zone*/temp    # millidegrees C

sudo jetson_clocks --show      # are clocks still pinned?
sudo nvpmodel -q               # did the power mode change?
```

Frequencies dropping while temperature climbs past ~80 C is throttling.

**Raspberry Pi:**

```bash
vcgencmd measure_temp
vcgencmd get_throttled     # 0x0 is healthy
vcgencmd measure_clock arm
```

`get_throttled` is a bitmask: bit 0 = undervoltage now, bit 1 = ARM frequency
capped now, bit 2 = currently throttled, bit 3 = soft temperature limit; bits
16-19 are the same conditions "since boot". A non-zero value with bit 16 set
and bit 0 clear means it happened earlier — often at boot, under a marginal
power supply.

**Portable check** (both platforms, and what `ThermalMonitor` reads):

```bash
for z in /sys/class/thermal/thermal_zone*; do
  echo "$(cat $z/type): $(( $(cat $z/temp) / 1000 )) C"
done
```

`edgevision` surfaces this automatically: `ThroughputWatchdog` establishes an
FPS baseline over the first frames and warns when throughput stays below 75%
of it, and `ThermalMonitor` reports the hottest zone in the run summary.

**If it is not thermal**, check in this order:

- **A second process on the GPU.** `sudo tegrastats` shows GPU utilisation;
  another container or a leftover process is common.
- **Memory pressure.** `free -h`. Swapping on an SD card is catastrophic and
  looks exactly like throttling. A Nano running a 640x640 model plus a
  desktop session is genuinely short of RAM.
- **The camera, not the pipeline.** Compare `frames_read` with
  `frames_delivered` in the capture stats. If `source_fps` itself fell, your
  camera reduced its rate (auto-exposure lengthening the exposure time in
  falling light is a classic) and nothing about your model changed.
- **Growth in scene complexity.** More objects means more NMS, more tracks,
  more boxes to draw. Check whether `track` and `draw` times grew rather than
  `inference`.
- **A log or a sink.** An unbuffered JSON write per frame to an SD card that
  has started garbage collecting will do this. Check the `sink` stage p99.

---

## Quick reference

| Symptom | Most likely cause | First thing to try |
|---|---|---|
| Detections lag reality by seconds | Frames queueing in a driver buffer | Threaded capture with a 1-deep latest-frame slot |
| Good mean FPS, visible stutter | Tail latency | Read p99, not mean; check thermals and GC |
| First frame 40x slower | No warmup | `detector.warmup(5)` and discard the timings |
| Fast on the bench, slow in the field | Power mode / clocks not set | `nvpmodel -m 0`, `jetson_clocks` |
| FPS decays over a minute | Thermal throttling | `tegrastats` / `vcgencmd get_throttled`; add cooling |
| Small/distant objects missed | Input resolution too low for them | Tiled inference, not a bigger model |
| INT8 much worse than FP16 | Calibration set unrepresentative | Recalibrate on deployment footage |
| CPU pinned with a USB camera | MJPEG decode + colour conversion | Move to CSI, or lower resolution |
| Boxes offset by a constant amount | Letterbox padding not undone | Use `unletterbox_boxes` with the real params |
