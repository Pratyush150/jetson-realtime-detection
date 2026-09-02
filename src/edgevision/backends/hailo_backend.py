"""Hailo-8 / Hailo-8L backend (HailoRT).

A Hailo M.2 or HAT accelerator changes the arithmetic on a Raspberry Pi: the
Pi CPU stops being the inference device and becomes a video plumbing device.
The practical consequences are different from a Jetson:

* The network runs as a compiled ``.hef``, produced offline by the Hailo
  Dataflow Compiler. You cannot build one on the Pi.
* Hailo is INT8 end to end. There is no FP16 fallback; the quantisation
  happened at compile time, and if accuracy is bad the fix is a better
  calibration set, not a runtime flag.
* Input is NHWC uint8, *not* NCHW float. Normalisation is usually folded into
  the compiled network, so pre-scaling on the CPU is wasted work.
* Many HEFs are compiled with NMS on-chip. In that case the output is a
  per-class list of ``[y_min, x_min, y_max, x_max, score]`` in *normalised*
  coordinates — note the y-first ordering, which is a very easy way to get
  boxes that look plausibly wrong.

Both output shapes are handled below.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

from .._compat import module_available
from ..preprocess import LetterboxParams, letterbox
from .base import Availability, Detector

LOGGER = logging.getLogger(__name__)

__all__ = ["HailoBackend"]


class HailoBackend(Detector):
    """Runs a compiled ``.hef`` on a Hailo device via HailoRT."""

    name = "hailo"
    priority = 70

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (640, 640),
        nms_on_device: Optional[bool] = None,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path=model_path, input_size=input_size, **kwargs)
        self.nms_on_device = nms_on_device
        self.batch_size = int(batch_size)
        self._hailo: Any = None
        self._target: Any = None
        self._network_group: Any = None
        self._network_group_params: Any = None
        self._input_vstream_info: Any = None
        self._input_vstreams_params: Any = None
        self._output_vstreams_params: Any = None
        self._input_name: str = ""

    @classmethod
    def probe(cls) -> Availability:
        if not module_available("hailo_platform"):
            return Availability(
                cls.name, False, "hailo_platform (HailoRT) not installed", cls.priority
            )
        try:  # pragma: no cover - requires the runtime
            import hailo_platform  # type: ignore

            version = getattr(hailo_platform, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover
            return Availability(
                cls.name, False, f"hailo_platform import failed: {exc}", cls.priority
            )
        return Availability(
            cls.name, True, f"HailoRT {version} importable", cls.priority,
            {"hailort": version},
        )

    def load(self) -> None:  # pragma: no cover - requires Hailo hardware
        from hailo_platform import (  # type: ignore
            HEF,
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )

        if not self.model_path:
            raise ValueError("HailoBackend requires a .hef model_path")

        self._hailo = __import__("hailo_platform")
        hef = HEF(self.model_path)
        self._target = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
        self._network_group = self._target.configure(hef, configure_params)[0]
        self._network_group_params = self._network_group.create_params()

        self._input_vstream_info = hef.get_input_vstream_infos()[0]
        self._input_name = self._input_vstream_info.name
        shape = tuple(self._input_vstream_info.shape)  # (H, W, C)
        if len(shape) >= 2:
            self.input_size = (int(shape[1]), int(shape[0]))

        # uint8 in, float32 out: the device wants quantised input and returns
        # dequantised scores, so no CPU-side normalisation is needed.
        self._input_vstreams_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.UINT8
        )
        self._output_vstreams_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )

        output_infos = hef.get_output_vstream_infos()
        if self.nms_on_device is None:
            self.nms_on_device = any(
                "nms" in info.name.lower() for info in output_infos
            )
        self.loaded = True
        LOGGER.info(
            "hailo hef %s loaded, input %s, on-device NMS=%s",
            self.model_path,
            self.input_size,
            self.nms_on_device,
        )

    def close(self) -> None:  # pragma: no cover - requires hardware
        target, self._target = self._target, None
        if target is not None:
            try:
                target.release()
            except Exception:
                LOGGER.debug("VDevice release raised", exc_info=True)
        self._network_group = None
        self.loaded = False

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, LetterboxParams]:
        # NHWC uint8, no scaling: the compiled network owns normalisation.
        padded, params = letterbox(frame, self.input_size)
        return np.ascontiguousarray(padded[None, ...].astype(np.uint8)), params

    def _forward(self, tensor: Any, params: LetterboxParams) -> Any:  # pragma: no cover
        from hailo_platform import InferVStreams  # type: ignore

        with InferVStreams(
            self._network_group,
            self._input_vstreams_params,
            self._output_vstreams_params,
        ) as pipeline:
            with self._network_group.activate(self._network_group_params):
                results = pipeline.infer({self._input_name: tensor})
        return results

    def _decode(self, raw: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Handle both on-device NMS output and a plain detection head."""
        if self.nms_on_device:
            return self.decode_hailo_nms(raw, self.conf_threshold, self.input_size)
        array = raw
        if isinstance(raw, dict):
            array = next(iter(raw.values()))
        return super()._decode(np.asarray(array))

    @staticmethod
    def decode_hailo_nms(
        raw: Any,
        conf_threshold: float,
        input_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode HailoRT's per-class NMS output into ``xyxy`` pixel boxes.

        The on-device format is a list indexed by class id, each entry an
        ``(n, 5)`` array of ``[y_min, x_min, y_max, x_max, score]`` normalised
        to ``[0, 1]``. Two traps: the y-first ordering, and the normalisation
        being relative to the *network input*, not the original frame — which
        is why this returns network-space pixels and lets the shared tail
        un-letterbox them.
        """
        if isinstance(raw, dict):
            raw = next(iter(raw.values()))
        if isinstance(raw, np.ndarray) and raw.ndim >= 1 and raw.dtype == object:
            raw = list(raw)
        if isinstance(raw, (list, tuple)) and len(raw) == 1 and isinstance(raw[0], (list, tuple)):
            raw = raw[0]

        width, height = int(input_size[0]), int(input_size[1])
        boxes: List[List[float]] = []
        scores: List[float] = []
        class_ids: List[int] = []

        for class_id, entries in enumerate(raw):
            entries = np.asarray(entries, dtype=np.float32).reshape(-1, 5)
            for y1, x1, y2, x2, score in entries:
                if score < conf_threshold:
                    continue
                boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
                scores.append(float(score))
                class_ids.append(int(class_id))

        if not boxes:
            return (
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int64),
            )
        return (
            np.asarray(boxes, dtype=np.float32),
            np.asarray(scores, dtype=np.float32),
            np.asarray(class_ids, dtype=np.int64),
        )

    def describe(self) -> dict:
        info = super().describe()
        info.update({"nms_on_device": self.nms_on_device, "precision": "int8"})
        return info
