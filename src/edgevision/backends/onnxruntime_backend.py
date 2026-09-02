"""ONNX Runtime backend.

This is the portable middle ground: one ``.onnx`` file runs on a Pi (CPU /
XNNPACK), on a Jetson (CUDA or the TensorRT execution provider) and on a
laptop, with identical pre/post-processing. It is the right thing to validate
against before you commit to a platform-specific engine, because if the ONNX
and TensorRT outputs disagree you now know the export is fine and the engine
build is not.

Provider selection matters more than anything else here. ``onnxruntime``
silently falls back to CPU if the GPU provider fails to initialise, so the
resolved provider list is logged rather than assumed.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .._compat import module_available
from ..preprocess import LetterboxParams, letterbox, to_nchw
from .base import Availability, Detector

LOGGER = logging.getLogger(__name__)

__all__ = ["OnnxRuntimeBackend"]

#: Ordered by "fastest where available".
DEFAULT_PROVIDERS: Tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "OpenVINOExecutionProvider",
    "XnnpackExecutionProvider",
    "CPUExecutionProvider",
)


class OnnxRuntimeBackend(Detector):
    """Runs a YOLO-style ONNX graph through onnxruntime."""

    name = "onnxruntime"
    priority = 40

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (640, 640),
        providers: Optional[Sequence[str]] = None,
        intra_op_threads: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path=model_path, input_size=input_size, **kwargs)
        self.requested_providers = tuple(providers) if providers else DEFAULT_PROVIDERS
        self.intra_op_threads = int(intra_op_threads)
        self.active_providers: Tuple[str, ...] = ()
        self._session: Any = None
        self._input_name: str = ""
        self._output_names: List[str] = []

    @classmethod
    def probe(cls) -> Availability:
        if not module_available("onnxruntime"):
            return Availability(
                cls.name, False, "onnxruntime is not installed", cls.priority
            )
        try:  # pragma: no cover - depends on environment
            import onnxruntime as ort  # type: ignore

            providers = tuple(ort.get_available_providers())
        except Exception as exc:  # pragma: no cover
            return Availability(
                cls.name, False, f"onnxruntime import failed: {exc}", cls.priority
            )
        accelerated = [p for p in providers if p != "CPUExecutionProvider"]
        priority = cls.priority + (10 if accelerated else 0)
        reason = (
            f"onnxruntime {ort.__version__} with providers: {', '.join(providers)}"
        )
        return Availability(
            cls.name, True, reason, priority, {"providers": list(providers)}
        )

    def load(self) -> None:  # pragma: no cover - requires onnxruntime + a model
        import onnxruntime as ort  # type: ignore

        if not self.model_path:
            raise ValueError("OnnxRuntimeBackend requires model_path")

        available = set(ort.get_available_providers())
        providers = [p for p in self.requested_providers if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.intra_op_threads:
            # On a 4-core Pi, letting ORT spawn 4 compute threads starves the
            # capture thread and the frame rate gets *worse*. Cap it at 3.
            options.intra_op_num_threads = self.intra_op_threads

        self._session = ort.InferenceSession(
            self.model_path, sess_options=options, providers=providers
        )
        self.active_providers = tuple(self._session.get_providers())
        LOGGER.info(
            "onnxruntime session for %s using providers %s",
            self.model_path,
            self.active_providers,
        )

        inputs = self._session.get_inputs()
        self._input_name = inputs[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        shape = inputs[0].shape  # typically [batch, 3, H, W]
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_size = (int(shape[3]), int(shape[2]))
            LOGGER.info("model has static input size %s", self.input_size)
        self.loaded = True

    def close(self) -> None:  # pragma: no cover
        self._session = None
        self.loaded = False

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, LetterboxParams]:
        padded, params = letterbox(frame, self.input_size)
        return to_nchw(padded), params

    def _forward(self, tensor: Any, params: LetterboxParams) -> Any:  # pragma: no cover
        outputs = self._session.run(self._output_names, {self._input_name: tensor})
        return outputs[0]

    def describe(self) -> dict:
        info = super().describe()
        info["providers"] = list(self.active_providers)
        return info
