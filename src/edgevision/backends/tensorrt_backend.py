"""TensorRT backend for Jetson.

This is the one that actually makes a Jetson usable. TensorRT fuses layers,
picks per-layer kernels for *your* GPU, and runs FP16 (or INT8) natively.

Three things bite people here, all handled below:

1. **An engine is not portable.** It is built for a specific GPU
   architecture, TensorRT version and JetPack. An engine built on an Orin
   will not load on a Nano and vice versa. Build on the target board.
2. **Host<->device copies are not free.** Buffers are allocated once at load
   time and reused. Allocating per frame is a classic way to lose most of the
   speedup you just bought.
3. **The API changed.** TensorRT 10 replaced the index-based binding API
   (``get_binding_shape``, ``execute_async_v2``) with the name-based tensor
   API (``get_tensor_shape``, ``execute_async_v3``). Both are supported here
   because Jetson boards in the field run everything from JetPack 4.6 to 6.x.

The heavy imports (``tensorrt``, ``pycuda``) happen inside :meth:`load`, not
at module import: ``import pycuda.autoinit`` creates a CUDA context as a side
effect, which you do not want to happen just because someone imported the
package to read a config.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._compat import module_available
from ..preprocess import LetterboxParams, letterbox, to_nchw
from .base import Availability, Detector

LOGGER = logging.getLogger(__name__)

__all__ = ["TensorRTBackend"]


class TensorRTBackend(Detector):
    """Runs a serialised TensorRT engine (``.engine`` / ``.plan``).

    Parameters
    ----------
    model_path:
        Path to a serialised engine. If it points at an ``.onnx`` file and
        ``build_if_missing`` is set, an engine is built next to it — that
        build takes minutes on a Nano, so it is opt-in and logged loudly.
    fp16:
        Only used when building. FP16 is nearly always the right default on
        Jetson: roughly 2x throughput for accuracy loss that is usually below
        measurement noise on COCO-scale detection.
    """

    name = "tensorrt"
    priority = 80

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (640, 640),
        fp16: bool = True,
        workspace_gb: float = 1.0,
        build_if_missing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path=model_path, input_size=input_size, **kwargs)
        self.fp16 = bool(fp16)
        self.workspace_gb = float(workspace_gb)
        self.build_if_missing = bool(build_if_missing)

        self._trt: Any = None
        self._cuda: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._stream: Any = None
        self._bindings: List[int] = []
        self._host_buffers: Dict[str, np.ndarray] = {}
        self._device_buffers: Dict[str, Any] = {}
        self._input_tensor: str = ""
        self._output_tensors: List[str] = []
        self._uses_tensor_api = False

    # -- capability ---------------------------------------------------------

    @classmethod
    def probe(cls) -> Availability:
        if not module_available("tensorrt"):
            return Availability(
                cls.name, False, "tensorrt python bindings not installed", cls.priority
            )
        has_pycuda = module_available("pycuda")
        has_cuda_python = module_available("cuda")
        if not (has_pycuda or has_cuda_python):
            return Availability(
                cls.name,
                False,
                "tensorrt present but no CUDA host bindings (install pycuda)",
                cls.priority,
            )
        details: Dict[str, Any] = {
            "pycuda": has_pycuda,
            "cuda_python": has_cuda_python,
        }
        try:  # pragma: no cover - depends on environment
            import tensorrt as trt  # type: ignore

            details["tensorrt"] = trt.__version__
            reason = f"tensorrt {trt.__version__} available"
        except Exception as exc:  # pragma: no cover
            return Availability(
                cls.name, False, f"tensorrt import failed: {exc}", cls.priority
            )
        return Availability(cls.name, True, reason, cls.priority, details)

    # -- lifecycle ----------------------------------------------------------

    def load(self) -> None:  # pragma: no cover - requires a Jetson / CUDA host
        import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
        import pycuda.driver as cuda  # type: ignore
        import tensorrt as trt  # type: ignore

        self._trt = trt
        self._cuda = cuda

        if not self.model_path:
            raise ValueError("TensorRTBackend requires model_path")

        engine_path = self.model_path
        if engine_path.endswith(".onnx"):
            candidate = os.path.splitext(engine_path)[0] + ".engine"
            if os.path.exists(candidate):
                engine_path = candidate
            elif self.build_if_missing:
                LOGGER.warning(
                    "building TensorRT engine from %s; this can take several "
                    "minutes on a Jetson and must be done on the target board",
                    self.model_path,
                )
                engine_path = self.build_engine(self.model_path, candidate)
            else:
                raise FileNotFoundError(
                    f"no engine at {candidate}; pass build_if_missing=True or run "
                    f"trtexec --onnx={self.model_path} --saveEngine={candidate}"
                )

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as handle, trt.Runtime(logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(handle.read())
        if self._engine is None:
            raise RuntimeError(
                f"failed to deserialise {engine_path}. An engine is tied to the "
                "GPU architecture and TensorRT version it was built with."
            )
        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()
        self._uses_tensor_api = hasattr(self._engine, "num_io_tensors")
        self._allocate_buffers()
        self.loaded = True
        LOGGER.info(
            "tensorrt engine %s loaded (%s API), input %s",
            engine_path,
            "tensor" if self._uses_tensor_api else "binding",
            self.input_size,
        )

    def _allocate_buffers(self) -> None:  # pragma: no cover - CUDA required
        """Allocate pinned host and device buffers once, and reuse them."""
        trt, cuda = self._trt, self._cuda
        self._host_buffers.clear()
        self._device_buffers.clear()
        self._bindings = []
        self._output_tensors = []

        if self._uses_tensor_api:
            names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
            for name in names:
                is_input = self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape = tuple(self._context.get_tensor_shape(name))
                dtype = trt.nptype(self._engine.get_tensor_dtype(name))
                host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                device = cuda.mem_alloc(host.nbytes)
                self._host_buffers[name] = host.reshape(shape)
                self._device_buffers[name] = device
                self._context.set_tensor_address(name, int(device))
                if is_input:
                    self._input_tensor = name
                    self.input_size = (int(shape[3]), int(shape[2]))
                else:
                    self._output_tensors.append(name)
        else:
            for index in range(self._engine.num_bindings):
                name = self._engine.get_binding_name(index)
                shape = tuple(self._context.get_binding_shape(index))
                dtype = trt.nptype(self._engine.get_binding_dtype(index))
                host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                device = cuda.mem_alloc(host.nbytes)
                self._host_buffers[name] = host.reshape(shape)
                self._device_buffers[name] = device
                self._bindings.append(int(device))
                if self._engine.binding_is_input(index):
                    self._input_tensor = name
                    self.input_size = (int(shape[3]), int(shape[2]))
                else:
                    self._output_tensors.append(name)

    def close(self) -> None:  # pragma: no cover - CUDA required
        self._context = None
        self._engine = None
        self._stream = None
        self._host_buffers.clear()
        self._device_buffers.clear()
        self.loaded = False

    # -- inference ----------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, LetterboxParams]:
        padded, params = letterbox(frame, self.input_size)
        return to_nchw(padded), params

    def _forward(self, tensor: Any, params: LetterboxParams) -> Any:  # pragma: no cover
        cuda = self._cuda
        host_in = self._host_buffers[self._input_tensor]
        np.copyto(host_in, tensor.astype(host_in.dtype, copy=False).reshape(host_in.shape))
        cuda.memcpy_htod_async(
            self._device_buffers[self._input_tensor], host_in, self._stream
        )

        if self._uses_tensor_api:
            self._context.execute_async_v3(stream_handle=self._stream.handle)
        else:
            self._context.execute_async_v2(
                bindings=self._bindings, stream_handle=self._stream.handle
            )

        for name in self._output_tensors:
            cuda.memcpy_dtoh_async(
                self._host_buffers[name], self._device_buffers[name], self._stream
            )
        self._stream.synchronize()
        return self._host_buffers[self._output_tensors[0]].copy()

    # -- engine building ----------------------------------------------------

    def build_engine(self, onnx_path: str, engine_path: str) -> str:  # pragma: no cover
        """Build and serialise an engine from ONNX. Build on the target board."""
        trt = self._trt
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as handle:
            if not parser.parse(handle.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

        config = builder.create_builder_config()
        workspace = int(self.workspace_gb * (1 << 30))
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        else:
            config.max_workspace_size = workspace
        if self.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")
        with open(engine_path, "wb") as handle:
            handle.write(serialized)
        LOGGER.info("wrote engine %s", engine_path)
        return engine_path

    def describe(self) -> dict:
        info = super().describe()
        info.update({"fp16": self.fp16, "api": "tensor" if self._uses_tensor_api else "binding"})
        return info
