"""Backend registry with runtime capability detection.

An edge fleet is never homogeneous. The same code runs on an Orin Nano with
TensorRT, a Pi 5 with a Hailo-8L, a Pi 4 with nothing but CPU, and a laptop
during development. Hard-coding the backend means four builds; probing at
runtime means one.

The rule this module follows: **always log why**. A silent fallback to CPU is
how a robot ends up running at 4 FPS in the field while everyone insists the
GPU path is enabled. Every probe returns a human-readable reason, success or
failure, and :func:`select_backend` records the full decision.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from .base import Availability, Detector
from .hailo_backend import HailoBackend
from .mock import MockBackend
from .onnxruntime_backend import OnnxRuntimeBackend
from .tensorrt_backend import TensorRTBackend
from .ultralytics_backend import UltralyticsBackend

LOGGER = logging.getLogger(__name__)

__all__ = [
    "Detector",
    "Availability",
    "MockBackend",
    "UltralyticsBackend",
    "OnnxRuntimeBackend",
    "TensorRTBackend",
    "HailoBackend",
    "register_backend",
    "get_backend_class",
    "backend_names",
    "probe_backends",
    "available_backends",
    "select_backend",
    "create_backend",
    "format_backend_table",
    "BackendChoice",
]

_REGISTRY: Dict[str, Type[Detector]] = {}

#: File extensions each backend can actually consume.
_MODEL_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    "tensorrt": (".engine", ".plan", ".onnx"),
    "hailo": (".hef",),
    "onnxruntime": (".onnx", ".ort"),
    "ultralytics": (".pt", ".onnx", ".engine", ".torchscript"),
    "mock": (),
}


def register_backend(cls: Type[Detector]) -> Type[Detector]:
    """Register a :class:`Detector` subclass under its ``name``."""
    if not issubclass(cls, Detector):
        raise TypeError(f"{cls!r} is not a Detector subclass")
    if not getattr(cls, "name", None) or cls.name == "base":
        raise ValueError(f"{cls!r} must define a unique class-level name")
    _REGISTRY[cls.name] = cls
    return cls


for _cls in (
    MockBackend,
    UltralyticsBackend,
    OnnxRuntimeBackend,
    TensorRTBackend,
    HailoBackend,
):
    register_backend(_cls)


def get_backend_class(name: str) -> Type[Detector]:
    """Look up a registered backend class by name."""
    try:
        return _REGISTRY[str(name).lower()]
    except KeyError:
        raise KeyError(
            f"unknown backend {name!r}; known: {', '.join(sorted(_REGISTRY))}"
        ) from None


def backend_names() -> List[str]:
    """All registered backend names, highest priority first."""
    return sorted(_REGISTRY, key=lambda n: -_REGISTRY[n].priority)


def probe_backends() -> Dict[str, Availability]:
    """Probe every registered backend. Never raises; failures become reasons."""
    results: Dict[str, Availability] = {}
    for name, cls in _REGISTRY.items():
        try:
            results[name] = cls.probe()
        except Exception as exc:  # pragma: no cover - defensive
            results[name] = Availability(name, False, f"probe raised: {exc}", cls.priority)
    return results


def available_backends() -> List[str]:
    """Names of backends that can run on this machine, best first."""
    probes = probe_backends()
    usable = [a for a in probes.values() if a.available]
    usable.sort(key=lambda a: -a.priority)
    return [a.name for a in usable]


class BackendChoice(Tuple[str, Availability]):
    """``(name, availability)`` with the full decision trail attached."""

    log: Tuple[str, ...] = ()

    def __new__(cls, name: str, availability: Availability, log: Sequence[str] = ()):
        obj = super().__new__(cls, (name, availability))
        obj.log = tuple(log)
        return obj

    @property
    def name(self) -> str:
        return self[0]

    @property
    def availability(self) -> Availability:
        return self[1]

    @property
    def reason(self) -> str:
        return self[1].reason


def _model_is_compatible(name: str, model_path: Optional[str]) -> bool:
    suffixes = _MODEL_SUFFIXES.get(name, ())
    if not suffixes:
        return True
    if not model_path:
        return False
    return str(model_path).lower().endswith(suffixes)


def select_backend(
    preferred: str = "auto",
    model_path: Optional[str] = None,
    allow_fallback: bool = True,
    exclude: Sequence[str] = (),
) -> BackendChoice:
    """Pick the best usable backend and explain the choice.

    Parameters
    ----------
    preferred:
        A backend name, or ``"auto"`` to rank by priority.
    model_path:
        Used as a second filter: TensorRT being installed is irrelevant if the
        only weights you have are a ``.pt``. This check is what stops "auto"
        from picking a backend that will throw on :meth:`Detector.load`.
    allow_fallback:
        If the preferred backend is unusable, drop to the next best instead of
        raising. The fallback is logged at WARNING, never silently.
    """
    probes = probe_backends()
    log: List[str] = []
    excluded = {str(e).lower() for e in exclude}

    for name in sorted(probes, key=lambda n: -probes[n].priority):
        probe = probes[name]
        status = "available" if probe.available else "unavailable"
        log.append(f"{name}: {status} - {probe.reason}")

    preferred = (preferred or "auto").lower()
    if preferred != "auto":
        if preferred not in _REGISTRY:
            raise KeyError(
                f"unknown backend {preferred!r}; known: {', '.join(sorted(_REGISTRY))}"
            )
        probe = probes[preferred]
        if probe.available and _model_is_compatible(preferred, model_path):
            log.append(f"selected {preferred}: explicitly requested and usable")
            LOGGER.info("backend %s selected (requested)", preferred)
            return BackendChoice(preferred, probe, log)
        detail = probe.reason if not probe.available else (
            f"model {model_path!r} is not loadable by {preferred}"
        )
        if not allow_fallback:
            raise RuntimeError(f"backend {preferred!r} unusable: {detail}")
        log.append(f"{preferred} requested but unusable: {detail}")
        LOGGER.warning("requested backend %s unusable: %s", preferred, detail)

    candidates = [
        probes[n]
        for n in probes
        if probes[n].available
        and n not in excluded
        and n != MockBackend.name
        and _model_is_compatible(n, model_path)
    ]
    candidates.sort(key=lambda a: -a.priority)

    if candidates:
        chosen = candidates[0]
        log.append(
            f"selected {chosen.name}: highest-priority usable backend "
            f"({chosen.reason})"
        )
        LOGGER.info("backend %s selected: %s", chosen.name, chosen.reason)
        return BackendChoice(chosen.name, chosen, log)

    reason = (
        "no accelerated backend is both installed and compatible with "
        f"model_path={model_path!r}"
    )
    log.append(f"selected mock: {reason}")
    LOGGER.warning("falling back to mock backend: %s", reason)
    return BackendChoice(MockBackend.name, probes[MockBackend.name], log)


def create_backend(
    backend: str = "auto",
    model_path: Optional[str] = None,
    allow_fallback: bool = True,
    **kwargs: Any,
) -> Detector:
    """Select and instantiate a backend in one call."""
    choice = select_backend(backend, model_path, allow_fallback=allow_fallback)
    cls = get_backend_class(choice.name)
    if choice.name == MockBackend.name:
        kwargs.pop("providers", None)
        return cls(model_path=None, **kwargs)
    return cls(model_path=model_path, **kwargs)


def format_backend_table(probes: Optional[Dict[str, Availability]] = None) -> str:
    """Render the capability probe as a fixed-width table for the CLI."""
    probes = probes or probe_backends()
    rows = sorted(probes.values(), key=lambda a: -a.priority)
    name_w = max(7, max(len(r.name) for r in rows))
    lines = [
        f"{'backend'.ljust(name_w)}  {'prio':>4}  {'ok':^3}  reason",
        f"{'-' * name_w}  {'-' * 4}  {'-' * 3}  {'-' * 40}",
    ]
    for row in rows:
        mark = "yes" if row.available else "no"
        lines.append(f"{row.name.ljust(name_w)}  {row.priority:>4}  {mark:^3}  {row.reason}")
    return "\n".join(lines)
