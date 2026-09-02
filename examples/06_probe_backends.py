#!/usr/bin/env python3
"""What can this board actually run, and why was that chosen?

    python3 examples/06_probe_backends.py [model_path]

Run this first on any new board. It answers "is my TensorRT install visible to
Python", "did onnxruntime find the CUDA provider or silently fall back to CPU",
and "which backend will the pipeline pick".
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from edgevision.backends import format_backend_table, probe_backends, select_backend  # noqa: E402


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else None

    print("capability probe")
    print("=" * 72)
    print(format_backend_table())

    for name, probe in probe_backends().items():
        if probe.details:
            print(f"\n{name} details: {probe.details}")

    print()
    print(f"selection for model_path={model!r}")
    print("=" * 72)
    choice = select_backend("auto", model_path=model)
    for line in choice.log:
        print(f"  {line}")
    print(f"\n-> {choice.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
