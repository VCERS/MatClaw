"""Robust torch device selection shared across MatClaw tools.

``torch.cuda.is_available()`` is not a reliable signal that CUDA actually works.
On a machine whose NVIDIA driver is too old for the installed torch build, the
NVML-based availability check (``PYTORCH_NVML_BASED_CUDA_CHECK=1``) still reports
``True`` even though any real CUDA operation raises "The NVIDIA driver on your
system is too old". Tools that trusted ``is_available()`` directly could therefore
pass their (mocked) unit tests yet crash at runtime with a CUDA error.

``get_torch_device()`` instead probes CUDA with a tiny allocation and falls back to
CPU if it fails, so tools run instead of crashing on an unusable GPU. Use it anywhere
you would otherwise write ``torch.device("cuda" if torch.cuda.is_available() else "cpu")``.

Set ``MATCLAW_FORCE_CPU=1`` to skip CUDA entirely.
"""

import os
import warnings


def get_torch_device():
    """Return a ``torch.device`` verified to work, preferring CUDA when it is usable."""
    import torch

    if os.getenv("MATCLAW_FORCE_CPU", "").strip().lower() in ("1", "true", "yes"):
        return torch.device("cpu")

    if torch.cuda.is_available():
        try:
            # Force a real CUDA context; raises if the driver is too old for this
            # torch build (or the device is otherwise unusable).
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:  # driver/build mismatch, init failure, etc.
            warnings.warn(
                f"CUDA reported as available but is not usable ({exc}); "
                "falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )

    return torch.device("cpu")
