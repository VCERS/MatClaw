"""Process-local cache for pretrained ML potentials.

``matgl.load_model()`` and ``matcalc.load_fp()`` were called on every tool
invocation. The weights are already on disk after the first use, so this is not
a download — it is deserializing the checkpoint, constructing the torch module
and moving it to the GPU, once per call. For a single-structure prediction the
inference itself is milliseconds, so nearly all of the tool's wall time was
model construction, repeated in a server process that never restarts.

Bounded on purpose
------------------
An unbounded cache would be a VRAM leak. Workers are separate OS processes, each
pinned to one GPU by ``CUDA_VISIBLE_DEVICES`` (see the deployment's
``matclaw-entrypoint.sh``), and each holds its own copy of whatever it has
loaded. A long-lived worker that sees several model names would accumulate all
of them on its card.

So this is a small explicit LRU rather than ``functools.lru_cache``: eviction
has to *do* something. Dropping the last reference is not enough to return the
memory — CPython frees the tensors when the refcount hits zero, but torch keeps
the freed blocks in its own caching allocator, so the VRAM stays checked out
until ``empty_cache()`` is called. ``lru_cache`` gives no eviction hook.

Thread safety
-------------
Guarded by a lock. A duplicated concurrent load would not be *wrong*, but it
would briefly hold two copies of a model on one GPU, which is exactly the
failure this module exists to avoid.
"""

import os
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple

# How many distinct models one worker keeps resident. Two covers the common case
# — a potential plus a property model — while leaving headroom on a card that is
# also running the calculation itself. Raise it on a large card via
# MATCLAW_MODEL_CACHE_SIZE if a workload cycles between more than that.
DEFAULT_CACHE_SIZE = 2


def _cache_size() -> int:
    raw = os.getenv("MATCLAW_MODEL_CACHE_SIZE", "").strip()
    if not raw:
        return DEFAULT_CACHE_SIZE
    try:
        # 0 disables caching entirely, which is the escape hatch if a model ever
        # turns out to carry per-call state.
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_CACHE_SIZE


_lock = threading.Lock()
# (kind, name) -> loaded object. Ordered by least-recently-used first.
_cache: "OrderedDict[Tuple[str, str], Any]" = OrderedDict()


def _release_vram() -> None:
    """Hand freed blocks back to the driver after an eviction.

    Best-effort: torch may not be importable in a CPU-only test environment, and
    a failure here costs memory, not correctness.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - see docstring
        pass


def _get_or_load(kind: str, name: str, loader: Callable[[str], Any]) -> Any:
    limit = _cache_size()
    if limit == 0:
        return loader(name)

    key = (kind, name)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]

    # Loaded outside the lock: a cold load takes seconds, and holding the lock
    # across it would serialize every worker thread behind the first request for
    # a new model. The cost of losing that race is one duplicate load, which the
    # insert below discards.
    loaded = loader(name)

    with _lock:
        if key in _cache:  # another thread won; keep theirs so callers share one
            _cache.move_to_end(key)
            return _cache[key]

        _cache[key] = loaded
        evicted = False
        while len(_cache) > limit:
            _cache.popitem(last=False)
            evicted = True

    if evicted:
        _release_vram()
    return loaded


def load_potential(name: str) -> Any:
    """A matgl pretrained model, cached per process.

    Wraps ``matgl.load_model``. Import is deferred so that importing this module
    does not pull in torch — the tool modules already import lazily to keep
    server start-up from paying for every optional dependency.
    """

    def _load(model_name: str) -> Any:
        import matgl

        return matgl.load_model(model_name)

    return _get_or_load("matgl", name, _load)


def load_calculator(name: str) -> Any:
    """A matcalc foundation potential (ASE calculator), cached per process.

    Wraps ``matcalc.load_fp``.
    """

    def _load(calculator_name: str) -> Any:
        import matcalc as mtc

        return mtc.load_fp(calculator_name)

    return _get_or_load("matcalc", name, _load)


def cache_info() -> Dict[str, Any]:
    """What this worker currently holds. For diagnostics and tests."""
    with _lock:
        return {"limit": _cache_size(), "entries": [f"{kind}:{name}" for kind, name in _cache]}


def clear_cache() -> None:
    """Drop everything. Used by tests; also the manual lever if a card needs
    freeing without restarting the worker."""
    with _lock:
        _cache.clear()
    _release_vram()
