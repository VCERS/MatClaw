"""Process-local model cache.

The behaviour under test is not "does it memoize" — it is that the cache stays
*bounded*. Workers are separate processes each pinned to one GPU, so an
unbounded cache is a VRAM leak that only shows up under a workload cycling
through several model names, which is exactly the workload nobody runs locally.

matgl and matcalc are stubbed: this module's job is the caching policy, and
loading a real potential would make the test slow, GPU-dependent and no more
informative.
"""

import sys
import types

import pytest


@pytest.fixture
def stub_libraries(monkeypatch):
    """Fake matgl/matcalc that count how often a model is actually loaded."""
    counts = {"matgl": 0, "matcalc": 0}

    matgl = types.ModuleType("matgl")

    def load_model(name):
        counts["matgl"] += 1
        return f"potential:{name}"

    matgl.load_model = load_model

    matcalc = types.ModuleType("matcalc")

    def load_fp(name):
        counts["matcalc"] += 1
        return f"calculator:{name}"

    matcalc.load_fp = load_fp

    monkeypatch.setitem(sys.modules, "matgl", matgl)
    monkeypatch.setitem(sys.modules, "matcalc", matcalc)
    return counts


@pytest.fixture(autouse=True)
def clean_cache():
    """The cache is module state, so a leftover entry would leak between tests."""
    from utils.model_cache import clear_cache

    clear_cache()
    yield
    clear_cache()


def test_a_model_is_loaded_once_and_reused(stub_libraries):
    """The whole point: this ran on every tool invocation, in a server process
    that never restarts."""
    from utils.model_cache import load_potential

    first = load_potential("TensorNet")
    second = load_potential("TensorNet")

    assert first is second
    assert stub_libraries["matgl"] == 1


def test_different_models_are_cached_separately(stub_libraries):
    from utils.model_cache import load_potential

    load_potential("TensorNet")
    load_potential("M3GNet")

    assert stub_libraries["matgl"] == 2


def test_the_two_libraries_do_not_collide(stub_libraries):
    """A matgl model and a matcalc calculator can share a name and are not
    interchangeable — one would be handed to code expecting the other."""
    from utils.model_cache import load_calculator, load_potential

    potential = load_potential("TensorNet")
    calculator = load_calculator("TensorNet")

    assert potential != calculator
    assert stub_libraries == {"matgl": 1, "matcalc": 1}


def test_the_cache_is_bounded(stub_libraries, monkeypatch):
    """Unbounded, this accumulates every model a worker has ever seen on one
    GPU. The failure is an OOM hours into a screening run, not at start-up."""
    monkeypatch.setenv("MATCLAW_MODEL_CACHE_SIZE", "2")
    from utils.model_cache import cache_info, load_potential

    for name in ("a", "b", "c", "d"):
        load_potential(name)

    assert len(cache_info()["entries"]) == 2


def test_eviction_is_least_recently_used(stub_libraries, monkeypatch):
    """A workload alternating between two models must not evict the one it is
    about to ask for again."""
    monkeypatch.setenv("MATCLAW_MODEL_CACHE_SIZE", "2")
    from utils.model_cache import load_potential

    load_potential("a")
    load_potential("b")
    load_potential("a")       # 'a' is now the most recent, so 'b' is next out
    load_potential("c")

    assert stub_libraries["matgl"] == 3
    load_potential("a")
    assert stub_libraries["matgl"] == 3, "'a' should still be resident"
    load_potential("b")
    assert stub_libraries["matgl"] == 4, "'b' should have been evicted"


def test_caching_can_be_switched_off(stub_libraries, monkeypatch):
    """The escape hatch, if a model ever turns out to carry per-call state."""
    monkeypatch.setenv("MATCLAW_MODEL_CACHE_SIZE", "0")
    from utils.model_cache import cache_info, load_potential

    load_potential("a")
    load_potential("a")

    assert stub_libraries["matgl"] == 2
    assert cache_info()["entries"] == []


def test_a_failed_load_is_not_cached(stub_libraries, monkeypatch):
    """Otherwise a transient failure — a half-written checkpoint, a busy GPU —
    would be remembered as the answer for the life of the worker."""
    from utils import model_cache

    boom = types.ModuleType("matgl")

    def explode(name):
        raise RuntimeError("checkpoint unreadable")

    boom.load_model = explode
    monkeypatch.setitem(sys.modules, "matgl", boom)

    with pytest.raises(RuntimeError):
        model_cache.load_potential("a")

    assert model_cache.cache_info()["entries"] == []
