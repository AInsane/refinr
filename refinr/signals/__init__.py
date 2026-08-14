"""Signal registry and protocol.

## The contract

A signal is a module (or any object) exposing:

    TIER: int          0 = free/deterministic, 1 = needs embeddings, 2 = needs an LLM
    NAME: str          stable identifier, appears in reports and benchmark output
    DETECTS: list[str] defect kinds it claims to find; the benchmark matches
                       these against testdata manifests
    run(corpus, config) -> list[Finding]

That is the whole interface. It is written as something a third party could
implement against, even though nothing external implements it yet -- the point
is that adding a signal touches one file instead of three.

## Why tiers

Cost. Tier 0 runs in milliseconds and needs no model, so it can run on every
save. Tier 2 costs an LLM call per chunk. Running cheap signals first and
stopping early (`--tier 0`) is the difference between a tool you run once and a
tool you leave on.
"""

from importlib import import_module

_MODULE_NAMES = [
    "exact_dupes",
    "boilerplate",
    "length",
    "near_dupes",
    "generic",
]

_registry = None


def registry():
    """All available signals, lazily imported, ordered by tier then name."""
    global _registry
    if _registry is None:
        modules = []
        for name in _MODULE_NAMES:
            module = import_module(f".{name}", __package__)
            _validate(module, name)
            modules.append(module)
        _registry = sorted(modules, key=lambda m: (m.TIER, m.NAME))
    return _registry


def _validate(module, name):
    for attribute in ("TIER", "NAME", "DETECTS", "run"):
        if not hasattr(module, attribute):
            raise TypeError(f"signal '{name}' is missing required attribute '{attribute}'")


def select(config):
    """Signals permitted by config: within max_tier, and enabled if a list is set."""
    chosen = [s for s in registry() if s.TIER <= config.max_tier]
    if config.enabled:
        allowed = set(config.enabled)
        chosen = [s for s in chosen if s.NAME in allowed]
    return chosen


def run_all(corpus, config, on_signal=None):
    """Run selected signals in tier order. Returns list[Finding].

    A signal that raises is reported and skipped rather than killing the run --
    one broken signal should not cost the user the other nine.
    """
    findings = []
    for signal in select(config):
        if on_signal:
            on_signal(signal)
        try:
            findings.extend(signal.run(corpus, config))
        except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
            print(f"  signal '{signal.NAME}' failed: {type(exc).__name__}: {exc}")
    return sorted(findings, key=lambda f: f.sort_key())
