"""Configuration: the file is the source of truth, flags override it.

A company's pipeline should be versionable, reviewable, and diffable -- which
means it lives in a file they commit, not in a shell history of CLI flags.
Recipes (named presets for support KBs, legal docs, API references) become
presets of this same structure later, so there is one shape to learn.
"""

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

CONFIG_NAME = "refinr.toml"


@dataclass
class Profile:
    """What the corpus is *required* to be, so the passport can grade it.

    Every field is optional: an empty profile grades nothing. Requirements
    are graded honestly -- a topic requirement can only PASS against a
    human-confirmed topic, never against an unconfirmed model guess.
    """

    intended_use: str = ""
    required_topics: list = field(default_factory=list)
    max_duplication_pct: float | None = None    # % of passages repeated elsewhere
    min_words: int | None = None


@dataclass
class Config:
    # chunking
    target_words: int = 180
    min_words: int = 20

    # signals
    enabled: list = field(default_factory=list)     # empty = all registered
    max_tier: int = 2                               # 0 = free only

    # thresholds
    near_duplicate: float = 0.95
    boilerplate_min_files: int = 3
    generic_threshold: float = 0.82                 # centroid similarity above this = generic
    length_short: int = 25
    length_long: int = 400

    # retrieval
    top_k: int = 5
    probe_sample: int = 20

    # inventory
    topic_similarity: float = 0.70   # stop merging clusters below this; fitted
                                     # to the handbook test corpus, provisional
    max_topics: int = 12             # caps LLM labeling calls per run
    label_words: int = 600           # max words shown to the model per cluster

    # profile ([profile] section; None when the file has none)
    profile: Profile | None = None

    def merged(self, **overrides):
        """Apply non-None overrides (CLI flags) on top of the file values."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean)


def find(start):
    """Walk up from start looking for refinr.toml."""
    path = Path(start).expanduser().resolve()
    for directory in [path, *path.parents]:
        candidate = directory / CONFIG_NAME
        if candidate.is_file():
            return candidate
    return None


def load(start="."):
    """Load config from the nearest refinr.toml, or defaults. Returns (config, path)."""
    path = find(start)
    if not path:
        return Config(), None

    with path.open("rb") as handle:
        data = tomllib.load(handle)

    sections = ("chunking", "signals", "thresholds", "retrieval", "inventory")
    unknown_sections = {k for k, v in data.items() if isinstance(v, dict)} \
        - set(sections) - {"profile"}
    if unknown_sections:
        raise ValueError(f"{path}: unknown sections {sorted(unknown_sections)} "
                         "-- a typo here would otherwise silently do nothing")

    fields = {f for f in Config.__dataclass_fields__}
    flat = {}
    for section in sections:
        flat.update(data.get(section, {}))
    flat.update({k: v for k, v in data.items() if not isinstance(v, dict)})

    unknown = set(flat) - fields
    if unknown:
        raise ValueError(f"{path}: unknown settings {sorted(unknown)}")

    # [profile] is its own dataclass, not flattened into Config
    profile = None
    if "profile" in data:
        profile_fields = set(Profile.__dataclass_fields__)
        bad = set(data["profile"]) - profile_fields
        if bad:
            raise ValueError(f"{path}: unknown [profile] settings {sorted(bad)}")
        profile = Profile(**data["profile"])

    return Config(profile=profile,
                  **{k: v for k, v in flat.items() if k in fields}), path
