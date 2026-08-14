"""What a signal produces.

Every finding records *where its evidence came from*, because the report must
not present a synthetic guess in the same voice as a measured fact. That
conflation is the fastest way for a tool like this to lose trust.
"""

from dataclasses import dataclass, field
from enum import Enum


class Evidence(str, Enum):
    """How much weight a finding's evidence can bear."""

    STRUCTURAL = "structural"          # corpus-intrinsic; true regardless of queries
    SYNTHETIC_QUERIES = "synthetic"    # generated questions; an estimate, labelled as one
    REAL_QUERIES = "real-queries"      # the user's actual traffic
    ABLATION = "ablation"              # measured: removed it, quality changed

    @property
    def confidence(self):
        return {
            Evidence.STRUCTURAL: "high",
            Evidence.SYNTHETIC_QUERIES: "low (estimate)",
            Evidence.REAL_QUERIES: "high",
            Evidence.ABLATION: "measured",
        }[self]


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def rank(self):
        return {"high": 0, "medium": 1, "low": 2, "info": 3}[self.value]


@dataclass
class Finding:
    signal: str                      # which signal produced this
    kind: str                        # defect kind, matched against benchmark manifests
    severity: Severity
    evidence: Evidence
    message: str                     # one line: what is wrong
    advice: str = ""                 # what to do about it
    fix_mode: str = "deterministic"  # how a future fix would run:
                                     #   deterministic -- pure code, reproducible
                                     #   llm           -- needs a model call per fix
                                     #   hybrid        -- code finds, judgment chooses
                                     # feeds the compile-pattern clean step; nothing
                                     # renders it yet, deliberately
    chunk_ids: list = field(default_factory=list)
    paragraph_ids: list = field(default_factory=list)
    sources: list = field(default_factory=list)   # file paths involved
    detail: dict = field(default_factory=dict)    # signal-specific extras

    def downgrade(self, severity, advice=None):
        """Re-rank without removing.

        The domain profile is allowed to change emphasis and remediation, never
        to delete a finding -- a profile that can hide findings becomes a way to
        make the report say whatever the user wants.
        """
        self.severity = severity
        if advice:
            self.advice = advice
        return self

    def sort_key(self):
        return (self.severity.rank, self.signal, self.message)
