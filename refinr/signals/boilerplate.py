"""Text that appears in many separate documents: footers, headers, disclaimers.

Deliberately distinct from `exact_dupes`, because the *advice* differs. A
duplicated paragraph is usually a mistake -- delete one. Boilerplate is usually
intentional, sometimes legally required, and must not simply be deleted.

Its cost is different too. Boilerplate is semantically similar to every
document that carries it, so it sits near the centre of embedding space and
gets retrieved for queries it cannot answer. It crowds real answers out of the
top-k. The fix is almost never "delete the disclaimer" -- it is "strip it at
index time, keep it at display time".
"""

from collections import defaultdict

from ..finding import Evidence, Finding, Severity

TIER = 0
NAME = "boilerplate"
DETECTS = ["boilerplate"]


def run(corpus, config):
    total_files = len(corpus.sources())
    threshold = min(config.boilerplate_min_files, max(2, total_files))

    by_fingerprint = defaultdict(list)
    for paragraph in corpus.paragraphs:
        by_fingerprint[paragraph.fp].append(paragraph)

    findings = []
    for group in by_fingerprint.values():
        sources = sorted({p.source for p in group})
        if len(sources) < threshold:
            continue

        share = len(sources) / max(total_files, 1)
        severity = Severity.HIGH if share >= 0.5 else Severity.MEDIUM

        findings.append(Finding(
            signal=NAME,
            kind="boilerplate",
            severity=severity,
            evidence=Evidence.STRUCTURAL,
            message=(f"boilerplate in {len(sources)}/{total_files} files "
                     f"({share:.0%}): \"{group[0].preview(48)}...\""),
            advice=("strip at index time, keep at display time -- it is retrieved "
                    "for queries it cannot answer and displaces real content"),
            paragraph_ids=[p.id for p in group],
            sources=sources,
            detail={"files": len(sources), "share": round(share, 3),
                    "words": group[0].words},
        ))

    return findings
