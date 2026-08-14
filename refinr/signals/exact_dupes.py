"""Byte-identical text repeated in the corpus.

**This is the fix for v0's headline defect.** v0 compared chunk fingerprints and
found zero duplicates in a corpus containing a footer planted verbatim in four
files -- because packing had glued each copy to a different neighbouring
paragraph, so no two chunks were byte-identical.

Comparison happens at paragraph level, where text identity actually lives.
Chunk boundaries are our invention; paragraphs are what the author wrote.
"""

from collections import defaultdict

from ..finding import Evidence, Finding, Severity

TIER = 0
NAME = "exact_dupes"
DETECTS = ["exact_duplicate"]


def run(corpus, config):
    buckets = defaultdict(list)
    for paragraph in corpus.paragraphs:
        buckets[paragraph.fp].append(paragraph)

    findings = []
    for group in buckets.values():
        if len(group) < 2:
            continue

        sources = sorted({p.source for p in group})
        # Repetition across many files is boilerplate; that signal owns it and
        # gives better advice. Here we only claim within-corpus exact repeats.
        spread_wide = len(sources) >= config.boilerplate_min_files

        findings.append(Finding(
            signal=NAME,
            kind="exact_duplicate",
            severity=Severity.LOW if spread_wide else Severity.MEDIUM,
            evidence=Evidence.STRUCTURAL,
            message=(f"identical paragraph appears {len(group)}x "
                     f"in {len(sources)} file(s): \"{group[0].preview(48)}...\""),
            advice=("see the boilerplate finding for this text"
                    if spread_wide else
                    "keep one copy; the others add nothing a retriever can use"),
            paragraph_ids=[p.id for p in group],
            sources=sources,
            detail={"count": len(group), "words": group[0].words},
        ))

    return findings
