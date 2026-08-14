"""Render findings, grouped by how much their evidence can bear.

The grouping is the point. A structural fact ("this paragraph is in four files")
and a synthetic estimate ("a generated question failed to retrieve this") are
not the same kind of claim, and presenting them in one undifferentiated list is
how a tool like this loses trust the first time a low-confidence guess is wrong.
"""

from collections import Counter, defaultdict

from .finding import Evidence, Severity

WIDTH = 74

_EVIDENCE_HEADINGS = {
    Evidence.STRUCTURAL:
        ("STRUCTURAL", "properties of the corpus itself -- true regardless of who queries it"),
    Evidence.REAL_QUERIES:
        ("MEASURED AGAINST REAL QUERIES", "from your own traffic"),
    Evidence.ABLATION:
        ("MEASURED BY ABLATION", "removed it and the answers changed"),
    Evidence.SYNTHETIC_QUERIES:
        ("ESTIMATED FROM GENERATED QUERIES", "low confidence -- generated questions "
         "are not real user questions"),
}

# Most-trustworthy first.
_ORDER = [Evidence.ABLATION, Evidence.REAL_QUERIES, Evidence.STRUCTURAL,
          Evidence.SYNTHETIC_QUERIES]


def render(corpus, findings, signals_run, elapsed=None, max_per_signal=4):
    lines = []
    add = lines.append

    add("=" * WIDTH)
    add("refinr")
    add("=" * WIDTH)

    words = sum(p.words for p in corpus.paragraphs)
    add(f"{len(corpus.sources())} files, {len(corpus.paragraphs)} paragraphs, "
        f"{len(corpus.chunks)} chunks, {words:,} words")
    ran = ", ".join(s.NAME for s in signals_run) or "none"
    add(f"signals: {ran}")
    if elapsed is not None:
        embedded = "no model calls" if not corpus.embedded else f"{corpus.embed_calls} embedded"
        add(f"time: {elapsed:.2f}s ({embedded})")
    add("")

    if not findings:
        add("No findings. Either the corpus is clean or the enabled signals")
        add("cannot see its problems -- check which signals ran, above.")
        return "\n".join(lines)

    by_evidence = defaultdict(list)
    for finding in findings:
        by_evidence[finding.evidence].append(finding)

    for evidence in _ORDER:
        group = by_evidence.get(evidence)
        if not group:
            continue
        heading, caveat = _EVIDENCE_HEADINGS[evidence]
        add("-" * WIDTH)
        add(f"{heading}   [confidence: {evidence.confidence}]")
        add(f"  {caveat}")
        add("-" * WIDTH)
        add("")
        _render_group(add, group, max_per_signal)

    add("=" * WIDTH)
    _render_summary(add, corpus, findings)
    add("=" * WIDTH)
    return "\n".join(lines)


def _render_group(add, group, max_per_signal):
    shown = Counter()
    for finding in group:
        shown[finding.signal] += 1
        if shown[finding.signal] > max_per_signal:
            continue
        add(f"  [{finding.severity.value:<6}] {finding.signal}")
        add(f"      {finding.message}")
        if finding.advice:
            add(f"      -> {finding.advice}")
        add("")

    for signal, count in shown.items():
        if count > max_per_signal:
            add(f"  ... and {count - max_per_signal} more from '{signal}' "
                f"(use --all to list every finding)")
            add("")


def _render_summary(add, corpus, findings):
    counts = Counter(f.severity for f in findings)
    parts = [f"{counts[s]} {s.value}" for s in Severity if counts[s]]
    add(f"{len(findings)} findings: {', '.join(parts)}")

    # Paragraphs implicated by any high/medium structural finding. Deliberately
    # a count of *affected units*, not an extrapolated percentage -- v0's
    # headline "% of your index is dead weight" mixed a measured count with an
    # estimate and presented the total as fact.
    flagged = set()
    for finding in findings:
        if finding.severity in (Severity.HIGH, Severity.MEDIUM):
            flagged.update(finding.paragraph_ids)
    if flagged:
        share = len(flagged) / max(len(corpus.paragraphs), 1)
        add(f"{len(flagged)}/{len(corpus.paragraphs)} paragraphs implicated ({share:.0%})")
