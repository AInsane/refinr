"""Chunks far outside the target size.

Both tails hurt retrieval, in opposite ways:

  too short -- a fragment carries too little context to answer anything. Often
    a stranded heading, a table row, or a sentence orphaned by a bad split.
  too long -- one embedding is asked to represent several unrelated ideas. The
    vector lands between them and is a good match for none. Usually means the
    splitter found no paragraph breaks (minified HTML, a wall of text).

This is INFO severity, not a defect claim. Outliers are a prompt to look at the
chunking strategy, and chunking is a tuning decision the corpus owner makes --
refinr's job here is to surface the distribution, not to overrule it.
"""

from ..finding import Evidence, Finding, Severity

TIER = 0
NAME = "length"
DETECTS = ["length_outlier"]


def run(corpus, config):
    findings = []
    for chunk in corpus.chunks:
        words = chunk.words
        if words < config.length_short:
            problem, advice = "too short", "too little context to answer anything on its own"
        elif words > config.length_long:
            problem, advice = ("too long",
                               "several ideas in one vector -- the embedding matches none of "
                               "them well; check whether the splitter found paragraph breaks")
        else:
            continue

        findings.append(Finding(
            signal=NAME,
            kind="length_outlier",
            severity=Severity.INFO,
            evidence=Evidence.STRUCTURAL,
            message=f"chunk {problem} ({words} words) in {chunk.source}: "
                    f"\"{chunk.preview(40)}...\"",
            advice=advice,
            chunk_ids=[chunk.id],
            sources=[chunk.source],
            detail={"words": words, "problem": problem},
        ))

    return findings
