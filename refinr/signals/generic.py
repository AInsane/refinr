"""Text that says what every other document says.

Measures cosine similarity to the corpus centroid. Text near the centroid is
*typical of everything* -- which is precisely the property that makes it get
retrieved for queries it cannot answer.

## Why this signal exists and `filler` does not

Three tier-0 heuristics for "vacuous text" were tried against
`testdata/handbook` and all three failed:

  1. Specificity markers (proper nouns, digits, coded tokens) -- anti-correlated.
     The boilerplate footer scored 0.250, the *highest* of any real paragraph,
     because "People Operations" is capitalized. Real technical prose scored
     0.000, because it is specific through lowercase domain nouns ("rollback",
     "staging", "penetration test").
  2. Common-word ratio with a hand-written list -- separated cleanly, but the
     list had been written after reading the filler. Re-run with a generic
     English frequency list, the separation vanished entirely and two real
     paragraphs outranked both filler paragraphs.
  3. Centroid similarity -- this one. Works, but for a narrower claim than
     "filler".

So this module claims only what it measured. On the handbook corpus the
boilerplate footer sits at 0.861 against a next-highest of 0.785 -- a real gap.
The off-topic filler in `misc.md` sits at 0.665 and this signal does not see it,
because centroid distance measures *typicality*, not informativeness.

Detecting genuinely vacuous-but-atypical text needs an LLM judgment (tier 2),
graded against the benchmark. It is not shipped yet, and this signal does not
pretend to cover it.
"""

import numpy as np

from ..finding import Evidence, Finding, Severity

TIER = 1
NAME = "generic"
DETECTS = ["generic"]

# Chosen from the observed gap on testdata/handbook (0.861 vs 0.785), then
# widened for margin. Provisional until the benchmark can tune it against
# several corpora -- a threshold fitted to one corpus is a guess.
DEFAULT_THRESHOLD = 0.82


def run(corpus, config):
    # Paragraph level, not chunk level. Run against chunks, this signal fired on
    # "The deployment freeze runs from December 20th..." (real content that
    # merely had the footer packed onto it) while missing the footer itself.
    # A threshold calibrated on paragraphs does not survive packing.
    paragraphs = corpus.embedded_paragraphs
    if len(paragraphs) < 4:
        return []

    vectors = corpus.paragraph_vectors
    centroid = vectors.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
    similarity = vectors @ centroid

    threshold = getattr(config, "generic_threshold", DEFAULT_THRESHOLD)

    findings = []
    for paragraph, score in zip(paragraphs, similarity):
        if score < threshold:
            continue
        findings.append(Finding(
            signal=NAME,
            kind="generic",
            severity=Severity.MEDIUM,
            evidence=Evidence.STRUCTURAL,
            message=(f"reads like every other document ({score:.3f} to corpus centre): "
                     f"\"{paragraph.preview(46)}...\""),
            advice=("sits near the centre of embedding space, so it is retrieved for "
                    "queries it cannot answer -- check whether it carries any content "
                    "specific to its own document"),
            paragraph_ids=[paragraph.id],
            sources=[paragraph.source],
            detail={"centroid_similarity": round(float(score), 4)},
            fix_mode="llm",  # rewriting vague text into specific text needs a model

        ))

    return findings
