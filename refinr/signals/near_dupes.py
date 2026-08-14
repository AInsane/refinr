"""Chunks that say the same thing in different words.

Tier 1: needs embeddings, because this is exactly what hashing cannot do. The
same policy reworded, a page lightly edited between versions, two teams
documenting one process -- byte comparison sees nothing, cosine similarity sees
0.95.

Cost in a RAG system: your top-5 becomes five phrasings of one fact. The
context window is spent, and the model sees one fact's worth of information in
five slots.

Not always a defect. Deliberate policy versioning looks identical to accidental
duplication from here, which is why the domain profile can re-rank these -- and
why refinr reports them either way rather than deciding on the user's behalf.
"""

import numpy as np

from ..finding import Evidence, Finding, Severity

TIER = 1
NAME = "near_dupes"
DETECTS = ["near_duplicate"]


def run(corpus, config):
    if len(corpus.chunks) < 2:
        return []

    similarity = corpus.similarity_matrix()
    # Upper triangle only: (a,b) and (b,a) are the same pair.
    rows, cols = np.triu_indices_from(similarity, k=1)
    scores = similarity[rows, cols]
    hits = np.where(scores >= config.near_duplicate)[0]

    findings = []
    for h in sorted(hits, key=lambda i: -scores[i]):
        a, b = int(rows[h]), int(cols[h])
        first, second = corpus.chunks[a], corpus.chunks[b]
        score = float(scores[h])

        identical = first.fp == second.fp
        findings.append(Finding(
            signal=NAME,
            kind="near_duplicate",
            severity=Severity.MEDIUM if score >= 0.97 else Severity.LOW,
            evidence=Evidence.STRUCTURAL,
            message=(f"{score:.3f} similar: {first.source} <-> {second.source} "
                     f"-- \"{first.preview(40)}...\""),
            advice=("identical text -- keep one" if identical else
                    "same content, different wording: keep the more specific one, "
                    "or confirm this is deliberate versioning"),
            chunk_ids=[a, b],
            sources=sorted({first.source, second.source}),
            detail={"similarity": round(score, 4), "identical_text": identical},
            fix_mode="hybrid",  # code finds the pair; choosing which wording
                                # survives is judgment

        ))

    return findings
