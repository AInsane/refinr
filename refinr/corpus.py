"""The object every signal receives.

Bundles what a signal might need so the protocol stays a single argument and
adding a capability later does not change every signal's function signature.

Embeddings are lazy: tier-0 signals never touch them, so a `--tier 0` run makes
no model calls at all and finishes in milliseconds.
"""

import numpy as np

from . import ingest


class Corpus:
    def __init__(self, folder, config, store=None):
        self.folder = folder
        self.config = config
        self.store = store
        self.paragraphs, self.chunks = ingest.load(
            folder, target_words=config.target_words, min_words=config.min_words,
        )
        self._vectors = None
        self._paragraph_vectors = None
        self._paragraph_index = {}
        self.embed_calls = 0

    # --- lazy embeddings -------------------------------------------------
    @property
    def vectors(self):
        """(N, D) L2-normalized chunk embeddings. Computed on first access."""
        if self._vectors is None:
            from .models import embed
            if self.store is not None:
                self._vectors, self.embed_calls = self.store.vectors_for(self.chunks, embed)
            else:
                raw = np.array(embed([c.text for c in self.chunks]), dtype=np.float32)
                self._vectors = raw / np.clip(
                    np.linalg.norm(raw, axis=1, keepdims=True), 1e-9, None)
                self.embed_calls = len(self.chunks)
        return self._vectors

    @property
    def paragraph_vectors(self):
        """(P, D) paragraph embeddings.

        Separate from chunk vectors on purpose. Any signal comparing *text* has
        to work at paragraph level -- chunk composition is our packing artifact,
        and a threshold calibrated on paragraphs does not transfer to chunks.
        Learned twice: once from the missed boilerplate, once from `generic`
        firing on a chunk that merely contained some.
        """
        if self._paragraph_vectors is None:
            from .models import embed
            usable = [p for p in self.paragraphs if p.words >= 10]
            self._paragraph_index = {p.id: i for i, p in enumerate(usable)}
            if self.store is not None:
                self._paragraph_vectors, calls = self.store.vectors_for(usable, embed)
                self.embed_calls += calls
            else:
                raw = np.array(embed([p.text for p in usable]), dtype=np.float32)
                self._paragraph_vectors = raw / np.clip(
                    np.linalg.norm(raw, axis=1, keepdims=True), 1e-9, None)
                self.embed_calls += len(usable)
        return self._paragraph_vectors

    @property
    def embedded_paragraphs(self):
        """Paragraphs matching paragraph_vectors, row for row."""
        self.paragraph_vectors  # noqa: B018 - populates the index
        return [self.paragraphs[pid] for pid in self._paragraph_index]

    @property
    def embedded(self):
        return self._vectors is not None or self._paragraph_vectors is not None

    # --- retrieval -------------------------------------------------------
    def search(self, query_vector, k=5):
        """[(chunk_index, score)] for the k nearest chunks."""
        query = np.asarray(query_vector, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-9)
        scores = self.vectors @ query
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def similarity_matrix(self):
        return self.vectors @ self.vectors.T

    # --- convenience -----------------------------------------------------
    def paragraph(self, paragraph_id):
        return self.paragraphs[paragraph_id]

    def sources(self):
        return sorted({p.source for p in self.paragraphs})

    def __repr__(self):
        return (f"Corpus({len(self.chunks)} chunks, {len(self.paragraphs)} paragraphs, "
                f"{len(self.sources())} files)")
