"""Persistent chunk and embedding cache, keyed by content hash.

Two jobs:

1. **Never re-embed unchanged text.** Embedding is the slow part of every run.
   Keying on a content fingerprint means editing one file in a thousand-document
   corpus costs one embedding call, not a thousand.

2. **Remember what the corpus already contains**, so a newly-arrived document
   can be checked *against it* -- is this a duplicate of something we already
   have? Does it collide with an existing answer? That check is impossible
   without persistent state, and it is the main reason this module exists
   rather than being a run-local dict.

The cache is invalidated wholesale if the embedding model changes: vectors from
two different models are not comparable, and silently mixing them would corrupt
every similarity number downstream.
"""

import json
from pathlib import Path

import numpy as np

STORE_DIRNAME = ".refinr"


class Store:
    def __init__(self, root, model):
        self.root = Path(root).expanduser().resolve() / STORE_DIRNAME / "store"
        self.model = model
        self._fps = []                  # row order of self._vectors
        self._row = {}                  # fp -> row index
        self._vectors = None            # (N, D) float32, L2-normalized
        self._load()

    # --- persistence -----------------------------------------------------
    @property
    def _meta_path(self):
        return self.root / "keys.json"

    @property
    def _vectors_path(self):
        return self.root / "vectors.npy"

    @property
    def _corpus_path(self):
        return self.root / "corpus.json"

    def _load(self):
        if not self._meta_path.is_file() or not self._vectors_path.is_file():
            return
        meta = json.loads(self._meta_path.read_text())
        if meta.get("model") != self.model:
            # Different embedding model: vectors are not comparable. Drop them.
            return
        self._fps = meta["fps"]
        self._row = {fp: i for i, fp in enumerate(self._fps)}
        self._vectors = np.load(self._vectors_path)

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None:
            np.save(self._vectors_path, self._vectors)
            self._meta_path.write_text(json.dumps(
                {"model": self.model, "fps": self._fps, "dim": int(self._vectors.shape[1])}
            ))

    # --- embeddings ------------------------------------------------------
    def cached(self, fps):
        return sum(1 for fp in fps if fp in self._row)

    def vectors_for(self, items, embed_fn):
        """Return an (N, D) matrix for items, embedding only cache misses.

        items: objects with .fp and .text (Chunk or Paragraph).
        embed_fn: list[str] -> list[list[float]]
        """
        missing = []
        seen = set()
        for item in items:
            if item.fp not in self._row and item.fp not in seen:
                missing.append(item)
                seen.add(item.fp)

        if missing:
            fresh = np.array(embed_fn([m.text for m in missing]), dtype=np.float32)
            fresh /= np.clip(np.linalg.norm(fresh, axis=1, keepdims=True), 1e-9, None)
            if self._vectors is None:
                self._vectors = fresh
            else:
                self._vectors = np.vstack([self._vectors, fresh])
            for offset, item in enumerate(missing):
                self._row[item.fp] = len(self._fps) + offset
            self._fps.extend(m.fp for m in missing)

        rows = [self._row[item.fp] for item in items]
        return self._vectors[rows], len(missing)

    # --- corpus snapshot -------------------------------------------------
    def save_corpus(self, paragraphs):
        self.root.mkdir(parents=True, exist_ok=True)
        self._corpus_path.write_text(json.dumps([
            {"id": p.id, "source": p.source, "position": p.position,
             "text": p.text, "fp": p.fp}
            for p in paragraphs
        ]))

    def load_corpus(self):
        """Previously-seen paragraphs as plain dicts, or [] if the store is new."""
        if not self._corpus_path.is_file():
            return []
        return json.loads(self._corpus_path.read_text())

    def known_fingerprints(self):
        return {p["fp"] for p in self.load_corpus()}
