"""Inventory: WHAT'S IN the corpus.

Two outputs with two different levels of trust, deliberately kept apart:

  1. MEASURED coverage -- paragraphs grouped by meaning, counted. Which
     paragraphs travel together and how much of the corpus each group is
     are facts of the text (Evidence.STRUCTURAL); they stay true whatever
     anyone decides to call the groups.
  2. PROPOSED labels -- one LLM call per group suggesting a name. A label
     is a guess until a human confirms it in `refinr tag`, same contract
     as every other guess in this tool.

## How the grouping works (worked example)

Paragraph embeddings are vectors pointing in a "meaning direction"; the dot
product of two of them is their cosine similarity (1.0 = same meaning). The
algorithm is greedy agglomerative clustering with centroid linkage:

  1. every paragraph starts as its own cluster; a cluster's *centroid* is
     the normalized average of its members ("the average meaning");
  2. find the two most similar centroids and merge them;
  3. repeat until no pair is more similar than `topic_similarity`.

On the handbook test corpus: the security-training paragraph in security.md
and its near-copy in handbook.md sit at ~0.95 similarity and merge first;
deployment paragraphs join each other around ~0.75-0.85; nothing reaches the
kitchen-restocking paragraph (~0.4 to everything), so it stays alone. Groups
of one go to an "unclustered" bucket -- measured, reported, never sent for
labeling, because a one-paragraph "topic" is noise.

Chosen over k-means because it is deterministic (no random starting points),
has no "pick k up front" problem, and fits in a sentence. Naive O(n^3) is
fine at this tool's scale (hundreds of paragraphs, not millions).

`topic_similarity` = 0.70 is provisional: fitted to the one handbook corpus,
same standing as `generic`'s 0.82. The benchmark harness will earn it a
real value.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .models import CHAT_MODEL, EMBED_MODEL, complete, parse_json
from .tags import TOPIC, Tag

INVENTORY_PATH = ".refinr/inventory.json"

LABEL_SYSTEM = """You name topics in document collections. You will see several passages that
an analysis grouped together because they discuss the same subject.

Reply with JSON only, exactly this shape:

{"label": "2-5 word topic name", "reasoning": "one sentence citing what you saw"}

Name the shared subject, not the document type. Be concrete: use the
vocabulary the passages themselves use."""


@dataclass
class Cluster:
    paragraphs: list                 # member Paragraph objects
    centroid: np.ndarray
    label_proposed: str = ""
    reasoning: str = ""

    @property
    def id(self):
        """Stable across re-runs while the member texts are unchanged."""
        joined = "".join(sorted(p.fp for p in self.paragraphs))
        return hashlib.sha256(joined.encode()).hexdigest()[:8]

    @property
    def words(self):
        return sum(p.words for p in self.paragraphs)

    def sources(self):
        counts = {}
        for p in self.paragraphs:
            counts[p.source] = counts.get(p.source, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _merge_closest(members, centroids, vectors, stop):
    """One round of 'merge the two most similar clusters'.

    members: list of index-lists into `vectors`; centroids: matching (K, D)
    array. stop(best) decides when to quit. Mutates nothing; returns new
    (members, centroids) or None when done.
    """
    while len(members) > 1:
        sims = centroids @ centroids.T
        np.fill_diagonal(sims, -1.0)
        i, j = np.unravel_index(int(np.argmax(sims)), sims.shape)
        if stop(float(sims[i, j]), len(members)):
            return members, centroids
        merged = members[i] + members[j]
        centroid = vectors[merged].mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-9)
        members = [m for k, m in enumerate(members) if k not in (i, j)] + [merged]
        centroids = np.vstack([
            np.delete(centroids, (i, j), axis=0), centroid[None, :]])
    return members, centroids


def cluster(corpus, config: Config):
    """Group paragraphs by meaning. Returns (clusters, unclustered_paragraphs).

    unclustered = groups of one, plus paragraphs too short to embed (<10
    words) -- together with the clusters they account for every paragraph,
    so coverage percentages add up.
    """
    vectors = corpus.paragraph_vectors
    embedded = corpus.embedded_paragraphs
    if len(embedded) == 0:
        return [], list(corpus.paragraphs)

    members = [[i] for i in range(len(embedded))]
    members, centroids = _merge_closest(
        members, vectors.copy(), vectors,
        stop=lambda best, _n: best < config.topic_similarity)

    # Cap how many topics can exist (bounds labeling calls). If the threshold
    # left more groups than the cap, fold the most similar *groups* together
    # -- the least destructive way to shorten the list.
    multi = [m for m in members if len(m) > 1]
    single = [m for m in members if len(m) == 1]
    if len(multi) > config.max_topics:
        cents = np.array([_centroid(vectors, m) for m in multi])
        multi, _ = _merge_closest(
            multi, cents, vectors,
            stop=lambda _best, n: n <= config.max_topics)

    clusters = [Cluster(paragraphs=[embedded[i] for i in m],
                        centroid=_centroid(vectors, m))
                for m in multi]
    clusters.sort(key=lambda c: -c.words)

    unclustered = [embedded[m[0]] for m in single]
    short = [p for p in corpus.paragraphs if p.words < 10]
    return clusters, unclustered + short


def _centroid(vectors, member_indices):
    c = vectors[member_indices].mean(axis=0)
    return c / max(float(np.linalg.norm(c)), 1e-9)


def label(clusters, corpus, config: Config, attempts=2, on_progress=None):
    """Ask the model to name each cluster. Mutates label_proposed/reasoning.

    One call per cluster, ~1.1K tokens of the 8K window each -- the guard in
    models.complete() cannot fire on a correctly capped sample, which is the
    reason the cap lives in config (label_words) and not in hope.
    """
    vectors = corpus.paragraph_vectors
    index = {p.id: i for i, p in enumerate(corpus.embedded_paragraphs)}

    for n, clu in enumerate(clusters, 1):
        ranked = sorted(
            clu.paragraphs,
            key=lambda p: -float(vectors[index[p.id]] @ clu.centroid))
        picked, words = [], 0
        for p in ranked[:6]:
            if picked and words + p.words > config.label_words:
                break
            picked.append(p)
            words += p.words

        files = sorted({p.source for p in picked})
        excerpts = "\n---\n".join(f"[{p.source}] {p.text}" for p in picked)
        prompt = (f"Passages grouped together (from files: {', '.join(files)}):\n\n"
                  f"{excerpts}")

        if on_progress:
            on_progress(f"  naming group {n}/{len(clusters)} "
                        f"({clu.words} words, {len(clu.paragraphs)} paragraphs)...")
        data = None
        for _ in range(attempts):
            data = parse_json(complete(prompt, system=LABEL_SYSTEM,
                                       temperature=0, max_tokens=150))
            if isinstance(data, dict) and str(data.get("label", "")).strip():
                break
            data = None
        if data:
            clu.label_proposed = str(data["label"]).strip()
            clu.reasoning = str(data.get("reasoning", "")).strip()
        elif on_progress:
            on_progress("    the model gave nothing usable; group stays unnamed")
    return clusters


def to_tags(clusters):
    """Topic proposals for the confirm flow. Unnamed clusters produce no tag
    -- there is nothing for a human to confirm."""
    return [Tag(group=TOPIC, label=c.label_proposed, reasoning=c.reasoning,
                evidence=sorted(c.sources()), key=c.id)
            for c in clusters if c.label_proposed]


def snapshot(corpus, clusters, unclustered, config: Config):
    """The persisted record. Paragraph fingerprints double as the staleness
    key: the passport re-ingests (free) and compares."""
    total_words = sum(p.words for p in corpus.paragraphs)
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "params": {"topic_similarity": config.topic_similarity,
                   "max_topics": config.max_topics},
        "totals": {"files": len(corpus.sources()),
                   "paragraphs": len(corpus.paragraphs),
                   "words": total_words},
        "clusters": [{
            "id": c.id,
            "label_proposed": c.label_proposed,
            "paragraphs": len(c.paragraphs),
            "words": c.words,
            "pct_words": round(100 * c.words / max(total_words, 1), 1),
            "sources": c.sources(),
            "paragraph_fps": sorted(p.fp for p in c.paragraphs),
            "representative": c.paragraphs[0].preview(120),
        } for c in clusters],
        "unclustered": {
            "paragraphs": len(unclustered),
            "words": sum(p.words for p in unclustered),
            "pct_words": round(100 * sum(p.words for p in unclustered)
                               / max(total_words, 1), 1),
            "paragraph_fps": sorted(p.fp for p in unclustered),
        },
    }


def save(folder, data):
    path = Path(folder).expanduser().resolve() / INVENTORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def load(folder):
    path = Path(folder).expanduser().resolve() / INVENTORY_PATH
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def render(data, width=74):
    """CLI view. Engineer vocabulary is fine here; the two voices are not:
    counts are measured, labels are guesses, and the layout says which is
    which on every line."""
    lines = ["=" * width,
             f"inventory -- {data['totals']['files']} files, "
             f"{data['totals']['paragraphs']} paragraphs, "
             f"{data['totals']['words']} words",
             "=" * width,
             "MEASURED COVERAGE  [structural -- true whatever the labels turn out to be]"]
    for c in data["clusters"]:
        srcs = " ".join(f"{s}({n})" for s, n in list(c["sources"].items())[:4])
        lines.append(f"  {c['pct_words']:>4}%  {c['paragraphs']} paragraphs, "
                     f"{c['words']} words   {srcs}")
        if c["label_proposed"]:
            lines.append(f"         proposed: \"{c['label_proposed']}\"   [? unconfirmed guess]")
        else:
            lines.append("         (no name proposed)")
        lines.append(f"         e.g. \"{c['representative']}\"")
    unc = data["unclustered"]
    if unc["paragraphs"]:
        lines.append(f"  {unc['pct_words']:>4}%  {unc['paragraphs']} paragraphs, "
                     f"{unc['words']} words   (no group -- one-off or too short)")
    lines += ["-" * width,
              "labels are model guesses -- run `refinr tag` to confirm or reject them"]
    return "\n".join(lines)
