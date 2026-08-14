"""Grade the signals against planted defects.

A benchmark case is a folder under the bench root containing a
manifest.json that declares what was planted:

    {"defects": [{"id": "footer", "kinds": ["boilerplate"],
                  "snippets": ["text that locates the paragraph"],
                  "note": "optional"}]}

Snippets locate defects: a defect's paragraphs are every paragraph whose
normalized text contains a normalized snippet. A snippet that matches
nothing is a hard error, never a warning -- that is how the harness
notices when someone edits a test file and silently breaks ground truth.

## Scoring

A finding hits a defect iff the finding's kind is in the defect's kinds
AND their paragraph sets overlap. Recall counts DEFECTS (found if at
least one matching finding overlaps it); precision counts FINDINGS (true
if it overlaps at least one matching defect). `kinds` is a list because
signals overlap by design: an exact duplicate spread across three files
is correctly claimed by both exact_dupes and boilerplate.

Defects whose kinds no selected signal DETECTS (e.g. "off_topic",
"encoding_artifact", or tier-1 kinds when running --tier 0) are reported
as *unclaimed* and excluded from all math. The manifest is ground truth;
the live registry decides what is gradeable -- when a future signal
claims a waiting kind, its defects enter scoring automatically.

The clean case has zero defects: every finding there is a false alarm.

## Discipline

Test corpora and manifests are authored and committed BEFORE the first
bench run (git history is the proof). Tuning a threshold on the same
case that grades it is flattery, not measurement -- sweep on one case,
sanity-check on another.

No model chat calls anywhere here: tier 0 is pure code, tier 1 re-uses
the fp-keyed embedding cache, so sweeps re-score in memory.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config as config_module
from . import signals
from .corpus import Corpus
from .store import Store

MANIFEST_NAME = "manifest.json"
WIDTH = 74


class BenchError(Exception):
    """Manifest rot or a bad sweep spec. Always fatal, always loud."""


@dataclass
class Defect:
    id: str
    kinds: list
    snippets: list
    note: str = ""
    paragraph_ids: set = field(default_factory=set)


# --- discovery + loading -------------------------------------------------

def discover_cases(root):
    root = Path(root).expanduser().resolve()
    if (root / MANIFEST_NAME).is_file():
        return [root]
    cases = sorted(d for d in root.iterdir()
                   if d.is_dir() and (d / MANIFEST_NAME).is_file())
    if not cases:
        raise BenchError(f"no {MANIFEST_NAME} found under {root} -- "
                         "a bench case is a folder with a manifest")
    return cases


def load_manifest(case_dir):
    path = Path(case_dir) / MANIFEST_NAME
    data = json.loads(path.read_text())

    unknown = set(data) - {"defects"}
    if unknown:
        raise BenchError(f"{path}: unknown top-level keys {sorted(unknown)}")

    defects, seen = [], set()
    for i, raw in enumerate(data.get("defects", [])):
        bad = set(raw) - {"id", "kinds", "snippets", "note"}
        if bad:
            raise BenchError(f"{path}: defect #{i} has unknown keys {sorted(bad)}")
        defect = Defect(id=raw.get("id", f"defect-{i}"),
                        kinds=raw.get("kinds", []),
                        snippets=raw.get("snippets", []),
                        note=raw.get("note", ""))
        if defect.id in seen:
            raise BenchError(f"{path}: duplicate defect id '{defect.id}'")
        seen.add(defect.id)
        if not defect.kinds:
            raise BenchError(f"{path}: defect '{defect.id}' has no kinds -- "
                             "unclaimed defects still carry a named kind")
        if not defect.snippets:
            raise BenchError(f"{path}: defect '{defect.id}' has no snippets")
        defects.append(defect)
    return defects


# --- resolution ----------------------------------------------------------

def _normalize(text):
    """ingest.fingerprint's normalization, minus the hash -- so a snippet
    still matches after line-wrapping."""
    return " ".join(text.split()).lower()


def resolve(defects, corpus, case_name):
    normalized = [(p.id, _normalize(p.text)) for p in corpus.paragraphs]
    for defect in defects:
        hits = set()
        for snippet in defect.snippets:
            want = _normalize(snippet)
            found = {pid for pid, text in normalized if want in text}
            if not found:
                raise BenchError(
                    f"{case_name}/{defect.id}: snippet matches nothing: "
                    f"\"{snippet[:60]}\" -- the manifest and the files have "
                    "drifted apart")
            hits |= found
        defect.paragraph_ids = hits
    return defects


def finding_paragraphs(finding, corpus):
    if finding.paragraph_ids:
        return set(finding.paragraph_ids)
    out = set()
    for cid in finding.chunk_ids:
        out.update(corpus.chunks[cid].paragraph_ids)
    return out


# --- scoring -------------------------------------------------------------

@dataclass
class SignalScore:
    planted: int = 0
    found: int = 0
    tp: int = 0
    fp: int = 0

    @property
    def missed(self):
        return self.planted - self.found

    @property
    def precision(self):
        total = self.tp + self.fp
        return self.tp / total if total else None

    @property
    def recall(self):
        return self.found / self.planted if self.planted else None


def score_case(defects, findings, corpus, selected):
    """Returns ({signal_name: SignalScore}, unclaimed_defects)."""
    claimed = set()
    for s in selected:
        claimed.update(s.DETECTS)

    graded = [d for d in defects if set(d.kinds) & claimed]
    unclaimed = [d for d in defects if not set(d.kinds) & claimed]

    scores = {s.NAME: SignalScore() for s in selected}
    kind_owner = {kind: s.NAME for s in selected for kind in s.DETECTS}

    for f in findings:
        f_paragraphs = finding_paragraphs(f, corpus)
        hit = any(f.kind in d.kinds and f_paragraphs & d.paragraph_ids
                  for d in graded)
        score = scores.get(f.signal)
        if score is None:
            continue
        if hit:
            score.tp += 1
        else:
            score.fp += 1

    for d in graded:
        d_signals = {kind_owner[k] for k in d.kinds if k in kind_owner}
        for name in d_signals:
            scores[name].planted += 1
        found_by = {
            f.signal for f in findings
            if f.kind in d.kinds and finding_paragraphs(f, corpus) & d.paragraph_ids
        }
        for name in found_by & set(scores):
            scores[name].found += 1

    return scores, unclaimed


def defect_level(defects, findings, corpus, selected):
    """(graded_count, found_count) at defect level -- a defect counts as
    found if ANY signal found it."""
    claimed = set()
    for s in selected:
        claimed.update(s.DETECTS)
    graded = [d for d in defects if set(d.kinds) & claimed]
    found = sum(
        1 for d in graded
        if any(f.kind in d.kinds and finding_paragraphs(f, corpus) & d.paragraph_ids
               for f in findings))
    return len(graded), found


# --- running -------------------------------------------------------------

def run_case(case_dir, tier=None, overrides=None, store=None):
    """Returns (corpus, findings, defects, selected). Config is loaded per
    case dir so each case may carry its own refinr.toml."""
    cfg, _ = config_module.load(case_dir)
    cfg = cfg.merged(max_tier=tier, **(overrides or {}))
    corpus = Corpus(case_dir, cfg, store=store)
    if not corpus.paragraphs:
        raise BenchError(f"{case_dir}: no readable documents")
    defects = resolve(load_manifest(case_dir), corpus, Path(case_dir).name)
    selected = signals.select(cfg)
    findings = signals.run_all(corpus, cfg)
    return corpus, findings, defects, selected


def _make_store(case_dir, tier):
    if tier is not None and tier < 1:
        return None
    from .models import EMBED_MODEL
    return Store(case_dir, EMBED_MODEL)


# --- sweep ---------------------------------------------------------------

def parse_sweep(spec):
    """'field=start:stop[:step]' or 'field=v1,v2,...' -> (field, [values])."""
    if "=" not in spec:
        raise BenchError(f"bad sweep spec '{spec}' -- expected field=start:stop:step "
                         "or field=v1,v2,...")
    name, _, rest = spec.partition("=")
    name = name.strip()
    default = config_module.Config()
    numeric = sorted(f for f in config_module.Config.__dataclass_fields__
                     if isinstance(getattr(default, f), (int, float)))
    if name not in numeric:
        raise BenchError(f"'{name}' is not a numeric config field -- "
                         f"sweepable: {', '.join(numeric)}")
    cast = type(getattr(default, name))

    if "," in rest:
        try:
            return name, [cast(v) for v in rest.split(",")]
        except ValueError as exc:
            raise BenchError(f"bad sweep values '{rest}': {exc}") from exc

    parts = rest.split(":")
    if len(parts) not in (2, 3):
        raise BenchError(f"bad sweep range '{rest}' -- expected start:stop[:step]")
    try:
        start, stop = cast(parts[0]), cast(parts[1])
        if len(parts) == 3:
            step = cast(parts[2])
        elif cast is int:
            step = 1
        else:
            raise BenchError("float sweeps need an explicit step: start:stop:step")
        if step <= 0:
            raise BenchError("sweep step must be positive")
    except ValueError as exc:
        raise BenchError(f"bad sweep range '{rest}': {exc}") from exc

    values, v = [], start
    while v <= stop + 1e-9:
        values.append(round(v, 10) if cast is float else v)
        v += step
    return name, values


def run_sweep(cases, name, values, tier=None):
    """[(value, {signal: SignalScore aggregated}, clean_fp)] -- one Store per
    case, shared across values, so embeddings are computed at most once."""
    stores = {case: _make_store(case, tier) for case in cases}
    rows = []
    for value in values:
        aggregate = {}
        clean_fp = 0
        for case in cases:
            corpus, findings, defects, selected = run_case(
                case, tier=tier, overrides={name: value}, store=stores[case])
            scores, _ = score_case(defects, findings, corpus, selected)
            for sig, sc in scores.items():
                agg = aggregate.setdefault(sig, SignalScore())
                agg.planted += sc.planted
                agg.found += sc.found
                agg.tp += sc.tp
                agg.fp += sc.fp
            if not defects:
                clean_fp += sum(sc.fp for sc in scores.values())
        rows.append((value, aggregate, clean_fp))
    for store in stores.values():
        if store is not None:
            store.save()
    return rows


# --- rendering -----------------------------------------------------------

def _fmt(value):
    return f"{value:.2f}" if value is not None else "   -"


def _table_row(name, sc):
    return (f"{name:<13}{sc.planted:>8}{sc.found:>7}{sc.missed:>8}"
            f"{sc.fp:>14}{_fmt(sc.precision):>11}{_fmt(sc.recall):>8}")


_HEADER = (f"{'signal':<13}{'planted':>8}{'found':>7}{'missed':>8}"
           f"{'false-alarms':>14}{'precision':>11}{'recall':>8}")


def render_case(case_name, corpus, scores, unclaimed, defects):
    lines = [f"case: {case_name}   {len(corpus.sources())} files, "
             f"{len(corpus.paragraphs)} paragraphs, "
             f"{len(defects)} defect(s)"
             + (f" ({len(unclaimed)} unclaimed)" if unclaimed else "")]
    if not defects:
        lines[-1] += " -- every finding is a false alarm"
    lines.append(_HEADER)
    for name in sorted(scores):
        lines.append(_table_row(name, scores[name]))
    for d in unclaimed:
        lines.append(f"  unclaimed: {d.id} (kinds {d.kinds} -- no selected "
                     "signal detects this)")
    return lines


def render_sweep(name, rows, moved):
    lines = [f"sweep: {name} = {rows[0][0]} .. {rows[-1][0]}   "
             f"({len(rows)} values)",
             "embeddings cached per corpus; each value re-scores in memory", ""]
    header = f"{'value':<9}"
    for sig in moved:
        header += f"{sig + ' prec':>13}{sig + ' rec':>12}"
    header += f"{'clean FPs':>11}"
    lines.append(header)
    for value, aggregate, clean_fp in rows:
        row = f"{value:<9g}"
        for sig in moved:
            sc = aggregate.get(sig, SignalScore())
            row += f"{_fmt(sc.precision):>13}{_fmt(sc.recall):>12}"
        row += f"{clean_fp:>11}"
        lines.append(row)
    return lines


# --- entry point ---------------------------------------------------------

def main(root, case=None, tier=None, sweep=None):
    try:
        cases = discover_cases(root)
        if case:
            cases = [c for c in cases if c.name == case]
            if not cases:
                raise BenchError(f"no case named '{case}' under {root}")

        if sweep:
            name, values = parse_sweep(sweep)
            rows = run_sweep(cases, name, values, tier=tier)
            keys = lambda sc: (sc.tp, sc.fp, sc.found)  # noqa: E731
            moved = sorted(
                sig for sig in rows[0][1]
                if len({keys(agg.get(sig, SignalScore())) for _, agg, _ in rows}) > 1)
            if not moved and len({fp for _, _, fp in rows}) == 1:
                print(f"sweep of '{name}' changed no signal's numbers -- "
                      "is this field consumed by any selected signal?")
                return 0
            print("\n".join(render_sweep(name, rows, moved)))
            return 0

        print("=" * WIDTH)
        print(f"refinr bench -- {len(cases)} case(s)"
              + (f", tier {tier}" if tier is not None else ""))
        print("=" * WIDTH)

        totals = {}
        graded_total = found_total = 0
        for case_dir in cases:
            store = _make_store(case_dir, tier)
            corpus, findings, defects, selected = run_case(
                case_dir, tier=tier, store=store)
            if store is not None:
                store.save()
            scores, unclaimed = score_case(defects, findings, corpus, selected)
            graded, found = defect_level(defects, findings, corpus, selected)
            graded_total += graded
            found_total += found
            for sig, sc in scores.items():
                agg = totals.setdefault(sig, SignalScore())
                agg.planted += sc.planted
                agg.found += sc.found
                agg.tp += sc.tp
                agg.fp += sc.fp

            print()
            print("\n".join(render_case(case_dir.name, corpus, scores,
                                        unclaimed, defects)))

        print()
        print("-" * WIDTH)
        print(f"aggregate across {len(cases)} case(s)")
        print(_HEADER)
        for name in sorted(totals):
            print(_table_row(name, totals[name]))
        print(f"OVERALL (defect level): {graded_total} graded, "
              f"{found_total} found, {graded_total - found_total} missed")
        print("=" * WIDTH)
        return 0
    except BenchError as exc:
        import sys
        print(f"bench: {exc}", file=sys.stderr)
        return 1
