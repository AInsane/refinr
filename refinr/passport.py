"""Passport: the one-page verdict. What this data is, what's in it, what it
is good for -- graded against the [profile] section of refinr.toml.

Zero model calls, by construction. A verdict document must be instant and
reproducible, so everything here is either already on disk (inventory.json,
tags.json), free to recompute (ingest + tier-0 signals, milliseconds), or
written by the user ([profile]). Anything that needs a model happens earlier,
in `guess` or `inventory`.

The grading rule that keeps it honest: **an unconfirmed guess can block a
PASS, but can never grant one.** A required topic passes only against a
human-confirmed topic tag. If only a model proposal matches, the check reads
UNVERIFIED and the overall verdict caps at INCOMPLETE. Checks are
three-valued (pass / fail / can't-say-yet), never squeezed into two.

Audience note: this page is for a mixed team. It talks about files,
passages, topics, and repeated text -- not chunks, embeddings, or clusters.
"""

import textwrap
from pathlib import Path

from . import inventory as inventory_module
from . import signals
from .config import Config
from .corpus import Corpus
from .tags import CONFIRMED, DATA_TYPE, PROPOSED, TOPIC, USE_CASE, TagSet

WIDTH = 74

PASS, FAIL, UNVERIFIED, NOT_ASSESSED = "PASS", "FAIL", "UNVERIFIED", "NOT GRADED"


class Check:
    def __init__(self, verdict, requirement, note):
        self.verdict = verdict
        self.requirement = requirement
        self.note = note


def _dup_percent(corpus, cfg):
    """% of passages that are exact repeats of text elsewhere -- measured by
    the free signals only, so the passport never waits on a model. Near-misses
    (rephrasings) are deliberately not counted here."""
    tier0 = cfg.merged(max_tier=0)
    findings = signals.run_all(corpus, tier0)
    implicated = set()
    for f in findings:
        if f.kind in ("exact_duplicate", "boilerplate"):
            implicated.update(f.paragraph_ids)
    return 100 * len(implicated) / max(len(corpus.paragraphs), 1)


def _staleness(inv, corpus):
    """How many passages changed since the inventory was taken."""
    recorded = set(inv["unclustered"]["paragraph_fps"])
    for c in inv["clusters"]:
        recorded.update(c["paragraph_fps"])
    current = {p.fp for p in corpus.paragraphs}
    return len(current ^ recorded)


def _match(required, tags_in_group):
    """Case-insensitive containment either way: a requirement 'security
    training' matches a topic 'Security Training policy' and vice versa."""
    want = required.lower().strip()
    for tag in tags_in_group:
        have = tag.label.lower().strip()
        if want in have or have in want:
            return tag
    return None


def _topic_check(required, tagset, inv, stale_count):
    if inv is None:
        return Check(NOT_ASSESSED, f'covers "{required}"',
                     "no inventory yet -- run: refinr inventory")
    if stale_count:
        return Check(NOT_ASSESSED, f'covers "{required}"',
                     f"the documents changed since the inventory "
                     f"({stale_count} passages) -- re-run: refinr inventory")
    confirmed = tagset.confirmed(TOPIC)
    hit = _match(required, confirmed)
    if hit:
        return Check(PASS, f'covers "{required}"',
                     f'confirmed topic "{hit.label}" matches')
    proposed = [t for t in tagset.group(TOPIC) if t.status == PROPOSED]
    hit = _match(required, proposed)
    if hit:
        return Check(UNVERIFIED, f'covers "{required}"',
                     f'the analysis proposed a matching topic ("{hit.label}") '
                     "but nobody has confirmed it -- run: refinr tag")
    if not confirmed:
        return Check(NOT_ASSESSED, f'covers "{required}"',
                     "no topics have been confirmed yet -- run: refinr tag")
    return Check(FAIL, f'covers "{required}"',
                 "no topic in the inventory matches -- the subject may still "
                 "appear in ungrouped passages, but nothing confirmed covers it")


def grade(corpus, cfg: Config, tagset, inv, stale_count):
    """All profile checks, three-valued. Empty when there is no profile."""
    profile = cfg.profile
    if profile is None:
        return []
    checks = []
    if profile.min_words is not None:
        words = sum(p.words for p in corpus.paragraphs)
        verdict = PASS if words >= profile.min_words else FAIL
        checks.append(Check(verdict, f"at least {profile.min_words} words",
                            f"{words} words counted"))
    if profile.max_duplication_pct is not None:
        pct = _dup_percent(corpus, cfg)
        verdict = PASS if pct <= profile.max_duplication_pct else FAIL
        checks.append(Check(
            verdict, f"repeated text at most {profile.max_duplication_pct:g}%",
            f"{pct:.0f}% of passages are exact repeats of text elsewhere"))
    for required in profile.required_topics:
        checks.append(_topic_check(required, tagset, inv, stale_count))
    return checks


def _overall(checks):
    if any(c.verdict == FAIL for c in checks):
        return "FAIL"
    waiting = sum(c.verdict in (UNVERIFIED, NOT_ASSESSED) for c in checks)
    if waiting:
        return (f"INCOMPLETE -- {waiting} requirement(s) cannot be graded yet; "
                "an unconfirmed guess never counts as a pass")
    return "PASS"


def run(folder, cfg: Config):
    """Assemble and render the page. Returns the page as a string."""
    corpus = Corpus(folder, cfg, store=None)
    tagset = TagSet(folder)
    inv = inventory_module.load(folder)
    stale_count = _staleness(inv, corpus) if inv else 0
    checks = grade(corpus, cfg, tagset, inv, stale_count)
    return render(folder, corpus, tagset, inv, stale_count, checks, cfg)


def render(folder, corpus, tagset, inv, stale_count, checks, cfg):
    words = sum(p.words for p in corpus.paragraphs)
    lines = ["=" * WIDTH,
             f"refinr passport -- {Path(folder).name}",
             "=" * WIDTH,
             f"{len(corpus.sources())} files, {words} words"]

    confirmed_types = tagset.confirmed(DATA_TYPE)
    lines.append("")
    lines.append("WHAT THIS IS")
    if confirmed_types:
        for t in confirmed_types:
            lines.append(f"  {t.label}   [confirmed by owner]")
    else:
        lines.append("  unknown -- run `refinr guess`, then `refinr tag`")

    lines.append("")
    lines.append("WHAT IT COVERS  (share of all text, measured)")
    if inv is None:
        lines.append("  not inventoried yet -- run `refinr inventory`")
    else:
        if stale_count:
            lines.append(f"  NOTE: taken {inv['created'][:10]}, but {stale_count} "
                         "passage(s) changed since -- re-run `refinr inventory`")
        by_key = {t.key: t for t in tagset.group(TOPIC) if t.key}
        for c in inv["clusters"]:
            tag = by_key.get(c["id"])
            if tag is not None and tag.status == CONFIRMED:
                mark, name = "[confirmed]", tag.label
            elif tag is not None and tag.status == PROPOSED:
                mark, name = "[proposed -- not yet confirmed]", tag.label
            elif c["label_proposed"]:
                mark, name = "[proposed -- not yet confirmed]", c["label_proposed"]
            else:
                mark, name = "", "(unnamed group)"
            lines.append(f"  {c['pct_words']:>4.0f}%  {name:<32} {mark}")
        unc = inv["unclustered"]
        if unc["paragraphs"]:
            lines.append(f"  {unc['pct_words']:>4.0f}%  (not part of any topic)")

    confirmed_uses = tagset.confirmed(USE_CASE)
    lines.append("")
    lines.append("WHAT IT IS GOOD FOR")
    if confirmed_uses:
        for t in confirmed_uses:
            lines.append(f"  {t.label}   [confirmed by owner]")
    else:
        lines.append("  not confirmed yet -- run `refinr guess`, then `refinr tag`")

    lines.append("-" * WIDTH)
    if cfg.profile is None:
        lines.append("no [profile] in refinr.toml -- nothing to grade against")
    else:
        lines.append("CHECKED AGAINST YOUR PROFILE  (refinr.toml)")
        if cfg.profile.intended_use:
            lines.append(f'  intended use: "{cfg.profile.intended_use}"')
        for c in checks:
            lines.append(f"  [{c.verdict:^10}] {c.requirement}")
            lines.extend(textwrap.wrap(c.note, WIDTH - 15,
                                       initial_indent=" " * 15,
                                       subsequent_indent=" " * 15))
        lines.append("-" * WIDTH)
        lines.append(f"VERDICT: {_overall(checks)}")
    lines.append("=" * WIDTH)
    return "\n".join(lines)
