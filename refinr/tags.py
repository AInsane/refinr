"""Tags: everything the model guesses, and what the user did about it.

The rule that keeps this honest: **a guess is not a fact until a human touches
it.** The model proposes; the user confirms, rejects, or edits. Only confirmed
tags are allowed to influence later analysis.

Stored as one small JSON file the user can read and edit by hand. If the format
needs a parser to understand, it is too complicated.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

TAGS_PATH = ".refinr/tags.json"

DATA_TYPE = "data_type"      # what this corpus IS
USE_CASE = "use_case"        # what it is GOOD FOR
TOPIC = "topic"              # what it COVERS (proposed by inventory)

GROUPS = (DATA_TYPE, USE_CASE, TOPIC)
TITLES = {
    DATA_TYPE: "WHAT THIS DATA IS",
    USE_CASE: "WHAT IT COULD BE USED FOR",
    TOPIC: "WHAT TOPICS IT COVERS",
}

PROPOSED = "proposed"
CONFIRMED = "confirmed"
REJECTED = "rejected"


@dataclass
class Tag:
    group: str
    label: str
    reasoning: str = ""
    status: str = PROPOSED
    source: str = "model"       # "model" or "user"
    evidence: list = field(default_factory=list)   # sources that prompted the guess
    key: str = ""               # stable link to what the tag describes (e.g. a
                                # topic cluster id) -- survives label edits,
                                # which the label string itself would not

    @property
    def confirmed(self):
        return self.status == CONFIRMED

    def line(self):
        mark = {PROPOSED: "?", CONFIRMED: "+", REJECTED: "-"}[self.status]
        return f"[{mark}] {self.label}"


class TagSet:
    def __init__(self, folder):
        self.path = Path(folder).expanduser().resolve() / TAGS_PATH
        self.tags = []
        self.load()

    def load(self):
        if self.path.is_file():
            self.tags = [Tag(**t) for t in json.loads(self.path.read_text())]
        return self

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(t) for t in self.tags], indent=2))
        return self

    def group(self, name):
        return [t for t in self.tags if t.group == name]

    def confirmed(self, name=None):
        return [t for t in self.tags
                if t.confirmed and (name is None or t.group == name)]

    def replace_proposals(self, new_tags, groups):
        """Swap in fresh proposals *within the named groups*, preserving
        anything the user already decided.

        Scoped to groups because different commands own different groups:
        `guess` proposes data_type/use_case, `inventory` proposes topics.
        The unscoped version dropped every pending proposal corpus-wide, so
        re-running guess would have silently deleted unconfirmed topics.
        """
        decided = {(t.group, t.label.lower()): t for t in self.tags
                   if t.status != PROPOSED}
        kept = [t for t in self.tags
                if t.status != PROPOSED or t.group not in groups]
        for tag in new_tags:
            if tag.group not in groups:
                raise ValueError(f"tag group '{tag.group}' outside scope {groups}")
            if (tag.group, tag.label.lower()) not in decided:
                kept.append(tag)
        self.tags = kept
        return self

    def add(self, group, label, source="user", status=CONFIRMED):
        self.tags.append(Tag(group=group, label=label, source=source, status=status))
        return self

    def context(self):
        """Confirmed tags as a prompt fragment, or '' if nothing is confirmed yet.

        Only confirmed tags appear here -- an unconfirmed guess must never feed
        back into the model and become self-reinforcing.
        """
        types = [t.label for t in self.confirmed(DATA_TYPE)]
        uses = [t.label for t in self.confirmed(USE_CASE)]
        topics = [t.label for t in self.confirmed(TOPIC)]
        parts = []
        if types:
            parts.append(f"This corpus is: {'; '.join(types)}.")
        if uses:
            parts.append(f"It is intended for: {'; '.join(uses)}.")
        if topics:
            parts.append(f"It covers: {'; '.join(topics)}.")
        return " ".join(parts)

    def __len__(self):
        return len(self.tags)
