"""Load documents into paragraphs, then pack paragraphs into chunks.

Two levels, deliberately kept separate:

  Paragraph -- what the author wrote. Stable. Comparable across documents.
  Chunk     -- what the retriever sees. An artifact of *our* packing choices.

v0 detected duplicates at chunk level and missed a footer planted verbatim in
four files, because packing had glued each copy to a different neighbour. Text
identity lives at the paragraph level; anything comparing source text has to
work there.

Chunking itself stays boring on purpose. Smarter strategies (semantic,
recursive, sentence-window) change retrieval a lot, and the point of refinr is
to *measure* that difference against a baseline rather than bake one in.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

TEXT_SUFFIXES = {".md", ".txt", ".rst", ".markdown", ".text"}


def fingerprint(text):
    """Hash of normalized text. Whitespace and case must not create identity."""
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


@dataclass
class Paragraph:
    id: int
    source: str
    position: int          # index within its source file
    text: str
    fp: str = field(init=False)

    def __post_init__(self):
        self.fp = fingerprint(self.text)

    @property
    def words(self):
        return len(self.text.split())

    def preview(self, width=60):
        return " ".join(self.text.split())[:width]


@dataclass
class Chunk:
    id: int
    source: str
    paragraph_ids: list
    text: str
    fp: str = field(init=False)

    def __post_init__(self):
        self.fp = fingerprint(self.text)

    @property
    def words(self):
        return len(self.text.split())

    def preview(self, width=60):
        return " ".join(self.text.split())[:width]


def read_paragraphs(folder):
    """Walk a folder and return every paragraph, in document order."""
    root = Path(folder).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"no such folder: {root}")

    paragraphs = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        source = str(path.relative_to(root))
        for position, block in enumerate(b.strip() for b in raw.split("\n\n")):
            if block:
                paragraphs.append(Paragraph(
                    id=len(paragraphs), source=source, position=position, text=block,
                ))
    return paragraphs


def pack(paragraphs, target_words=180, min_words=20):
    """Pack consecutive paragraphs from the same file into chunks."""
    chunks = []
    current, count = [], 0

    def flush():
        nonlocal current, count
        if not current:
            return
        text = "\n\n".join(p.text for p in current)
        if len(text.split()) >= min_words:
            chunks.append(Chunk(
                id=len(chunks),
                source=current[0].source,
                paragraph_ids=[p.id for p in current],
                text=text,
            ))
        current, count = [], 0

    for paragraph in paragraphs:
        crosses_file = current and paragraph.source != current[0].source
        overflows = current and count + paragraph.words > target_words
        if crosses_file or overflows:
            flush()
        current.append(paragraph)
        count += paragraph.words

    flush()
    return chunks


def load(folder, target_words=180, min_words=20):
    """Convenience: returns (paragraphs, chunks)."""
    paragraphs = read_paragraphs(folder)
    return paragraphs, pack(paragraphs, target_words, min_words)
