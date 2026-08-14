"""Step 2: the model reads a sample and proposes what the data is and what it is for.

Everything it returns is a *proposal*, never a conclusion. The user confirms or
rejects in step 3, and only confirmed tags are allowed to influence anything
else. That constraint is the whole design: a local 14B model guessing about a
domain it has never seen is a useful starting point and a terrible authority.

Two questions, deliberately no more:

  data_type -- what IS this corpus?
  use_case  -- what is it GOOD FOR? (RAG, fine-tuning, or neither, with reasons)
"""

from .models import complete, parse_json
from .tags import DATA_TYPE, USE_CASE, Tag

SYSTEM = """You analyse document collections to judge what they are and what they are useful for.

You will see excerpts from one collection. Reply with JSON only, in exactly this shape:

{
  "data_type": [
    {"label": "short name for what this collection is", "reasoning": "one sentence citing what you saw"}
  ],
  "use_case": [
    {"mode": "RAG", "label": "what it would answer", "reasoning": "one sentence"}
  ]
}

Each use_case entry has exactly ONE mode. Choose the single best fit:
  "RAG"        - worth retrieving from to answer questions
  "FINETUNE"   - worth training on to teach a style, format, or vocabulary
  "UNSUITABLE" - not useful for either, and why

Do not combine modes in one entry. If a collection suits both RAG and
fine-tuning, write two separate entries.

Give 2-3 options in each list, ordered most to least likely. Be concrete: name
things you actually saw in the text. "UNSUITABLE" is a valid and useful answer,
not a failure -- say it when the collection is too thin or too inconsistent."""


def sample(corpus, max_chunks=12, max_words=900):
    """Spread the sample across files so one big document cannot dominate."""
    by_source = {}
    for chunk in corpus.chunks:
        by_source.setdefault(chunk.source, []).append(chunk)

    picked, words = [], 0
    round_index = 0
    while len(picked) < max_chunks and words < max_words:
        added = False
        for chunks in by_source.values():
            if round_index < len(chunks):
                chunk = chunks[round_index]
                picked.append(chunk)
                words += chunk.words
                added = True
                if len(picked) >= max_chunks or words >= max_words:
                    break
        if not added:
            break
        round_index += 1
    return picked


def run(corpus, attempts=2, context=""):
    """Returns (tags, sample_used). Empty tag list means the model gave nothing usable.

    context: confirmed facts from the data owner (tags.context()). This is the
    only model output allowed back into the prompt -- and only because a human
    confirmed it first. Unconfirmed guesses feeding themselves would be a
    self-reinforcing loop.
    """
    chunks = sample(corpus)
    excerpts = "\n\n---\n\n".join(
        f"[{c.source}]\n{c.text}" for c in chunks
    )
    owner_note = (f"\n\nThe data owner has confirmed: {context}\n"
                  "Refine your proposals with that knowledge -- be more specific, "
                  "and drop hypotheses the confirmation rules out." if context else "")
    prompt = (f"Excerpts from {len(corpus.sources())} files "
              f"({len(corpus.chunks)} chunks total):{owner_note}\n\n{excerpts}")

    data = None
    for _ in range(attempts):
        data = parse_json(complete(prompt, system=SYSTEM, temperature=0, max_tokens=700))
        if isinstance(data, dict) and (data.get("data_type") or data.get("use_case")):
            break
        data = None

    if not data:
        return [], chunks

    sources = sorted({c.source for c in chunks})
    tags = []
    for group in (DATA_TYPE, USE_CASE):
        for entry in data.get(group, []) or []:
            if isinstance(entry, str):
                entry = {"label": entry}
            label = str(entry.get("label", "")).strip()
            mode = str(entry.get("mode", "")).strip().upper()
            if group == USE_CASE and mode:
                label = f"{mode}: {label}"
            if label:
                tags.append(Tag(
                    group=group,
                    label=label,
                    reasoning=str(entry.get("reasoning", "")).strip(),
                    evidence=sources,
                ))
    return tags, chunks
