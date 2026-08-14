"""Turn measurements into advice: one model call, clearly voiced as a proposal.

The deterministic half of "what should I do?" needs no model at all -- every
Finding already carries per-signal advice, and the web layer renders those
directly. This module is the other half: the model reads a sample plus a
summary of what the signals measured, and proposes corpus-level advice a
mixed team can act on.

Same contract as guess.py: everything returned here is a *proposal*. The web
layer renders it under the "model output" voice, never in the measured voice.
Confirmed tags (and only confirmed tags) may steer the prompt.
"""

from .guess import sample
from .models import complete, parse_json

SYSTEM = """You advise teams on their document collections. You will see excerpts from a
collection plus a list of problems an automated analysis measured in it.

Reply with JSON only, in exactly this shape:

{
  "advices": [
    {"title": "short imperative advice", "why": "one sentence: the cost of not doing it", "action": "one concrete first step"}
  ],
  "suitability": "one sentence: what this collection is currently good for, and what holds it back"
}

Rules:
- 3 to 5 advices, ordered most to least important.
- Ground every advice in something you actually saw -- quote or name it.
- Plain language for a mixed team: never use words like "chunk", "embedding",
  "vector", "cosine", or "index". Say "passage", "repeated text", "search".
- If the collection looks healthy, say so in fewer advices rather than
  inventing problems."""


def _findings_summary(findings, limit=12):
    """Compact, model-facing digest of what the signals measured."""
    if not findings:
        return "The analysis found no problems."
    lines = []
    for finding in findings[:limit]:
        lines.append(f"- [{finding.severity.value}] {finding.message}")
    if len(findings) > limit:
        lines.append(f"- ...and {len(findings) - limit} more findings")
    return "\n".join(lines)


def _normalize(data):
    """Validate the model's reply into a predictable dict, or None."""
    if not isinstance(data, dict):
        return None
    advices = []
    for entry in data.get("advices", []) or []:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title", "")).strip()
        if not title:
            continue
        advices.append({
            "title": title,
            "why": str(entry.get("why", "")).strip(),
            "action": str(entry.get("action", "")).strip(),
        })
    if not advices:
        return None
    return {
        "advices": advices[:5],
        "suitability": str(data.get("suitability", "")).strip(),
    }


def run(corpus, findings, context="", attempts=2):
    """Returns an advice dict, or None if the model gave nothing usable.

    context: confirmed facts from the data owner (tags.context()) -- the only
    model output allowed back into a prompt, because a human confirmed it.
    """
    chunks = sample(corpus)
    excerpts = "\n\n---\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
    owner_note = (f"\nThe data owner has confirmed: {context}\n" if context else "")
    prompt = (f"Excerpts from {len(corpus.sources())} files:{owner_note}\n"
              f"What the analysis measured:\n{_findings_summary(findings)}\n\n"
              f"The excerpts:\n\n{excerpts}")

    for _ in range(attempts):
        data = _normalize(parse_json(
            complete(prompt, system=SYSTEM, temperature=0, max_tokens=600)))
        if data:
            return data
    return None
