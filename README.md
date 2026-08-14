<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset=".github/assets/refinr-banner-dark.png" />
    <source media="(prefers-color-scheme: light)" srcset=".github/assets/refinr-banner-light.png" />
    <img src=".github/assets/refinr-banner-light.png" alt="refinr" width="720" />
  </picture>
</p>

Know your data before an LLM does. refinr measures a folder of documents,
tells you what's in it, what it's good for, and what is not earning its
place — before you build a RAG system or fine-tune a model on it.

Runs **fully local** (Ollama). Your documents never leave your machine.

## Why data preparation matters

Whatever you build on a corpus — retrieval or fine-tuning — the model can
only be as good as what you feed it. The failure modes are specific and
mechanical, not vague "bad data":

- **Duplicates.** Retrieval returns the top-k most relevant passages. If one
  fact exists in five copies, your top-5 is five phrasings of one fact — the
  context window is spent, and the model sees one fact's worth of
  information.
- **Boilerplate.** Footers and disclaimers are similar to *every* document,
  so they sit near the center of meaning-space and get retrieved for
  questions they cannot answer, displacing real content. Deleting them is
  usually wrong (sometimes they're legally required) — excluding them from
  search while keeping them on the page is right.
- **Vague, say-nothing text.** Matches every query, answers none.
- **Coverage gaps.** If a topic isn't in the corpus, no amount of prompt
  engineering will retrieve it. You need to know what's *missing*, not just
  what's wrong.
- **Awkward sizing.** A stranded heading can't answer anything; a wall of
  text mixes several ideas into one search unit and matches none of them
  well.

Three concepts carry the whole tool:

- A **passage** (paragraph) is what the author wrote — the stable unit of
  text identity.
- A **fingerprint** is a hash of a passage's normalized text: an instant
  "have I seen exactly this before?" check. It powers duplicate detection
  and caching, but one changed word makes a new fingerprint — it cannot see
  *near*-copies.
- An **embedding** turns text into a vector (a direction in meaning-space),
  so "same content, different words" becomes a measurable similarity score.
  That's what catches everything hashing can't.

**The rule behind the design:** the tool never treats a model's guess as a
fact. Code measures; the model proposes; only you confirm. Confirmed tags
are the only model output allowed to influence later analysis.

## The workflow

```
refinr look      ./docs    measure     free, instant — duplicates, boilerplate, sizing
refinr guess     ./docs    propose     the model says what this data is and is for
refinr tag       ./docs    confirm     you accept, reject, or edit each guess
refinr inventory ./docs    map         topic groups + coverage %, measured; names proposed
refinr passport  ./docs    verdict     one page: pass/fail vs your requirements profile
```

Start with `look --tier 0`: it costs nothing and finishes in milliseconds.
Add embeddings (`--tier 1`) when you want near-duplicates and typicality;
they run once and cache — re-running on an unchanged corpus makes zero model
calls. After confirming tags, `guess` and `inventory` become domain-aware:
what you confirmed steers the next pass.

The **passport** grades the corpus against a `[profile]` section you write
in `refinr.toml` (required topics, duplication ceiling, minimum size). Its
honesty rule: an unconfirmed model guess can *block* a pass, but can never
*grant* one. Verdicts are three-valued — pass, fail, or can't-say-yet — and
it never squeezes uncertainty into a yes.

## What `look` finds today

| Signal | Cost | What it catches |
|---|---|---|
| `exact_dupes` | free | the same passage pasted into several files |
| `boilerplate` | free | footers/disclaimers repeated across the corpus |
| `length` | free | passages too small to answer anything, or too big to search well |
| `near_dupes` | embeddings | the same content in different words |
| `generic` | embeddings | text so typical of the corpus it matches every query and answers none |

Findings are grouped by evidence: a **structural fact** ("this passage is in
4 files") is never presented in the same voice as an **estimate**.

## Measured, not promised

Detectors are graded against corpora with known planted defects:

```
refinr bench testdata            # precision/recall per signal, per case
refinr bench testdata --sweep near_duplicate=0.90:0.98:0.01
```

Current honest numbers on the built-in cases: the free (tier-0) signals are
at 100% precision/recall; `near_dupes` and `generic` are under active
repair — the benchmark exists precisely so that claim can't be fudged. No
new detector merges without numbers.

## Try it in the browser

```
refinr web
```

A local demo (127.0.0.1) for small file sets: drop in documents, see
measured findings and a prioritized "what to fix first" list instantly, get
the model's guesses and corpus-level advice, confirm or reject with clicks.
Small limits on purpose — a sample-based demo is honest at demo scale; the
CLI has no limits.

## Install / run

Needs Python ≥ 3.11 and [Ollama](https://ollama.com) running two local
models (defaults: `qwen2.5:14b` for chat, `mxbai-embed-large` for
embeddings — override with `REFINR_CHAT_MODEL` / `REFINR_EMBED_MODEL`).

```bash
uv pip install -e .
ollama serve                          # in another terminal
refinr look ./your-docs --tier 0      # the free pass — start here
```

Configuration lives in `refinr.toml` (thresholds, chunk sizes, enabled
signals, your `[profile]`); CLI flags override it. State lives in `.refinr/`
next to your docs: `tags.json` (your decisions — hand-editable JSON),
`inventory.json` (the topic map), and `store/` (embedding cache, safe to
delete).

## Honest limits, current version

- Embedding-signal thresholds were calibrated on small test corpora; the
  benchmark harness is how they earn real values, and `generic` currently
  fails its own benchmark (the mechanism is being reworked, not the number).
- Retrievability testing ("can questions actually find this passage?") was
  removed: v0's version tested passages with questions generated *from*
  those same passages, which nearly always succeeds and proves nothing. It
  returns when it can be done honestly.
- "This data suits RAG / fine-tuning" tags are the model's proposal plus
  your confirmation — a judgment, not a measurement. Proving it requires
  building the pipeline and evaluating it.
