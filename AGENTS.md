# refinr — agent context

**KYD (Know Your Data):** a local-first tool that tells a team what their
corpus *is*, what's *in* it, what it's *good for* — then enhances it. Runs
entirely against local Ollama; the wedge is proprietary data that can't be
uploaded anywhere. This is an open-source library for builders and
businesses: engineering judgment in the commit log matters as much as
features.

## Run it

There is **no venv committed in this repo** — point `PY` at any Python 3.11+
with the deps installed (e.g. `python3 -m venv .venv` first):

```bash
PY=.venv/bin/python
PYTHONPATH=. $PY -m refinr.cli look testdata/handbook --tier 0   # free, instant
PYTHONPATH=. $PY -m refinr.cli guess testdata/handbook           # ~60-90s
PYTHONPATH=. $PY -m refinr.cli inventory testdata/handbook       # topics, ~2-5min
PYTHONPATH=. $PY -m refinr.cli passport testdata/handbook        # instant, no model
PYTHONPATH=. $PY -m refinr.cli web                               # demo UI :7358
```

Needs `ollama serve` with `qwen2.5:14b` (chat/tools) and `mxbai-embed-large`
(embeddings). NOTE: `qwen2.5-coder:*` variants are not tool-trained — never
use them for anything agentic. Install deps into that venv with
`uv pip install --python $PY <pkg>` (it has no pip).

**Model access is the native Ollama API** (`models.py`, stdlib urllib, no
openai dep) with `num_ctx` pinned to 8192 — the OpenAI-compat endpoint
*ignores* num_ctx and silently truncates (measured 2026-08-13); 8K is the
ceiling on the 16GB M2 Pro this targets (10GB, 100% GPU). `complete()`
raises `ContextOverflow` rather than let anything truncate silently.

## Architecture

Three layers: **1** demo web app (`webapp.py`, stdlib-only, 127.0.0.1, hard
limits 8 files/100KB/250KB — limits are honesty, not laziness); **2**
CLI/engine (`cli.py`: look / guess / tag / web / signals); **3** framework
(signal protocol, later).

Core loop: `look` (code measures) → `guess` (model proposes what the data is
and what it's for) → `tag` (user confirms/rejects/edits). **A guess is not a
fact until a human touches it** — only confirmed tags (`tags.context()`) may
feed back into any prompt. This rule is load-bearing; do not weaken it.

Slice 4 added: `inventory` (greedy centroid-agglomerative clustering of
paragraph embeddings = measured coverage; one LLM call per cluster proposes
a name → TOPIC tag group, confirmed via `tag`; persists
`.refinr/inventory.json` with paragraph fps as the staleness key) and
`passport` (one-page verdict vs `[profile]` in refinr.toml; zero model
calls; three-valued checks — an unconfirmed guess can block a PASS, never
grant one). `inventory`'s 0.70 threshold is corpus-fitted and provisional:
on the handbook, 0.65 starts chain-merging and 0.60 collapses to one blob.
`Finding.fix_mode` (deterministic|llm|hybrid) is carried but rendered
nowhere until the compile-pattern clean step consumes it.

Key modules: `ingest.py` (Paragraph → packed Chunk; **text identity lives at
paragraph level** — chunk packing destroys it, this bit us twice),
`corpus.py` (lazy chunk + paragraph embeddings), `store.py` (embeddings
cached by content hash; invalidated on embed-model change), `finding.py`
(every Finding carries an Evidence source; the report never presents an
estimate in the same voice as a measured fact), `signals/__init__.py`
(protocol: TIER 0 free / 1 embeddings / 2 LLM; NAME; DETECTS; run(corpus,
config)), `config.py` (`refinr.toml` is source of truth, flags override).

## Current signals

exact_dupes, boilerplate, length (tier 0) · near_dupes, generic (tier 1).
**Benchmark state (first run, 2026-08-13):** tier-0 signals are perfect
(1.00/1.00, zero clean-corpus FPs). `near_dupes` scored **0% recall** — it
compares chunks, packing dilutes a 0.956 paragraph pair to 0.82, and even
the unpacked refund plant sits at 0.9475, under the 0.95 default by 0.0025;
the sweep says 0.92–0.94 gives precision 1.00. Fix is paragraph-level
comparison — benchmark-justified, not yet done. `generic` scored **0%/0%**:
a mechanism failure, not tuning — duplicate paragraphs drag the centroid
toward themselves (footer copies score 0.861 "generic" while real filler
sits below every threshold); no sweep value works. Rethink or retire it.
There is deliberately **no filler signal**: three heuristics failed (one only
"worked" because its wordlist was written after reading the test data — see
slice-1 commit message). No retrievability signal either: v0's version was
tautological (questions generated from the chunk under test) and was removed.

## Roadmap (agreed 2026-08-13)

1. **Inventory + passport** — ✅ shipped 2026-08-13 (slice 4), minus
   entities (topics only) and the web-app passport view. `fix_mode` landed.
2. **Clean + delta** via the **compile pattern**: LLM advises at authoring
   time → emits rules the user confirms → pipeline runs as pure deterministic
   code (100% reproducible). Corpus-fitted rules are fine *here* — they
   declare their scope; that's what redeems the failed-wordlist experiment.
3. **Enrich** (summaries/metadata), then benchmark-gated extras.

**Benchmark discipline (ACTIVE):** the harness is `refinr bench testdata`
(`--sweep field=a:b:step`, `--tier 0` needs no Ollama). No detector merges
without a bench run pasted into the commit message. Three cases: `handbook`
(documented plants), `clean` (FP meter, target zero), `dirty` (real-world
mess; manifest authored before the harness existed — git history is the
proof; the slice-1 wordlist scar is why the ordering is load-bearing).
Unclaimed kinds planted and waiting for signals: `off_topic`,
`encoding_artifact`. Manifest `kinds` are decided up front — widening them
after seeing results is the peeking the discipline prevents.

**Product vision (2026-08-13):** real data is very dirty; the tool
should tell the user what the artifacts are (signals), what the data is
good for (passport), and eventually recommend a **suitable storing model**
and **what analytics it could feed** — future passport sections, model-
proposed and human-confirmed, same guess-is-not-a-fact contract. Named
follow-ups: near_dupes paragraph-level fix (evidence in the bench sweep),
generic rethink, encoding/export-junk signal (dirt already planted),
inventory-clustering quality benchmark.

## Working with the maintainer

- **Plan before building.** Enter plan mode for anything non-trivial; agents
  have been interrupted for building without asking. Option questions get
  answered readily — use them.
- Explain example-first: name every symbol/term before using it, run
  code to prove claims instead of asserting them.
- **Audience for all user-facing output is a mixed team**: the web layer and
  passport never say "chunk", "cosine", "embedding", "centroid". Engineer
  vocabulary stays in the CLI.
- Report honestly: if a fix's delta is flat, say so; never print an
  unconsumed capability as if it worked (this was a real caught bug).
- Commit style: substantive messages that record what failed and why —
  they're part of the project's public record.
