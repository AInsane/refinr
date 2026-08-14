"""Local model access.

Everything runs against Ollama so that corpora never leave the machine -- that
is the whole point of the tool.

This module speaks Ollama's *native* API (/api/chat, /api/embed), not its
OpenAI-compatible /v1 endpoint, for one measured reason: the compat endpoint
ignores the num_ctx option (verified 2026-08-13: options sent, `ollama ps`
stayed at the 4096-token default) and silently truncates any prompt that
exceeds the window -- the model simply never sees the start of the prompt, and
nothing errors. The native API honors options.num_ctx per request. On the
16GB M2 Pro this targets, qwen2.5:14b at 8192 loads at ~10GB fully on GPU;
that is the ceiling, so 8192 is the default here.

Rather than let any endpoint truncate, complete() estimates the prompt size
up front and raises ContextOverflow loudly. An estimate (words x 1.3) is fine
because callers are expected to budget with wide margins, not to shave the
limit.

Known follow-up, deliberately not handled here: /api/embed silently truncates
inputs beyond the embedding model's own window (512 tokens for
mxbai-embed-large). Chunk packing keeps inputs well under that today.
"""

import json
import os
import re
import urllib.error
import urllib.request

_raw_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
BASE_URL = _raw_base[:-3] if _raw_base.rstrip("/").endswith("/v1") else _raw_base
BASE_URL = BASE_URL.rstrip("/")
CHAT_MODEL = os.environ.get("REFINR_CHAT_MODEL", "qwen2.5:14b")
EMBED_MODEL = os.environ.get("REFINR_EMBED_MODEL", "mxbai-embed-large")
NUM_CTX = int(os.environ.get("REFINR_NUM_CTX", "8192"))

_MARGIN = 64  # tokens held back for chat template wrapping


class ContextOverflow(RuntimeError):
    """Prompt would exceed the model's context window. Raised instead of
    letting Ollama truncate silently, which produces answers about a prompt
    the model never fully read."""


def estimate_tokens(text):
    """Rough token count for English/markdown: ~1.3 tokens per word."""
    return int(len(text.split()) * 1.3)


def _post(path, payload, timeout=300):
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            detail = json.loads(body).get("error", body)
        except json.JSONDecodeError:
            detail = body
        raise RuntimeError(f"ollama {path} -> HTTP {exc.code}: {detail}") from exc


def embed(texts, batch_size=32):
    """Embed a list of strings. Returns a list of float vectors."""
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = _post("/api/embed", {"model": EMBED_MODEL, "input": batch})
        vectors.extend(response["embeddings"])
    return vectors


def complete(prompt, system=None, temperature=0, max_tokens=200):
    """One-shot completion, num_ctx pinned so nothing is silently dropped."""
    estimated = estimate_tokens((system or "") + " " + prompt)
    if estimated + max_tokens > NUM_CTX - _MARGIN:
        raise ContextOverflow(
            f"prompt is ~{estimated} tokens + {max_tokens} reply, over the "
            f"{NUM_CTX}-token window -- shrink the sample instead of letting "
            f"the model read a truncated prompt"
        )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = _post("/api/chat", {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    })
    return response["message"]["content"].strip()


def parse_json(raw):
    """Pull a JSON object out of a local model's reply.

    Small models wrap JSON in prose or fences often enough that assuming clean
    output is a bug, not a simplification. Lives here because coping with
    local-model output quirks is this module's job, not each caller's.
    """
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        brace = text.find("{")
        if brace == -1:
            return None
        text = text[brace:]
    depth, end = 0, None
    for i, char in enumerate(text):
        depth += (char == "{") - (char == "}")
        if depth == 0 and char == "}":
            end = i + 1
            break
    try:
        return json.loads(text[:end] if end else text)
    except json.JSONDecodeError:
        return None


def health_check():
    """Verify both models respond. Returns (ok, message)."""
    try:
        embed(["health check"])
        complete("Reply with: ok", max_tokens=5)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a message
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"chat={CHAT_MODEL} embed={EMBED_MODEL} at {BASE_URL} ctx={NUM_CTX}"
