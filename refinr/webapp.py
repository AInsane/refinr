"""Layer 1: the demo web app.

A local, single-page showcase for people who will never open a terminal:
choose a few small files, watch the two steps happen (code measures, model
guesses), confirm or reject the guesses, add your own context, run again.

Deliberately limited -- few files, small sizes -- because a sample-based demo
is honest at demo scale and misleading at audit scale. The CLI (layer 2) is
for real corpora.

Plain-language rule: this layer never says "chunk", "cosine", "embedding" or
"centroid". Engineer vocabulary lives in the CLI.

Zero dependencies beyond the stdlib: files are read in the browser and posted
as JSON, so there is no multipart parsing and no framework.
"""

import json
import shutil
import tempfile
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import advice as advice_module
from . import guess as guess_module
from . import signals, tags as tags_module
from .config import Config
from .corpus import Corpus

# --- demo limits (the point, not a workaround) --------------------------
MAX_FILES = 8
MAX_FILE_BYTES = 100_000
MAX_TOTAL_BYTES = 250_000
ALLOWED_SUFFIXES = {".md", ".txt", ".rst", ".markdown", ".text"}

# --- plain-language rendering of engine findings ------------------------
PLAIN = {
    "boilerplate": ("Repeated block",
                    "The same text appears in {files} of your files. Search will keep "
                    "returning it instead of real answers."),
    "exact_dupes": ("Copy-pasted text",
                    "An identical passage is pasted in more than one place. Only one "
                    "copy is useful."),
    "near_dupes": ("Same thing, different words",
                   "Two passages say the same thing in different wording. A reader "
                   "gets no more from both than from one."),
    "generic": ("Says what everything says",
                "This text is so typical of the collection that it matches any "
                "question and answers none."),
    "length": ("Awkward size",
               "A section is too short or too long to work well in search."),
}

_SESSIONS = {}
_LOCK = threading.Lock()


def _plain_findings(findings):
    out = []
    for f in findings:
        title, template = PLAIN.get(f.signal, (f.signal, f.message))
        out.append({
            "title": title,
            "text": template.format(files=f.detail.get("files", "several")),
            "quote": (f.message.split('"')[1] if '"' in f.message else "")[:90],
            "severity": f.severity.value,
            "confidence": f.evidence.confidence,
            "advice": f.advice,
        })
    return out


class Session:
    def __init__(self, files):
        self.id = uuid.uuid4().hex[:12]
        self.dir = Path(tempfile.mkdtemp(prefix="refinr-demo-"))
        for name, text in files:
            safe = Path(name).name or "file.txt"
            (self.dir / safe).write_text(text, encoding="utf-8")
        self.tags = tags_module.TagSet(self.dir)
        self.guess_state = "idle"          # idle | running | ready | failed
        self.guess_error = ""
        self.advice_state = "idle"         # same lifecycle as guess_state
        self.advice_error = ""
        self.advice = None
        cfg = Config(target_words=120)
        self.corpus = Corpus(self.dir, cfg, store=None)
        self.findings = signals.run_all(self.corpus, cfg)

    def start_guess(self):
        self.guess_state = "running"
        threading.Thread(target=self._guess, daemon=True).start()

    def _guess(self):
        try:
            proposals, _ = guess_module.run(self.corpus, context=self.tags.context())
            with _LOCK:
                if proposals:
                    self.tags.replace_proposals(
                        proposals, groups=(tags_module.DATA_TYPE, tags_module.USE_CASE)).save()
                    self.guess_state = "ready"
                else:
                    self.guess_state = "failed"
                    self.guess_error = "the model returned nothing usable -- try again"
        except Exception as exc:  # noqa: BLE001 - shown to the user
            with _LOCK:
                self.guess_state = "failed"
                self.guess_error = f"{type(exc).__name__}: {exc}"

    def start_advice(self):
        self.advice_state = "running"
        threading.Thread(target=self._advise, daemon=True).start()

    def _advise(self):
        try:
            result = advice_module.run(self.corpus, self.findings,
                                       context=self.tags.context())
            with _LOCK:
                if result:
                    self.advice = result
                    self.advice_state = "ready"
                else:
                    self.advice_state = "failed"
                    self.advice_error = "the model returned nothing usable -- try again"
        except Exception as exc:  # noqa: BLE001 - shown to the user
            with _LOCK:
                self.advice_state = "failed"
                self.advice_error = f"{type(exc).__name__}: {exc}"

    def payload(self):
        return {
            "session": self.id,
            "files": len(self.corpus.sources()),
            "words": sum(p.words for p in self.corpus.paragraphs),
            "findings": _plain_findings(self.findings),
            "guess_state": self.guess_state,
            "guess_error": self.guess_error,
            "advice_state": self.advice_state,
            "advice_error": self.advice_error,
            "advice": self.advice,
            "tags": [{"group": t.group, "label": t.label, "reasoning": t.reasoning,
                      "status": t.status} for t in self.tags.tags],
            "context": self.tags.context(),
        }

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def _validate(files):
    if not files:
        return "no files received"
    if len(files) > MAX_FILES:
        return f"demo limit: at most {MAX_FILES} files"
    total = 0
    for name, text in files:
        if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
            return f"'{name}': only text files ({', '.join(sorted(ALLOWED_SUFFIXES))})"
        size = len(text.encode("utf-8"))
        total += size
        if size > MAX_FILE_BYTES:
            return f"'{name}' is {size//1000}KB -- demo limit is {MAX_FILE_BYTES//1000}KB per file"
    if total > MAX_TOTAL_BYTES:
        return f"{total//1000}KB total -- demo limit is {MAX_TOTAL_BYTES//1000}KB"
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the terminal quiet
        pass

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_TOTAL_BYTES * 2:
            return None
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/state/"):
            session = _SESSIONS.get(self.path.rsplit("/", 1)[-1])
            if not session:
                return self._json({"error": "unknown session"}, 404)
            with _LOCK:
                self._json(session.payload())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            data = self._body()
        except (json.JSONDecodeError, ValueError):
            return self._json({"error": "bad request"}, 400)
        if data is None:
            return self._json({"error": "request too large for the demo"}, 413)

        if self.path == "/api/analyze":
            files = [(f.get("name", ""), f.get("text", "")) for f in data.get("files", [])]
            problem = _validate(files)
            if problem:
                return self._json({"error": problem}, 400)
            session = Session(files)
            if not session.corpus.paragraphs:
                session.cleanup()
                return self._json({"error": "no readable text in those files"}, 400)
            with _LOCK:
                _SESSIONS[session.id] = session
            session.start_guess()
            with _LOCK:
                return self._json(session.payload())

        if self.path == "/api/tag":
            session = _SESSIONS.get(data.get("session", ""))
            if not session:
                return self._json({"error": "unknown session"}, 404)
            with _LOCK:
                for tag in session.tags.tags:
                    if tag.group == data.get("group") and tag.label == data.get("label"):
                        action = data.get("action")
                        if action == "confirm":
                            tag.status = tags_module.CONFIRMED
                        elif action == "reject":
                            tag.status = tags_module.REJECTED
                session.tags.save()
                return self._json(session.payload())

        if self.path == "/api/add-context":
            session = _SESSIONS.get(data.get("session", ""))
            if not session:
                return self._json({"error": "unknown session"}, 404)
            label = str(data.get("label", "")).strip()[:200]
            group = data.get("group")
            if label and group in (tags_module.DATA_TYPE, tags_module.USE_CASE):
                with _LOCK:
                    session.tags.add(group, label)
                    session.tags.save()
            with _LOCK:
                return self._json(session.payload())

        if self.path == "/api/rerun":
            session = _SESSIONS.get(data.get("session", ""))
            if not session:
                return self._json({"error": "unknown session"}, 404)
            if session.guess_state == "running":
                return self._json({"error": "already running"}, 409)
            session.start_guess()
            with _LOCK:
                return self._json(session.payload())

        if self.path == "/api/advise":
            session = _SESSIONS.get(data.get("session", ""))
            if not session:
                return self._json({"error": "unknown session"}, 404)
            if session.advice_state == "running":
                return self._json({"error": "already running"}, 409)
            session.start_advice()
            with _LOCK:
                return self._json(session.payload())

        return self._json({"error": "not found"}, 404)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>refinr — know your data</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#0f1117; --card:#181b24; --text:#e8eaf0; --dim:#9aa0b0;
          --accent:#5eead4; --warn:#fbbf24; --bad:#f87171; --line:#2a2e3b; }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--text);
         font:16px/1.55 system-ui,-apple-system,sans-serif; padding:2rem 1rem; }
  main { max-width:680px; margin:0 auto; }
  h1 { font-size:1.5rem; } h1 span { color:var(--accent); }
  .sub { color:var(--dim); margin:.3rem 0 1.6rem; }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:12px; padding:1.2rem; margin-bottom:1rem; }
  .drop { border:2px dashed var(--line); border-radius:12px; padding:2.2rem;
          text-align:center; color:var(--dim); cursor:pointer; }
  .drop.over { border-color:var(--accent); color:var(--accent); }
  .limits { font-size:.8rem; color:var(--dim); margin-top:.6rem; }
  button { background:var(--accent); color:#08251f; border:0; border-radius:8px;
           padding:.55rem 1.1rem; font-weight:600; cursor:pointer; font-size:.95rem; }
  button.ghost { background:transparent; color:var(--dim);
                 border:1px solid var(--line); }
  button:disabled { opacity:.45; cursor:default; }
  .flist { margin:.8rem 0; font-size:.9rem; color:var(--dim); }
  .sect { font-size:.75rem; letter-spacing:.12em; text-transform:uppercase;
          color:var(--dim); margin:1.4rem 0 .5rem; }
  .badge { font-size:.72rem; padding:.1rem .5rem; border-radius:99px;
           border:1px solid var(--line); color:var(--dim); margin-left:.5rem; }
  .finding { padding:.7rem 0; border-top:1px solid var(--line); }
  .finding b.warn { color:var(--warn); } .finding b.bad { color:var(--bad); }
  .finding .q { color:var(--dim); font-style:italic; font-size:.85rem; }
  .tag { display:flex; gap:.7rem; align-items:flex-start;
         padding:.7rem 0; border-top:1px solid var(--line); }
  .tag .why { color:var(--dim); font-size:.85rem; }
  .tag .acts { margin-left:auto; display:flex; gap:.4rem; flex-shrink:0; }
  .tag .acts button { padding:.25rem .7rem; font-size:.85rem; }
  .confirmed { color:var(--accent); } .rejected { color:var(--dim);
               text-decoration:line-through; }
  .spin { color:var(--dim); }
  .spin::after { content:""; display:inline-block; width:.8em; height:.8em;
    margin-left:.5em; border:2px solid var(--dim); border-top-color:transparent;
    border-radius:50%; animation:r 1s linear infinite; vertical-align:-.1em; }
  @keyframes r { to { transform:rotate(360deg); } }
  .err { color:var(--bad); margin:.6rem 0; }
  input[type=text] { background:var(--bg); border:1px solid var(--line);
    color:var(--text); border-radius:8px; padding:.5rem .7rem; width:100%;
    font-size:.9rem; }
  .row { display:flex; gap:.5rem; margin-top:.6rem; }
  .note { font-size:.8rem; color:var(--dim); margin-top:1rem; }
</style></head><body><main>
<h1>refinr <span>· know your data</span></h1>
<p class="sub">Drop in a few documents. Code measures them; a local model guesses
what they are and what they're good for; you decide what's true.
Nothing leaves this machine.</p>

<div class="card" id="pick">
  <div class="drop" id="drop">choose or drop files
    <div class="limits">demo limits: 8 files · 100KB each · text formats
      (.md .txt .rst) — for real corpora use the CLI</div>
  </div>
  <input type="file" id="fileinput" multiple accept=".md,.txt,.rst,.markdown,.text"
         style="display:none">
  <div class="flist" id="flist"></div>
  <div class="err" id="pickerr"></div>
  <button id="go" disabled>analyze</button>
</div>

<div id="results" style="display:none">
  <div class="card">
    <div class="sect">measured by code <span class="badge">certain</span></div>
    <div id="findings"></div>
    <div id="fixes"></div>
  </div>
  <div class="card">
    <div class="sect">the model's advice <span class="badge">proposal — your judgment applies</span></div>
    <div id="advice">
      <button id="advise">get advice</button>
      <p class="note">One model pass over a sample of your files plus the measured
      findings. Takes about a minute. Advice is a proposal, not a verdict.</p>
    </div>
  </div>
  <div class="card">
    <div class="sect">the model's guesses <span class="badge">need your confirmation</span></div>
    <div id="guesses"><span class="spin">the model is reading your data</span></div>
    <div id="ctx" style="display:none">
      <div class="sect">add what the model missed</div>
      <div class="row">
        <input type="text" id="ctxinput"
               placeholder="e.g. these are support tickets from our billing team">
        <button class="ghost" id="ctxadd">add</button>
      </div>
      <div class="row">
        <button id="rerun">run again with my context</button>
      </div>
    </div>
    <p class="note">A guess is not a fact until you confirm it. Only confirmed
      lines are used when you run again.</p>
  </div>
</div>

<script>
let picked = [], sid = null, poller = null;
const $ = id => document.getElementById(id);
// Everything that came from a file or a model goes through esc() before
// touching innerHTML -- documents are untrusted input.
const esc = s => String(s).replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

$("drop").onclick = () => $("fileinput").click();
$("drop").ondragover = e => { e.preventDefault(); $("drop").classList.add("over"); };
$("drop").ondragleave = () => $("drop").classList.remove("over");
$("drop").ondrop = e => { e.preventDefault(); $("drop").classList.remove("over");
                          addFiles(e.dataTransfer.files); };
$("fileinput").onchange = e => addFiles(e.target.files);

function addFiles(list) {
  $("pickerr").textContent = "";
  for (const f of list) {
    if (picked.length >= 8) { $("pickerr").textContent = "demo limit: 8 files"; break; }
    if (f.size > 100000) { $("pickerr").textContent = f.name + " is over 100KB"; continue; }
    picked.push(f);
  }
  $("flist").textContent = picked.map(f => f.name).join("  ·  ");
  $("go").disabled = picked.length === 0;
}

$("go").onclick = async () => {
  $("go").disabled = true; $("go").textContent = "reading...";
  const files = [];
  for (const f of picked)
    files.push({ name: f.name, text: await f.text() });
  const res = await fetch("/api/analyze", { method:"POST",
    headers:{"Content-Type":"application/json"}, body: JSON.stringify({files}) });
  const data = await res.json();
  if (data.error) { $("pickerr").textContent = data.error;
                    $("go").disabled = false; $("go").textContent = "analyze"; return; }
  sid = data.session;
  $("pick").style.display = "none"; $("results").style.display = "block";
  render(data); poll();
};

function poll() {
  clearInterval(poller);
  poller = setInterval(async () => {
    const data = await (await fetch("/api/state/" + sid)).json();
    render(data);
    if (data.guess_state !== "running" && data.advice_state !== "running")
      clearInterval(poller);
  }, 1500);
}

function render(d) {
  $("findings").innerHTML = d.findings.length
    ? d.findings.map(f => `<div class="finding">
        <b class="${f.severity === 'high' ? 'bad' : 'warn'}">${esc(f.title)}</b>
        — ${esc(f.text)}${f.quote ? `<div class="q">“${esc(f.quote)}…”</div>` : ""}</div>`).join("")
    : `<div class="finding">Nothing wrong found in ${d.files} file(s),
       ${d.words.toLocaleString()} words. That's a real result, not a failure.</div>`;

  const fixes = [];
  for (const f of d.findings)
    if (f.advice && !fixes.some(x => x.advice === f.advice)) fixes.push(f);
  $("fixes").innerHTML = fixes.length
    ? `<div class="sect">what to fix first</div>` + fixes.slice(0, 5).map((f, i) =>
        `<div class="finding"><b>${i + 1}.</b> ${esc(f.title)} —
         ${esc(f.advice)}</div>`).join("")
    : "";

  renderAdvice(d);

  const g = $("guesses");
  if (d.guess_state === "running")
    g.innerHTML = `<span class="spin">the model is reading your data</span>`;
  else if (d.guess_state === "failed")
    g.innerHTML = `<div class="err">${esc(d.guess_error)}</div>`;
  else if (d.guess_state === "ready") {
    const groups = { data_type: "What this data is", use_case: "What it could be used for" };
    g.innerHTML = Object.entries(groups).map(([key, title]) => {
      const rows = d.tags.filter(t => t.group === key);
      if (!rows.length) return "";
      return `<div class="sect">${title}</div>` + rows.map(t => `
        <div class="tag"><div>
          <div class="${t.status}">${esc(t.label)}</div>
          ${t.reasoning ? `<div class="why">${esc(t.reasoning)}</div>` : ""}</div>
          <div class="acts">${t.status === "proposed" ? `
            <button data-g="${key}" data-l="${esc(t.label)}" data-a="confirm">yes</button>
            <button class="ghost" data-g="${key}" data-l="${esc(t.label)}" data-a="reject">no</button>`
            : `<span class="badge">${t.status}</span>`}</div>
        </div>`).join("");
    }).join("");
    g.querySelectorAll("button[data-a]").forEach(b =>
      b.onclick = () => act(b.dataset.g, b.dataset.l, b.dataset.a));
    $("ctx").style.display = "block";
  }
}

function renderAdvice(d) {
  const a = $("advice");
  if (d.advice_state === "running")
    a.innerHTML = `<span class="spin">the model is thinking about your data</span>`;
  else if (d.advice_state === "failed")
    a.innerHTML = `<div class="err">${esc(d.advice_error)}</div>
      <button id="advise">try again</button>`;
  else if (d.advice_state === "ready" && d.advice)
    a.innerHTML = d.advice.advices.map((x, i) => `<div class="finding">
        <b>${i + 1}. ${esc(x.title)}</b>${x.why ? ` — ${esc(x.why)}` : ""}
        ${x.action ? `<div class="q">first step: ${esc(x.action)}</div>` : ""}</div>`).join("")
      + (d.advice.suitability ? `<p class="note">${esc(d.advice.suitability)}</p>` : "");
  const btn = $("advise");
  if (btn) btn.onclick = advise;
}

async function advise() {
  await fetch("/api/advise", { method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({session: sid}) });
  $("advice").innerHTML = `<span class="spin">the model is thinking about your data</span>`;
  poll();
}

async function act(group, label, action) {
  const data = await (await fetch("/api/tag", { method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({session: sid, group, label, action}) })).json();
  render(data);
}

$("ctxadd").onclick = async () => {
  const label = $("ctxinput").value.trim();
  if (!label) return;
  $("ctxinput").value = "";
  const data = await (await fetch("/api/add-context", { method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({session: sid, group: "data_type", label}) })).json();
  render(data);
};

$("rerun").onclick = async () => {
  await fetch("/api/rerun", { method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({session: sid}) });
  $("guesses").innerHTML = `<span class="spin">re-reading with your context</span>`;
  poll();
};
</script>
</main></body></html>"""


def serve(port=7358, open_browser=True):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"refinr demo at {url}  (Ctrl+C to stop)")
    print(f"limits: {MAX_FILES} files, {MAX_FILE_BYTES//1000}KB/file, "
          f"{MAX_TOTAL_BYTES//1000}KB total -- the CLI has none")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        for session in _SESSIONS.values():
            session.cleanup()
