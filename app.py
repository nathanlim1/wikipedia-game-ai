import os
import json
import random
import re
import time
import secrets
from typing import List, Dict, Any, Tuple, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import tinker
from tinker import types
from transformers import AutoTokenizer

from bs4 import BeautifulSoup
from urllib.parse import unquote
from concurrent.futures import TimeoutError as FutureTimeoutError


# -----------------------------
# Config
# -----------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "WikiMazeLLMAgent/1.0 (edu project)"

# Not user-facing. Prevents infinite runs / runaway sessions.
SAFETY_MAX_MOVES = 800
SAFETY_MAX_SECONDS = 7 * 60  # 7 minutes

# Wikipedia requests
HTTP_TIMEOUT_S = 25
HTTP_RETRIES = 2

# Link scraping
MAX_HTML_LINKS = 6000

# Candidate pool + prompt size
CANDIDATE_POOL = 120     # pool after heuristic ranking
LLM_CHOICES = 28         # what we actually show to the LLM (keeps it fast/reliable)

# LLM timeouts/fallbacks
LLM_TIMEOUT_S = 20

MODEL_NAME = os.getenv("TINKER_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

# Tinker client
service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)

# Tokenizer (remote inference via Tinker; local tokenizer only)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# HTTP session
http = requests.Session()

app = FastAPI()

# In-memory sessions (local only)
SESSIONS: Dict[str, Dict[str, Any]] = {}


# -----------------------------
# UI
# -----------------------------
INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Wiki Maze Solver (Tinker)</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; max-width: 1100px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: end; }
    label { display: block; font-size: 12px; opacity: 0.8; margin-bottom: 6px; }
    input { width: 350px; padding: 10px; border: 1px solid #ccc; border-radius: 10px; }
    button { padding: 10px 14px; border: 0; border-radius: 12px; cursor: pointer; font-weight: 700; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .card { margin-top: 16px; padding: 14px; border: 1px solid #e5e5e5; border-radius: 14px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New"; }
    #log { height: 460px; overflow: auto; white-space: pre-wrap; background: #0b0b0b; color: #eaeaea; padding: 12px; border-radius: 12px; }
    .small { font-size: 12px; opacity: 0.8; }
    .ok { color: #0a7a2a; font-weight: 900; }
    .bad { color: #b3261e; font-weight: 900; }
    #pathLine { white-space: pre-wrap; background: #f6f6f6; padding: 10px; border-radius: 12px; }
    #stepsBox { white-space: pre-wrap; background: #f6f6f6; padding: 10px; border-radius: 12px; }
    .pill { display:inline-block; font-size:12px; padding:3px 8px; border-radius:999px; background:#f1f1f1; margin-left:8px; }
  </style>
</head>
<body>
  <h1>Wiki Maze Solver <span class="small">(single agent, sequential, backtracks when stuck)</span></h1>

  <div class="row">
    <div>
      <label>Start Wikipedia page title</label>
      <input id="start" value=""/>
    </div>
    <div>
      <label>Target Wikipedia page title</label>
      <input id="target" value=""/>
    </div>
    <div style="display:flex; gap:10px;">
      <button id="go">Run</button>
      <button id="stop" disabled>Stop</button>
    </div>
  </div>

  <div class="card">
    <h3>Result</h3>
    <div id="statusLine"></div>
    <div class="small" style="margin-top:6px;">
      Each hop prints the exact <b>anchor text</b> you can Ctrl+F on the current Wikipedia page, then click.
    </div>

    <div style="margin-top:10px;"><b>Current path:</b></div>
    <div id="pathLine" class="mono"></div>

    <div style="margin-top:14px;"><b>Steps (links clicked):</b></div>
    <div id="stepsBox" class="mono"></div>
  </div>

  <div class="card">
    <h3>Console log</h3>
    <div id="log" class="mono"></div>
  </div>

<script>
  const el = (id) => document.getElementById(id);
  let running = false;
  let sessionId = null;

  function logLine(s) {
    const box = el("log");
    box.textContent += s + "\\n";
    box.scrollTop = box.scrollHeight;
  }
  function setStatus(html) { el("statusLine").innerHTML = html; }
  function setPath(text) { el("pathLine").textContent = text; }
  function setSteps(text) { el("stepsBox").textContent = text; }

  async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  el("stop").addEventListener("click", () => {
    running = false;
    el("stop").disabled = true;
    el("go").disabled = false;
    logLine(">>> Stop requested.");
    setStatus(`<span class="bad">⏹ Stopped</span>`);
  });

  el("go").addEventListener("click", async () => {
    el("go").disabled = true;
    el("stop").disabled = false;
    running = true;
    sessionId = null;

    el("log").textContent = "";
    setPath("");
    setSteps("");
    setStatus("");

    const payload = {
      start_title: el("start").value.trim(),
      target_title: el("target").value.trim()
    };

    logLine(">>> Starting run...");
    try {
      const startRes = await fetch("/api/start", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const startData = await startRes.json();

      if (!startRes.ok) {
        setStatus(`<span class="bad">❌ Error</span>: ${startData.failure_reason || "unknown"}`);
        logLine("ERROR: " + (startData.failure_reason || "unknown"));
        running = false;
        el("stop").disabled = true;
        el("go").disabled = false;
        return;
      }

      sessionId = startData.session_id;
      setStatus(`Running… <span class="pill">Start: ${startData.resolved_start}</span><span class="pill">Target: ${startData.resolved_target}</span>`);
      logLine("RESOLVED_START: " + startData.resolved_start);
      logLine("RESOLVED_TARGET: " + startData.resolved_target);
      logLine("");

      while (running) {
        const stepRes = await fetch("/api/step", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ session_id: sessionId })
        });
        const stepData = await stepRes.json();

        if (!stepRes.ok) {
          setStatus(`<span class="bad">❌ Error</span>: ${stepData.failure_reason || "unknown"}`);
          logLine("ERROR: " + (stepData.failure_reason || "unknown"));
          break;
        }

        setPath(stepData.chain || "");
        setSteps(stepData.steps_text || "");

        if (stepData.event) {
          if (stepData.event.type === "move") {
            const m = stepData.event.move;
            logLine(`${m.step}. ${m.from_title} --["${m.anchor_text}"]--> ${m.to_title}`);
            if (m.analysis) logLine("   Why: " + m.analysis);
            logLine("");
          } else if (stepData.event.type === "backtrack") {
            logLine(`<<< BACKTRACK: ${stepData.event.from_title} -> ${stepData.event.to_title} (${stepData.event.reason})`);
            logLine("");
          }
        }

        if (stepData.done) {
          if (stepData.success) {
            setStatus(`<span class="ok">✅ Success</span> in <b>${stepData.hops}</b> hops <span class="pill">Target: ${stepData.resolved_target}</span>`);
          } else {
            setStatus(`<span class="bad">❌ Failed</span>: ${stepData.failure_reason} <span class="pill">Target: ${stepData.resolved_target}</span>`);
          }
          logLine(">>> Done.");
          break;
        }

        await sleep(60);
      }
    } catch (e) {
      setStatus(`<span class="bad">❌ Error</span>: ${e}`);
      logLine("ERROR: " + e);
      console.error(e);
    } finally {
      running = false;
      el("stop").disabled = true;
      el("go").disabled = false;
    }
  });
</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(INDEX_HTML)


# -----------------------------
# API Models
# -----------------------------
class StartRequest(BaseModel):
    start_title: str
    target_title: str

class StepRequest(BaseModel):
    session_id: str


# -----------------------------
# Wikipedia helpers
# -----------------------------
def wiki_get(params: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT}
    last_err = None
    for attempt in range(HTTP_RETRIES + 1):
        try:
            r = http.get(WIKI_API, params=params, headers=headers, timeout=HTTP_TIMEOUT_S)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.45 * (attempt + 1))
    raise last_err  # type: ignore

def resolve_title_exact(title: str) -> str:
    title = title.strip()
    if not title:
        raise ValueError("Empty title")

    data = wiki_get({"action": "query", "format": "json", "redirects": 1, "titles": title})
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    if page.get("missing") is None and page.get("title"):
        return page["title"]
    raise ValueError(f"Wikipedia page not found (exact): {title}")

def resolve_title_fuzzy_start(title: str) -> str:
    title = title.strip()
    if not title:
        raise ValueError("Empty title")
    try:
        return resolve_title_exact(title)
    except Exception:
        pass

    os_data = wiki_get({"action": "opensearch", "format": "json", "search": title, "limit": 1, "namespace": 0})
    if isinstance(os_data, list) and len(os_data) >= 2 and os_data[1]:
        return resolve_title_exact(os_data[1][0])

    raise ValueError(f"Wikipedia page not found: {title}")

def get_extract(title: str, max_chars: int = 650) -> str:
    data = wiki_get({
        "action": "query", "format": "json",
        "prop": "extracts", "explaintext": 1, "exintro": 1,
        "redirects": 1, "titles": title
    })
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    extract = (page.get("extract") or "").strip()
    extract = re.sub(r"\s+", " ", extract)
    return extract[:max_chars]

def get_visible_outgoing_links(title: str, max_total: int = MAX_HTML_LINKS) -> Tuple[List[str], Dict[str, str]]:
    """
    Extract human-clickable links from rendered article body.
    Returns (unique page titles list, title->anchor_text mapping).
    """
    data = wiki_get({
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text",
        "redirects": 1
    })
    if "error" in data:
        raise ValueError(f"Wikipedia parse error for '{title}': {data['error']}")

    html = data["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "lxml")
    root = soup.find("div", class_="mw-parser-output") or soup

    # Remove sections that create “weird/ghosty” paths or non-body navigation.
    for selector in [
        "div.navbox", "div.vertical-navbox", "table.navbox",
        "div.reflist", "ol.references", "div.mw-references-wrap",
        "div.catlinks", "div.toc", "span.mw-editsection",
        "sup.reference"
    ]:
        for node in root.select(selector):
            node.decompose()

    titles: List[str] = []
    seen = set()
    title_to_anchor: Dict[str, str] = {}

    for a in root.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/wiki/"):
            continue

        classes = a.get("class") or []
        if "new" in classes:  # redlink
            continue

        slug = href.split("/wiki/", 1)[1].split("#", 1)[0]
        if not slug:
            continue

        # main namespace only
        if ":" in slug:
            continue

        # destination title
        t = a.get("title")
        if not t:
            t = unquote(slug).replace("_", " ")
        t = t.strip()
        if not t or t == "Main Page":
            continue

        # anchor text EXACTLY as displayed (preserve spaces between nested spans)
        anchor = a.get_text(" ", strip=True)
        if not anchor:
            anchor = t

        if t not in seen:
            seen.add(t)
            titles.append(t)
            title_to_anchor[t] = anchor

        if len(titles) >= max_total:
            break

    return titles, title_to_anchor


# -----------------------------
# Heuristics + "avoid list hell"
# -----------------------------
def tokenize_simple(s: str) -> List[str]:
    return [tok for tok in re.split(r"[^a-zA-Z0-9]+", s.lower()) if tok]

def title_penalty(title: str) -> float:
    p = 0.0
    tl = title.lower()
    if tl.startswith("list of "):
        p += 2.2
    if "(disambiguation)" in tl:
        p += 3.5
    if tl.startswith("outline of "):
        p += 2.2
    if tl.startswith("index of "):
        p += 2.2
    return p

def heuristic_score(title: str, target_title: str, target_keywords: set) -> float:
    lt = set(tokenize_simple(title))
    tt = set(tokenize_simple(target_title))
    overlap_title = len(lt & tt)
    overlap_kw = len(lt & target_keywords)
    contains = 2.0 if target_title.lower() in title.lower() else 0.0
    return (1.8 * overlap_title) + (1.0 * overlap_kw) + contains - title_penalty(title)

def build_candidate_list(
    outgoing: List[str],
    title_to_anchor: Dict[str, str],
    target_title: str,
    target_keywords: set,
    path_set: set,
    tried_edges_from_current: set,
) -> List[str]:
    filtered = []
    for t in outgoing:
        if t in path_set:
            continue
        if t not in title_to_anchor:
            continue
        if (t,) in tried_edges_from_current:
            continue
        filtered.append(t)

    if not filtered:
        return []

    scored = sorted(
        filtered,
        key=lambda t: heuristic_score(t, target_title, target_keywords),
        reverse=True
    )

    pool = scored[: min(CANDIDATE_POOL, len(scored))]
    rest = scored[len(pool):]
    if rest:
        pool += random.sample(rest, min(20, len(rest)))

    seen = set()
    out = []
    for t in pool:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= CANDIDATE_POOL:
            break

    return out


# -----------------------------
# LLM: destination-aware reasoning + choose index
# -----------------------------
DECISION_INSTRUCTIONS = """Return ONLY valid JSON. No extra text.

Schema:
{
  "choice_index": <integer>,
  "analysis": "<2-4 sentences, destination-aware, explain bridge logic>"
}

Rules:
- choice_index must be one of the provided candidate indices.
- Make your analysis explicitly reference BOTH the current page AND the target page.
- Prefer moves that create a clear chain toward the target topic (characters/TV show/franchise/people/company/etc.).
- Avoid "List of ..." and "(disambiguation)" unless they are clearly a bridge.
"""

def build_prompt(
    current_title: str,
    current_extract: str,
    target_title: str,
    target_extract: str,
    candidates: List[str],
    candidate_scores: List[float],
    recent_path: List[str],
) -> str:
    lines = []
    for i, t in enumerate(candidates):
        flags = []
        if t.lower().startswith("list of "):
            flags.append("LIST")
        if "(disambiguation)" in t.lower():
            flags.append("DISAMBIG")
        f = f" [{', '.join(flags)}]" if flags else ""
        lines.append(f"{i}: {t}{f}  (score={candidate_scores[i]:.2f})")

    user = (
        f"CURRENT: {current_title}\n"
        f"CURRENT_INTRO: {current_extract}\n\n"
        f"TARGET: {target_title}\n"
        f"TARGET_INTRO: {target_extract}\n\n"
        f"RECENT_PATH: {recent_path}\n\n"
        f"CANDIDATES:\n" + "\n".join(lines) + "\n\n"
        "Pick the best next hop and explain the bridge logic toward the target."
    )

    messages = [
        {"role": "system", "content": DECISION_INSTRUCTIONS},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return DECISION_INSTRUCTIONS + "\n\n" + user

def parse_json_first(output_text: str) -> Dict[str, Any]:
    output_text = output_text.strip()
    m = re.search(r"\{.*\}", output_text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found. Raw: {output_text[:200]}")
    return json.loads(m.group(0))

def llm_choose(
    current_title: str,
    current_extract: str,
    target_title: str,
    target_extract: str,
    candidates: List[str],
    candidate_scores: List[float],
    recent_path: List[str],
) -> Tuple[int, str]:
    prompt_text = build_prompt(
        current_title=current_title,
        current_extract=current_extract,
        target_title=target_title,
        target_extract=target_extract,
        candidates=candidates,
        candidate_scores=candidate_scores,
        recent_path=recent_path,
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    params = types.SamplingParams(
        max_tokens=260,
        temperature=0.2,
        top_p=0.9
    )

    fut = sampling_client.sample(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=params,
    )

    try:
        res = fut.result(timeout=LLM_TIMEOUT_S)
    except TypeError:
        res = fut.result()
    except FutureTimeoutError:
        raise

    toks = res.sequences[0].tokens
    text = tokenizer.decode(toks)
    data = parse_json_first(text)
    idx = int(data["choice_index"])
    analysis = str(data.get("analysis", "")).strip()
    return idx, analysis


# -----------------------------
# Search display helpers
# -----------------------------
def chain_string(start: str, moves: List[Dict[str, Any]]) -> str:
    s = start
    for m in moves:
        s += f' --["{m["anchor_text"]}"]--> {m["to_title"]}'
    return s

def steps_text_from_moves(start: str, target: str, moves: List[Dict[str, Any]]) -> str:
    """
    What the user wants:
    Start: <start page>
    Step 1: <anchor text clicked>
    Step 2: <anchor text clicked>
    ...
    Destination: <target page>
    """
    lines = [f"Start: {start}"]
    for i, m in enumerate(moves, start=1):
        lines.append(f"Step {i}: {m['anchor_text']}")
    lines.append(f"Destination: {target}")
    return "\n".join(lines)

def target_keywords_from_extract(target_title: str, target_extract: str) -> set:
    toks = tokenize_simple(target_title) + tokenize_simple(target_extract)
    stop = {"the","and","of","to","in","a","an","for","on","by","with","as","at","is","was","are","from","that","it"}
    return {t for t in toks if t not in stop and len(t) >= 3}

def make_frame(
    page_title: str,
    target_title: str,
    target_keywords: set,
    path_set: set,
    tried_edges: Dict[str, set],
) -> Dict[str, Any]:
    outgoing, title_to_anchor = get_visible_outgoing_links(page_title)
    tried_from = tried_edges.setdefault(page_title, set())

    candidates = build_candidate_list(
        outgoing=outgoing,
        title_to_anchor=title_to_anchor,
        target_title=target_title,
        target_keywords=target_keywords,
        path_set=path_set,
        tried_edges_from_current={(c,) for c in tried_from},
    )

    scores = [heuristic_score(c, target_title, target_keywords) for c in candidates]

    return {
        "page": page_title,
        "title_to_anchor": title_to_anchor,
        "candidates": candidates,
        "scores": scores,
    }

def best_fallback_index(scores: List[float]) -> int:
    if not scores:
        return 0
    best_i = 0
    best_v = scores[0]
    for i, v in enumerate(scores):
        if v > best_v:
            best_i = i
            best_v = v
    return best_i


# -----------------------------
# API
# -----------------------------
@app.post("/api/start")
def api_start(req: StartRequest):
    try:
        resolved_start = resolve_title_fuzzy_start(req.start_title)
        resolved_target = resolve_title_exact(req.target_title)
    except Exception as e:
        return JSONResponse({"failure_reason": str(e)}, status_code=400)

    target_extract = get_extract(resolved_target)
    target_keywords = target_keywords_from_extract(resolved_target, target_extract)

    sid = secrets.token_urlsafe(12)
    SESSIONS[sid] = {
        "started_at": time.time(),
        "resolved_start": resolved_start,
        "resolved_target": resolved_target,
        "target_extract": target_extract,
        "target_keywords": target_keywords,

        "path": [resolved_start],
        "path_set": {resolved_start},
        "moves": [],

        "stack": [],

        "tried_edges": {},  # page -> set(next_titles_tried)

        "done": False,
        "success": False,
        "failure_reason": "",
    }

    return {"session_id": sid, "resolved_start": resolved_start, "resolved_target": resolved_target}


@app.post("/api/step")
def api_step(req: StepRequest):
    s = SESSIONS.get(req.session_id)
    if not s:
        return JSONResponse({"failure_reason": "Invalid session_id. Click Run again."}, status_code=400)

    def payload(event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "done": s["done"],
            "success": s["success"],
            "failure_reason": s["failure_reason"],
            "resolved_start": s["resolved_start"],
            "resolved_target": s["resolved_target"],
            "hops": len(s["moves"]),
            "chain": chain_string(s["resolved_start"], s["moves"]),
            "steps_text": steps_text_from_moves(s["resolved_start"], s["resolved_target"], s["moves"]),
            "event": event,
        }

    if s["done"]:
        return payload()

    # Hidden safety stops
    if (time.time() - s["started_at"]) > SAFETY_MAX_SECONDS:
        s["done"] = True
        s["success"] = False
        s["failure_reason"] = f"Stopped by safety timer ({SAFETY_MAX_SECONDS}s)."
        return payload()

    if len(s["moves"]) >= SAFETY_MAX_MOVES:
        s["done"] = True
        s["success"] = False
        s["failure_reason"] = f"Stopped by safety move cap ({SAFETY_MAX_MOVES})."
        return payload()

    target = s["resolved_target"]
    current = s["path"][-1]

    if current == target:
        s["done"] = True
        s["success"] = True
        return payload()

    # Ensure stack top frame corresponds to current
    if not s["stack"] or s["stack"][-1]["page"] != current:
        try:
            frame = make_frame(
                page_title=current,
                target_title=target,
                target_keywords=s["target_keywords"],
                path_set=s["path_set"],
                tried_edges=s["tried_edges"],
            )
        except Exception as e:
            if len(s["path"]) <= 1:
                s["done"] = True
                s["success"] = False
                s["failure_reason"] = f"Wikipedia scrape failed at start: {e}"
                return payload()

            prev = s["path"][-2]
            popped = s["path"].pop()
            s["path_set"].remove(popped)
            if s["moves"]:
                s["moves"].pop()
            if s["stack"] and s["stack"][-1].get("page") == popped:
                s["stack"].pop()

            return payload({"type": "backtrack", "from_title": popped, "to_title": prev, "reason": f"scrape failed: {e}"})

        s["stack"].append(frame)

    frame = s["stack"][-1]
    candidates = frame["candidates"]
    scores = frame["scores"]

    # Dead end
    if not candidates:
        if len(s["path"]) <= 1:
            s["done"] = True
            s["success"] = False
            s["failure_reason"] = f"No usable outgoing links from '{current}'."
            return payload()

        prev = s["path"][-2]
        popped = s["path"].pop()
        s["path_set"].remove(popped)
        if s["moves"]:
            s["moves"].pop()
        s["stack"].pop()

        return payload({"type": "backtrack", "from_title": popped, "to_title": prev, "reason": "no candidates"})

    # LLM subset
    current_extract = get_extract(current)
    target_extract = s["target_extract"]

    llm_view = candidates[: min(LLM_CHOICES, len(candidates))]
    llm_scores = scores[: len(llm_view)]

    idx = None
    analysis = ""

    try:
        llm_idx, analysis = llm_choose(
            current_title=current,
            current_extract=current_extract,
            target_title=target,
            target_extract=target_extract,
            candidates=llm_view,
            candidate_scores=llm_scores,
            recent_path=s["path"][-8:],
        )
        if 0 <= llm_idx < len(llm_view):
            idx = llm_idx
    except FutureTimeoutError:
        idx = best_fallback_index(llm_scores)
        analysis = "(LLM timeout; heuristic fallback)"
    except Exception:
        idx = best_fallback_index(llm_scores)
        analysis = "(LLM error; heuristic fallback)"

    chosen = llm_view[idx] if idx is not None else llm_view[0]

    tried_from = s["tried_edges"].setdefault(current, set())

    if chosen in tried_from or chosen in s["path_set"]:
        chosen = None
        for cand in candidates:
            if cand not in tried_from and cand not in s["path_set"]:
                chosen = cand
                break
        if not chosen:
            if len(s["path"]) <= 1:
                s["done"] = True
                s["success"] = False
                s["failure_reason"] = f"Exhausted all outgoing options from '{current}'."
                return payload()

            prev = s["path"][-2]
            popped = s["path"].pop()
            s["path_set"].remove(popped)
            if s["moves"]:
                s["moves"].pop()
            s["stack"].pop()
            return payload({"type": "backtrack", "from_title": popped, "to_title": prev, "reason": "exhausted options"})

    tried_from.add(chosen)
    anchor_text = frame["title_to_anchor"].get(chosen, chosen)

    move = {
        "step": len(s["moves"]) + 1,
        "from_title": current,
        "to_title": chosen,
        "anchor_text": anchor_text,
        "analysis": analysis
    }
    s["moves"].append(move)
    s["path"].append(chosen)
    s["path_set"].add(chosen)

    if chosen == target:
        s["done"] = True
        s["success"] = True

    return payload({"type": "move", "move": move})
