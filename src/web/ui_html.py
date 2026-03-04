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
    input { width: 300px; padding: 10px; border: 1px solid #ccc; border-radius: 10px; }
    select { padding: 10px; border: 1px solid #ccc; border-radius: 10px; font-size: 14px; background: #fff; }
    .model-wrap { position: relative; }
    .model-row { display: flex; align-items: center; gap: 8px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .spinner { display: none; width: 14px; height: 14px; border: 2px solid #ddd; border-top-color: #555; border-radius: 50%; animation: spin 0.7s linear infinite; flex-shrink: 0; }
    #modelStatus { position: absolute; top: 100%; left: 0; margin-top: 4px; white-space: nowrap; font-size: 12px; font-weight: 600; pointer-events: none; }
    .model-loading-state { color: #b06000; }
    .model-ready-state   { color: #0a7a2a; }
    .model-error-state   { color: #b3261e; }
    button { padding: 10px 14px; border: 0; border-radius: 12px; cursor: pointer; font-weight: 700; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .card { margin-top: 16px; padding: 14px; border: 1px solid #e5e5e5; border-radius: 14px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New"; }
    #log { height: 520px; overflow: auto; white-space: pre-wrap; background: #0b0b0b; color: #eaeaea; padding: 12px; border-radius: 12px; font-size: 13px; line-height: 1.5; }
    .small { font-size: 12px; opacity: 0.8; }
    .ok { color: #0a7a2a; font-weight: 900; }
    .bad { color: #b3261e; font-weight: 900; }
    #pathLine { white-space: pre-wrap; background: #f6f6f6; padding: 10px; border-radius: 12px; }
    #stepsBox { white-space: pre-wrap; background: #f6f6f6; padding: 10px; border-radius: 12px; }
    .pill { display:inline-block; font-size:12px; padding:3px 8px; border-radius:999px; background:#f1f1f1; margin-left:8px; }
    /* Planning-agent log colours */
    .log-section   { color: #6ec6ff; }
    .log-hyp       { color: #c3e88d; }
    .log-search    { color: #ffcb6b; }
    .log-decision  { color: #f78c6c; }
    .log-move      { color: #eaeaea; }
    .log-backtrack { color: #ff5370; }
    .log-dim       { color: #777; }
  </style>
</head>
<body>
  <h1>Wiki Maze Solver <span class="small">(Tinker AI)</span></h1>

  <div class="row">
    <div>
      <label>Start Wikipedia page title</label>
      <input id="start" value=""/>
    </div>
    <div>
      <label>Target Wikipedia page title</label>
      <input id="target" value=""/>
    </div>
    <div>
      <label>Agent</label>
      <select id="agentSelect"><option value="">loading…</option></select>
    </div>
    <div class="model-wrap">
      <label>Model</label>
      <div class="model-row">
        <select id="modelSelect" style="min-width:260px;"><option value="">loading…</option></select>
        <span id="modelSpinner" class="spinner"></span>
      </div>
      <div id="modelStatus"></div>
    </div>
    <div style="display:flex; gap:10px; align-items:end;">
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

  // ── Coloured log helpers ────────────────────────────────────────────────
  function logSpan(text, cssClass) {
    const box = el("log");
    const span = document.createElement("span");
    if (cssClass) span.className = cssClass;
    span.textContent = text + "\n";
    box.appendChild(span);
    box.scrollTop = box.scrollHeight;
  }
  function logLine(s, cssClass) { logSpan(s, cssClass || "log-move"); }
  function logBlank() { logLine("", "log-dim"); }

  function setStatus(html) { el("statusLine").innerHTML = html; }
  function setPath(text)   { el("pathLine").textContent = text; }
  function setSteps(text)  { el("stepsBox").textContent = text; }

  async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  // ── Populate agent selector on load ────────────────────────────────────
  (async () => {
    try {
      const res = await fetch("/api/agents");
      const data = await res.json();
      const sel = el("agentSelect");
      sel.innerHTML = "";
      for (const id of data.agents) {
        const opt = document.createElement("option");
        opt.value = id;
        opt.textContent = id;
        if (id === data.default) opt.selected = true;
        sel.appendChild(opt);
      }
    } catch (e) {
      console.error("Could not load agent list:", e);
    }
  })();

  // ── Populate model selector on load ────────────────────────────────────
  (async () => {
    try {
      const res = await fetch("/api/models");
      const data = await res.json();
      const sel = el("modelSelect");
      sel.innerHTML = "";
      for (const m of data.models) {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.label;
        if (m.id === data.current) opt.selected = true;
        sel.appendChild(opt);
      }
    } catch (e) {
      console.error("Could not load model list:", e);
    }
  })();

  // ── Model selector change ───────────────────────────────────────────────
  el("modelSelect").addEventListener("change", async () => {
    const sel = el("modelSelect");
    const modelId = sel.value;
    if (!modelId) return;

    const modelLabel = sel.options[sel.selectedIndex].textContent;
    const status = el("modelStatus");
    const spinner = el("modelSpinner");

    sel.disabled = true;
    el("go").disabled = true;
    spinner.style.display = "inline-block";
    status.className = "model-loading-state";
    status.textContent = `Initializing ${modelLabel}… this may take a few seconds`;

    try {
      const res = await fetch("/api/models", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({model_id: modelId})
      });
      const data = await res.json();
      if (!res.ok) {
        status.className = "model-error-state";
        status.textContent = `✗ Failed to load model: ${data.error || "unknown error"}`;
      } else {
        status.className = "model-ready-state";
        status.textContent = `✓ ${modelLabel} ready`;
        setTimeout(() => { status.textContent = ""; status.className = ""; }, 3000);
      }
    } catch (e) {
      status.className = "model-error-state";
      status.textContent = `✗ Error: ${e}`;
      console.error("Model switch error:", e);
    } finally {
      spinner.style.display = "none";
      sel.disabled = false;
      el("go").disabled = false;
    }
  });

  // ── Stop button ─────────────────────────────────────────────────────────
  el("stop").addEventListener("click", () => {
    running = false;
    el("stop").disabled = true;
    el("go").disabled = false;
    logLine(">>> Stop requested.", "log-dim");
    setStatus(`<span class="bad">Stopped</span>`);
  });

  // ── Render planning-agent details from an event ─────────────────────────
  function logPlanningDetails(event) {
    const subheadings = event.subheadings || [];
    const hypotheses  = event.hypotheses  || [];
    const searchLog   = event.search_log  || [];

    if (subheadings.length) {
      logLine("  [SECTIONS]  " + subheadings.slice(0, 12).join(" | "), "log-section");
    }
    for (let i = 0; i < hypotheses.length; i++) {
      logLine(`  [HYPOTHESIS ${i + 1}]  ${hypotheses[i]}`, "log-hyp");
    }
    for (const s of searchLog) {
      const hits = (s.top_hits || []).slice(0, 5).join(", ");
      logLine(`  [SEARCH: "${s.query}"]  → ${hits || "(no results)"}`, "log-search");
    }
  }

  // ── Run button ──────────────────────────────────────────────────────────
  el("go").addEventListener("click", async () => {
    el("go").disabled = true;
    el("stop").disabled = false;
    running = true;
    sessionId = null;

    el("log").textContent = "";
    setPath("");
    setSteps("");
    setStatus("");

    const agentId = el("agentSelect").value;
    const modelId = el("modelSelect").value;
    const payload = {
      start_title: el("start").value.trim(),
      target_title: el("target").value.trim(),
      agent_id: agentId || null,
    };

    logLine(">>> Starting run…", "log-dim");
    try {
      const startRes = await fetch("/api/start", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const startData = await startRes.json();

      if (!startRes.ok) {
        setStatus(`<span class="bad">Error</span>: ${startData.failure_reason || "unknown"}`);
        logLine("ERROR: " + (startData.failure_reason || "unknown"), "log-backtrack");
        running = false;
        el("stop").disabled = true;
        el("go").disabled = false;
        return;
      }

      sessionId = startData.session_id;
      const modelSel = el("modelSelect");
      const modelLabel = modelSel.options[modelSel.selectedIndex]?.textContent || modelId;
      setStatus(`Running… <span class="pill">Start: ${startData.resolved_start}</span><span class="pill">Target: ${startData.resolved_target}</span><span class="pill">Agent: ${agentId}</span><span class="pill">Model: ${modelLabel}</span>`);
      logLine("RESOLVED_START:  " + startData.resolved_start, "log-dim");
      logLine("RESOLVED_TARGET: " + startData.resolved_target, "log-dim");
      logBlank();

      while (running) {
        const stepRes = await fetch("/api/step", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ session_id: sessionId })
        });
        const stepData = await stepRes.json();

        if (!stepRes.ok) {
          setStatus(`<span class="bad">Error</span>: ${stepData.failure_reason || "unknown"}`);
          logLine("ERROR: " + (stepData.failure_reason || "unknown"), "log-backtrack");
          break;
        }

        setPath(stepData.chain || "");
        setSteps(stepData.steps_text || "");

        if (stepData.event) {
          const ev = stepData.event;

          if (ev.type === "move") {
            const m = ev.move;
            logLine(`${m.step}. ${m.from_title}  --["${m.anchor_text}"]-->  ${m.to_title}`, "log-move");
            if (m.analysis) logLine(`   Why: ${m.analysis}`, "log-dim");
            logPlanningDetails(ev);
            logBlank();

          } else if (ev.type === "backtrack") {
            logLine(`<<< BACKTRACK: ${ev.from_title} -> ${ev.to_title}  (${ev.reason})`, "log-backtrack");
            logPlanningDetails(ev);
            logBlank();
          }
        }

        if (stepData.done) {
          if (stepData.success) {
            setStatus(`<span class="ok">Success</span> in <b>${stepData.hops}</b> hops <span class="pill">Target: ${stepData.resolved_target}</span>`);
          } else {
            setStatus(`<span class="bad">Failed</span>: ${stepData.failure_reason} <span class="pill">Target: ${stepData.resolved_target}</span>`);
          }
          logLine(">>> Done.", "log-dim");
          break;
        }

        await sleep(60);
      }
    } catch (e) {
      setStatus(`<span class="bad">Error</span>: ${e}`);
      logLine("ERROR: " + e, "log-backtrack");
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
