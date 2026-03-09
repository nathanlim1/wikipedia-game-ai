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
    /* Comparison card styles */
    #optimalPathLine { white-space: pre-wrap; background: #f0f7f0; padding: 10px; border-radius: 12px; border-left: 3px solid #0a7a2a; }
    #comparisonCard { display: none; } /* Hidden by default, shown via JavaScript */
    .metrics-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    .metrics-table th, .metrics-table td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #e5e5e5; }
    .metrics-table th { font-size: 13px; opacity: 0.7; font-weight: 600; }
    .metrics-table td { font-size: 14px; }
    .metrics-table .metric-value { font-weight: 700; font-size: 18px; }
    .metric-good { color: #0a7a2a; }
    .metric-warn { color: #b06000; }
    .metric-bad  { color: #b3261e; }
    .wp-unavailable { color: #888; font-style: italic; }
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
  <h1>Wikipedia Game AI <span class="small">(Developed by The Code Ninjas)</span></h1>

  <div class="row">
    <div>
      <label>Start Wikipedia page title</label>
      <input id="start" value=""/>
    </div>
    <div>
      <label>Target Wikipedia page title</label>
      <input id="target" value=""/>
    </div>
  </div>

  <div class="row" style="margin-top:12px; margin-bottom:24px;">
    <div>
      <label>Agent</label>
      <select id="agentSelect"><option value="">loading…</option></select>
    </div>
    <div id="llmChoicesWrap" style="display:none;">
      <label>LLM candidates</label>
      <input id="llmChoices" type="number" min="1" max="4096" value="28" style="width:90px;"/>
    </div>
    <div id="retrievalTopKWrap" style="display:none;">
      <label>Results per query</label>
      <input id="retrievalTopK" type="number" min="1" max="200" value="10" style="width:90px;"/>
    </div>
    <div id="totSettingsWrap" style="display:none;">
      <label>ToT: links to LLM</label>
      <input id="totLlmCandidates" type="number" min="1" max="1000" value="50" style="width:70px;" title="Links sent to LLM for scoring per expansion"/>
      <label>ToT: top k</label>
      <input id="totK" type="number" min="1" max="50" value="5" style="width:60px;" title="Best k candidates added as children"/>
    </div>
    <div id="totExpansionsWrap" style="display:none;">
      <label>ToT: expansions/step</label>
      <input id="totExpansions" type="number" min="1" max="50" value="15" style="width:60px;"/>
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

  <div class="card" id="comparisonCard">
    <h3>Path Statistics</h3>
    <div id="wpError" class="wp-unavailable" style="display:none;"></div>
    <div id="wpContent">
      <div><b>Optimal shortest path</b> <span class="small">(via Wikipath)</span> <span id="optimalLengthBadge" class="pill"></span></div>
      <div id="optimalPathLine" class="mono" style="margin-top:6px;"></div>

      <div id="metricsWrap" style="display:none; margin-top:14px;">
        <b>Agent Final Path Performance Metrics</b>
        <table class="metrics-table">
          <tr>
            <th>Metric</th>
            <th>Value</th>
          </tr>
          <tr>
            <td>Optimal path length</td>
            <td id="metricOptimal" class="metric-value">-</td>
          </tr>
          <tr>
            <td>LLM path length</td>
            <td id="metricLLM" class="metric-value">-</td>
          </tr>
          <tr>
            <td>Efficiency (optimal / LLM)</td>
            <td id="metricEfficiency" class="metric-value">-</td>
          </tr>
          <tr>
            <td>Number of shortest paths that exist</td>
            <td id="metricCount" class="metric-value">-</td>
          </tr>
          <tr>
            <td>Time elapsed</td>
            <td id="metricTime" class="metric-value">-</td>
          </tr>
        </table>
      </div>
    </div>
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

  // ── Comparison panel helpers ────────────────────────────────────────────
  let optimalData = null;  // stored from /api/start response

  function resetComparison() {
    optimalData = null;
    el("comparisonCard").style.display = "none";
    el("wpError").style.display = "none";
    el("wpContent").style.display = "block";
    el("optimalPathLine").textContent = "";
    el("optimalLengthBadge").textContent = "";
    el("metricsWrap").style.display = "none";
  }

  function showOptimalPath(startData) {
    // Debug logging
    console.log("showOptimalPath called with:", {
      optimal_path: startData.optimal_path,
      optimal_length: startData.optimal_length,
      wikipath_error: startData.wikipath_error
    });

    const card = el("comparisonCard");
    if (!card) {
      console.error("comparisonCard element not found!");
      return;
    }

    // Always show the comparison card (must use "block", not "" — CSS default is display:none)
    card.style.display = "block";
    console.log("Comparison card display set to block");

    if (startData.wikipath_error || startData.optimal_path == null) {
      el("wpError").style.display = "";
      el("wpError").textContent = "Shortest path data unavailable" +
        (startData.wikipath_error ? ": " + startData.wikipath_error : 
         " (Wikipath service may be unreachable or no path found)");
      el("wpContent").style.display = "none";
      console.log("Showing error message, hiding content");
      return;
    }

    // Hide error, show content
    el("wpError").style.display = "none";
    el("wpContent").style.display = "";

    optimalData = {
      path: startData.optimal_path,
      length: startData.optimal_length,
      count: startData.optimal_count,
    };

    const pathStr = optimalData.path.join("  -->  ");
    console.log("Setting path string:", pathStr);
    el("optimalPathLine").textContent = pathStr || "(same page)";

    const len = optimalData.length != null ? optimalData.length : "?";
    const badgeText = len + " hop" + (len !== 1 ? "s" : "");
    console.log("Setting length badge:", badgeText);
    el("optimalLengthBadge").textContent = badgeText;
    
    console.log("Comparison card should now be visible with path:", pathStr);
  }

  function showMetrics(stepData) {
    if (!optimalData) return;
    el("metricsWrap").style.display = "";

    const opt = optimalData.length != null ? optimalData.length : "-";
    const llm = stepData.hops != null ? stepData.hops : "-";
    el("metricOptimal").textContent = opt;
    el("metricLLM").textContent = llm;

    if (stepData.efficiency != null) {
      const pct = (stepData.efficiency * 100).toFixed(1) + "%";
      const effEl = el("metricEfficiency");
      effEl.textContent = pct;
      effEl.className = "metric-value " + (
        stepData.efficiency >= 1.0 ? "metric-good" :
        stepData.efficiency >= 0.5 ? "metric-warn" : "metric-bad"
      );
    } else if (stepData.success === false) {
      el("metricEfficiency").textContent = "N/A (did not reach target)";
      el("metricEfficiency").className = "metric-value metric-bad";
    } else {
      el("metricEfficiency").textContent = "-";
    }

    el("metricCount").textContent = optimalData.count != null ? optimalData.count.toLocaleString() : "-";

    const timeSec = stepData.llm_to_target_seconds;
    el("metricTime").textContent = timeSec != null ? timeSec + "s" : "N/A";
  }

  async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  function updateAgentSettings() {
    const agent = el("agentSelect").value;
    el("llmChoicesWrap").style.display    = agent === "default"  ? "" : "none";
    el("retrievalTopKWrap").style.display = agent === "planning" ? "" : "none";
    el("totSettingsWrap").style.display   = agent === "tot" ? "" : "none";
    el("totExpansionsWrap").style.display = agent === "tot" ? "" : "none";
  }

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
      updateAgentSettings();
    } catch (e) {
      console.error("Could not load agent list:", e);
    }
  })();

  el("agentSelect").addEventListener("change", updateAgentSettings);

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
    resetComparison();

    const agentId = el("agentSelect").value;
    const modelId = el("modelSelect").value;
    const payload = {
      start_title: el("start").value.trim(),
      target_title: el("target").value.trim(),
      agent_id: agentId || null,
      llm_choices: agentId === "default" ? parseInt(el("llmChoices").value, 10) : null,
      retrieval_top_k: agentId === "planning" ? parseInt(el("retrievalTopK").value, 10) : null,
      tot_k: agentId === "tot" ? parseInt(el("totK").value, 10) : null,
      tot_llm_candidates: agentId === "tot" ? parseInt(el("totLlmCandidates").value, 10) : null,
      tot_expansions_per_step: agentId === "tot" ? parseInt(el("totExpansions").value, 10) : null,
      tot_score_samples: agentId === "tot" ? 1 : null,
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

      // Show optimal shortest path from Wikipath
      try {
        showOptimalPath(startData);
        if (startData.optimal_length != null) {
          logLine("OPTIMAL_LENGTH:  " + startData.optimal_length + " hops (via Wikipath)", "log-dim");
        }
      } catch (err) {
        console.error("Error showing optimal path:", err);
        logLine("ERROR showing optimal path: " + err, "log-backtrack");
      }

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

        const evList = stepData.events || (stepData.event ? [stepData.event] : []);
        for (const ev of evList) {
          if (!ev || !ev.type) continue;

          if (ev.type === "exploration_start") {
            logLine(`[ToT] ${ev.message || "Exploring..."} (${ev.expansions_planned || "?"} expansions planned)`, "log-section");
            logBlank();
          } else if (ev.type === "exploration_progress") {
            const n = ev.expansion_n != null ? ev.expansion_n : "?";
            const node = ev.node_expanded || "?";
            const size = ev.frontier_size != null ? ev.frontier_size : "?";
            const back = ev.backtracked ? " [branch switch]" : "";
            logLine(`[ToT] Expansion ${n}: expanded ${node}, frontier size ${size}${back}`, "log-section");
            logBlank();
          } else if (ev.type === "move") {
            const m = ev.move;
            logLine(`${m.step}. ${m.from_title}  --["${m.anchor_text}"]-->  ${m.to_title}`, "log-move");
            if (m.analysis) logLine(`   Why: ${m.analysis}`, "log-dim");
            if (m.tot_score != null) logLine(`   Score: ${m.tot_score}/100`, "log-dim");
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
            const timeStr = stepData.llm_to_target_seconds != null ? ` (${stepData.llm_to_target_seconds}s)` : "";
            setStatus(`<span class="ok">Success</span> in <b>${stepData.hops}</b> hops${timeStr} <span class="pill">Target: ${stepData.resolved_target}</span>`);
          } else {
            setStatus(`<span class="bad">Failed</span>: ${stepData.failure_reason} <span class="pill">Target: ${stepData.resolved_target}</span>`);
          }
          showMetrics(stepData);
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
