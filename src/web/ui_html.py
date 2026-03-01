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
    box.textContent += s + "\n";
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
