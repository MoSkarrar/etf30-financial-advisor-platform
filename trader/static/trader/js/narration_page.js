(function () {
  const root = document.getElementById("narration-page");
  if (!root) return;

  const sessionId = root.dataset.sessionId;
  const wsPath = root.dataset.wsPath || `/ws/narration/${sessionId}/`;
  const latestRunId = root.dataset.latestRunId || "";
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${window.location.host}${wsPath}`);

  const runSelect = document.getElementById("run-select");
  const loadRunBtn = document.getElementById("load-run-btn");
  const terminal = document.getElementById("narration-terminal");
  const chatMessages = document.getElementById("chat-messages");
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const allocationBody = document.getElementById("allocation-table-body");
  const benchmarkCard = document.getElementById("benchmark-card");
  const riskCard = document.getElementById("risk-card");
  const scenarioCard = document.getElementById("scenario-card");
  const explainSummary = document.getElementById("explain-summary");
  const latestSummary = document.getElementById("latest-summary");
  const enginePolicyPanel = document.getElementById("engine-policy-panel");
  const ruleSummaryCard = document.getElementById("rule-summary-card");
  const explanationBundleCard = document.getElementById("explanation-bundle-card");
  const explanationLabCard = document.getElementById("explanation-lab-card");

  let latestEngineInfo = {};
  let latestPolicyCheck = {};

  function meaningfulQuestion(value) {
    const q = String(value || "").trim();
    if (!q) return false;
    if (q.length === 1) return false;
    const words = q.match(/[A-Za-z0-9]+/g) || [];
    if (!words.length) return false;
    if (words.length === 1 && words[0].length === 1) return false;
    return true;
  }

  function fmtPct(value) {
    const n = Number(value || 0);
    return `${(n * 100).toFixed(2)}%`;
  }

  function appendTerminal(text) {
    terminal.textContent += String(text || "") + "\n";
    terminal.scrollTop = terminal.scrollHeight;
  }

  function appendMessage(kind, text) {
    const clean = String(text || "").trim();
    if (!clean) return;

    const wrapper = document.createElement("div");
    wrapper.className = "mb-3";

    const label = document.createElement("div");
    label.className = "small text-uppercase text-muted mb-1";
    label.textContent = kind === "user" ? "You" : kind === "assistant" ? "Advisor" : "System";

    const body = document.createElement("div");
    body.style.whiteSpace = "pre-wrap";
    body.style.padding = "10px 12px";
    body.style.borderRadius = "8px";
    body.style.border = "1px solid rgba(255,255,255,0.10)";
    body.style.background = kind === "user" ? "#111827" : "#0f172a";
    body.textContent = clean;

    wrapper.appendChild(label);
    wrapper.appendChild(body);
    chatMessages.appendChild(wrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function pretty(payload) {
    if (payload == null || payload === "") return "";
    if (typeof payload === "string") return payload;
    try {
      return JSON.stringify(payload, null, 2);
    } catch (err) {
      return String(payload);
    }
  }

  function isEmptyObject(payload) {
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) return false;
    return Object.keys(payload).length === 0;
  }

  function setPretty(el, payload, emptyText) {
    const text = pretty(payload);
    el.textContent = text || (emptyText || "");
  }

  function renderAllocation(message) {
    const targetWeights = (message && message.target_weights) || {};
    const deltas = (message && message.rebalance_deltas) || {};
    const rows = Object.entries(targetWeights).sort((a, b) => Number(b[1]) - Number(a[1]));

    allocationBody.innerHTML = "";
    if (!rows.length) {
      allocationBody.innerHTML = '<tr><td colspan="3" class="text-muted p-3">No weights available.</td></tr>';
      return;
    }

    rows.forEach(([ticker, weight]) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><button type="button" class="btn btn-link p-0 explain-weight" data-ticker="${ticker}">${ticker}</button></td>
        <td class="text-right">${fmtPct(weight)}</td>
        <td class="text-right">${fmtPct(deltas[ticker] || 0)}</td>
      `;
      allocationBody.appendChild(tr);
    });
  }

  function renderBenchmark(message) {
    if (!message) {
      benchmarkCard.textContent = "No benchmark comparison loaded.";
      return;
    }
    benchmarkCard.textContent = [
      `Benchmark: ${message.benchmark_name || "equal_weight"}`,
      `Portfolio return: ${fmtPct(message.portfolio_return)}`,
      `Benchmark return: ${fmtPct(message.benchmark_return)}`,
      `Active return: ${fmtPct(message.active_return)}`,
      `Tracking error: ${fmtPct(message.tracking_error)}`,
      `Information ratio: ${Number(message.information_ratio || 0).toFixed(2)}`,
    ].join("\n");
  }

  function renderRisk(message) {
    if (!message) {
      riskCard.textContent = "No risk snapshot loaded.";
      return;
    }
    riskCard.textContent = [
      `Realized volatility: ${fmtPct(message.realized_volatility)}`,
      `Downside volatility: ${fmtPct(message.downside_volatility)}`,
      `Max drawdown: ${fmtPct(message.max_drawdown)}`,
      `Concentration HHI: ${Number(message.concentration_hhi || 0).toFixed(3)}`,
      `Turnover: ${fmtPct(message.turnover)}`,
      `Cash weight: ${fmtPct(message.cash_weight)}`,
    ].join("\n");
  }

  function renderScenario(message) {
    if (!message) {
      scenarioCard.textContent = "No scenario analysis available for this run.";
      return;
    }

    const lines = [];
    if (message.request && message.request.scenario_name) {
      lines.push(`Scenario: ${message.request.scenario_name}`);
    }
    if (message.projected_return !== undefined && message.projected_return !== null) {
      lines.push(`Projected return: ${fmtPct(message.projected_return)}`);
    }
    if (message.projected_volatility !== undefined && message.projected_volatility !== null) {
      lines.push(`Projected volatility: ${fmtPct(message.projected_volatility)}`);
    }
    if (message.projected_drawdown !== undefined && message.projected_drawdown !== null) {
      lines.push(`Projected drawdown: ${fmtPct(message.projected_drawdown)}`);
    }
    if (message.narrative) {
      lines.push(String(message.narrative));
    }

    const impact = message.derived_policy_impact || {};
    if (impact && typeof impact === "object" && Object.keys(impact).length) {
      lines.push(`Estimated turnover: ${fmtPct(impact.estimated_turnover)}`);
      lines.push(`Additional cash needed: ${fmtPct(impact.additional_cash_needed)}`);
      const impacted = Array.isArray(impact.positions_impacted) ? impact.positions_impacted : [];
      if (impacted.length) {
        lines.push(`Most affected positions: ${impacted.map((item) => item.ticker).join(", ")}`);
      } else {
        const topCurrent = Array.isArray(impact.top_current_positions) ? impact.top_current_positions : [];
        if (topCurrent.length) {
          lines.push(`Largest current positions: ${topCurrent.map((item) => `${item.ticker} (${fmtPct(item.weight)})`).join(", ")}`);
        }
      }
    }

    scenarioCard.textContent = lines.filter(Boolean).join("\n") || "No scenario analysis available for this run.";
  }

  function renderEnginePolicyPanel(engineInfo, policyCheck) {
    const blocks = [];
    if (engineInfo && Object.keys(engineInfo).length) {
      blocks.push("Engine info");
      blocks.push(pretty(engineInfo));
    }
    if (policyCheck && Object.keys(policyCheck).length) {
      blocks.push("Policy check");
      blocks.push(pretty(policyCheck));
    }
    enginePolicyPanel.textContent = blocks.join("\n\n") || "Engine and policy details will appear here.";
  }

  function renderExplanationLab(message) {
    if (!message || typeof message !== "object") {
      explanationLabCard.textContent = "No explanation audit stored for this run.";
      return;
    }

    const hypotheses = Array.isArray(message.hypotheses) ? message.hypotheses : [];
    const contradictions = Array.isArray(message.contradictions) ? message.contradictions : [];
    const openQuestions = Array.isArray(message.open_questions) ? message.open_questions : [];
    const confidence = Number(message.confidence_score || 0);
    const agreement = Number(message.cross_method_agreement || 0);
    const finalInterpretation = String(message.final_interpretation || "").trim();

    const lines = [];

    if (hypotheses.length) {
      lines.push("Hypotheses:");
      hypotheses.forEach((h) => lines.push(`- ${h}`));
      lines.push("");
    }

    if (contradictions.length) {
      lines.push("Contradictions:");
      contradictions.forEach((c) => lines.push(`- ${c}`));
      lines.push("");
    }

    if (openQuestions.length) {
      lines.push("Open questions:");
      openQuestions.forEach((q) => lines.push(`- ${q}`));
      lines.push("");
    }

    if (agreement || confidence) {
      lines.push(`Cross-method agreement: ${agreement.toFixed(2)}`);
      lines.push(`Confidence score: ${confidence.toFixed(2)}`);
      lines.push("");
    }

    if (finalInterpretation) {
      lines.push("Interpretation:");
      lines.push(finalInterpretation);
    }

    if (!lines.length) {
      lines.push("Explanation audit was generated, but this run produced only a low-information audit with little stored disagreement or confidence detail.");
    }

    explanationLabCard.textContent = lines.join("\n").trim();
  }

  function sendEvent(payload) {
    socket.send(JSON.stringify(payload));
  }

  function loadRun() {
    const runId = (runSelect.value || "").trim();
    if (!runId) {
      appendMessage("system", "Choose a run first.");
      return;
    }
    sendEvent({ type: "load_run", run_id: runId });
  }

  socket.onopen = function () {
    appendTerminal("[WS] Advisor console connected.");
    if (latestRunId) {
      const option = Array.from(runSelect.options).find((opt) => opt.value === latestRunId);
      if (option) {
        runSelect.value = latestRunId;
        loadRun();
      }
    }
  };

  socket.onclose = function () {
    appendTerminal("[WS] Advisor console closed.");
  };

  socket.onerror = function () {
    appendTerminal("[WS] Advisor console error.");
  };

  socket.onmessage = function (event) {
    const data = JSON.parse(event.data);

    if (data.type === "terminal") {
      appendTerminal(data.message);
      return;
    }
    if (data.type === "advisor_answer") {
      appendMessage("assistant", data.message);
      return;
    }
    if (data.type === "allocation") {
      renderAllocation(data.message);
      return;
    }
    if (data.type === "benchmark") {
      renderBenchmark(data.message);
      return;
    }
    if (data.type === "risk") {
      renderRisk(data.message);
      return;
    }
    if (data.type === "scenario") {
      renderScenario(data.message);
      return;
    }
    if (data.type === "explain") {
      explainSummary.textContent = String(data.message || "");
      return;
    }
    if (data.type === "advisory_summary") {
      latestSummary.textContent = String(data.message || "");
      return;
    }
    if (data.type === "engine_status") {
      latestEngineInfo = data.message || {};
      renderEnginePolicyPanel(latestEngineInfo, latestPolicyCheck);
      return;
    }
    if (data.type === "policy_check") {
      latestPolicyCheck = data.message || {};
      renderEnginePolicyPanel(latestEngineInfo, latestPolicyCheck);
      return;
    }
    if (data.type === "rule_summary") {
      if (isEmptyObject(data.message)) {
        ruleSummaryCard.textContent = "No rule or policy summary available for this run.";
      } else {
        setPretty(ruleSummaryCard, data.message, "No rule or policy summary available for this run.");
      }
      return;
    }
    if (data.type === "explanation_bundle") {
      if (isEmptyObject(data.message)) {
        explanationBundleCard.textContent = "No structured explanation bundle stored for this run.";
      } else {
        setPretty(explanationBundleCard, data.message, "No structured explanation bundle stored for this run.");
      }
      return;
    }
    if (data.type === "explanation_lab") {
      renderExplanationLab(data.message || {});
      return;
    }
  };

  loadRunBtn.addEventListener("click", loadRun);
  runSelect.addEventListener("change", function () {
    if (runSelect.value) loadRun();
  });

  chatForm.addEventListener("submit", function (event) {
    event.preventDefault();
    const question = (chatInput.value || "").trim();
    if (!meaningfulQuestion(question)) {
      appendMessage("system", "Please ask a fuller question, for example: why this allocation, how risky is it, or what if the client is more conservative?");
      chatInput.focus();
      return;
    }
    appendMessage("user", question);
    sendEvent({ type: "ask_advisor", message: question });
    chatInput.value = "";
    chatInput.focus();
  });

  document.querySelectorAll(".quick-chat").forEach((btn) => {
    btn.addEventListener("click", function () {
      const type = btn.dataset.event || "ask_advisor";
      const message = btn.dataset.message || "";
      if (!message) return;
      appendMessage("user", message);
      sendEvent({ type, message });
    });
  });

  allocationBody.addEventListener("click", function (event) {
    const btn = event.target.closest(".explain-weight");
    if (!btn) return;
    const ticker = btn.dataset.ticker || "";
    const message = `Why did the weight change for ${ticker}?`;
    appendMessage("user", message);
    sendEvent({ type: "explain_weight_change", ticker, message });
  });
})();
