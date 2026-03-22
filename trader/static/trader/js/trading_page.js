(function () {
  const root = document.getElementById("trading-page");
  if (!root) return;

  const wsPath = root.dataset.wsPath || "/ws/execute-model/";
  const narrationBase = root.dataset.narrationBase || "/stocks/narration/session/";
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${window.location.host}${wsPath}`);

  const form = document.getElementById("trading-form");
  const btnClear = document.getElementById("btnClear");
  const btnOpenConsole = document.getElementById("btnOpenConsole");
  const linkOpenConsoleNewTab = document.getElementById("linkOpenConsoleNewTab");
  const runOpenState = document.getElementById("run_open_state");

  const terminalResult = document.getElementById("the_result");
  const explainBox = document.getElementById("the_explain");
  const narrationBox = document.getElementById("the_narration");
  const allocationPreview = document.getElementById("allocation_preview");
  const benchmarkPreview = document.getElementById("benchmark_preview");
  const riskPreview = document.getElementById("risk_preview");
  const engineCard = document.getElementById("engine_status_card");
  const explainStatusCard = document.getElementById("explain_status_card");
  const ruleSummaryCard = document.getElementById("rule_summary_card");

  let latestSessionId = "";

  function appendLine(el, text) {
    el.textContent += String(text || "") + "\n";
    el.scrollTop = el.scrollHeight;
  }

  function setText(el, text) {
    el.textContent = String(text || "");
  }

  function fmtPct(value) {
    const n = Number(value || 0);
    return `${(n * 100).toFixed(2)}%`;
  }

  function cleanLabel(value) {
    return String(value || "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (m) => m.toUpperCase())
      .trim();
  }

  function summarizePolicyCheck(message) {
    if (!message) return "No policy check received yet.";
    const breaches = Array.isArray(message.breaches) ? message.breaches : [];
    if (!breaches.length) {
      return message.human_summary || "Policy constraints satisfied.";
    }
    return [
      `${breaches.length} issue${breaches.length > 1 ? "s" : ""} found`,
      breaches.slice(0, 3).map((x) => `• ${x}`).join("\n"),
    ].filter(Boolean).join("\n");
  }

  function summarizeRuleSummary(message) {
    if (!message) return "No policy or rule summary yet.";
    if (typeof message === "string") return message;
    const parts = [];
    if (message.summary_text) parts.push(message.summary_text);
    const rules = Array.isArray(message.rules_triggered) ? message.rules_triggered : [];
    if (rules.length) parts.push(`Rules: ${rules.slice(0, 4).map(cleanLabel).join(", ")}`);
    return parts.join("\n") || "No policy or rule summary yet.";
  }

  function summarizeEngine(message) {
    if (!message) return "Waiting for engine selection…";
    if (typeof message === "string") return message;
    const parts = [];
    const engine = message.engine_name || message.engine || "";
    const algo = message.algorithm_name || message.selected_model || "";
    const backend = message.backend_type || "";
    if (engine) parts.push(`Engine: ${cleanLabel(engine)}`);
    if (algo) parts.push(`Algorithm: ${algo}`);
    if (backend) parts.push(`Backend: ${cleanLabel(backend)}`);
    return parts.join("\n") || "Waiting for engine selection…";
  }

  function summarizeExplain(message) {
    if (!message) return "Waiting for explanation phase…";
    if (typeof message === "string") return message;
    const parts = [];
    if (message.advisor_summary) parts.push("Advisor summary ready");
    const cp = Array.isArray(message.consensus_points) ? message.consensus_points.length : 0;
    const dp = Array.isArray(message.disagreement_points) ? message.disagreement_points.length : 0;
    if (cp || dp) parts.push(`Consensus: ${cp} • Disagreements: ${dp}`);
    if (message.final_interpretation) parts.push("Interpretation ready");
    return parts.join("\n") || "Explanation artifacts received.";
  }

  function resetUi() {
    terminalResult.textContent = "";
    explainBox.textContent = "Waiting for explanation...";
    narrationBox.textContent = "Waiting for advisor summary...";
    allocationPreview.textContent = "Allocation preview will appear here.";
    benchmarkPreview.textContent = "Benchmark comparison will appear here.";
    riskPreview.textContent = "Risk snapshot will appear here.";
    engineCard.textContent = "Waiting for engine selection…";
    explainStatusCard.textContent = "Waiting for explanation phase…";
    ruleSummaryCard.textContent = "No policy or rule summary yet.";
    latestSessionId = "";
    btnOpenConsole.disabled = true;
    linkOpenConsoleNewTab.classList.add("disabled");
    linkOpenConsoleNewTab.href = "#";
    runOpenState.textContent = "The advisor console link will appear after a successful run.";
  }

  function renderAllocation(message) {
    if (!message || !message.target_weights) {
      allocationPreview.textContent = "No allocation payload received yet.";
      return;
    }
    const items = Object.entries(message.target_weights)
      .sort((a, b) => Number(b[1]) - Number(a[1]))
      .slice(0, 8)
      .map(([ticker, weight]) => `${ticker}: ${fmtPct(weight)}`);
    allocationPreview.textContent = `Top weights\n${items.join("\n")}`;
  }

  function renderBenchmark(message) {
    if (!message) {
      benchmarkPreview.textContent = "No benchmark comparison received yet.";
      return;
    }
    benchmarkPreview.textContent = [
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
      riskPreview.textContent = "No risk snapshot received yet.";
      return;
    }
    riskPreview.textContent = [
      `Realized volatility: ${fmtPct(message.realized_volatility)}`,
      `Downside volatility: ${fmtPct(message.downside_volatility)}`,
      `Max drawdown: ${fmtPct(message.max_drawdown)}`,
      `Concentration HHI: ${Number(message.concentration_hhi || 0).toFixed(3)}`,
      `Turnover: ${fmtPct(message.turnover)}`,
      `Cash weight: ${fmtPct(message.cash_weight)}`,
    ].join("\n");
  }

  function getNarrationUrl(sessionId) {
    const cleanBase = narrationBase.endsWith("/") ? narrationBase : `${narrationBase}/`;
    return `${cleanBase}${sessionId}/`;
  }

  function enableConsoleOpen(sessionId) {
    latestSessionId = String(sessionId || "").trim();
    if (!latestSessionId) return;
    const url = getNarrationUrl(latestSessionId);
    btnOpenConsole.disabled = false;
    linkOpenConsoleNewTab.classList.remove("disabled");
    linkOpenConsoleNewTab.href = url;
    runOpenState.textContent = "Run finished. Open the advisor console when you are ready.";
  }

  function validatePercent(id, min, max) {
    const value = Number(document.getElementById(id).value);
    if (!Number.isFinite(value) || value < min || value > max) {
      throw new Error(`${id} must be between ${min} and ${max}`);
    }
    return value;
  }

  function buildPayload() {
    const initialAmount = Number(form.initial_amount.value);
    if (!Number.isFinite(initialAmount) || initialAmount <= 0) {
      throw new Error("Initial capital must be greater than zero.");
    }

    const maxCap = validatePercent("max_single_position_cap", 0.01, 1);
    const minCash = validatePercent("min_cash_weight", 0, 1);
    const turnoverBudget = validatePercent("turnover_budget", 0, 1);
    const riskTolerance = validatePercent("risk_tolerance", 0, 1);
    const targetVolatility = validatePercent("target_volatility", 0, 1);
    const maxDrawdown = validatePercent("max_drawdown_preference", 0, 1);
    const turnoverAversion = validatePercent("turnover_aversion", 0, 1);
    const cadence = Number(document.getElementById("rebalance_cadence_days").value);

    if (!Number.isFinite(cadence) || cadence < 5 || cadence > 126) {
      throw new Error("Rebalance cadence must be between 5 and 126 days.");
    }

    return {
      market: "etf30",
      initial_amount: initialAmount,
      robustness: form.robustness_option.value,
      date_train: form.date_train.value,
      date_trade_1: form.date_trade_1.value,
      date_trade_2: form.date_trade_2.value,
      benchmark_choice: document.getElementById("benchmark_choice").value,
      engine: document.getElementById("engine").value,
      risk_mode: document.getElementById("risk_mode").value,
      explanation_depth: document.getElementById("explanation_depth").value,
      scenario_mode: document.getElementById("scenario_mode").checked,
      rebalance_cadence_days: cadence,
      max_position_weight: maxCap,
      min_cash_weight: minCash,
      max_turnover: turnoverBudget,
      investor_profile: {
        profile_name: document.getElementById("profile_name").value,
        target_style: document.getElementById("target_style").value,
        advisor_mode: document.getElementById("advisor_mode").value,
        risk_tolerance: riskTolerance,
        target_volatility: targetVolatility,
        max_drawdown_preference: maxDrawdown,
        turnover_aversion: turnoverAversion,
        min_cash_preference: minCash,
        benchmark_preference: document.getElementById("benchmark_choice").value,
      },
      policy_settings: {
        max_single_position_cap: maxCap,
        min_cash_weight: minCash,
        turnover_budget: turnoverBudget,
        rebalance_cadence_days: cadence,
        allow_cash_sleeve: document.getElementById("allow_cash_sleeve").checked,
        long_only: document.getElementById("long_only").checked,
      },
    };
  }

  socket.onopen = function () {
    appendLine(terminalResult, "[WS] ETF30 advisor socket connected.");
  };

  socket.onclose = function () {
    appendLine(terminalResult, "[WS] Advisor socket closed.");
  };

  socket.onerror = function () {
    appendLine(terminalResult, "[WS] Advisor socket error.");
  };

  socket.onmessage = function (event) {
    const data = JSON.parse(event.data);

    if (data.type === "terminal") {
      appendLine(terminalResult, data.message);
      return;
    }
    if (data.type === "explain") {
      setText(explainBox, data.message);
      return;
    }
    if (data.type === "advisor_summary") {
      setText(narrationBox, data.message);
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
    if (data.type === "engine_status") {
      setText(engineCard, summarizeEngine(data.message));
      return;
    }
    if (data.type === "explain_status") {
      setText(explainStatusCard, summarizeExplain(data.message));
      return;
    }
    if (data.type === "rule_summary") {
      setText(ruleSummaryCard, summarizeRuleSummary(data.message));
      return;
    }
    if (data.type === "policy_check") {
      setText(ruleSummaryCard, summarizePolicyCheck(data.message));
      return;
    }
    if (data.type === "explanation_bundle") {
      setText(explainStatusCard, summarizeExplain(data.message));
      return;
    }
    if (data.type === "done_session") {
      appendLine(terminalResult, "Session complete.");
      enableConsoleOpen(data.session_id);
    }
  };

  btnClear.addEventListener("click", resetUi);

  btnOpenConsole.addEventListener("click", function () {
    if (!latestSessionId) return;
    window.location.href = getNarrationUrl(latestSessionId);
  });

  form.addEventListener("submit", function (event) {
    event.preventDefault();

    if (socket.readyState !== WebSocket.OPEN) {
      appendLine(terminalResult, "Execution error: advisor socket is not open.");
      return;
    }

    try {
      const payload = buildPayload();
      resetUi();
      appendLine(terminalResult, "Submitting advisory mandate...");
      socket.send(JSON.stringify(payload));
    } catch (err) {
      appendLine(terminalResult, `Validation error: ${err.message || err}`);
    }
  });
})();
