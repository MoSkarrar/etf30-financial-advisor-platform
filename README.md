# ETF30 Financial Advisor Bot

An explainable, risk-aware advisory platform for ETF30 portfolio analysis and model-driven investment recommendations.

---

## Repository Description

This repository contains the current version of the **ETF30 Financial Advisor Bot**, a Django-based advisory platform that trains, evaluates, and presents reinforcement-learning portfolio decisions through a structured two-stage workflow:

- **Page 1:** advisory intake, execution, run status, and results generation
- **Page 2:** narration/session view for human-readable advisory interpretation

The project is focused on **ETF30-only advisory workflows**, not a broad multi-market trading interface. It combines portfolio optimisation logic, risk controls, explainability outputs, artifact persistence, and narration support inside one modular system.

---

## What the Current Project Does

The current project is designed as an **advisory platform**, not just a raw trading runner.

### Core capabilities
- Accept a user advisory mandate from the web interface
- Run backend validation and orchestration for the selected advisory scenario
- Train and compare candidate RL models such as **A2C**, **PPO**, and **DDPG**
- Select a final model using comparative validation and policy-aware filtering
- Produce portfolio allocation outputs, benchmark comparisons, and risk summaries
- Generate explainability artifacts for model behaviour and decision support
- Persist run outputs into a structured `results/` artifact directory
- Load saved run context into a second narration page for user-facing explanation

---

## Current Project Focus

This repository is now centred on:

- **ETF30 advisory runs** rather than broad market switching
- **Explainable investment recommendations** rather than only buy/sell execution
- **Risk-aware evaluation** using metrics such as drawdown, volatility, turnover, concentration, and downside risk
- **Artifact persistence** so each run can be inspected, narrated, and reproduced
- **Clean separation between explanation and narration**
- **Django + Channels workflow** for browser-driven execution and session handling

---

## Main Features

### 1. Web-based advisory workflow
The Django interface provides a browser-based advisory experience where the user can submit a mandate, monitor execution, and inspect results.

### 2. Two-page output design
The platform separates:
- **execution and result generation**
- **narration and session-based interpretation**

This keeps the advisory flow clearer for end users.

### 3. Candidate-model comparison
The system compares multiple DRL candidates during validation, including:
- **A2C**
- **PPO**
- **DDPG**

### 4. Risk-aware evaluation
Runs are evaluated using portfolio and risk metrics rather than return alone.

### 5. Explainability stack
The platform includes explainability-oriented outputs to support model transparency and interpretability.

### 6. Structured artifact storage
Generated outputs are saved into organised result folders so each run can be revisited and narrated later.

### 7. Automated test coverage
The project includes a pytest suite covering key logic such as pipeline stages, services, views/consumers, artifact handling, risk logic, and narration-related behaviour.

### 8. Ollama-powered page-2 narration
The second page can answer grounded follow-up questions from persisted run artifacts using a local Ollama API connection rather than inventing responses from scratch.

---

## Technology Stack

- **Python 3.10**
- **Django** for the web application
- **Django Channels / WebSocket workflow** for interactive execution flows
- **Stable-Baselines3** for reinforcement-learning algorithms
- **Pytest** for automated tests
- **Matplotlib / result rendering utilities** for reporting assets
- **JSON / CSV / Parquet artifact outputs** for persisted run data
- **Ollama** for page-2 narrated advisory responses

---

## High-Level Workflow

1. **User submits advisory mandate** from the web interface
2. **Backend validates** configuration and launches the advisory run
3. **Pipeline stages execute** for data preparation, training, selection, trading/evaluation, and explanation
4. **Artifacts are saved** under the structured `results/` directory
5. **Narration/session page loads** the saved advisory run for human-readable interpretation
6. **Ollama answers page-2 questions** from the saved run context

---

## Project Structure

```text
.
├── README.md
├── manage.py
├── trader/
│   └── drl_stock_trader/
│       ├── main.py
│       ├── preprocess.py
│       ├── algorithms.py
│       ├── artifact_store.py
│       ├── trading_service.py
│       ├── narration_service.py
│       ├── narration_chat_service.py
│       ├── narration_context.py
│       ├── narration_xai_adapter.py
│       ├── session_models.py
│       ├── config/
│       │   ├── app_config.py
│       │   └── paths.py
│       ├── pipeline/
│       │   ├── data_stage.py
│       │   ├── train_stage.py
│       │   ├── selection_stage.py
│       │   ├── trade_stage.py
│       │   └── explain_stage.py
│       ├── engines/
│       │   ├── engine_registry.py
│       │   ├── finrl_engine.py
│       │   └── legacy_rl_engine.py
│       ├── risk/
│       │   ├── risk_metrics.py
│       │   ├── risk_overlay.py
│       │   └── policy_checks.py
│       └── RL_envs/
│           └── wrappers/
│               └── ollama_narrator.py
├── templates/
│   ├── home.html
│   ├── narration_session.html
│   └── register.html
└── tests/
    ├── test_artifact_store.py
    ├── test_config_paths.py
    ├── test_consumers_and_views.py
    ├── test_data_stage.py
    ├── test_engine_layer.py
    ├── test_explainability_stack.py
    ├── test_frontend_assets.py
    ├── test_main_helpers.py
    ├── test_narration_layer.py
    ├── test_risk_layer.py
    ├── test_session_models.py
    ├── test_trading_service.py
    └── test_train_and_selection.py
```

> Adjust paths if your local checkout uses a slightly different root layout.

---

## Results Directory Layout

The current project saves advisory outputs into structured result folders such as:

```text
results/
├── advisory/
├── benchmarks/
├── engines/
├── explainability/
├── narration/
├── portfolio_allocation/
├── risk/
├── run_manifests/
├── scenarios/
└── snapshots/
```

Typical stored outputs include:
- run manifests
- portfolio allocation JSON files
- benchmark summaries
- risk reports
- explainability bundles
- narration/session caches
- snapshots used for later review

---

## Quickstart (Windows + Anaconda)

### 1. Open Anaconda Prompt
Use **Anaconda Prompt**.

### 2. Go to the project folder
```bat
cd C:\Users\RAUL7\Downloads\fab224\fab224
```

### 3. Create and activate the environment
```bat
conda create -n fab224 python=3.10 -y
conda activate fab224
```

### 4. Install dependencies
```bat
pip install -r requirements.txt
```

### 5. Configure Ollama
Follow the **Ollama setup** section below before using page-2 narration.

### 6. Run the server
```bat
python manage.py runserver
```

### 7. Open the app
Visit:

```text
http://127.0.0.1:8000/
```

---

## Recommended Default Setup

Suggested default dates for a stable first run:
- **Train start:** 2014-01-01
- **Trade start:** 2015-01-01
- **Trade end:** 2025-12-24

If cached ETF30 data appears outdated, rebuild the cached ETF30 dataset files and rerun the app.

---

## Ollama Setup for New Users

Page-2 narration uses **Ollama** as the local model gateway. In this project, the application sends prompts to Ollama's local HTTP API and expects Ollama to be reachable before the narration page can answer questions.

### What the project expects by default
The current runtime defaults are:

- **Base URL:** `http://127.0.0.1:11434`
- **Model:** `gpt-oss:120b-cloud`
- **Temperature:** `0.0`
- **Top-p:** `0.95`
- **Connect timeout:** `3` seconds
- **Read timeout:** `180` seconds
- **Generation budget:** `1400` tokens for both normal and chat responses

These defaults come from `APP_CONFIG.ollama` in `app_config.py` and are used directly by `OllamaNarrationConfig` in `ollama_narrator.py`.

### 1. Install Ollama on Windows
The easiest official Windows installation method is the Ollama installer, and Ollama also documents a PowerShell installation command.

### 2. Start Ollama
After installing, start Ollama so the local API is available on port `11434`.

### 3. Sign in if you want to use the current default model
This project is currently configured to use:

```text
gpt-oss:120b-cloud
```

Because that is a cloud model name, sign in first:

```bat
ollama signin
```

Then load the configured model at least once:

```bat
ollama run gpt-oss:120b-cloud
```

### 5. Verify Ollama before opening page-2
You can test the local API with:

```bat
curl http://127.0.0.1:11434/api/generate -d "{\"model\":\"gpt-oss:120b-cloud\",\"prompt\":\"hello\"}"
```

If Ollama is working, you should receive a JSON response.

---
