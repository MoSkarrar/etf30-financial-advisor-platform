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

---

## Technology Stack

- **Python 3.10**
- **Django** for the web application
- **Django Channels / WebSocket workflow** for interactive execution flows
- **Stable-Baselines3** for reinforcement-learning algorithms
- **Pytest** for automated tests
- **Matplotlib / result rendering utilities** for reporting assets
- **JSON / CSV / Parquet artifact outputs** for persisted run data

---

## High-Level Workflow

1. **User submits advisory mandate** from the web interface  
2. **Backend validates** configuration and launches the advisory run  
3. **Pipeline stages execute** for data preparation, training, selection, trading/evaluation, and explanation  
4. **Artifacts are saved** under the structured `results/` directory  
5. **Narration/session page loads** the saved advisory run for human-readable interpretation  

---

## Project Structure

```text
.
├── README.md
├── QUICKSTART_WINDOWS.md
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

## Quickstart


### 1. Create and activate an environment
```bash
conda create -n fab224 python=3.10 -y
conda activate fab224
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the server
```bash
python manage.py runserver
```

### 4. Open the app
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

## Running Tests

```bash
python -m pytest -q
```

A recent user-provided pytest result reported:
- **42 passed**
- **4 warnings**
- **0 failed**
- **0 skipped**

---


### 1. Open Anaconda Prompt
Use **Anaconda Prompt**, not a random system terminal.

### 2. Go to the project folder



### 3. Create the environment
```bat
conda create -n fab224 python=3.10 -y
conda activate fab224
```

### 4. Install everything
```bat
pip install -r requirements.txt
```

### 5. Start the project
```bat
python manage.py runserver
```

### 6. Open the browser
Go to:
```text
http://127.0.0.1:8000/
```

### 7. Best default date setup
Use:
- Train start: 2014-01-01
- Trade start: 2015-01-01
- Trade end: 2025-12-24

### 8. Run tests
```bat
python -m pytest -q
```



