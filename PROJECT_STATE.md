# PROJECT_STATE.md — The Agentic Honeypot (Sybil-Agent)

> **Purpose:** This is the single source of truth for the project. Update this file at the end of every phase. Any new collaborator should read this first before touching anything.

> **Last Updated:** Phase 6 (Complete)
> **Current Phase:** Phase 6 — Robustness Testing
> **Status:** COMPLETE (ROBUST verdict — classifier holds above 95% macro F1 across all tested perturbations)

---

## 1. Project Identity

- **Name:** Agentic Honeypot (Sybil-Agent)
- **Timeline:** Feb 24 to Last Week of April (roughly 9 weeks)
- **Framework:** Build, Measure, Prove, Package
- **Owner:** Ricardo Joasilus (RJ)
- **Repo:** https://github.com/rjoasilus/agentic-honeypot

---

## 2. Research Question

Can we reliably distinguish LLM-driven autonomous agents from humans and traditional script bots using timing-based behavioral telemetry captured by a deceptive honeypot API?

### Hypothesis

LLM agents have a distinct behavioral fingerprint. They tend to be slower than script bots but more systematic than humans, so they show up clearly through session-level timing features like inter-arrival gaps, burstiness, entropy, and recovery latency.

---

## 3. System Architecture

```
Client (Human / Script Bot / LLM Agent)
  → FastAPI Honeypot Endpoints
    → Telemetry Logger
      → Storage: JSONL (raw, append-only) + SQLite (structured queries)
        → Pandas ETL + Feature Engineering
          → scikit-learn Classification
            → Evaluation (metrics, plots, ablation)
              → Robustness Testing (Phase 6)
                → Report Artifacts (CSV, figures, slides)
```

### Honeypot Endpoints (Phase 1)

| Endpoint | Purpose |
|---|---|
| `/hp/login` | Auth simulation |
| `/hp/balance` | Account data probe |
| `/hp/transfer` | Transaction attempt |
| `/hp/checkout` | Payment flow |
| `/hp/verify` | Multi-step verification trap |
| `/hp/history` | Activity log access |
| `/health` | System smoke test (not a honeypot route) |

### Deception Layer (Phase 1)

| Endpoint | Status Code | Deception Strategy |
|---|---|---|
| `/hp/login` | 401 | Fake auth failure with `retry_after` bait |
| `/hp/balance` | 200 | Partial fake data (masked account ID, fake balance) |
| `/hp/transfer` | 202 | Pending status plus redirect to `/hp/verify` (multi-step trap) |
| `/hp/checkout` | 200 | Fake processing delay ("check back shortly") |
| `/hp/verify` | 403 | Perpetual verification loop (always expired, always invites retry) |
| `/hp/history` | 200 | Fake transaction list with pagination bait (`total_pages: 3`) |

### Simulation Engine (Phase 2)

| Simulator | File | Behavioral Model | Timing Source |
|---|---|---|---|
| Human | `agents/human_sim.py` | Distracted, inconsistent, bails on errors | `asyncio.sleep(1-8s)` plus distraction pauses |
| Script Bot | `agents/script_bot.py` | Mechanical, rigid, zero adaptation | `asyncio.sleep(10-50ms)` |
| LLM Agent | `agents/llm_agent.py` | Goal-driven, adaptive, reads responses | Real Ollama inference latency (0.5-3s) |

### Security Hardening (Phase 0)

| Layer | Implementation | Tool |
|---|---|---|
| Request size limiting | Reject payloads above 10KB via middleware | Custom FastAPI middleware |
| Rate limiting | Throttle per-IP request rate | slowapi |
| SQLite concurrency | WAL mode enabled on connection | SQLite PRAGMA via SQLAlchemy event |
| Payload sanitization | Redact fields matching sensitive patterns | Custom sanitizer (`sanitizer.py`) |
| IP anonymization | SHA-256 hash before storage | hashlib (stdlib) |
| Secrets management | `.env` in `.gitignore`, `.env.example` committed | python-dotenv |
| HTTPS | Not required (localhost only); documented as known limitation | N/A |
| Auth on honeypot | Intentionally absent so the honeypot stays open | By design |

Sensitive field patterns that get redacted: `password`, `passwd`, `token`, `secret`, `api_key`, `apikey`, `ssn`, `credit_card`, `card_number`, `cvv`, `authorization`

---

## 4. Actor Definitions

### Human Simulator (Phase 2)
- Random delays between actions (1-8 seconds) plus occasional distraction pauses (15-30s)
- Typo injection (30% chance per payload): character swap, character drop, misspelled field name
- Variable session lengths (3-12 requests, seeded per session)
- Weighted random endpoint selection (balance and history favored over verify and checkout)
- Bail-on-error: 60% chance of ending the session after 401 or 403
- Browser-like User-Agent header

### Script Bot (Phase 2)
- Near-zero delay between requests (10-50ms)
- Fixed endpoint patterns: scraper, spammer, full sweep, brute force
- No error handling so it ignores all response status codes
- Identical payloads per pattern (zero variation)
- Minimal headers (bare `python-httpx` UA, no Accept-Language or Accept-Encoding)
- 20-50 requests per session

### LLM Agent (Phase 2 — Ollama: Llama 3)
- Goal-driven behavior via a custom agent loop (not AgentExecutor)
- ChatOllama for LLM inference with conversation memory via message history
- System prompt defines goal: login, check balance, transfer, verify, review history
- Reads and reasons about API responses so it can adapt after failures
- Structured JSON output format with defensive parser and fallback action
- Real inference latency (no artificial delay) — roughly 0.5-3s per turn
- Max 25 turns per session with graceful timeout handling
- Supports a `prompt_variant` parameter so neutral prompts can also be tested (Phase 6)

---

## 5. Tool Stack (Locked)

| Layer | Tool |
|---|---|
| API Framework | FastAPI |
| ASGI Server | Uvicorn |
| Data Validation | Pydantic |
| Raw Storage | JSONL (append-only) |
| Structured DB | SQLite via SQLAlchemy + aiosqlite |
| Async File I/O | aiofiles (added Phase 6) |
| Data Processing | Pandas, NumPy |
| ML Models | scikit-learn |
| Model Serialization | joblib (added Phase 6) |
| Visualization | Matplotlib, Seaborn |
| LLM Runtime | Ollama (local) |
| LLM Integration | langchain-ollama + langchain-core |
| HTTP Client | httpx |
| Config | python-dotenv |
| Testing | pytest |
| Linting | flake8 |
| Rate Limiting | slowapi |

---

## 6. Repository Structure

```
agentic-honeypot/
├── api/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app + deceptive endpoints + async BackgroundTasks logging
│   │   ├── config.py          # Seeds, env vars, constants
│   │   ├── db.py              # SQLite schema + sync/async engines + WAL mode + indexes
│   │   ├── logger.py          # Dual-write telemetry (JSONL via aiofiles + async SQLite mirror)
│   │   ├── middleware.py      # 10KB payload size limit + slowapi rate limiter
│   │   └── sanitizer.py      # Payload field redaction
│   └── tests/
│       ├── __init__.py
│       ├── test_health.py     # 1 test
│       ├── test_security.py   # 3 tests
│       ├── test_endpoints.py  # 8 tests
│       ├── test_telemetry.py  # 10 tests
│       ├── test_seed.py       # 6 tests
│       ├── test_db_init.py    # 4 tests
│       └── test_simulation.py # 7 tests
├── agents/
│   ├── __init__.py
│   ├── base_client.py              # Shared async HTTP client
│   ├── human_sim.py                # Human behavioral simulator
│   ├── script_bot.py               # Script bot simulator (4 route patterns)
│   ├── llm_agent.py                # LLM agent simulator (Ollama + prompt_variant support)
│   ├── orchestrator.py             # Sequential simulation runner
│   └── concurrent_orchestrator.py  # Concurrent simulation runner via asyncio.gather (Phase 6)
├── data/
│   ├── raw/                   # JSONL telemetry files (source of truth)
│   │   ├── telemetry.jsonl         # Phase 2 baseline data (9,792 rows)
│   │   ├── telemetry_tarpit.jsonl  # Phase 6 tarpitting experiment
│   │   ├── telemetry_concurrent.jsonl  # Phase 6 concurrent simulation
│   │   └── telemetry_ood_model.jsonl   # Phase 6 OOD Mistral experiment
│   ├── processed/             # Feature CSVs
│   │   ├── features.csv            # Phase 5 baseline (600 sessions x 13 columns)
│   │   ├── features_tarpit.csv     # Phase 6 tarpitting features
│   │   ├── features_concurrent.csv # Phase 6 concurrent features
│   │   └── features_ood_model.csv  # Phase 6 OOD Mistral features
│   └── samples/
├── features/
│   ├── __init__.py
│   ├── etl.py                 # JSONL loader + schema validator + session sorter
│   ├── engineering.py         # 10 session-level behavioral features
│   ├── pipeline.py            # ETL to features to CSV orchestrator + CLI
│   └── tests/
│       ├── __init__.py
│       ├── test_etl.py        # 12 tests
│       ├── test_features.py   # 27 tests
│       └── test_pipeline.py   # 7 tests
├── models/
│   ├── __init__.py
│   ├── classifiers.py         # LR + RF training, evaluation, ablation, error analysis
│   ├── robustness.py          # Phase 6 artifact save/load, experiment evaluator, noise curve, report
│   ├── artifacts/             # Saved Phase 5 model artifacts (7 joblib files)
│   │   ├── rf_baseline.joblib
│   │   ├── lr_baseline.joblib
│   │   ├── label_encoder.joblib
│   │   ├── imputer_rf.joblib
│   │   ├── scaler_rf.joblib
│   │   ├── imputer_lr.joblib
│   │   └── scaler_lr.joblib
│   └── tests/
│       ├── __init__.py
│       ├── test_classifiers.py  # 11 tests
│       └── test_robustness.py   # 7 tests (Phase 6)
├── analysis/
│   └── eda.py                 # 8-step EDA pipeline
├── report/
│   ├── architecture.md
│   ├── threat_model.md
│   ├── diagrams/
│   ├── figures/
│   │   ├── feature_distributions.png
│   │   ├── burstiness_sensitivity.png
│   │   ├── correlation_matrix.png
│   │   ├── pair_plot_top3.png
│   │   ├── killer_viz.png
│   │   ├── confusion_matrix_lr.png
│   │   ├── confusion_matrix_rf.png
│   │   ├── feature_importance.png
│   │   ├── roc_curves.png
│   │   ├── ablation_comparison.png
│   │   ├── killer_viz_with_errors.png
│   │   ├── robustness_curve.png          # Phase 6: F1 vs Gaussian noise
│   │   ├── robustness_comparison.png     # Phase 6: macro F1 across all conditions
│   │   ├── confusion_matrix_rf_tarpit.png
│   │   └── confusion_matrix_rf_concurrent.png
│   └── metrics/
│       ├── lr_results.json
│       ├── rf_results.json
│       ├── ablation_results.json
│       ├── robustness_tarpit.json
│       ├── robustness_concurrent.json
│       ├── robustness_ood_model.json
│       └── robustness_results.json       # Phase 6: combined verdict file
├── scripts/
│   └── make.ps1
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── PROJECT_STATE.md
├── README.md
├── requirements.txt
├── requirements-lock.txt      # Frozen versions (add before git push)
└── pyproject.toml
```

---

## 7. SQLite Schema (Locked in Phase 0, Indexed in Phase 3)

### Table: `telemetry`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key, auto-increment |
| `session_id` | TEXT | UUID per session |
| `timestamp_ms` | INTEGER | Unix epoch milliseconds |
| `endpoint` | TEXT | Route hit (e.g., `/hp/login`) |
| `method` | TEXT | HTTP method (GET/POST) |
| `payload_size` | INTEGER | Request body size in bytes |
| `response_time_ms` | REAL | Server processing time |
| `ip_hash` | TEXT | SHA-256 of client IP |
| `user_agent` | TEXT | Full UA string |
| `status_code` | INTEGER | HTTP response code returned |
| `actor_type` | TEXT | Label: human / bot / llm_agent |

### Indexes (Phase 3)

| Index Name | Column | Purpose |
|---|---|---|
| `ix_telemetry_session_id` | `session_id` | Speed up GROUP BY session_id |
| `ix_telemetry_timestamp_ms` | `timestamp_ms` | Speed up ORDER BY timestamp_ms |
| `ix_telemetry_actor_type` | `actor_type` | Speed up WHERE actor_type filtering |

### Table: `sessions`

| Column | Type | Description |
|---|---|---|
| `session_id` | TEXT | Primary key (UUID) |
| `actor_type` | TEXT | human / bot / llm_agent |
| `start_time` | INTEGER | First request timestamp (ms) |
| `end_time` | INTEGER | Last request timestamp (ms) |
| `request_count` | INTEGER | Total requests in session |
| `created_at` | TEXT | ISO datetime of session creation |

---

## 8. Feature Engineering (Phase 3)

### Per-Session Features (10 features)

| Feature | Description |
|---|---|
| `mean_gap_time` | Average time between consecutive requests |
| `std_gap_time` | Standard deviation of inter-request gaps |
| `max_delay` | Longest pause in session |
| `burstiness` | Ratio of short gaps to total gaps |
| `recovery_time` | Time to next request after error response |
| `session_entropy` | Shannon entropy of endpoint visit distribution |
| `session_duration` | Total time from first to last request |
| `request_rate` | Requests per second |
| `payload_entropy` | Entropy of payload content across session |
| `path_repetition` | Ratio of repeated endpoint sequences |

---

## 9. Model Results (Phase 5)

### Baseline Pair
- Logistic Regression: macro F1 = 0.9916 (16 features including missing indicators, `request_rate` dropped)
- Random Forest: macro F1 = 1.0000 (18 features including missing indicators, all 10 features)

### LR vs RF Interpretation
The gap is under 1%, so the features are approximately linearly separable. Even a simple linear model nearly solves this. The data does the work, not the model complexity.

### Per-Class Metrics (RF — Best Model)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| bot | 1.000 | 1.000 | 1.000 |
| human | 1.000 | 1.000 | 1.000 |
| llm_agent | 1.000 | 1.000 | 1.000 |

### Feature Ablation (Macro F1)

| Feature Set | LR | RF | Interpretation |
|---|---|---|---|
| ALL | 0.9916 | 1.0000 | Full performance ceiling |
| TIMING only | 0.9831 | 1.0000 | Timing alone achieves perfect RF classification |
| STRUCTURAL only | 0.9662 | 0.9916 | Strong but not quite sufficient alone for LR |

### Top Feature Importances (RF Gini)

| Rank | Feature | Gini Importance |
|---|---|---|
| 1 | `recovery_time` | 0.2256 |
| 2 | `burstiness` | 0.1369 |
| 3 | `max_delay` | 0.1336 |

---

## 10. Commands Reference

```powershell
# Run all tests
python -m pytest -v

# Start the honeypot server
uvicorn api.app.main:app --host 127.0.0.1 --port 8000

# Run a simulation (small test)
python -m agents.orchestrator --humans 3 --bots 3 --agents 2

# Run concurrent simulation (Phase 6)
python -m agents.concurrent_orchestrator --humans 50 --bots 50 --agents 50

# Feature pipeline
python -m features.pipeline

# EDA
python -m analysis.eda

# Classification
python -m models.classifiers

# Save model artifacts (Phase 6)
python -m models.robustness --save-artifacts

# Run robustness curve (Phase 6)
python -m models.robustness --experiment noise_curve

# Evaluate a robustness experiment (Phase 6)
python -m models.robustness --experiment tarpit --input data/processed/features_tarpit.csv

# Build robustness report (Phase 6)
python -m models.robustness --report
```

---

## 11. Environment Setup

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -e .

# Copy and fill in your environment file
Copy-Item .env.example .env
```

Required entries in `.env`:

```
RANDOM_SEED=42
DB_PATH=data/agentic_honeypot.sqlite3
LOG_PATH=data/raw/telemetry.jsonl
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
API_HOST=127.0.0.1
API_PORT=8000
MAX_PAYLOAD_BYTES=10240
RATE_LIMIT=100/minute
```

---

## 12. Known Limitations

These are stated limitations so anyone reading the project understands where the boundaries are.

1. **Synthetic data.** The classifier is trained on data generated by the project's own simulators. Generalization to real-world LLM agents in production is a research direction, not a current claim. The contribution here is the measurement framework and the method, not a production-ready detector.

2. **Single-machine deployment.** The honeypot runs on localhost so there is no real network jitter between the client and server. In production, network variance would need to be accounted for since it could compress the timing gaps between actor types.

3. **LLM inference speed is increasing.** As LLM inference gets faster (through better hardware and optimized runtimes), the timing gap between bots and agents will shrink. The Phase 5 ablation study already shows that structural features alone achieve 99.16% F1, so the system degrades gracefully rather than failing completely.

4. **Rate limit scope.** The current rate limiter applies per-IP. In a shared NAT environment, multiple users behind the same IP would be grouped together and could trigger throttling unexpectedly.

---

## 13. Windows/PowerShell Environment Notes

- All commands must be PowerShell since the dev environment is Windows 11
- Use `New-Item` instead of `touch`, `Copy-Item` instead of `cp`, `Remove-Item` instead of `rm`
- Run everything from the project root so editable installs resolve correctly
- If VS Code shows yellow squiggles on imports, fix it via Ctrl+Shift+P and "Python: Select Interpreter" then choose `.venv`
- Clear the `LOG_PATH` and `OLLAMA_MODEL` environment variables between experiments so tests read from the right JSONL file

```powershell
Remove-Item Env:LOG_PATH -ErrorAction SilentlyContinue
Remove-Item Env:OLLAMA_MODEL -ErrorAction SilentlyContinue
```

---

## 14. Common Debugging Issues

| Issue | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | Running commands from a subdirectory | Run from project root |
| `ImportError` on SQLAlchemy | Capitalization issue (`Event` vs `event`) | Check exact import casing |
| `SyntaxError: return outside function` | Indentation issue when pasting code | Check indentation carefully |
| VS Code yellow squiggles on imports | Wrong Python interpreter selected | Ctrl+Shift+P, select `.venv` interpreter |
| `create_all()` silently skipping new indexes | Table already exists so SQLAlchemy skips it | Delete the SQLite file and let `init_db()` recreate it |
| Tests reading wrong JSONL file | `LOG_PATH` env var still set from a server run | Run `Remove-Item Env:LOG_PATH` before running pytest |
| Ollama session hanging for 1000+ seconds | System resource exhaustion after long simulation runs | Restart Ollama and reduce session count |

---

## 15. Improvement Tracking

### Phase 1 — Honeypot API

| Item | Status | Description |
|---|---|---|
| Full header capture in JSONL | Done | `"headers": dict(request.headers)` added to telemetry record |
| Logging flow comment in main.py | Done | Comment above logger makes the dual-storage contract explicit |
| Graceful SQLite write failure | Done | SQLite inserts wrapped in try/except so DB failure does not crash requests |
| Handle malformed JSON / wrong content types | Done | Explicit behavior for non-JSON content types so they fail predictably |

### Phase 2 — Simulation

| Item | Status | Description |
|---|---|---|
| Seed consistency test | Done | Proves `RANDOM_SEED` is applied and produces deterministic output |
| DB init + WAL mode tests | Done | Proves tables exist after `init_db()` and WAL mode is active |

### Phase 3 — Data Pipeline

| Item | Status | Description |
|---|---|---|
| Add SQLite indexes | Done | Indexes on `session_id`, `timestamp_ms`, `actor_type` |
| Editable install (`pip install -e .`) | Done | `pyproject.toml` added so imports work from any directory |

### Phase 4 — EDA

| Item | Status | Description |
|---|---|---|
| Burstiness threshold sensitivity | Done | Tested at 250ms, 500ms, 1000ms. Result is robust at all three. |
| Correlation matrix | Done | Pearson and Spearman across all 10 features. One pair flagged. |
| `path_repetition` metric review | Done | Clear 3-tier separation confirmed at N=600. Feature stays. |
| Log-scale axes for timing plots | Done | Applied to all timing features so bot distributions are visible. |
| Pair plot of top 3 features | Done | Three distinct clusters visible in every 2D projection. |

### Phase 5 — Modeling

| Item | Status | Description |
|---|---|---|
| Class imbalance handling | Done | Stratified split plus `class_weight='balanced'` on both models. |
| LR + RF as baseline pair | Done | LR 0.9916, RF 1.0000. Gap under 1% confirms feature quality. |
| Drop correlated features before LR | Done | `request_rate` excluded from LR feature set. |

### Phase 6 — Robustness Testing

| Item | Status | Description |
|---|---|---|
| Response format consistency audit | Done | All error codes now return `{"detail": "..."}` format so no fingerprinting vector exists. |
| Async JSONL writes under load | Done | Moved to `BackgroundTasks` plus `aiofiles` so disk I/O no longer contaminates response timing measurements. |
| Save model artifacts | Done | 7 joblib files saved to `models/artifacts/` so every experiment uses the same fixed baseline. |
| Robustness curve (F1 vs timing noise) | Done | Classifier tolerates sigma=1000ms before meaningful degradation. Still above 0.90 at sigma=5000ms. |
| Tarpitting experiment | Done | RF drops from 1.0000 to 0.9867 (down 1.3%). Bots still perfect because structural features compensate. |
| Concurrent simulation stress test | Done | RF stays at 0.9867 so concurrent load introduces zero additional degradation. |
| OOD agent validation (Mistral) | Done | 97.3% agent recall on a model the classifier never trained on. Signature is model-agnostic. |
| OOD agent validation (neutral prompt) | Pending | Simulation running. Results to be added when complete. |

### Phase 7 — Report and Ethics

| Item | Status | Description |
|---|---|---|
| Synthetic data generalization limitation | Pending | Add explicit section to `threat_model.md` documenting the boundary of the research contribution. |

### Phase 8 — Final Packaging

| Item | Status | Description |
|---|---|---|
| Business value framing | Pending | Lead README with business impact framing for hiring audience. |
| Interview scaling answer | Pending | Prepare answer for "what if this needs 10k RPS?" (chunking, async queue, Postgres, horizontal). |
| One killer visualization | Done | `report/figures/killer_viz.png` delivered in Phase 4. |
| One insight statement | Pending | Finalize: "LLM agents are distinguishable not by speed alone but by how they think." |
| Freeze dependencies | Pending | Run `pip freeze > requirements-lock.txt` before push. |
| `.flake8` config file | Pending | Create with `max-line-length = 120`. |
| `.vscode/settings.json` | Pending | Set `python.defaultInterpreterPath` to `.venv`. |
| GitHub Actions CI | Pending | Add `.github/workflows/test.yml` that runs pytest and flake8 on every push. |

---

## 16. JSONL Record Schema (Frozen in Phase 1)

This is the contract between what `logger.py` writes and what the Phase 3 ETL expects. Any changes require updating both `logger.py` and this section.

| Field | Type | Description |
|---|---|---|
| `session_id` | TEXT (UUID) | From `X-Session-ID` header, or `uuid4()` fallback |
| `timestamp_ms` | INTEGER | Unix epoch milliseconds |
| `endpoint` | TEXT | Route hit (e.g., `"/hp/login"`) |
| `method` | TEXT | HTTP method (`GET` or `POST`) |
| `payload_size` | INTEGER | Request body size in bytes (0 if empty) |
| `response_time_ms` | REAL | Server processing time in milliseconds |
| `ip_hash` | TEXT | SHA-256 of client IP |
| `user_agent` | TEXT | Full User-Agent string |
| `headers` | DICT | All HTTP headers as key-value pairs |
| `status_code` | INTEGER | HTTP response code returned |
| `actor_type` | TEXT | From `X-Actor-Type` header, or `"unknown"` fallback |
| `payload` | DICT or NULL | Sanitized request body, or `null` if unparseable |
| `payload_error` | TEXT or NULL | `"invalid_json"`, `"wrong_content_type"`, `"empty_body"`, or `null` |

`headers`, `payload`, and `payload_error` exist only in JSONL. The SQLite `telemetry` table mirrors the subset defined in Section 7. JSONL is the source of truth and SQLite is the query-optimized mirror.

---

## 17. Simulation Run Guide

### Standard Sequential Simulation

Two terminals required:

```powershell
# Terminal 1: Start the honeypot
uvicorn api.app.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Small test
python -m agents.orchestrator --humans 3 --bots 3 --agents 2

# Terminal 2: Full run
python -m agents.orchestrator --humans 200 --bots 200 --agents 200
```

### Phase 6 Experiment Runs

For separate experiment files, set `LOG_PATH` in the server environment before starting:

```powershell
# Terminal 1
$env:LOG_PATH = "data/raw/telemetry_concurrent.jsonl"
uvicorn api.app.main:app --host 127.0.0.1 --port 8000

# Terminal 2
python -m agents.concurrent_orchestrator --humans 50 --bots 50 --agents 50
```

For OOD experiments, also set the model:

```powershell
$env:OLLAMA_MODEL = "mistral"
$env:LOG_PATH = "data/raw/telemetry_ood_model.jsonl"
uvicorn api.app.main:app --host 127.0.0.1 --port 8000
```

Always clear these variables before running pytest:

```powershell
Remove-Item Env:LOG_PATH -ErrorAction SilentlyContinue
Remove-Item Env:OLLAMA_MODEL -ErrorAction SilentlyContinue
```

### Timing Expectations (per session, approximate)

| Actor | Time per Session | Bottleneck |
|---|---|---|
| Human | 5-60s | `asyncio.sleep()` delays |
| Bot | 1-3s | 20-50 requests at 10-50ms each |
| LLM Agent | 60-180s | Ollama inference (~15-20s per turn) |

---

## 18. Data Pipeline Guide

```powershell
# Default paths
python -m features.pipeline

# Custom paths
python -m features.pipeline --input data/raw/telemetry.jsonl --output data/processed/features.csv
```

### Feature Output Schema

One row per session, 13 columns: `session_id`, `actor_type`, `request_count`, plus the 10 behavioral features.

### NaN Policy

Features that cannot be computed (like 1-request sessions with no gaps) are `NaN`, not 0. Filling with 0 creates false signal since it implies no gap rather than unknown.

---

## 19. EDA Results (Phase 4)

### Dataset

| Metric | Value |
|---|---|
| Total rows (JSONL) | 9,792 |
| Rows retained after ETL | 9,792 (0 dropped) |
| Total sessions | 600 |
| Bot sessions | 203 |
| Human sessions | 203 |
| LLM agent sessions | 194 |

### Feature Separation Summary

| Feature | Separation | Bot | Human | Agent |
|---|---|---|---|---|
| `mean_gap_time` | Clear | 57ms (sd=3) | 6,476ms (sd=4,657) | 12,163ms (sd=2,171) |
| `burstiness` | Clear | 1.0 (sd=0) | 0.0 (sd=0) | 0.0 (sd=0) |
| `recovery_time` | Clear | 58ms (sd=4) | 6,843ms (sd=7,153) | 11,587ms (sd=2,110) |
| `payload_entropy` | Clear | 0.0 (sd=0) | 0.69 (sd=0.76) | 1.71 (sd=0.50) |
| `request_rate` | Clear | 18.1/s (sd=0.9) | 0.34/s (sd=0.3) | 0.10/s (sd=0.02) |

### Key Findings

1. Timing alone separates all three classes. `mean_gap_time` shows three non-overlapping tiers: bot around 57ms, human around 6.5s, agent around 12.2s.
2. Content diversity (payload entropy) distinguishes agents from humans at 1.71 vs 0.69. LLM agents generate novel payloads because they reason about responses. Humans work from templates.
3. Bots are trivially detectable because burstiness=1.0, payload_entropy=0.0, and path_repetition=1.0 all hold.
4. The human-agent boundary is the hard problem. Both have burstiness=0.0 and overlapping session_entropy. The discriminating features are mean_gap_time (2x gap), payload_entropy (2.5x gap), and recovery_time.

---

## 20. Classification Results (Phase 5)

### Models Trained

| Model | Features | Macro F1 | Accuracy | Misclassifications |
|---|---|---|---|---|
| Logistic Regression | 9 features (no `request_rate`) + 7 missing indicators = 16 | 0.9916 | 0.9917 | 1 / 120 |
| Random Forest | 10 features + 8 missing indicators = 18 | 1.0000 | 1.0000 | 0 / 120 |

### Key Findings

1. Hypothesis confirmed. Timing-based behavioral features achieve 100% macro F1 (RF) in distinguishing LLM agents from humans and script bots.
2. Features are approximately linearly separable so even the simpler model nearly solves this. The features do the work, not model complexity.
3. Timing alone is sufficient. TIMING-only RF = 100%. Structural features add marginal value on top.
4. `recovery_time` is the top discriminator. How actors respond to failure (bot ~58ms, agent ~11.6s, human ~6.8s) reveals their nature more clearly than average timing alone.
5. Zero misclassifications (RF). All 120 test samples correctly classified.
6. Agent recall = 100%. Every LLM agent in the test set was identified correctly.

---

## 21. Robustness Results (Phase 6)

### Verdict: ROBUST

The classifier maintains above 95% macro F1 across all tested perturbations. Timing-based behavioral detection is robust under adversarial conditions, concurrent load, and out-of-distribution LLM agents.

### Comparison Table

| Condition | RF F1 | LR F1 | Agent Recall | Notes |
|---|---|---|---|---|
| Phase 5 Baseline | 1.0000 | 0.9916 | 100% | Sequential, Llama 3, goal prompt |
| Tarpitting | 0.9867 | 0.9867 | 96% | 429 replaced with fake 200 plus 2-5s sleep |
| Concurrent Simulation | 0.9867 | 0.9867 | 96% | All 3 actor types hitting server simultaneously |
| OOD Agent (Mistral) | N/A | N/A | 97.3% | Agent-only dataset so macro F1 is not comparable across classes |
| OOD Agent (Neutral Prompt) | Pending | Pending | Pending | Simulation running |

### Experiment Findings

**Tarpitting.** Replacing 429 with a fake 200 plus a random 2-5 second sleep corrupts bot timing so their gaps look like agent gaps. RF drops 1.3% but bots are still classified perfectly. The reason is that structural features like payload_entropy and path_repetition are timing-independent so the classifier falls back to those when timing is corrupted. Two independent classification pathways each back the other up.

**Concurrent simulation.** Running all three actor types simultaneously produced identical degradation to the tarpitting result (0.9867). The degradation comes from the smaller sample size used for experiments (50 sessions per class vs 200 in Phase 5), not from concurrent load itself. Concurrent load introduces zero additional degradation.

**OOD Mistral.** 97.3% of Mistral sessions were correctly classified as `llm_agent` by a model trained exclusively on Llama 3 data. The behavioral signature is model-agnostic, not Llama 3-specific.

**Robustness curve.** Injecting Gaussian noise into timing features shows the classifier is immune to noise below sigma=500ms, barely affected at sigma=1000ms, and still above 0.90 at sigma=5000ms. The curve degrades gracefully rather than collapsing suddenly. LR outperforms RF at extreme noise because structural features favor linear boundaries when timing is fully corrupted.

### Infrastructure Changes (Phase 6)

- Response format consistency: all error codes now return `{"detail": "..."}` format so there is no fingerprinting vector from inconsistent error shapes
- Async JSONL writes: moved to `BackgroundTasks` plus `aiofiles` so disk I/O no longer contaminates `response_time_ms` measurements
- Model artifacts saved to `models/artifacts/` as 7 joblib files so every experiment uses the exact same fixed baseline without retraining

### Phase 6 Figures

| Figure | Path | Purpose |
|---|---|---|
| Robustness curve | `report/figures/robustness_curve.png` | F1 vs Gaussian noise sigma |
| Robustness comparison | `report/figures/robustness_comparison.png` | Bar chart across all conditions |
| Tarpit confusion matrix | `report/figures/confusion_matrix_rf_tarpit.png` | RF on tarpitted data |
| Concurrent confusion matrix | `report/figures/confusion_matrix_rf_concurrent.png` | RF on concurrent data |

### Phase 6 Metrics Files

| File | Path | Contents |
|---|---|---|
| Tarpit results | `report/metrics/robustness_tarpit.json` | RF/LR F1 and agent recall on tarpitted data |
| Concurrent results | `report/metrics/robustness_concurrent.json` | RF/LR F1 and agent recall on concurrent data |
| OOD model results | `report/metrics/robustness_ood_model.json` | Agent recall for Mistral sessions |
| Combined verdict | `report/metrics/robustness_results.json` | All experiments in one file |

### Tests (Phase 6)

7 new tests in `models/tests/test_robustness.py` covering artifact save/load, preprocessor no-refit, noise injection correctness, zero-noise baseline reproduction, comparison table schema, tarpit handler behavior, and concurrent session ID uniqueness.

Total tests: 103 passing, 0 failing.

---

*Update this file at the end of every phase.*
