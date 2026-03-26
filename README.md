# Agentic Honeypot (Sybil-Agent)

A FastAPI honeypot that captures behavioral telemetry to classify visitors as:
- **Human** — organic, unpredictable browsing
- **Script Bot** — mechanical, repetitive automation
- **LLM Agent** — AI-driven, goal-oriented exploration

## Research Question

Can timing-based behavioral features reliably distinguish LLM agents from humans and bots?

## Quickstart

### Windows (PowerShell)
```powershell
git clone <REPO_URL>
cd agentic-honeypot
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env

# Initialize database
.\scripts\make.ps1 init-db

# Run the honeypot
.\scripts\make.ps1 run

# Run tests
.\scripts\make.ps1 test
```

### Mac/Linux (Bash)
```bash
git clone <REPO_URL>
cd agentic-honeypot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

make init-db
make run
make test
```

## Architecture

See: [`report/architecture.md`](report/architecture.md) for the full system design.

## Threat Model

See: [`report/threat_model.md`](report/threat_model.md) for actor definitions, risks, and security mitigations.

## Project State

See: [`PROJECT_STATE.md`](PROJECT_STATE.md) for current phase, decisions, and deliverables tracker.

## Project Structure
```
agentic-honeypot/
├── api/app/          # FastAPI server, config, db, middleware, sanitizer
├── api/tests/        # pytest test suite
├── agents/           # Human, bot, and LLM agent simulators
├── data/raw/         # JSONL telemetry logs
├── data/processed/   # Cleaned CSVs and feature tables
├── features/         # Session-level feature extraction
├── models/           # ML classifiers 
├── analysis/         # Exploratory data analysis 
├── report/           # Architecture, threat model, diagrams, figures
├── scripts/          # Task runner and utilities
```