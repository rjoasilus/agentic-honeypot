# System Architecture — Agentic Honeypot (Sybil-Agent)

## End-to-End Pipeline

Client (human / script bot / LLM agent)
→ FastAPI honeypot endpoints (with deception layer)
→ Telemetry logger (per-request recording)
→ Dual storage: JSONL (raw, append-only) + SQLite (structured)
→ Pandas ETL: parse, clean, aggregate to session level
→ Feature engineering: timing-based behavioral features
→ scikit-learn: classification (3-class)
→ Evaluation: metrics, confusion matrix, ROC, ablation
→ Artifacts: CSV dataset, figures, PDF report, slides

## Storage Strategy

- **JSONL** (`data/raw/telemetry.jsonl`): Source of truth. Every request appended as-is. Never modified.
- **SQLite** (`data/agentic_honeypot.sqlite3`): Structured mirror for fast queries, joins, aggregation.

## Design Principle

Append raw first, structure second. If SQLite corrupts or schema changes, JSONL can always rebuild it.

## Security Hardening

- **Payload sanitization:** Sensitive fields (password, token, ssn, credit_card, etc.) redacted to `[REDACTED]` before logging.
- **Request size limiting:** Payloads exceeding 10KB rejected with HTTP 413 via middleware.
- **Rate limiting:** Per-IP throttling at 100 req/min via slowapi. Prevents simulator runaway.
- **SQLite WAL mode:** Write-Ahead Logging prevents database lock errors during concurrent writes.
- **IP anonymization:** Client IPs SHA-256 hashed before storage. Raw IPs never persisted.
- **Secrets isolation:** `.env` excluded from Git. `.env.example` committed as template.

## Technology Stack

| Layer           | Tool                              |
|-----------------|-----------------------------------|
| API Framework   | FastAPI + Uvicorn                 |
| Data Validation | Pydantic                          |
| Raw Storage     | JSONL (append-only)               |
| Structured DB   | SQLite via SQLAlchemy + aiosqlite |
| Data Processing | Pandas, NumPy                     |
| ML Models       | scikit-learn                      |
| Visualization   | Matplotlib                        |
| LLM Runtime     | Ollama (local)                    |
| HTTP Client     | httpx                             |
| Testing         | pytest                            |
| Rate Limiting   | slowapi                           |