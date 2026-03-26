# Threat Model — Agentic Honeypot (Phase 0)

## Actors

### 1. Human
- Browses normally with varied timing
- Natural navigation with inconsistent patterns
- Occasional input errors and hesitation
- Session lengths vary widely

### 2. Script Bot
- Near-zero delay between requests
- Fixed, repetitive endpoint patterns
- Low entropy headers and payloads
- Predictable timing signature

### 3. LLM Agent (Ollama: Llama 3 / Mistral)
- Human-like language but tool-driven behavior
- Structured probing and systematic exploration
- "Polite" retries with error recovery
- Goal-oriented session flow
- Unusual endpoint discovery patterns

## Assumptions
- Honeypot endpoints are non-production; no real user data exists.
- All captured data is synthetic telemetry (headers, timing, payload fields).
- Agents are run locally via Ollama (no external API dependency).

## Risks
- **False positives:** Fast humans or accessibility tools may resemble bots.
- **Evasion:** Adaptive agents could slow down and randomize to mimic humans.
- **Privacy:** Must avoid storing secrets; redact obvious credentials in payloads.
- **Synthetic bias:** All data is simulated; real-world generalization is unproven for now.

## Security Mitigations (Implemented in Phase 0)
- **Payload sanitization:** Fields matching sensitive patterns (password, token, ssn, credit_card, etc.) are redacted to `[REDACTED]` before logging.
- **Request size limiting:** Payloads exceeding 10KB are rejected (HTTP 413) to prevent log bloat and local DoS.
- **Rate limiting:** Per-IP throttling via slowapi (100 req/min default) prevents runaway simulators from crashing the system.
- **SQLite WAL mode:** Write-Ahead Logging prevents database lock errors during concurrent agent simulations.
- **IP hashing:** Client IPs are SHA-256 hashed before storage; raw IPs are never persisted.
- **Secrets isolation:** `.env` excluded from version control; `.env.example` committed as template.

## Known Limitations (Accepted)
- **No HTTPS:** Running on localhost only; plaintext is acceptable for local research.
- **No authentication:** Intentional — honeypot must remain open to all actor types by design.
- **No cloud deployment security:** System is local-only; cloud hardening is out of scope.

## What We Will Measure
- **Timing features:** Inter-arrival deltas, jitter, burstiness, session duration.
- **Content features:** Payload entropy, repetition, lexical markers.
- **Protocol features:** Header consistency, UA anomalies, path exploration behavior.