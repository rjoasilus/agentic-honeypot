import hashlib
import json
import logging
import aiofiles
import time
import uuid
from fastapi import Request
from api.app.config import LOG_PATH
from api.app.db import async_insert_telemetry
from api.app.sanitizer import sanitize_payload

# Ensure the log directory exists before any writes
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
# Named logger so warnings show up as "honeypot.telemetry: WARNING: ..."
logger = logging.getLogger("honeypot.telemetry")

async def log_request(
    request: Request, # The raw FastAPI request object (for headers, IP, method)
    endpoint: str, # Route string like "/hp/login"
    body: dict | None, # Parsed JSON body, or None if parsing failed
    payload_error: str | None, # Why parsing failed ("invalid_json", 
                               # "wrong_content_type","empty_body"), 
                               #  or None if parsing succeeded
    status_code: int, # The HTTP status code the endpoint is returning
    response_time_ms: float, # How long the endpoint took to process (milliseconds)
    actor_type: str = "unknown", # Traffic label
) -> None:
    """Central telemetry pipeline. Every honeypot endpoint calls this once.
    Writes to JSONL (source of truth), then best-effort mirrors to SQLite."""

    # Session ID
    # Use explicit session ID from header, or generate a fresh UUID.
    # simulators send X-Session-ID; unlabeled traffic gets
    # unique UUIDs that ETL can stitch into sessions via ip_hash
    # + time-window gaps.
    session_id = request.headers.get("x-session-id") or str(uuid.uuid4())

    # Actor Type
    # simulators send X-Actor-Type header with the label.
    # If present, it overrides the default "unknown" parameter.
    actor_type = request.headers.get("x-actor-type", actor_type)

    # IP Anonymization 
    # Hash the raw IP before it touches any storage. PII never leaves this function
    raw_ip = request.client.host if request.client else "unknown"
    ip_hash = hashlib.sha256(raw_ip.encode()).hexdigest()

    # Payload Size
    # request.body() returns cached bytes since _safe_parse_body already read it
    try:
        raw_body = await request.body()
        payload_size = len(raw_body)
    except Exception:
        payload_size = 0

    # Sanitize Payload
    # Redact sensitive fields (passwords, tokens, etc.) before any storage
    sanitized_body = sanitize_payload(body) if body else None

    # Build the JSONL Record
    record = {
        "session_id": session_id,
        "timestamp_ms": int(time.time() * 1000),
        "endpoint": endpoint,
        "method": request.method,
        "payload_size": payload_size,
        "response_time_ms": round(response_time_ms, 2),
        "ip_hash": ip_hash,
        "user_agent": request.headers.get("user-agent", ""),
        "headers": dict(request.headers),  
        "status_code": status_code,
        "actor_type": actor_type,
        "payload": sanitized_body,
        "payload_error": payload_error,
    }

    #JSONL Write
    # Append-only, one JSON object per line. This file is the canonical record.
    # default=str is a safety net: if any value isn't JSON-serializable
    # (e.g., a UUID or datetime object), str() is called instead of crashing.
    async with aiofiles.open(LOG_PATH, "a", encoding="utf-8") as f:
        await f.write(json.dumps(record, default=str) + "\n")

    # SQLite Mirror (Best-Effort) 
    # async write so a locked DB doesn't block the event loop.
    # If this fails, JSONL already has the record so we just log a warning.
    try:
        await async_insert_telemetry(record)
    except Exception as e:
        logger.warning("SQLite write failed (JSONL still has the record): %s", e)