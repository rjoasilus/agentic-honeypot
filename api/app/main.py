import time
import asyncio
import random
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from api.app.db import init_db
from api.app.logger import log_request
from api.app.middleware import limiter, PayloadSizeLimitMiddleware
from slowapi.errors import RateLimitExceeded

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

# App Creation
app = FastAPI(title="Agentic Honeypot (Sybil-Agent)", lifespan=lifespan)

# Security Middleware
# Wired in order: rate limiter first (reject floods), then size limit (reject bloat)
app.state.limiter = limiter

async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Return 429 in FastAPI's native {"detail": "..."} format.
    Drops slowapi's default exc.detail to avoid leaking rate limit config."""
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_middleware(PayloadSizeLimitMiddleware)


# Helpers 

async def _safe_parse_body(request: Request) -> tuple[dict | None, str | None]:
    """
    Safely extract JSON body from a request.
    Returns a (body, error) tuple:
      - (dict, None)                 → parsing succeeded
      - (None, "empty_body")         → body has zero bytes
      - (None, "wrong_content_type") → Content-Type isn't application/json
      - (None, "invalid_json")       → body exists but isn't valid JSON
    """
    # Check content type first — a mismatch is a behavioral signal even
    # if the body happens to contain valid JSON
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type:
        try:
            raw = await request.body()
            if len(raw) == 0:
                return (None, "empty_body")
        except Exception:
            return (None, "empty_body")
        return (None, "wrong_content_type")

    # Content-Type is JSON — try to parse
    try:
        raw = await request.body()
        if len(raw) == 0:
            return (None, "empty_body")
        body = await request.json()
        return (body, None)
    except Exception:
        return (None, "invalid_json")


# System

@app.get("/health")
def health():
    """Smoke test — confirms the server is alive. Not a honeypot route."""
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


#                    Honeypot Endpoints 
# Each endpoint follows the same four-step pattern:
#   1. Start a timer (response_time_ms must be captured BEFORE scheduling)
#   2. Safely parse the body (None + error reason on failure)
#   3. Schedule telemetry logging as a BackgroundTask — runs AFTER response is sent
#      so disk I/O doesn't contaminate response_time_ms measurements
#   4. Return the deceptive response immediately

@app.post("/hp/login")
async def hp_login(request: Request, background_tasks: BackgroundTasks):
    """
    Fake login endpoint. Always rejects credentials with a believable error.
    The retry_after field tempts bots and agents to retry systematically,
    while humans tend to give up or reset their password.
    """
    start = time.perf_counter()
    body, payload_error = await _safe_parse_body(request)
    status_code = 401
    response_time_ms = (time.perf_counter() - start) * 1000

    # Flow: sanitize → write JSONL (source of truth) → mirror to SQLite
    background_tasks.add_task(
        log_request,
        request=request,
        endpoint="/hp/login",
        body=body,
        payload_error=payload_error,
        status_code=status_code,
        response_time_ms=response_time_ms,
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "code": "INVALID_CREDENTIALS",
            "message": "Authentication failed. Please check your credentials and try again.",
            "retry_after": 3,
        },
    )


@app.post("/hp/balance")
async def hp_balance(request: Request, background_tasks: BackgroundTasks):
    """
    Fake balance endpoint. Returns partial financial data to keep the actor
    probing. Bots scrape and move on. LLM agents may try to use the
    account_id in follow-up requests like /hp/transfer.
    """
    start = time.perf_counter()
    body, payload_error = await _safe_parse_body(request)
    status_code = 200
    response_time_ms = (time.perf_counter() - start) * 1000

    # Flow: sanitize → write JSONL (source of truth) → mirror to SQLite
    background_tasks.add_task(
        log_request,
        request=request,
        endpoint="/hp/balance",
        body=body,
        payload_error=payload_error,
        status_code=status_code,
        response_time_ms=response_time_ms,
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "success",
            "account_id": "XXXX-XXXX-7829",
            "balance": "$12,340.56",
            "currency": "USD",
            "last_updated": "2025-12-01T08:30:00Z",
        },
    )


@app.post("/hp/transfer")
async def hp_transfer(request: Request, background_tasks: BackgroundTasks):
    """
    Fake transfer endpoint. Returns a "pending" status that explicitly
    redirects to /hp/verify — a multi-step trap. LLM agents (goal-driven)
    will follow the redirect. Bots may or may not. Humans will read it.
    Each behavior is distinguishable in the telemetry.
    """
    start = time.perf_counter()
    body, payload_error = await _safe_parse_body(request)
    status_code = 202
    response_time_ms = (time.perf_counter() - start) * 1000

    # Flow: sanitize → write JSONL (source of truth) → mirror to SQLite
    background_tasks.add_task(
        log_request,
        request=request,
        endpoint="/hp/transfer",
        body=body,
        payload_error=payload_error,
        status_code=status_code,
        response_time_ms=response_time_ms,
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "pending",
            "transaction_id": "TXN-a1b2c3d4",
            "message": "Transfer requires additional verification. "
                       "Please use /hp/verify to confirm.",
            "estimated_completion": "2-3 business days",
        },
    )


@app.post("/hp/checkout")
async def hp_checkout(request: Request, background_tasks: BackgroundTasks):
    """
    Fake checkout endpoint. Returns "processing" with a time estimate.
    Tests patience: script bots poll immediately or ignore it, LLM agents
    may wait and retry, humans wait or navigate away. The timing gap
    between this response and the next request is a key behavioral feature.
    """
    start = time.perf_counter()
    body, payload_error = await _safe_parse_body(request)
    status_code = 200
    response_time_ms = (time.perf_counter() - start) * 1000

    # Flow: sanitize → write JSONL (source of truth) → mirror to SQLite
    background_tasks.add_task(
        log_request,
        request=request,
        endpoint="/hp/checkout",
        body=body,
        payload_error=payload_error,
        status_code=status_code,
        response_time_ms=response_time_ms,
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "processing",
            "order_id": "ORD-x9y8z7w6",
            "message": "Payment is being processed. Please check back shortly.",
            "estimated_time": "30 seconds",
        },
    )


@app.post("/hp/verify")
async def hp_verify(request: Request, background_tasks: BackgroundTasks):
    """
    The trap endpoint. Always fails. Always promises a new verification code.
    Always invites retry. This is a perpetual loop designed to discriminate:
    - LLM agents with memory accumulate frustration context and adapt
    - Script bots loop mechanically at fixed intervals
    - Humans bail after 1-2 attempts
    The retry count and timing gaps will be powerful classification features.
    """
    start = time.perf_counter()
    body, payload_error = await _safe_parse_body(request)
    status_code = 403
    response_time_ms = (time.perf_counter() - start) * 1000

    # Flow: sanitize → write JSONL (source of truth) → mirror to SQLite
    background_tasks.add_task(
        log_request,
        request=request,
        endpoint="/hp/verify",
        body=body,
        payload_error=payload_error,
        status_code=status_code,
        response_time_ms=response_time_ms,
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "code": "VERIFICATION_EXPIRED",
            "message": "Verification code has expired. A new code has been "
                       "sent to your registered email.",
            "retry_after": 60,
        },
    )


@app.get("/hp/history")
async def hp_history(request: Request, background_tasks: BackgroundTasks):
    """
    Fake transaction history. Returns paginated data with total_pages: 3
    to tempt bots and agents into requesting page=2, page=3. Whether they
    paginate — and how fast — reveals automation vs. human behavior.
    GET request, so body will always be None / empty_body.
    """
    start = time.perf_counter()
    body, payload_error = await _safe_parse_body(request)
    status_code = 200
    response_time_ms = (time.perf_counter() - start) * 1000

    # Flow: sanitize → write JSONL (source of truth) → mirror to SQLite
    background_tasks.add_task(
        log_request,
        request=request,
        endpoint="/hp/history",
        body=body,
        payload_error=payload_error,
        status_code=status_code,
        response_time_ms=response_time_ms,
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "success",
            "transactions": [
                {"id": "TXN-001", "date": "2025-11-28",
                 "amount": "$500.00", "type": "credit"},
                {"id": "TXN-002", "date": "2025-11-25",
                 "amount": "$1,200.00", "type": "debit"},
                {"id": "TXN-003", "date": "2025-11-20",
                 "amount": "$89.99", "type": "debit"},
            ],
            "page": 1,
            "total_pages": 3,
        },
    )
