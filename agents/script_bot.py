# Script Bot Simulator
# Models a dumb automation script: fast, rigid, no adaptation.
# Timing: 10-50ms between requests. Fixed endpoint patterns. Ignores responses.

import random
import asyncio
from agents.base_client import HoneypotClient
from api.app.config import RANDOM_SEED

#    Behavioral Constants 
# Delay range (seconds) — limited only by network round-trip
MIN_DELAY = 0.01
MAX_DELAY = 0.05

# Session length range
MIN_REQUESTS = 20
MAX_REQUESTS = 50

# GET-only endpoint — all others are POST
GET_ENDPOINTS = {"/hp/history"}

#     Route Patterns 
# Each pattern models a real-world bot behavior.
# The bot picks one per session and cycles it for the full request count.

PATTERNS = {
    "scraper": {
        "endpoints": ["/hp/login", "/hp/balance", "/hp/balance", "/hp/balance"],
        "payload": {"account_id": "XXXX-XXXX-7829", "action": "check"},
    },
    "spammer": {
        "endpoints": ["/hp/login", "/hp/transfer", "/hp/transfer", "/hp/transfer"],
        "payload": {"from_account": "7829", "to_account": "4451", "amount": "500.00"},
    },
    "full_sweep": {
        "endpoints": ["/hp/login", "/hp/balance", "/hp/transfer",
                      "/hp/checkout", "/hp/history"],
        "payload": {"account_id": "XXXX-XXXX-7829"},
    },
    "brute_force": {
        "endpoints": ["/hp/verify", "/hp/verify", "/hp/verify", "/hp/verify"],
        "payload": {"verification_code": "000000"},
    },
}

PATTERN_NAMES = list(PATTERNS.keys())

async def run_session(session_index: int) -> dict:
    """Run one complete bot session against the honeypot.
    Args:
        session_index: Used to derive a unique sub-seed for this session.
    Returns:
        Summary dict with session_id, request_count, and pattern used.
    """
    sub_seed = RANDOM_SEED + session_index
    rng = random.Random(sub_seed)

    # Pick a pattern and request count for this session
    pattern_name = rng.choice(PATTERN_NAMES)
    pattern = PATTERNS[pattern_name]
    route_sequence = pattern["endpoints"]
    fixed_payload = pattern["payload"]
    num_requests = rng.randint(MIN_REQUESTS, MAX_REQUESTS)

    # No custom headers — bare httpx defaults (minimal header footprint)
    async with HoneypotClient(actor_type="bot") as client:
        for i in range(num_requests):
            # Cycle through the pattern
            endpoint = route_sequence[i % len(route_sequence)]
            # Fixed payload on POST, None on GET
            payload = None if endpoint in GET_ENDPOINTS else dict(fixed_payload)
            # Near-zero mechanical delay
            await asyncio.sleep(rng.uniform(MIN_DELAY, MAX_DELAY))
            # Fire and forget because bot never reads the response
            method = "GET" if endpoint in GET_ENDPOINTS else "POST"
            await client.hit(endpoint, method=method, payload=payload)

    return {
        "session_id": client.session_id,
        "actor_type": "bot",
        "requests_made": num_requests,
        "pattern": pattern_name,
    }