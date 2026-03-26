# Human Behavioral Simulator
# Models a distracted, inconsistent human interacting with a banking API.
# Timing: 1-8s between requests, occasional long pauses, early bail on errors.
import random
import asyncio
from agents.base_client import HoneypotClient
from api.app.config import RANDOM_SEED

# Behavioral Constants 

ENDPOINTS = ["/hp/login", "/hp/balance", "/hp/transfer",
             "/hp/checkout", "/hp/verify", "/hp/history"]
# Humans check balance/history more than they attempt transfers or verify
ENDPOINT_WEIGHTS = [0.20, 0.25, 0.15, 0.10, 0.10, 0.20]

# GET-only endpoint — all others accept POST
GET_ENDPOINTS = {"/hp/history"}

# Delay ranges (seconds)
MIN_DELAY = 1.0
MAX_DELAY = 8.0
LONG_PAUSE_MIN = 15.0
LONG_PAUSE_MAX = 30.0
LONG_PAUSE_CHANCE = 0.10       # 10% chance of a distraction pause

# Session length range (number of requests)
MIN_REQUESTS = 3
MAX_REQUESTS = 12

# Behavioral probabilities
SEND_PAYLOAD_CHANCE = 0.50     # 50% chance of sending a body on POST
TYPO_CHANCE = 0.30             # 30% chance of typo in payload
BAIL_ON_ERROR_CHANCE = 0.60    # 60% chance of quitting after 401/403
ERROR_STATUS_CODES = {401, 403}

# Browser-like User-Agent (not python-httpx)
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Sample payload templates (realistic banking inputs)
PAYLOAD_TEMPLATES = [
    {"username": "jdoe_2024", "password": "myP@ss123"},
    {"account_id": "XXXX-XXXX-7829", "action": "check"},
    {"from_account": "7829", "to_account": "4451", "amount": "250.00"},
    {"order_id": "ORD-x9y8z7w6", "confirm": True},
    {"verification_code": "884521"},
]


def _inject_typo(payload: dict, rng: random.Random) -> dict:
    """Mutate a random field value to simulate human typos.
    Picks one string field and applies one of three mutations."""
    result = dict(payload)
    # Find fields with string values
    string_keys = [k for k, v in result.items() if isinstance(v, str)]
    if not string_keys:
        return result

    key = rng.choice(string_keys)
    value = result[key]
    if len(value) < 2:
        return result

    mutation = rng.choice(["swap", "drop", "wrong_field"])

    if mutation == "swap":
        # Swap two adjacent characters
        i = rng.randint(0, len(value) - 2)
        chars = list(value)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        result[key] = "".join(chars)
    elif mutation == "drop":
        # Drop a random character
        i = rng.randint(0, len(value) - 1)
        result[key] = value[:i] + value[i + 1:]
    elif mutation == "wrong_field":
        # Move value to a misspelled field name
        del result[key]
        result[key + "x"] = value

    return result


async def run_session(session_index: int) -> dict:
    """Run one complete human session against the honeypot.
    Args:
        session_index: Used to derive a unique sub-seed for this session.
    Returns:
        Summary dict with session_id, request_count, and bail_reason.
    """
    # Per-session RNG for deterministic but varied behavior
    sub_seed = RANDOM_SEED + session_index
    rng = random.Random(sub_seed)

    num_requests = rng.randint(MIN_REQUESTS, MAX_REQUESTS)
    requests_made = 0
    bail_reason = "completed"

    async with HoneypotClient(
        actor_type="human",
        headers={"User-Agent": BROWSER_UA},
    ) as client:

        for i in range(num_requests):
            # Pick endpoint 
            endpoint = rng.choices(ENDPOINTS, weights=ENDPOINT_WEIGHTS, k=1)[0]

            # Build payload
            payload = None
            if endpoint not in GET_ENDPOINTS and rng.random() < SEND_PAYLOAD_CHANCE:
                payload = dict(rng.choice(PAYLOAD_TEMPLATES))
                if rng.random() < TYPO_CHANCE:
                    payload = _inject_typo(payload, rng)

            #  Human delay 
            delay = rng.uniform(MIN_DELAY, MAX_DELAY)
            if rng.random() < LONG_PAUSE_CHANCE:
                delay += rng.uniform(LONG_PAUSE_MIN, LONG_PAUSE_MAX)
            await asyncio.sleep(delay)

            #  Make request 
            method = "GET" if endpoint in GET_ENDPOINTS else "POST"
            response = await client.hit(endpoint, method=method, payload=payload)
            requests_made += 1

            #  Bail on error 
            if response.status_code in ERROR_STATUS_CODES:
                if rng.random() < BAIL_ON_ERROR_CHANCE:
                    bail_reason = f"quit_after_{response.status_code}"
                    break

    return {
        "session_id": client.session_id,
        "actor_type": "human",
        "requests_made": requests_made,
        "bail_reason": bail_reason,
    }