import re
import copy

# Sensitive Field Detection 
# Compiled once at import time for performance
# Matches field names that could contain credentials, financial data, or PII
SENSITIVE_PATTERNS = re.compile(
    r"(password|passwd|token|secret|api_key|apikey|ssn|"
    r"credit_card|card_number|cvv|authorization)",
    re.IGNORECASE,
)

def sanitize_payload(payload: dict) -> dict:
    """Deep-copy payload and redact fields matching sensitive patterns.
    Recursively walks nested dicts. Any key matching SENSITIVE_PATTERNS
    gets its value replaced with '[REDACTED]' before logging.
    """
    # Guard: if it's not a dict, nothing to sanitize
    if not isinstance(payload, dict):
        return payload

    # Clone so  the original request data is never mutated
    sanitized = copy.deepcopy(payload)

    for key in list(sanitized.keys()):
        if SENSITIVE_PATTERNS.search(key):
            # Sensitive field found? then replace value, original never persisted
            sanitized[key] = "[REDACTED]"
        elif isinstance(sanitized[key], dict):
            # Nested dict — recurse to catch deeply buried sensitive fields
            sanitized[key] = sanitize_payload(sanitized[key])
    return sanitized
