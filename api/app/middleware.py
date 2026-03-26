from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from api.app.config import MAX_PAYLOAD_BYTES, RATE_LIMIT

# Rate Limiter (slowapi)
# Throttles requests per IP address
# Default: 100 requests/minute (configurable in .env)
# Exceeding the limit returns HTTP 429 (Too Many Requests)
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])


#  Request Size Limit Middleware 
# Intercepts every request BEFORE it reaches any endpoint
# Rejects payloads larger than MAX_PAYLOAD_BYTES (default 10KB)
class PayloadSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Read Content-Length header to check size WITHOUT loading the body
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_PAYLOAD_BYTES:
            # Return 413 error
            return JSONResponse(
                status_code=413,
                content={"detail": f"Payload exceeds {MAX_PAYLOAD_BYTES} byte limit"}
            )
        # if payload is within limits then pass request through to the endpoint
        return await call_next(request)