# base_client.py — Shared HTTP Client for Honeypot Simulators
# Wraps httpx.AsyncClient with automatic session/actor headers.
# All three simulators (human, bot, LLM agent) use this as their foundation.

import uuid
import httpx
from api.app.config import API_HOST, API_PORT

class HoneypotClient:
    """Async HTTP client that attaches X-Session-ID and X-Actor-Type
    to every request. Use as an async context manager:
        async with HoneypotClient(actor_type="human") as client:
            resp = await client.hit("/hp/login", method="POST", payload={...})
    """

    def __init__(self, actor_type: str, session_id: str = None, headers: dict = None):
        self.actor_type = actor_type
        self.session_id = session_id or str(uuid.uuid4())
        # Extra headers from the simulator
        self._extra_headers = headers or {}
        self.base_url = f"http://{API_HOST}:{API_PORT}"
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        """Open the HTTP client with default headers baked in."""
        default_headers = {
            "X-Session-ID": self.session_id,
            "X-Actor-Type": self.actor_type,
            **self._extra_headers,
        }
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=default_headers,
            timeout=30.0,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the HTTP client and release connection resources."""
        if self._client:
            await self._client.aclose()
        # Returning None (falsy) lets exceptions propagate normally
        return None

    async def hit(self, endpoint: str, method: str = "GET", payload: dict = None) -> httpx.Response:
        """Make a single request to a honeypot endpoint.
        Args:
            endpoint: Route path (e.g., "/hp/login")
            method:   HTTP method — "GET" or "POST"
            payload:  Optional JSON body (POST requests)
        Returns:
            The full httpx.Response (status, headers, body).
        """
        if method.upper() == "POST":
            return await self._client.post(endpoint, json=payload)
        return await self._client.get(endpoint)