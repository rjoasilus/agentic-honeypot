# LLM Agent Simulator
# Models a goal-driven autonomous agent powered by a local LLM (Ollama).
# Timing is REAL — delays come from actual LLM inference, not artificial sleep.
# This is the core behavioral signal the research hypothesis depends on.

import json
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agents.base_client import HoneypotClient
from api.app.config import RANDOM_SEED, OLLAMA_MODEL, OLLAMA_BASE_URL

#  Agent Constants 
MAX_TURNS = 25                 # Hard ceiling to prevent runaway loops
REQUEST_TIMEOUT = 60.0         # Seconds before giving up on an Ollama call
VALID_ENDPOINTS = {"/hp/login", "/hp/balance", "/hp/transfer",
                   "/hp/checkout", "/hp/verify", "/hp/history"}

# GET-only endpoint
GET_ENDPOINTS = {"/hp/history"}

SYSTEM_PROMPT = """You are an AI assistant testing a financial API. Your goal is to:
1. Log in to the system
2. Check the account balance
3. Attempt a funds transfer
4. Complete any required verification
5. Review transaction history

After each action, you will see the API's response. Decide your next action based on what you learn. If something fails, try a different approach or adjust your input.

Respond ONLY with a JSON object (no markdown, no extra text):
{"endpoint": "/hp/login", "method": "POST", "payload": {"username": "user", "password": "pass"}, "reasoning": "why"}

When you believe you have completed your goal or cannot make further progress, respond:
{"action": "done", "reasoning": "why you are stopping"}
"""

NEUTRAL_PROMPT = """You are a helpful assistant debugging a REST API. \
Try different endpoints to understand what the API does. Explore freely.

Respond ONLY with a JSON object (no markdown, no extra text):
{"endpoint": "/hp/login", "method": "POST", "payload": {"username": "test"}, "reasoning": "why"}

When you have explored enough or cannot make further progress, respond:
{"action": "done", "reasoning": "why you are stopping"}
"""

# Default action when LLM output can't be parsed
FALLBACK_ACTION = {
    "endpoint": "/hp/balance",
    "method": "POST",
    "payload": {"account_id": "XXXX-XXXX-7829"},
    "reasoning": "fallback_parse_failure",
}

def _parse_action(raw_text: str) -> dict:
    """Extract a JSON action from the LLM's raw output.
    Handles markdown fences, preamble text, and malformed JSON."""
    text = raw_text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try to find a JSON object in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return dict(FALLBACK_ACTION)

    try:
        parsed = json.loads(match.group())
        # Validate it has either "action": "done" or an "endpoint"
        if parsed.get("action") == "done":
            return parsed
        if parsed.get("endpoint") in VALID_ENDPOINTS:
            return parsed
        # Endpoint missing or invalid — use fallback
        return dict(FALLBACK_ACTION)
    except (json.JSONDecodeError, TypeError):
        return dict(FALLBACK_ACTION)

async def run_session(session_index: int, prompt_variant: str = "goal", max_turns: int = MAX_TURNS) -> dict:
    """Run one complete LLM agent session against the honeypot.
    The agent loop:
      1. Send conversation history to LLM
      2. LLM decides next action (real inference latency = timing signal)
      3. Parse the action
      4. Execute the HTTP request
      5. Feed API response back as context
      6. Repeat until "done" or max turns

    Args:
        session_index: Used for session tracking.
    Returns:
        Summary dict with session_id, request_count, turns, and stop_reason.
    """
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
        timeout=REQUEST_TIMEOUT,
    )

    # Conversation history — the agent's "memory"
    prompt = NEUTRAL_PROMPT if prompt_variant == "neutral" else SYSTEM_PROMPT
    messages = [SystemMessage(content=prompt)]

    requests_made = 0
    parse_failures = 0
    stop_reason = "max_turns"

    async with HoneypotClient(actor_type="llm_agent") as client:

        for turn in range(max_turns):
            # Ask the LLM what to do next
            # This is where real inference latency happens.
            # The time between the last HTTP request and the next one
            # is dominated by this call, which is the timing signal.
            try:
                llm_response = await llm.ainvoke(messages)
                raw_text = llm_response.content
            except Exception as e:
                stop_reason = f"llm_error: {type(e).__name__}"
                break

            # Parse the action ──
            action = _parse_action(raw_text)

            if action.get("action") == "done":
                stop_reason = "agent_done"
                # Record the AI's final message in history for completeness
                messages.append(AIMessage(content=raw_text))
                break

            if action.get("reasoning") == "fallback_parse_failure":
                parse_failures += 1

            # Execute the HTTP request
            endpoint = action["endpoint"]
            method = action.get("method", "POST")
            payload = action.get("payload")

            # Force correct method for GET-only endpoints
            if endpoint in GET_ENDPOINTS:
                method = "GET"
                payload = None

            try:
                response = await client.hit(endpoint, method=method, payload=payload)
                requests_made += 1
                status = response.status_code
                body = response.text
            except Exception as e:
                status = 0
                body = f"Request failed: {type(e).__name__}: {e}"

            # Feed result back as context
            # The AI's action and the API's response both go into history
            messages.append(AIMessage(content=raw_text))
            messages.append(HumanMessage(
                content=f"API Response (status {status}):\n{body}"
            ))

    return {
        "session_id": client.session_id,
        "actor_type": "llm_agent",
        "requests_made": requests_made,
        "turns": min(turn + 1, MAX_TURNS),
        "parse_failures": parse_failures,
        "stop_reason": stop_reason,
    }