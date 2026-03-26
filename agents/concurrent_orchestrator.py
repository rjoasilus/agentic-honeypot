# this is the concurrent Simulation Orchestrator
# Runs all three simulators simultaneously using asyncio.gather().
# Tests whether timing features remain separable under real concurrent load.
# Usage:
#   python -m agents.concurrent_orchestrator --humans 50 --bots 50 --agents 50
# Important: Set LOG_PATH in the server's environment before starting it.
#   Terminal 1:
#     $env:LOG_PATH = "data/raw/telemetry_concurrent.jsonl"
#     uvicorn api.app.main:app --host 127.0.0.1 --port 8000
#   Terminal 2:
#     python -m agents.concurrent_orchestrator --humans 50 --bots 50 --agents 50

import asyncio
import argparse
import time
import httpx
from api.app.config import API_HOST, API_PORT, OLLAMA_BASE_URL
from agents import human_sim, script_bot, llm_agent

# Same offset as sequential orchestrator — bots get a separate seed namespace
# so session seeds never collide with human session seeds across actor types
BOT_INDEX_OFFSET = 100_000
AGENT_INDEX_OFFSET = 200_000  # agents get their own namespace too


# Pre-Flight Checks (reused from orchestrator.py) 

async def check_honeypot() -> bool:
    url = f"http://{API_HOST}:{API_PORT}/health"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            return resp.status_code == 200
    except Exception:
        return False

async def check_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(OLLAMA_BASE_URL)
            return resp.status_code == 200
    except Exception:
        return False


# Per-Actor Concurrent Runner 

async def run_actor_concurrent(actor_name: str, run_func, count: int, offset: int, skip: bool = False) -> list:
    """Run N sessions for one actor type. Designed to run concurrently alongside
    other actor types via asyncio.gather(). Each session is awaited in sequence
    within this coroutine — concurrency comes from gather(), not from this loop.
    Args:
        actor_name: Display label ("human", "bot", "agent")
        run_func:   The simulator's run_session() coroutine
        count:      Number of sessions to run
        offset:     Starting session index (for seed uniqueness)
        skip:       If True, skip entirely (used when Ollama is down)
    Returns:
        List of session summary dicts
    """
    if skip or count == 0:
        print(f"[{actor_name:6s}] SKIPPED")
        return []

    print(f"[{actor_name:6s}] Starting {count} sessions concurrently (offset={offset})")
    results = []
    errors = 0

    for i in range(count):
        session_index = offset + i
        session_start = time.perf_counter()
        try:
            summary = await run_func(session_index)
            elapsed = time.perf_counter() - session_start
            reqs = summary.get("requests_made", "?")
            detail = (
                summary.get("bail_reason")
                or summary.get("pattern")
                or summary.get("stop_reason")
                or ""
            )
            print(f"  [{actor_name:6s}] {i+1:>4d}/{count} — {reqs:>3} reqs — {detail} ({elapsed:.1f}s)")
            results.append(summary)
        except Exception as e:
            elapsed = time.perf_counter() - session_start
            errors += 1
            print(f"  [{actor_name:6s}] {i+1:>4d}/{count} — ERROR: {type(e).__name__}: {e} ({elapsed:.1f}s)")

    print(f"[{actor_name:6s}] Done. {len(results)} succeeded, {errors} failed.")
    return results


# Main

async def main(args):
    """Run all three simulators concurrently via asyncio.gather()."""

    # Pre-flight
    print("*" * 60)
    print("CONCURRENT SIMULATION — PRE-FLIGHT")
    print("*" * 60)

    honeypot_ok = await check_honeypot()
    print(f"  Honeypot ({API_HOST}:{API_PORT}): {'OK' if honeypot_ok else 'FAIL'}")

    ollama_ok = await check_ollama()
    print(f"  Ollama ({OLLAMA_BASE_URL}): {'OK' if ollama_ok else 'FAIL'}")

    if not honeypot_ok:
        print("\n[ERROR] Honeypot not running.")
        print("  Start with: $env:LOG_PATH='data/raw/telemetry_concurrent.jsonl'")
        print("              uvicorn api.app.main:app --host 127.0.0.1 --port 8000")
        return

    if not ollama_ok:
        print("\n[WARN] Ollama not running — agent sessions will be skipped.")

    print("*" * 60)
    print(f"  Sessions: {args.humans} human | {args.bots} bot | {args.agents} agent")
    print(f"  Mode: CONCURRENT (asyncio.gather)")
    print("*" * 60)

    total_start = time.perf_counter()

    #  Launch all three actor types simultaneously 
    # asyncio.gather() starts all three coroutines and runs them concurrently
    # on the same event loop. Each one yields control at every 'await' point,
    # allowing the other two to advance. This interleaves all three traffic
    # streams the way a real deployment would experience them.
    human_results, bot_results, agent_results = await asyncio.gather(
        run_actor_concurrent(
            "human", human_sim.run_session,
            count=args.humans, offset=args.offset,
        ),
        run_actor_concurrent(
            "bot", script_bot.run_session,
            count=args.bots, offset=args.offset + BOT_INDEX_OFFSET,
        ),
        run_actor_concurrent(
            "agent", llm_agent.run_session,
            count=args.agents, offset=args.offset + AGENT_INDEX_OFFSET,
            skip=(not ollama_ok and args.agents > 0),
        ),
    )

    #  Summary 
    total_elapsed = time.perf_counter() - total_start
    total_requests = (
        sum(r.get("requests_made", 0) for r in human_results)
        + sum(r.get("requests_made", 0) for r in bot_results)
        + sum(r.get("requests_made", 0) for r in agent_results)
    )

    print("\n" + "*" * 60)
    print("CONCURRENT SIMULATION COMPLETE")
    print("*" * 60)
    print(f"  Human sessions:  {len(human_results)}")
    print(f"  Bot sessions:    {len(bot_results)}")
    print(f"  Agent sessions:  {len(agent_results)}")
    print(f"  Total requests:  {total_requests}")
    print(f"  Total time:      {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    print("*" * 60)


def cli():
    parser = argparse.ArgumentParser(
        description="Agentic Honeypot — Concurrent Simulation Orchestrator"
    )
    parser.add_argument("--humans", type=int, default=50,
                        help="Number of human sessions (default: 50)")
    parser.add_argument("--bots", type=int, default=50,
                        help="Number of bot sessions (default: 50)")
    parser.add_argument("--agents", type=int, default=50,
                        help="Number of LLM agent sessions (default: 50)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Starting session index offset (default: 0)")
    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
