# Simulation Orchestrator
# Runs human, bot, and LLM agent simulators sequentially against the live honeypot.
# Usage: python -m agents.orchestrator --humans 200 --bots 200 --agents 200

import asyncio
import argparse
import time
import httpx
from api.app.config import API_HOST, API_PORT, OLLAMA_BASE_URL
from agents import human_sim, script_bot, llm_agent

# Seed offset so bot sessions never collide with human sessions
BOT_INDEX_OFFSET = 100_000

# Pre-Flight Checks 
async def check_honeypot() -> bool:
    """Verify the honeypot server is running."""
    url = f"http://{API_HOST}:{API_PORT}/health"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            return resp.status_code == 200
    except Exception:
        return False

async def check_ollama() -> bool:
    """Verify Ollama is running and reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(OLLAMA_BASE_URL)
            return resp.status_code == 200
    except Exception:
        return False


async def preflight() -> bool:
    """Run all pre-flight checks. Returns True if everything is ready."""
    print("*" * 60)
    print("PRE-FLIGHT CHECKS")
    print("*" * 60)

    honeypot_ok = await check_honeypot()
    status = "OK" if honeypot_ok else "FAIL"
    print(f"  Honeypot server ({API_HOST}:{API_PORT}): {status}")

    ollama_ok = await check_ollama()
    status = "OK" if ollama_ok else "FAIL"
    print(f"  Ollama ({OLLAMA_BASE_URL}):              {status}")

    if not honeypot_ok:
        print("\n[ERROR] Honeypot not running. Start it first:")
        print("  python -m uvicorn api.app.main:app --host 127.0.0.1 --port 8000")
        return False

    if not ollama_ok:
        print("\n[WARN] Ollama not running. LLM agent sessions will be skipped.")
        print("  Start Ollama with: ollama serve")
        # Don't block — human and bot sims can still run

    print("-" * 60)
    return True


#  Session Runners 

async def run_actor_sessions(actor_name, run_func, count, offset, skip=False, prompt_variant="goal"):
    """Run N sessions for one actor type with progress reporting.
    Args:
        actor_name: Display label ("human", "bot", "agent")
        run_func:   The simulator's run_session() coroutine
        count:      Number of sessions to run
        offset:     Starting session index (for seed uniqueness across runs)
        skip:       If True, skip entirely (used when Ollama is down)
    Returns:
        List of session summary dicts
    """
    if skip:
        print(f"\n[{actor_name:6s}] SKIPPED ({count} sessions)")
        return []

    print(f"\n[{actor_name:6s}] Starting {count} sessions (offset={offset})")
    results = []
    errors = 0

    for i in range(count):
        session_index = offset + i
        session_start = time.perf_counter()

        try:
            kwargs = {}
            if prompt_variant != "goal":
                kwargs["prompt_variant"] = prompt_variant
                kwargs["max_turns"] = 8
            summary = await run_func(session_index, **kwargs)
            elapsed = time.perf_counter() - session_start

            # Build progress line from summary fields
            reqs = summary.get("requests_made", "?")
            detail = (summary.get("bail_reason")
                      or summary.get("pattern")
                      or summary.get("stop_reason")
                      or "")
            print(f"  [{actor_name:6s}] Session {i+1:>4d}/{count} "
                  f"— {reqs:>3} reqs — {detail} ({elapsed:.1f}s)")
            results.append(summary)

        except Exception as e:
            elapsed = time.perf_counter() - session_start
            errors += 1
            print(f"  [{actor_name:6s}] Session {i+1:>4d}/{count} "
                  f"— ERROR: {type(e).__name__}: {e} ({elapsed:.1f}s)")

    print(f"[{actor_name:6s}] Done. {len(results)} succeeded, {errors} failed.")
    return results


#  Main 

async def main(args):
    """Run the full simulation pipeline."""
    # Override JSONL output path if specified
    if args.output:
        import api.app.logger as _logger
        from pathlib import Path as _Path
        _logger.LOG_PATH = _Path(args.output)
        _logger.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[OUTPUT] Writing telemetry to: {_logger.LOG_PATH}")
    
    # Pre-flight
    ready = await preflight()
    if not ready:
        return
    
    ollama_ok = await check_ollama()
    total_start = time.perf_counter()

    # Run each actor type sequentially
    human_results = await run_actor_sessions(
        "human", human_sim.run_session,
        count=args.humans, offset=args.offset,
    )

    bot_results = await run_actor_sessions(
        "bot", script_bot.run_session,
        count=args.bots, offset=args.offset + BOT_INDEX_OFFSET,
    )

    agent_results = await run_actor_sessions(
        "agent", llm_agent.run_session,
        count=args.agents, offset=args.offset,
        skip=(not ollama_ok and args.agents > 0),
        prompt_variant=args.prompt,
    )

    #    Summary 
    total_elapsed = time.perf_counter() - total_start
    total_requests = (
        sum(r.get("requests_made", 0) for r in human_results)
        + sum(r.get("requests_made", 0) for r in bot_results)
        + sum(r.get("requests_made", 0) for r in agent_results)
    )

    print("\n" + "*" * 60)
    print("SIMULATION COMPLETE")
    print("*" * 60)
    print(f"  Human sessions:  {len(human_results)}")
    print(f"  Bot sessions:    {len(bot_results)}")
    print(f"  Agent sessions:  {len(agent_results)}")
    print(f"  Total requests:  {total_requests}")
    print(f"  Total time:      {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    print("*" * 60)

def cli():
    """Parse command-line arguments and run the orchestrator."""
    parser = argparse.ArgumentParser(
        description="Agentic Honeypot — Simulation Orchestrator"
    )
    parser.add_argument("--humans", type=int, default=200,
                        help="Number of human sessions (default: 200)")
    parser.add_argument("--bots", type=int, default=200,
                        help="Number of bot sessions (default: 200)")
    parser.add_argument("--agents", type=int, default=200,
                        help="Number of LLM agent sessions (default: 200)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Starting session index for seed uniqueness across runs (default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Override JSONL output path (default: LOG_PATH from config)")
    parser.add_argument("--prompt", type=str, default="goal",
                        choices=["goal", "neutral"],
                        help="Agent system prompt variant (default: goal)")

    args = parser.parse_args()
    asyncio.run(main(args))

if __name__ == "__main__":
    cli()