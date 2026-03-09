from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src.agents.default_agent import DefaultAgent
from src.agents.registry import AgentRegistry
from src.game.runner import GameRunner
from src.game.session_store import InMemorySessionStore
from src.llm_timing import TimingLLMWrapper
from src.tinker_llm import TinkerLLM
from src.wikipedia import WikipediaClient
from src.wikipath_client import WikipathClient

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

# Default model from main harness
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# LLM choices to sweep
LLM_CHOICES_SWEEP = [64, 128, 256, 1024]

# 2 most complex paths (conceptually distant - require many hops)
COMPLEX_PATHS = [
    ("Patrick Star", "John Wall"),
    ("Billy Joel", "VANOS"),
]


def run_one(
    wiki_client: WikipediaClient,
    wikipath_client: WikipathClient,
    path: tuple[str, str],
    model_id: str,
    llm_choices: int,
) -> dict:
    """Run a single evaluation: one path, one llm_choices value."""
    start_title, target_title = path
    llm_client = TimingLLMWrapper(TinkerLLM(model=model_id))
    registry = AgentRegistry()
    registry.register(
        DefaultAgent(wiki_client=wiki_client, llm_client=llm_client),
        is_default=True,
    )
    session_store = InMemorySessionStore()
    runner = GameRunner(
        session_store=session_store,
        agent_registry=registry,
        wiki_client=wiki_client,
        wikipath_client=wikipath_client,
        safety_max_moves=800,
        safety_max_seconds=7 * 60,
    )

    try:
        result = runner.start(
            start_title=start_title,
            target_title=target_title,
            agent_id="default",
            llm_choices=llm_choices,
        )
    except Exception as exc:
        return {
            "path": {"start": start_title, "target": target_title},
            "model_id": model_id,
            "llm_choices": llm_choices,
            "success": False,
            "hops": 0,
            "time_s": 0.0,
            "failure_reason": f"Start failed: {exc}",
            "optimal_length": None,
            "efficiency": None,
        }

    session_id = result["session_id"]
    optimal_length = result.get("optimal_length")
    t0 = time.time()
    final_payload = None

    while True:
        payload, status_code = runner.step(session_id)
        if status_code != 200:
            return {
                "path": {"start": start_title, "target": target_title},
                "model_id": model_id,
                "llm_choices": llm_choices,
                "success": False,
                "hops": 0,
                "time_s": time.time() - t0,
                "failure_reason": str(payload),
                "optimal_length": optimal_length,
                "efficiency": None,
            }
        if payload.get("done"):
            final_payload = payload
            break

    elapsed = time.time() - t0
    return {
        "path": {"start": start_title, "target": target_title},
        "model_id": model_id,
        "llm_choices": llm_choices,
        "success": final_payload["success"],
        "hops": final_payload["hops"],
        "time_s": round(elapsed, 2),
        "failure_reason": final_payload.get("failure_reason"),
        "optimal_length": optimal_length,
        "efficiency": final_payload.get("efficiency"),
    }


def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"llm_sweep_{timestamp}.json"

    wiki_client = WikipediaClient()
    wikipath_client = WikipathClient()

    results: list[dict] = []
    total = len(COMPLEX_PATHS) * len(LLM_CHOICES_SWEEP)
    run_num = 0

    for path in COMPLEX_PATHS:
        for llm_choices in LLM_CHOICES_SWEEP:
            run_num += 1
            print(f"[{run_num}/{total}] {path[0]} -> {path[1]} | llm_choices={llm_choices}")
            try:
                r = run_one(
                    wiki_client=wiki_client,
                    wikipath_client=wikipath_client,
                    path=path,
                    model_id=DEFAULT_MODEL,
                    llm_choices=llm_choices,
                )
                results.append(r)
                status = "OK" if r["success"] else "FAIL"
                print(f"  -> {status} | {r['hops']} hops | {r['time_s']:.1f}s")
            except Exception as exc:
                print(f"  -> ERROR: {exc}")
                results.append({
                    "path": {"start": path[0], "target": path[1]},
                    "model_id": DEFAULT_MODEL,
                    "llm_choices": llm_choices,
                    "success": False,
                    "hops": 0,
                    "time_s": 0.0,
                    "failure_reason": str(exc),
                    "optimal_length": None,
                    "efficiency": None,
                })

    meta = {
        "script": "run_llm_sweep.py",
        "timestamp": timestamp,
        "model": DEFAULT_MODEL,
        "paths": [f"{p[0]} -> {p[1]}" for p in COMPLEX_PATHS],
        "llm_choices_sweep": LLM_CHOICES_SWEEP,
        "agent": "default",
    }

    output_data = {"meta": meta, "results": results}
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults written to {output_path}")

    print("\n" + "=" * 60)
    print("LLM SWEEP SUMMARY")
    print("=" * 60)
    for llm in LLM_CHOICES_SWEEP:
        rows = [r for r in results if r["llm_choices"] == llm]
        ok = sum(1 for r in rows if r["success"])
        avg_hops = sum(r["hops"] for r in rows if r["success"]) / ok if ok else 0
        avg_time = sum(r["time_s"] for r in rows) / len(rows) if rows else 0
        print(f"llm_choices={llm}: {ok}/{len(rows)} success | avg hops: {avg_hops:.1f} | avg time: {avg_time:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
