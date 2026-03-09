from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agents.default_agent import DefaultAgent
from src.agents.planning_agent import PlanningAgent
from src.agents.registry import AgentRegistry
from src.agents.tot_agent import ToTAgent
from src.game.runner import GameRunner
from src.game.session_store import InMemorySessionStore
from src.llm_timing import TimingLLMWrapper
from src.tinker_llm import TinkerLLM
from src.wikipedia import WikipediaClient
from src.wikipath_client import WikipathClient

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

# Models: (model_id, default_llm_choices, planning_retrieval_top_k, tot_llm_candidates)
MODELS = [
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", 128, 12, 128),
    ("meta-llama/Llama-3.3-70B-Instruct", 1024, 128, 1024),
    ("openai/gpt-oss-120b", 1024, 128, 1024),
]

AGENTS = ["default", "planning", "tot"]


def parse_paths(path_file: Path) -> List[Tuple[str, str]]:
    """Parse Start -> End pairs from path file. Skip blank and non-path lines."""
    paths: List[Tuple[str, str]] = []
    pattern = re.compile(r"^\s*(?:\d+\.\s*)?(.+?)\s*->\s*(.+?)\s*$")
    for line in path_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = pattern.match(line)
        if m:
            start, target = m.group(1).strip(), m.group(2).strip()
            if start and target:
                paths.append((start, target))
    return paths


def build_registry(
    wiki_client: WikipediaClient,
    llm_client: TinkerLLM,
) -> AgentRegistry:
    """Register all three agents in the registry."""
    registry = AgentRegistry()
    registry.register(
        DefaultAgent(wiki_client=wiki_client, llm_client=llm_client),
        is_default=True,
    )
    registry.register(PlanningAgent(wiki_client=wiki_client, llm_client=llm_client))
    registry.register(ToTAgent(wiki_client=wiki_client, llm_client=llm_client))
    return registry


def run_one(
    wiki_client: WikipediaClient,
    wikipath_client: WikipathClient,
    path: Tuple[str, str],
    model_id: str,
    agent_id: str,
    agent_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a single evaluation: one path, one model, one agent."""
    start_title, target_title = path
    llm_client = TimingLLMWrapper(TinkerLLM(model=model_id))
    registry = build_registry(wiki_client, llm_client)
    session_store = InMemorySessionStore()
    runner = GameRunner(
        session_store=session_store,
        agent_registry=registry,
        wiki_client=wiki_client,
        wikipath_client=wikipath_client,
        safety_max_moves=800,
        safety_max_seconds=7 * 60,
    )

    start_kwargs: Dict[str, Any] = {
        "start_title": start_title,
        "target_title": target_title,
        "agent_id": agent_id,
    }
    if agent_id == "default":
        start_kwargs["llm_choices"] = agent_config.get("llm_choices")
    elif agent_id == "planning":
        start_kwargs["retrieval_top_k"] = agent_config.get("retrieval_top_k")
    elif agent_id == "tot":
        start_kwargs["tot_llm_candidates"] = agent_config.get("tot_llm_candidates")

    try:
        result = runner.start(**start_kwargs)
    except Exception as exc:
        return {
            "path": {"start": start_title, "target": target_title},
            "model_id": model_id,
            "agent_id": agent_id,
            "agent_config": agent_config,
            "success": False,
            "hops": 0,
            "final_path": None,
            "chain": None,
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
                "agent_id": agent_id,
                "agent_config": agent_config,
                "success": False,
                "hops": 0,
                "final_path": None,
                "chain": None,
                "time_s": time.time() - t0,
                "failure_reason": str(payload),
                "optimal_length": optimal_length,
                "efficiency": None,
            }
        if payload.get("done"):
            final_payload = payload
            break

    elapsed = time.time() - t0
    session = session_store.get(session_id)
    final_path = session.get("path") if session else None

    return {
        "path": {"start": start_title, "target": target_title},
        "model_id": model_id,
        "agent_id": agent_id,
        "agent_config": agent_config,
        "success": final_payload["success"],
        "hops": final_payload["hops"],
        "final_path": final_path,
        "chain": final_payload.get("chain"),
        "time_s": round(elapsed, 2),
        "failure_reason": final_payload.get("failure_reason"),
        "optimal_length": optimal_length,
        "efficiency": final_payload.get("efficiency"),
    }


def get_agent_config(model_row: Tuple[str, int, int, int], agent_id: str) -> Dict[str, Any]:
    """Return agent-specific config for the given model row."""
    _model_id, llm_choices, retrieval_top_k, tot_llm_candidates = model_row
    if agent_id == "default":
        return {"llm_choices": llm_choices}
    if agent_id == "planning":
        return {"retrieval_top_k": retrieval_top_k}
    if agent_id == "tot":
        return {"tot_llm_candidates": tot_llm_candidates}
    return {}


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Group by agent
    by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_agent.setdefault(r["agent_id"], []).append(r)

    for agent_id in AGENTS:
        rows = by_agent.get(agent_id, [])
        if not rows:
            continue
        success_count = sum(1 for r in rows if r["success"])
        total_hops = sum(r["hops"] for r in rows if r["success"])
        total_time = sum(r["time_s"] for r in rows)
        avg_hops = total_hops / success_count if success_count else 0
        print(f"\n{agent_id.upper()}: {success_count}/{len(rows)} success | "
              f"avg hops (success): {avg_hops:.1f} | total time: {total_time:.0f}s")
        for model_id in [m[0] for m in MODELS]:
            model_rows = [r for r in rows if r["model_id"] == model_id]
            if not model_rows:
                continue
            ok = sum(1 for r in model_rows if r["success"])
            hops_sum = sum(r["hops"] for r in model_rows if r["success"])
            avg = hops_sum / ok if ok else 0
            print(f"  {model_id}: {ok}/{len(model_rows)} | avg hops: {avg:.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation harness on Wikipedia game paths")
    parser.add_argument(
        "--paths",
        type=Path,
        default=EVAL_DIR / "PATHS_TO_TEST.TXT",
        help="Path file with Start -> End pairs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: results/eval_harness_YYYYMMDD_HHMMSS.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per config (default: 1)",
    )
    args = parser.parse_args()

    paths = parse_paths(args.paths)
    if not paths:
        print(f"No paths found in {args.paths}", file=sys.stderr)
        return 1

    runs: List[Tuple[Tuple[str, str], str, str, Dict[str, Any], int]] = []
    for path in paths:
        for model_row in MODELS:
            model_id = model_row[0]
            for agent_id in AGENTS:
                agent_config = get_agent_config(model_row, agent_id)
                for run_idx in range(args.runs):
                    runs.append((path, model_id, agent_id, agent_config, run_idx))

    if args.dry_run:
        print(f"Dry run: {len(runs)} runs planned")
        for i, (path, model_id, agent_id, agent_config, run_idx) in enumerate(runs):
            run_suffix = f" (run {run_idx + 1}/{args.runs})" if args.runs > 1 else ""
            print(f"  {i + 1}. {path[0]} -> {path[1]} | {model_id} | {agent_id} | {agent_config}{run_suffix}")
        return 0

    wiki_client = WikipediaClient()
    wikipath_client = WikipathClient()

    results: List[Dict[str, Any]] = []
    total = len(runs)

    RESULTS_DIR.mkdir(exist_ok=True)
    for i, (path, model_id, agent_id, agent_config, run_idx) in enumerate(runs):
        run_num = i + 1
        run_suffix = f" [run {run_idx + 1}/{args.runs}]" if args.runs > 1 else ""
        print(f"[{run_num}/{total}] {path[0]} -> {path[1]} | {model_id} | {agent_id}{run_suffix}")
        try:
            r = run_one(
                wiki_client=wiki_client,
                wikipath_client=wikipath_client,
                path=path,
                model_id=model_id,
                agent_id=agent_id,
                agent_config=agent_config,
            )
            r["run_index"] = run_idx
            results.append(r)
            status = "OK" if r["success"] else "FAIL"
            print(f"  -> {status} | {r['hops']} hops | {r['time_s']:.1f}s")
        except Exception as exc:
            print(f"  -> ERROR: {exc}")
            results.append({
                "path": {"start": path[0], "target": path[1]},
                "model_id": model_id,
                "agent_id": agent_id,
                "agent_config": agent_config,
                "run_index": run_idx,
                "success": False,
                "hops": 0,
                "final_path": None,
                "chain": None,
                "time_s": 0.0,
                "failure_reason": str(exc),
                "optimal_length": None,
                "efficiency": None,
            })

    output_path = args.output or RESULTS_DIR / f"eval_harness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        "meta": {
            "script": "run_eval_harness.py",
            "runs_per_config": args.runs,
            "paths_file": str(args.paths),
        },
        "results": results,
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults written to {output_path}")

    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
