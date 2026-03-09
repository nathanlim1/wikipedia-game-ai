from __future__ import annotations

import heapq
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.agents.default_agent import CANDIDATE_POOL, DefaultAgent
from src.tinker_llm import TinkerLLM
from src.wikipedia import WikipediaClient


TOT_K = 5
TOT_LLM_CANDIDATES = 50  # Links sent to LLM for scoring; top k are added as children
TOT_EXPANSIONS_PER_STEP = 15
TOT_SCORE_SAMPLES = 1
LLM_TIMEOUT_S = 60

# Set TOT_DEBUG=1 to print exploration trace
TOT_DEBUG = os.environ.get("TOT_DEBUG", "").strip() in ("1", "true", "yes")

SCORE_INSTRUCTIONS = """Return ONLY valid JSON. No extra text.

Given the target topic and candidate pages, score each page 0-100 for how promising it is toward reaching the target.
- 0 = completely unrelated
- 100 = the target topic itself

Consider: bridge logic, topic relevance, avoiding dead ends (List of..., disambiguation).

Schema:
{"scores": [n0, n1, n2, ...]}

Each n is an integer 0-100. The order must match the candidate order."""

RESCORE_INSTRUCTIONS = """Return ONLY valid JSON. No extra text.

Given the target topic and a list of nodes (each with path and current page), score each node 0-100 for how promising it is toward reaching the target.
- 0 = completely unrelated
- 100 = the target topic itself

Consider: bridge logic, topic relevance, path context (how we got here), avoiding dead ends.

Schema:
{"scores": [n0, n1, n2, ...]}

Each n is an integer 0-100. The order must match the node order."""

PICK_TOP_K_INSTRUCTIONS = """Return ONLY valid JSON. No extra text.

You are given a list of candidate links from a Wikipedia page. Your task: PICK the best k candidates for reaching the target, and score each 0-100.
- 0 = completely unrelated
- 100 = the target topic itself

Do NOT score all candidates. Only pick and score the k best.

Consider: bridge logic, topic relevance, avoiding dead ends (List of..., disambiguation).

Schema:
{"picks": [{"index": <0-based index>, "score": <0-100>}, ...]}

Return exactly k picks. Each index must be from the candidate list. Order by score descending."""


class ToTAgent(DefaultAgent):
    agent_id = "tot"

    def __init__(self, wiki_client: WikipediaClient, llm_client: TinkerLLM) -> None:
        super().__init__(wiki_client, llm_client)

    def initialize_session(self, start_title: str, target_title: str) -> Dict[str, Any]:
        session = super().initialize_session(start_title, target_title)
        session["tot_k"] = TOT_K
        session["tot_llm_candidates"] = TOT_LLM_CANDIDATES
        session["tot_expansions_per_step"] = TOT_EXPANSIONS_PER_STEP
        session["tot_score_samples"] = TOT_SCORE_SAMPLES
        session["tot_frontier"] = []  # min-heap: (-score, id, node)
        session["tot_node_id"] = 0
        session["tot_nodes"] = {}  # id -> {path, current, score, title_to_anchor}
        return session

    def step(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        target = session["resolved_target"]
        path = session["path"]
        path_set = session["path_set"]
        committed_path = list(path)
        frontier = session.get("tot_frontier", [])
        nodes = session.get("tot_nodes", {})
        node_id_counter = session.get("tot_node_id", 0)
        k = session.get("tot_k", TOT_K)
        llm_candidates = session.get("tot_llm_candidates", TOT_LLM_CANDIDATES)
        n_expansions = session.get("tot_expansions_per_step", TOT_EXPANSIONS_PER_STEP)

        events: List[Dict[str, Any]] = []

        # Ensure frontier is rooted at committed path
        if not frontier and not session.get("tot_initialized"):
            session["tot_initialized"] = True
            current = committed_path[-1]
            if current == target:
                session["done"] = True
                session["success"] = True
                return []
            # Start page: no progress toward target yet; rescore will update before first expansion
            root_score = 0
            nid = node_id_counter
            node_id_counter += 1
            nodes[nid] = {
                "path": list(committed_path),
                "current": current,
                "score": root_score,
                "title_to_anchor": {},
            }
            heapq.heappush(frontier, (-root_score, nid, current))
            session["tot_frontier"] = frontier
            session["tot_nodes"] = nodes
            session["tot_node_id"] = node_id_counter
            if TOT_DEBUG:
                print(f"[ToT] Root init: {current} (score={root_score})")

        # Check if any frontier node is the target (commit full path to it)
        for neg_score, nid, _ in list(frontier):
            nd = nodes.get(nid)
            if nd and nd["current"] == target:
                best_path = nd["path"]
                if len(best_path) > len(committed_path):
                    return self._commit_full_path(session, committed_path, best_path, events)
                return []

        events.append({
            "type": "exploration_start",
            "message": "Exploring with best-first search",
            "expansions_planned": n_expansions,
        })
        if TOT_DEBUG:
            print(f"[ToT] === Step {len(session['moves'])+1}: {n_expansions} expansions planned ===")

        # Rescore ALL nodes in the frontier at the start of each step
        if frontier:
            self._rescore_frontier(session, frontier, nodes, target)
            frontier = session["tot_frontier"]

        last_expanded_node: Optional[str] = None
        tried_edges = session.get("tried_edges", {})

        for exp_n in range(n_expansions):
            if not frontier:
                break

            neg_score, nid, node_current = heapq.heappop(frontier)
            nd = nodes.get(nid)
            if not nd or nd["current"] != node_current:
                continue

            if TOT_DEBUG:
                print(f"[ToT] Expansion {exp_n+1}: POP {nd['current']} (score={-neg_score}) path_depth={len(nd['path'])}")

            if nd["current"] == target:
                best_path = nd["path"]
                if len(best_path) > len(committed_path):
                    # Commit full path to target (all remaining moves)
                    return self._commit_full_path(session, committed_path, best_path, events)
                break

            # Expand: get up to llm_candidates links, LLM scores all, we add top k
            try:
                frame = self.make_frame(
                    page_title=nd["current"],
                    target_title=target,
                    target_keywords=session["target_keywords"],
                    path_set=set(nd["path"]),
                    tried_edges=tried_edges,
                    candidate_pool=max(llm_candidates, session.get("candidate_pool", CANDIDATE_POOL)),
                )
            except Exception as exc:
                if len(committed_path) <= 1:
                    session["done"] = True
                    session["success"] = False
                    session["failure_reason"] = f"Wikipedia scrape failed: {exc}"
                    return [{"type": "backtrack", "from_title": nd["current"], "to_title": committed_path[-1], "reason": str(exc)}]
                # Backtrack
                prev = committed_path[-2]
                popped = committed_path.pop()
                path_set.discard(popped)
                if session["moves"]:
                    session["moves"].pop()
                path[:] = list(committed_path)
                return events + [{"type": "backtrack", "from_title": popped, "to_title": prev, "reason": f"scrape failed: {exc}"}]

            all_candidates = frame["candidates"][:llm_candidates]
            title_to_anchor = frame["title_to_anchor"]

            if not all_candidates:
                continue

            # LLM picks the best k from the list and scores only those (not all)
            scored_pairs = self.llm_pick_top_k(
                target_title=target,
                target_extract=session["target_extract"],
                candidate_titles=all_candidates,
                current_page=nd["current"],
                k=k,
            )

            prev_expanded = last_expanded_node
            # True backtrack: we're expanding a node whose parent is NOT the previous node.
            # (i.e. we jumped to a different branch instead of going deeper)
            parent_of_current = nd["path"][-2] if len(nd["path"]) >= 2 else None
            backtracked = prev_expanded is not None and parent_of_current != prev_expanded
            last_expanded_node = nd["current"]
            if TOT_DEBUG and backtracked:
                print(f"[ToT]   ^ BRANCH SWITCH: jumped from {prev_expanded} to {nd['current']} (different branch)")

            if TOT_DEBUG:
                for c, s in scored_pairs[:k]:
                    print(f"[ToT]   + {c}: score={s} (top {k} of {len(all_candidates)})")

            # Add top k children to frontier
            added = 0
            for cand, score in scored_pairs[:k]:
                if cand in path_set:
                    continue
                tried_from = tried_edges.setdefault(nd["current"], set())
                if cand in tried_from:
                    continue
                child_path = nd["path"] + [cand]
                child_nid = node_id_counter
                node_id_counter += 1
                nodes[child_nid] = {
                    "path": child_path,
                    "current": cand,
                    "score": score,
                    "title_to_anchor": title_to_anchor,
                }
                heapq.heappush(frontier, (-score, child_nid, cand))
                added += 1

            if TOT_DEBUG:
                print(f"[ToT]   Added {added} children, frontier size: {len(frontier)}")

            session["tot_frontier"] = frontier
            session["tot_nodes"] = nodes
            session["tot_node_id"] = node_id_counter

            events.append({
                "type": "exploration_progress",
                "expansion_n": exp_n + 1,
                "node_expanded": nd["current"],
                "frontier_size": len(frontier),
                "backtracked": backtracked,
            })

        # Commit: best leaf
        if not frontier:
            if len(committed_path) <= 1:
                session["done"] = True
                session["success"] = False
                session["failure_reason"] = "No usable outgoing links."
                return events
            prev = committed_path[-2]
            popped = committed_path.pop()
            path_set.discard(popped)
            if session["moves"]:
                session["moves"].pop()
            path[:] = list(committed_path)
            return events + [{"type": "backtrack", "from_title": popped, "to_title": prev, "reason": "no candidates"}]

        # Find best leaf: highest score, then shortest path (prefer target, prefer fewer hops)
        def key_fn(item: Tuple[float, int, str]) -> Tuple[float, int]:
            neg_score, nid, _ = item
            nd = nodes.get(nid)
            path_len = len(nd["path"]) if nd else 999
            return (neg_score, path_len)  # min score wins, then min path_len
        best_item = min(frontier, key=key_fn)
        best_nid = best_item[1]
        best_nd = nodes[best_nid]
        best_path = best_nd["path"]

        if TOT_DEBUG:
            print(f"[ToT] COMMIT: best_path={best_path} (score={best_nd['score']})")
            print(f"[ToT] Committed was: {committed_path}")

        # Commit: extend committed_path along best_path (may backtrack if paths diverge)
        if len(best_path) > len(committed_path):
            return self._commit_move(session, committed_path, best_path, events)
        # Same depth but different path: backtrack to common ancestor, then move
        if best_path != committed_path:
            return self._commit_move(session, committed_path, best_path, events)

        return events

    def _commit_move(
        self,
        session: Dict[str, Any],
        committed_path: List[str],
        best_path: List[str],
        events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Commit the first uncommitted step of best_path. Backtrack if paths diverge."""
        path = session["path"]
        path_set = session["path_set"]
        moves = session["moves"]

        # Find common prefix
        common_len = 0
        for i in range(min(len(committed_path), len(best_path))):
            if committed_path[i] == best_path[i]:
                common_len = i + 1
            else:
                break

        # Backtrack if we need to go back to common ancestor
        while len(path) > common_len:
            prev = path[-2]
            popped = path.pop()
            path_set.discard(popped)
            if moves:
                moves.pop()
            if TOT_DEBUG:
                print(f"[ToT] BACKTRACK: {popped} -> {prev} (branch switch)")
            events.append({
                "type": "backtrack",
                "from_title": popped,
                "to_title": prev,
                "reason": "switching to better branch",
            })

        # Now path == committed_path[:common_len]. Get next step from best_path
        if common_len >= len(best_path):
            return events
        next_title = best_path[common_len]
        current = path[-1]

        # Get anchor text from a node that has it
        title_to_anchor = {}
        for nd in session.get("tot_nodes", {}).values():
            title_to_anchor.update(nd.get("title_to_anchor", {}))
        anchor_text = title_to_anchor.get(next_title, next_title)

        move = {
            "step": len(moves) + 1,
            "from_title": current,
            "to_title": next_title,
            "anchor_text": anchor_text,
            "analysis": "",
            "tot_score": None,
            "tot_alternatives": session.get("tot_k", TOT_K),
        }

        # Get score from nodes if available
        for nd in session.get("tot_nodes", {}).values():
            if nd.get("current") == next_title:
                move["tot_score"] = nd.get("score")
                break

        moves.append(move)
        path.append(next_title)
        path_set.add(next_title)
        tried_from = session.setdefault("tried_edges", {}).setdefault(current, set())
        tried_from.add(next_title)

        if TOT_DEBUG:
            print(f"[ToT] MOVE: {current} -> {next_title} (score={move.get('tot_score')})")

        if next_title == session["resolved_target"]:
            session["done"] = True
            session["success"] = True

        # Option B: Keep full frontier (no filtering by committed path)
        return events + [{"type": "move", "move": move}]

    def _commit_full_path(
        self,
        session: Dict[str, Any],
        committed_path: List[str],
        best_path: List[str],
        events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Commit the full path to the target (all remaining moves)."""
        result = events
        current_committed = list(committed_path)
        while len(current_committed) < len(best_path):
            step_events = self._commit_move(session, current_committed, best_path, [])
            result = result + step_events
            current_committed = list(session["path"])
            if session.get("done"):
                break
        return result

    def _rescore_frontier(
        self,
        session: Dict[str, Any],
        frontier: List[Tuple[float, int, str]],
        nodes: Dict[int, Dict[str, Any]],
        target: str,
    ) -> None:
        """Rescore every node in the frontier via LLM and rebuild the heap."""
        if not frontier:
            return
        # Build ordered list of (nid, nd) for prompt
        node_list: List[Tuple[int, Dict[str, Any]]] = []
        seen = set()
        for neg_score, nid, _ in frontier:
            if nid in seen:
                continue
            seen.add(nid)
            nd = nodes.get(nid)
            if nd:
                node_list.append((nid, nd))

        if not node_list:
            return

        lines = []
        for idx, (nid, nd) in enumerate(node_list):
            path = nd.get("path", [])
            current = nd.get("current", "?")
            path_str = " -> ".join(path) if len(path) > 1 else current
            lines.append(f"{idx}: {current} (path: {path_str})")

        user_prompt = (
            f"TARGET: {target}\n"
            f"TARGET_INTRO: {session.get('target_extract', '')}\n\n"
            f"NODES TO SCORE (current page and path):\n" + "\n".join(lines) + "\n\n"
            "Score each node 0-100. Return JSON: {\"scores\": [n0, n1, ...]}"
        )
        messages = [
            {"role": "system", "content": RESCORE_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ]

        # Rescore can have many nodes; allow longer timeout
        text = self.llm.chat(messages, max_tokens=2048, temperature=0.2, timeout_s=180)
        data = self._parse_score_json(text)
        scores = data.get("scores", [])
        if len(scores) < len(node_list):
            scores = scores + [50] * (len(node_list) - len(scores))
        scores = [max(0, min(100, int(s))) for s in scores[: len(node_list)]]

        for i, (nid, nd) in enumerate(node_list):
            if i < len(scores):
                nd["score"] = scores[i]

        # Rebuild frontier heap with new scores
        new_frontier = []
        for neg_score, nid, current in frontier:
            nd = nodes.get(nid)
            score = nd.get("score", 50) if nd else 50
            new_frontier.append((-score, nid, nd["current"] if nd else current))
        heapq.heapify(new_frontier)
        session["tot_frontier"] = new_frontier

        if TOT_DEBUG:
            print(f"[ToT] RESCORED {len(node_list)} nodes: ", end="")
            for (nid, nd), s in zip(node_list, scores):
                print(f"{nd['current']}={s} ", end="")
            print()

    def llm_pick_top_k(
        self,
        target_title: str,
        target_extract: str,
        candidate_titles: List[str],
        current_page: str,
        k: int,
    ) -> List[Tuple[str, int]]:
        """Ask the LLM to pick the best k candidates and score only those. Returns [(title, score), ...]."""
        if not candidate_titles or k <= 0:
            return []
        k = min(k, len(candidate_titles))

        lines = []
        for idx, title in enumerate(candidate_titles):
            flags = []
            if title.lower().startswith("list of "):
                flags.append("LIST")
            if "(disambiguation)" in title.lower():
                flags.append("DISAMBIG")
            suffix = f" [{', '.join(flags)}]" if flags else ""
            lines.append(f"{idx}: {title}{suffix}")

        user_prompt = (
            f"TARGET: {target_title}\n"
            f"TARGET_INTRO: {target_extract}\n\n"
            f"CURRENT PAGE (where these links are from): {current_page}\n\n"
            f"CANDIDATE PAGES ({len(candidate_titles)} total):\n" + "\n".join(lines) + "\n\n"
            f"Pick the best {k} candidates for reaching the target. Score each 0-100. "
            f'Return JSON: {{"picks": [{{"index\": <0-based>, \"score\": <0-100>}}, ...]}}'
        )
        messages = [
            {"role": "system", "content": PICK_TOP_K_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ]

        text = self.llm.chat(messages, max_tokens=512, temperature=0.2, timeout_s=LLM_TIMEOUT_S)
        data = self._parse_pick_json(text, candidate_titles, k)
        return data

    def llm_score_candidates(
        self,
        target_title: str,
        target_extract: str,
        candidate_titles: List[str],
        current_page: str,
        score_samples: int = 1,
    ) -> List[int]:
        """Score each candidate 0-100. Optionally sample n times and average."""
        lines = []
        for idx, title in enumerate(candidate_titles):
            flags = []
            if title.lower().startswith("list of "):
                flags.append("LIST")
            if "(disambiguation)" in title.lower():
                flags.append("DISAMBIG")
            suffix = f" [{', '.join(flags)}]" if flags else ""
            lines.append(f"{idx}: {title}{suffix}")

        user_prompt = (
            f"TARGET: {target_title}\n"
            f"TARGET_INTRO: {target_extract}\n\n"
            f"CURRENT PAGE (where these links are from): {current_page}\n\n"
            f"CANDIDATE PAGES:\n" + "\n".join(lines) + "\n\n"
            "Score each candidate 0-100. Return JSON: {\"scores\": [n0, n1, ...]}"
        )
        messages = [
            {"role": "system", "content": SCORE_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ]

        all_scores: List[List[int]] = []
        for _ in range(score_samples):
            text = self.llm.chat(messages, max_tokens=256, temperature=0.2, timeout_s=LLM_TIMEOUT_S)
            data = self._parse_score_json(text)
            scores = data.get("scores", [])
            if len(scores) < len(candidate_titles):
                scores = scores + [0] * (len(candidate_titles) - len(scores))
            scores = [max(0, min(100, int(s))) for s in scores[: len(candidate_titles)]]
            all_scores.append(scores)

        if score_samples == 1:
            return all_scores[0]
        # Average
        result = []
        for i in range(len(candidate_titles)):
            vals = [row[i] for row in all_scores if i < len(row)]
            result.append(round(sum(vals) / len(vals)) if vals else 50)
        return result

    @staticmethod
    def _parse_pick_json(
        output_text: str, candidate_titles: List[str], k: int
    ) -> List[Tuple[str, int]]:
        """Parse LLM pick response: picks with index and score. Returns [(title, score), ...]."""
        output_text = output_text.strip()
        output_text = re.sub(r"^```(?:json)?\s*", "", output_text)
        output_text = re.sub(r"\s*```\s*$", "", output_text)
        match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found. Raw: {output_text[:200]}")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON. Raw: {output_text[:200]}")
        picks = data.get("picks", [])
        result = []
        seen_indices = set()
        for p in picks[:k]:
            idx = p.get("index", -1)
            if not isinstance(idx, int):
                try:
                    idx = int(idx)
                except (ValueError, TypeError):
                    continue
            if idx < 0 or idx >= len(candidate_titles) or idx in seen_indices:
                continue
            seen_indices.add(idx)
            score = p.get("score", 50)
            if not isinstance(score, (int, float)):
                try:
                    score = int(score)
                except (ValueError, TypeError):
                    score = 50
            score = max(0, min(100, int(score)))
            result.append((candidate_titles[idx], score))
        return result

    @staticmethod
    def _parse_score_json(output_text: str) -> Dict[str, Any]:
        output_text = output_text.strip()
        # Remove markdown code blocks if present
        output_text = re.sub(r"^```(?:json)?\s*", "", output_text)
        output_text = re.sub(r"\s*```\s*$", "", output_text)
        match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if not match:
            # Fallback: try to extract scores array directly (handles truncated JSON)
            arr_match = re.search(r'"scores"\s*:\s*\[([0-9,\s]*)', output_text)
            if arr_match:
                nums = re.findall(r"\d+", arr_match.group(1))
                return {"scores": [int(n) for n in nums]}
            raise ValueError(f"No JSON found. Raw: {output_text[:200]}")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            # Truncated JSON: try to extract scores array
            arr_match = re.search(r'"scores"\s*:\s*\[([0-9,\s]*)', output_text)
            if arr_match:
                nums = re.findall(r"\d+", arr_match.group(1))
                return {"scores": [int(n) for n in nums]}
            raise
