from __future__ import annotations

import json
import random
import re
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, List, Optional, Tuple

from src.tinker_llm import TinkerLLM
from src.wikipedia import WikipediaClient


CANDIDATE_POOL = 120
LLM_CHOICES = 28
LLM_TIMEOUT_S = 20

DECISION_INSTRUCTIONS = """Return ONLY valid JSON. No extra text.

Schema:
{
  "choice_index": <integer>,
  "analysis": "<2-4 sentences, destination-aware, explain bridge logic>"
}

Rules:
- choice_index must be one of the provided candidate indices.
- Make your analysis explicitly reference BOTH the current page AND the target page.
- Prefer moves that create a clear chain toward the target topic (characters/TV show/franchise/people/company/etc.).
- Avoid "List of ..." and "(disambiguation)" unless they are clearly a bridge.
"""


class DefaultAgent:
    agent_id = "default"

    def __init__(self, wiki_client: WikipediaClient, llm_client: TinkerLLM) -> None:
        self.wiki = wiki_client
        self.llm = llm_client

    def initialize_session(self, start_title: str, target_title: str) -> Dict[str, Any]:
        resolved_start = self.wiki.resolve_title_fuzzy_start(start_title)
        resolved_target = self.wiki.resolve_title_exact(target_title)
        target_extract = self.wiki.get_extract(resolved_target)
        target_keywords = self.target_keywords_from_extract(resolved_target, target_extract)

        return {
            "resolved_start": resolved_start,
            "resolved_target": resolved_target,
            "target_extract": target_extract,
            "target_keywords": target_keywords,
            "path": [resolved_start],
            "path_set": {resolved_start},
            "moves": [],
            "stack": [],
            "tried_edges": {},
            "done": False,
            "success": False,
            "failure_reason": "",
            "chain_builder": self.chain_string,
            "steps_builder": self.steps_text_from_moves,
        }

    def step(self, session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        target = session["resolved_target"]
        current = session["path"][-1]

        if current == target:
            session["done"] = True
            session["success"] = True
            return None

        if not session["stack"] or session["stack"][-1]["page"] != current:
            try:
                frame = self.make_frame(
                    page_title=current,
                    target_title=target,
                    target_keywords=session["target_keywords"],
                    path_set=session["path_set"],
                    tried_edges=session["tried_edges"],
                )
            except Exception as exc:
                if len(session["path"]) <= 1:
                    session["done"] = True
                    session["success"] = False
                    session["failure_reason"] = f"Wikipedia scrape failed at start: {exc}"
                    return None

                prev = session["path"][-2]
                popped = session["path"].pop()
                session["path_set"].remove(popped)
                if session["moves"]:
                    session["moves"].pop()
                if session["stack"] and session["stack"][-1].get("page") == popped:
                    session["stack"].pop()
                return {
                    "type": "backtrack",
                    "from_title": popped,
                    "to_title": prev,
                    "reason": f"scrape failed: {exc}",
                }

            session["stack"].append(frame)

        frame = session["stack"][-1]
        candidates = frame["candidates"]
        scores = frame["scores"]

        if not candidates:
            if len(session["path"]) <= 1:
                session["done"] = True
                session["success"] = False
                session["failure_reason"] = f"No usable outgoing links from '{current}'."
                return None

            prev = session["path"][-2]
            popped = session["path"].pop()
            session["path_set"].remove(popped)
            if session["moves"]:
                session["moves"].pop()
            session["stack"].pop()
            return {"type": "backtrack", "from_title": popped, "to_title": prev, "reason": "no candidates"}

        current_extract = self.wiki.get_extract(current)
        target_extract = session["target_extract"]

        llm_view = candidates[: min(LLM_CHOICES, len(candidates))]
        llm_scores = scores[: len(llm_view)]

        idx = None
        analysis = ""

        try:
            llm_idx, analysis = self.llm_choose(
                current_title=current,
                current_extract=current_extract,
                target_title=target,
                target_extract=target_extract,
                candidates=llm_view,
                candidate_scores=llm_scores,
                recent_path=session["path"][-8:],
            )
            if 0 <= llm_idx < len(llm_view):
                idx = llm_idx
        except FutureTimeoutError:
            idx = self.best_fallback_index(llm_scores)
            analysis = "(LLM timeout; heuristic fallback)"
        except Exception:
            idx = self.best_fallback_index(llm_scores)
            analysis = "(LLM error; heuristic fallback)"

        chosen = llm_view[idx] if idx is not None else llm_view[0]

        tried_from = session["tried_edges"].setdefault(current, set())
        if chosen in tried_from or chosen in session["path_set"]:
            chosen = None
            for cand in candidates:
                if cand not in tried_from and cand not in session["path_set"]:
                    chosen = cand
                    break
            if not chosen:
                if len(session["path"]) <= 1:
                    session["done"] = True
                    session["success"] = False
                    session["failure_reason"] = f"Exhausted all outgoing options from '{current}'."
                    return None

                prev = session["path"][-2]
                popped = session["path"].pop()
                session["path_set"].remove(popped)
                if session["moves"]:
                    session["moves"].pop()
                session["stack"].pop()
                return {"type": "backtrack", "from_title": popped, "to_title": prev, "reason": "exhausted options"}

        tried_from.add(chosen)
        anchor_text = frame["title_to_anchor"].get(chosen, chosen)

        move = {
            "step": len(session["moves"]) + 1,
            "from_title": current,
            "to_title": chosen,
            "anchor_text": anchor_text,
            "analysis": analysis,
        }
        session["moves"].append(move)
        session["path"].append(chosen)
        session["path_set"].add(chosen)

        if chosen == target:
            session["done"] = True
            session["success"] = True

        return {"type": "move", "move": move}

    @staticmethod
    def tokenize_simple(text: str) -> List[str]:
        return [tok for tok in re.split(r"[^a-zA-Z0-9]+", text.lower()) if tok]

    @staticmethod
    def title_penalty(title: str) -> float:
        penalty = 0.0
        lower_title = title.lower()
        if lower_title.startswith("list of "):
            penalty += 2.2
        if "(disambiguation)" in lower_title:
            penalty += 3.5
        if lower_title.startswith("outline of "):
            penalty += 2.2
        if lower_title.startswith("index of "):
            penalty += 2.2
        return penalty

    @classmethod
    def heuristic_score(cls, title: str, target_title: str, target_keywords: set[str]) -> float:
        title_tokens = set(cls.tokenize_simple(title))
        target_title_tokens = set(cls.tokenize_simple(target_title))
        overlap_title = len(title_tokens & target_title_tokens)
        overlap_keywords = len(title_tokens & target_keywords)
        contains_target = 2.0 if target_title.lower() in title.lower() else 0.0
        return (1.8 * overlap_title) + (1.0 * overlap_keywords) + contains_target - cls.title_penalty(title)

    @classmethod
    def build_candidate_list(
        cls,
        outgoing: List[str],
        title_to_anchor: Dict[str, str],
        target_title: str,
        target_keywords: set[str],
        path_set: set[str],
        tried_edges_from_current: set[Tuple[str]],
    ) -> List[str]:
        filtered = []
        for title in outgoing:
            if title in path_set:
                continue
            if title not in title_to_anchor:
                continue
            if (title,) in tried_edges_from_current:
                continue
            filtered.append(title)

        if not filtered:
            return []

        scored = sorted(
            filtered,
            key=lambda page_title: cls.heuristic_score(page_title, target_title, target_keywords),
            reverse=True,
        )

        pool = scored[: min(CANDIDATE_POOL, len(scored))]
        rest = scored[len(pool) :]
        if rest:
            pool += random.sample(rest, min(20, len(rest)))

        seen = set()
        output = []
        for title in pool:
            if title not in seen:
                seen.add(title)
                output.append(title)
            if len(output) >= CANDIDATE_POOL:
                break
        return output

    @classmethod
    def target_keywords_from_extract(cls, target_title: str, target_extract: str) -> set[str]:
        tokens = cls.tokenize_simple(target_title) + cls.tokenize_simple(target_extract)
        stopwords = {
            "the",
            "and",
            "of",
            "to",
            "in",
            "a",
            "an",
            "for",
            "on",
            "by",
            "with",
            "as",
            "at",
            "is",
            "was",
            "are",
            "from",
            "that",
            "it",
        }
        return {token for token in tokens if token not in stopwords and len(token) >= 3}

    def make_frame(
        self,
        page_title: str,
        target_title: str,
        target_keywords: set[str],
        path_set: set[str],
        tried_edges: Dict[str, set[str]],
    ) -> Dict[str, Any]:
        outgoing, title_to_anchor = self.wiki.get_visible_outgoing_links(page_title)
        tried_from = tried_edges.setdefault(page_title, set())
        candidates = self.build_candidate_list(
            outgoing=outgoing,
            title_to_anchor=title_to_anchor,
            target_title=target_title,
            target_keywords=target_keywords,
            path_set=path_set,
            tried_edges_from_current={(candidate,) for candidate in tried_from},
        )
        scores = [self.heuristic_score(candidate, target_title, target_keywords) for candidate in candidates]
        return {
            "page": page_title,
            "title_to_anchor": title_to_anchor,
            "candidates": candidates,
            "scores": scores,
        }

    @staticmethod
    def best_fallback_index(scores: List[float]) -> int:
        if not scores:
            return 0
        best_index = 0
        best_score = scores[0]
        for idx, score in enumerate(scores):
            if score > best_score:
                best_index = idx
                best_score = score
        return best_index

    @staticmethod
    def chain_string(start: str, moves: List[Dict[str, Any]]) -> str:
        chain = start
        for move in moves:
            chain += f' --["{move["anchor_text"]}"]--> {move["to_title"]}'
        return chain

    @staticmethod
    def steps_text_from_moves(start: str, target: str, moves: List[Dict[str, Any]]) -> str:
        lines = [f"Start: {start}"]
        for i, move in enumerate(moves, start=1):
            lines.append(f"Step {i}: {move['anchor_text']}")
        lines.append(f"Destination: {target}")
        return "\n".join(lines)

    @staticmethod
    def parse_json_first(output_text: str) -> Dict[str, Any]:
        output_text = output_text.strip()
        match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found. Raw: {output_text[:200]}")
        return json.loads(match.group(0))

    @classmethod
    def build_prompt(
        cls,
        current_title: str,
        current_extract: str,
        target_title: str,
        target_extract: str,
        candidates: List[str],
        candidate_scores: List[float],
        recent_path: List[str],
    ) -> List[Dict[str, str]]:
        lines = []
        for idx, title in enumerate(candidates):
            flags = []
            if title.lower().startswith("list of "):
                flags.append("LIST")
            if "(disambiguation)" in title.lower():
                flags.append("DISAMBIG")
            suffix = f" [{', '.join(flags)}]" if flags else ""
            lines.append(f"{idx}: {title}{suffix}  (score={candidate_scores[idx]:.2f})")

        user_prompt = (
            f"CURRENT: {current_title}\n"
            f"CURRENT_INTRO: {current_extract}\n\n"
            f"TARGET: {target_title}\n"
            f"TARGET_INTRO: {target_extract}\n\n"
            f"RECENT_PATH: {recent_path}\n\n"
            f"CANDIDATES:\n" + "\n".join(lines) + "\n\n"
            "Pick the best next hop and explain the bridge logic toward the target."
        )
        return [
            {"role": "system", "content": DECISION_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ]

    def llm_choose(
        self,
        current_title: str,
        current_extract: str,
        target_title: str,
        target_extract: str,
        candidates: List[str],
        candidate_scores: List[float],
        recent_path: List[str],
    ) -> Tuple[int, str]:
        messages = self.build_prompt(
            current_title=current_title,
            current_extract=current_extract,
            target_title=target_title,
            target_extract=target_extract,
            candidates=candidates,
            candidate_scores=candidate_scores,
            recent_path=recent_path,
        )
        text = self.llm.chat(messages, max_tokens=260, temperature=0.2, timeout_s=LLM_TIMEOUT_S)
        data = self.parse_json_first(text)
        idx = int(data["choice_index"])
        analysis = str(data.get("analysis", "")).strip()
        return idx, analysis
