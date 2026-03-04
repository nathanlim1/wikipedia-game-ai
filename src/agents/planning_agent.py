from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from src.retrieval import WikiPageIndex, build_page_index, load_models
from src.tinker_llm import TinkerLLM
from src.wikipedia import WikipediaClient


LLM_TIMEOUT_S = 30
MAX_SEARCH_ROUNDS = 3
FINAL_TOP_K = 10

HYPOTHESIZE_SYSTEM = """You are a strategic Wikipedia navigator.

Given the section headings of the CURRENT Wikipedia page and information about the TARGET article, generate explicit hypotheses about which sections or topics most likely bridge toward the target — then produce focused search queries to find those links.

Return ONLY valid JSON. No extra text outside the JSON object.

Schema:
{
  "hypotheses": [
    "<explicit hypothesis: e.g. 'The History section likely links to Political movements which connects to the target via shared ideology'>",
    "..."
  ],
  "search_queries": [
    "<specific keyword/concept to search for among this page's links>",
    "..."
  ]
}

Rules:
- 2–4 hypotheses. Each must name a SPECIFIC section or topic visible on this page AND explain WHY it leads toward the target.
- 2–4 search_queries. Each should be a short, targeted phrase (not a full sentence) — e.g. "Cold War diplomacy", "German philosophy", "semiconductor manufacturing".
- Vary the queries: some narrow (exact target name), some broad (topic domain).
- Prioritize domain bridging over surface-level name similarity.
"""

DECIDE_SYSTEM = """You are a strategic Wikipedia navigator choosing the next hop in a path to reach a target article.

Return ONLY valid JSON. No extra text outside the JSON object.

Schema:
{
  "chosen_link": "<exact title from the candidates list, or null>",
  "reasoning": "<2–3 sentences: reference both the current page and the target, explain the bridge logic>",
  "request_more_searches": false,
  "additional_queries": []
}

Rules:
- chosen_link must be an EXACT title string from the candidates list (case-sensitive), or null.
- Set request_more_searches=true only if none of the candidates are plausible bridges AND search rounds remain.
- If request_more_searches=true, provide 1–3 new queries in additional_queries.
- Avoid "List of …" and "(disambiguation)" pages unless they are clearly the only bridge.
- reasoning must reference BOTH the current page context AND the target article.
"""


class StepState(TypedDict):
    # ── Inputs (set before graph invocation) ───────────────────────────────
    current_page: str
    target_page: str
    target_extract: str
    path: List[str]
    path_set: set
    tried_from_current: set

    # ── Page structure (filled by inspect_page) ─────────────────────────────
    subheadings: List[str]
    links: List[str]
    link_sections: Dict[str, str]
    anchor_map: Dict[str, str]
    page_index: Optional[Any]  # WikiPageIndex; stored as Any for TypedDict compat

    # ── Hypotheses + queries (filled by hypothesize) ────────────────────────
    hypotheses: List[str]
    search_queries: List[str]

    # ── Search results (accumulated across rounds) ───────────────────────────
    search_results: List[Dict[str, Any]]
    candidate_pool: List[str]
    search_round: int
    retrieval_top_k: int

    # ── Decision (filled by decide) ──────────────────────────────────────────
    chosen_link: Optional[str]
    chosen_reasoning: str
    request_more_searches: bool
    additional_queries: List[str]


class PlanningAgent:
    agent_id = "planning"

    def __init__(self, wiki_client: WikipediaClient, llm_client: TinkerLLM) -> None:
        self.wiki = wiki_client
        self.llm = llm_client
        self._bi_encoder = None
        self._cross_encoder = None
        self._graph = None

    # ── Model lazy-loading ───────────────────────────────────────────────────

    def _ensure_models(self) -> None:
        if self._bi_encoder is None:
            print("PlanningAgent: loading retrieval models (first use)…")
            self._bi_encoder, self._cross_encoder = load_models()
            print("PlanningAgent: retrieval models ready.")

    # ── Graph construction ───────────────────────────────────────────────────

    def _get_graph(self):
        if self._graph is not None:
            return self._graph

        builder: StateGraph = StateGraph(StepState)
        builder.add_node("inspect_page", self._node_inspect_page)
        builder.add_node("hypothesize", self._node_hypothesize)
        builder.add_node("search", self._node_search)
        builder.add_node("decide", self._node_decide)

        builder.set_entry_point("inspect_page")
        builder.add_edge("inspect_page", "hypothesize")
        builder.add_edge("hypothesize", "search")
        builder.add_edge("search", "decide")
        builder.add_conditional_edges("decide", self._route_after_decide)

        self._graph = builder.compile()
        return self._graph

    @staticmethod
    def _route_after_decide(state: StepState) -> str:
        if state["request_more_searches"] and state["search_round"] < MAX_SEARCH_ROUNDS:
            return "search"
        return END

    # ── LangGraph nodes ──────────────────────────────────────────────────────

    def _node_inspect_page(self, state: StepState) -> dict:
        self._ensure_models()
        try:
            structure = self.wiki.get_page_with_structure(state["current_page"])
        except Exception as exc:
            return {
                "subheadings": [],
                "links": [],
                "link_sections": {},
                "anchor_map": {},
                "page_index": None,
            }

        page_index = build_page_index(
            structure,
            bi_encoder=self._bi_encoder,
            cross_encoder=self._cross_encoder,
        )
        return {
            "subheadings": structure["subheadings"],
            "links": structure["links"],
            "link_sections": structure["link_sections"],
            "anchor_map": structure["anchor_map"],
            "page_index": page_index,
        }

    def _node_hypothesize(self, state: StepState) -> dict:
        subheadings = state["subheadings"]
        current_page = state["current_page"]
        target_page = state["target_page"]
        target_extract = state["target_extract"]
        path = state["path"]

        sections_str = (
            " | ".join(subheadings[:20]) if subheadings else "(no section headings found)"
        )
        user_content = (
            f"CURRENT PAGE: {current_page}\n"
            f"PAGE SECTIONS: {sections_str}\n\n"
            f"TARGET: {target_page}\n"
            f"TARGET INTRO: {target_extract[:400]}\n\n"
            f"PATH SO FAR (last 6 hops): {' → '.join(path[-6:])}\n\n"
            "Generate hypotheses and search queries."
        )

        messages = [
            {"role": "system", "content": HYPOTHESIZE_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        hypotheses: List[str] = []
        search_queries: List[str] = []
        try:
            text = self.llm.chat(messages, max_tokens=400, temperature=0.3, timeout_s=LLM_TIMEOUT_S)
            data = _parse_json_first(text)
            hypotheses = [str(h) for h in data.get("hypotheses", []) if h]
            search_queries = [str(q) for q in data.get("search_queries", []) if q]
        except Exception:
            pass

        if not search_queries:
            search_queries = [target_page, f"{current_page} {target_page}"]

        return {"hypotheses": hypotheses, "search_queries": search_queries}

    def _node_search(self, state: StepState) -> dict:
        page_index: Optional[WikiPageIndex] = state["page_index"]
        if page_index is None:
            return {
                "search_results": state["search_results"],
                "candidate_pool": state["candidate_pool"],
                "search_round": state["search_round"] + 1,
            }

        tried_from = state["tried_from_current"]
        path_set = state["path_set"]

        queries = (
            state["search_queries"]
            if state["search_round"] == 0
            else state["additional_queries"]
        )

        new_results: List[Dict[str, Any]] = []
        new_candidates: List[str] = []

        if state["search_round"] == 0:
            target_hits = page_index.search(state["target_page"], k=state["retrieval_top_k"])
            valid_target_hits = [
                h for h in target_hits
                if h[0] not in tried_from and h[0] not in path_set
            ]
            new_results.append({"query": state["target_page"], "top_hits": valid_target_hits})
            for title, _, _ in valid_target_hits:
                if title not in new_candidates:
                    new_candidates.append(title)

        for query in queries:
            if not query:
                continue
            hits = page_index.search(query, k=state["retrieval_top_k"])
            valid_hits = [h for h in hits if h[0] not in tried_from and h[0] not in path_set]
            new_results.append({"query": query, "top_hits": valid_hits})
            for title, _, _ in valid_hits:
                if title not in new_candidates:
                    new_candidates.append(title)

        combined_pool = list(state["candidate_pool"])
        for title in new_candidates:
            if title not in combined_pool:
                combined_pool.append(title)

        return {
            "search_results": state["search_results"] + new_results,
            "candidate_pool": combined_pool,
            "search_round": state["search_round"] + 1,
        }

    def _node_decide(self, state: StepState) -> dict:
        candidate_pool = state["candidate_pool"]
        hypotheses = state["hypotheses"]
        target_page = state["target_page"]
        current_page = state["current_page"]
        link_sections = state["link_sections"]
        search_round = state["search_round"]

        if not candidate_pool:
            if search_round < MAX_SEARCH_ROUNDS:
                return {
                    "chosen_link": None,
                    "chosen_reasoning": "",
                    "request_more_searches": True,
                    "additional_queries": [
                        f"{target_page}",
                        f"{target_page} history origin",
                    ],
                }
            return {
                "chosen_link": None,
                "chosen_reasoning": "No viable candidates found after all search rounds.",
                "request_more_searches": False,
                "additional_queries": [],
            }

        candidate_lines = []
        for i, title in enumerate(candidate_pool[:30]):
            section = link_sections.get(title, "")
            suffix = f"  [section: {section}]" if section else ""
            candidate_lines.append(f"{i}: {title}{suffix}")

        hyp_text = (
            "\n".join(f"- {h}" for h in hypotheses)
            if hypotheses
            else "(no explicit hypotheses)"
        )
        user_content = (
            f"CURRENT PAGE: {current_page}\n"
            f"TARGET: {target_page}\n\n"
            f"HYPOTHESES:\n{hyp_text}\n\n"
            f"CANDIDATES (from retrieval search):\n"
            + "\n".join(candidate_lines)
            + "\n\nPick the best next hop toward the target."
        )

        messages = [
            {"role": "system", "content": DECIDE_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        chosen_link: Optional[str] = None
        reasoning = ""
        request_more = False
        additional_queries: List[str] = []

        try:
            text = self.llm.chat(messages, max_tokens=350, temperature=0.2, timeout_s=LLM_TIMEOUT_S)
            data = _parse_json_first(text)
            chosen_link = data.get("chosen_link") or None
            reasoning = str(data.get("reasoning", "")).strip()
            request_more = bool(data.get("request_more_searches", False))
            additional_queries = [str(q) for q in data.get("additional_queries", []) if q]

            if chosen_link and chosen_link not in state["links"]:
                chosen_link = None
        except Exception as exc:
            chosen_link = candidate_pool[0] if candidate_pool else None
            reasoning = f"(LLM error — picked top retrieval result as fallback: {exc})"

        if request_more and search_round >= MAX_SEARCH_ROUNDS:
            request_more = False
            if not chosen_link and candidate_pool:
                chosen_link = candidate_pool[0]
                reasoning = (
                    f"Max search rounds reached ({MAX_SEARCH_ROUNDS}); "
                    "selecting top-ranked candidate from pool."
                )

        return {
            "chosen_link": chosen_link,
            "chosen_reasoning": reasoning,
            "request_more_searches": request_more and bool(additional_queries),
            "additional_queries": additional_queries,
        }

    # ── Public GameAgent interface ────────────────────────────────────────────

    def initialize_session(self, start_title: str, target_title: str) -> Dict[str, Any]:
        resolved_start = self.wiki.resolve_title_fuzzy_start(start_title)
        resolved_target = self.wiki.resolve_title_exact(target_title)
        target_extract = self.wiki.get_extract(resolved_target)

        return {
            "resolved_start": resolved_start,
            "resolved_target": resolved_target,
            "target_extract": target_extract,
            "path": [resolved_start],
            "path_set": {resolved_start},
            "moves": [],
            "tried_edges": {},
            "done": False,
            "success": False,
            "failure_reason": "",
            "chain_builder": _chain_string,
            "steps_builder": _steps_text_from_moves,
        }

    def step(self, session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        target = session["resolved_target"]
        current = session["path"][-1]

        if current == target:
            session["done"] = True
            session["success"] = True
            return None

        tried_from_current: set = session["tried_edges"].get(current, set())

        initial_state: StepState = {
            "current_page": current,
            "target_page": target,
            "target_extract": session["target_extract"],
            "path": list(session["path"]),
            "path_set": set(session["path_set"]),
            "tried_from_current": set(tried_from_current),
            # page structure — filled by inspect_page
            "subheadings": [],
            "links": [],
            "link_sections": {},
            "anchor_map": {},
            "page_index": None,
            # hypotheses — filled by hypothesize
            "hypotheses": [],
            "search_queries": [],
            # search state — filled by search (accumulated)
            "search_results": [],
            "candidate_pool": [],
            "search_round": 0,
            "retrieval_top_k": session.get("retrieval_top_k", FINAL_TOP_K),
            # decision — filled by decide
            "chosen_link": None,
            "chosen_reasoning": "",
            "request_more_searches": False,
            "additional_queries": [],
        }

        final_state: StepState = self._get_graph().invoke(initial_state)

        chosen = final_state["chosen_link"]

        # Validate the LLM's choice is actually navigable
        if chosen and (chosen in tried_from_current or chosen in session["path_set"]):
            chosen = None

        # If graph produced no valid choice, try any untried candidate from pool
        if not chosen:
            for link in final_state["candidate_pool"]:
                if link not in tried_from_current and link not in session["path_set"]:
                    chosen = link
                    break

        # No viable next hop — backtrack
        if not chosen:
            if len(session["path"]) <= 1:
                session["done"] = True
                session["success"] = False
                session["failure_reason"] = (
                    f"No usable outgoing links from '{current}' after exhaustive search."
                )
                return None

            prev = session["path"][-2]
            popped = session["path"].pop()
            session["path_set"].discard(popped)
            if session["moves"]:
                session["moves"].pop()
            return {
                "type": "backtrack",
                "from_title": popped,
                "to_title": prev,
                "reason": "exhausted retrieval search options",
                "subheadings": final_state["subheadings"],
                "hypotheses": final_state["hypotheses"],
                "search_log": _format_search_log(final_state["search_results"]),
            }

        # Execute the move
        tried_edges_from = session["tried_edges"].setdefault(current, set())
        tried_edges_from.add(chosen)
        anchor_text = final_state["anchor_map"].get(chosen, chosen)

        move = {
            "step": len(session["moves"]) + 1,
            "from_title": current,
            "to_title": chosen,
            "anchor_text": anchor_text,
            "analysis": final_state["chosen_reasoning"],
        }
        session["moves"].append(move)
        session["path"].append(chosen)
        session["path_set"].add(chosen)

        if chosen == target:
            session["done"] = True
            session["success"] = True

        return {
            "type": "move",
            "move": move,
            "subheadings": final_state["subheadings"],
            "hypotheses": final_state["hypotheses"],
            "search_log": _format_search_log(final_state["search_results"]),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_first(text: str) -> Dict[str, Any]:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        print(f"ERROR: No JSON found. Raw: {text}")
        raise ValueError(f"No JSON found in LLM output: {text[:200]}")
    return json.loads(match.group(0))


def _format_search_log(search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "query": r["query"],
            "top_hits": [hit[0] for hit in r["top_hits"][:5]],
        }
        for r in search_results
    ]


def _chain_string(start: str, moves: List[Dict[str, Any]]) -> str:
    chain = start
    for move in moves:
        chain += f' --["{move["anchor_text"]}"]--> {move["to_title"]}'
    return chain


def _steps_text_from_moves(start: str, target: str, moves: List[Dict[str, Any]]) -> str:
    lines = [f"Start: {start}"]
    for i, move in enumerate(moves, start=1):
        lines.append(f"Step {i}: {move['anchor_text']}")
    lines.append(f"Destination: {target}")
    return "\n".join(lines)
