# wikipedia-game-ai

Quarter-long project for Cal Poly SLO CSC581: Computer Support for Knowledge Management.

An AI system that autonomously plays the Wikipedia game, navigating from a start article to a target article by clicking links. Driven purely by LLM-based reasoning.

## Structure

```
src/
  agents/         # game agents: default (heuristic), planning (LangGraph), tot (Tree of Thought)
  game/           # game runner, session store, path revision
  web/            # FastAPI app, routes, UI, model manager
  retrieval.py    # BM25 + bi-encoder + cross-encoder link retrieval
  wikipedia.py    # Wikipedia API/HTML client
  tinker_llm.py   # LLM client wrapper
  llm_timing.py   # LLM timing wrapper for evaluation
  wikipath_client.py  # Wikipath API for optimal path lengths
main.py           # entrypoint
```

## Running

Create a `.env` file in the project root with your Tinker API key:

```
TINKER_API_KEY=your_api_key_here
```

Then:

```bash
pip install -r requirements.txt
python main.py
```

Opens at `http://127.0.0.1:8000` by default.

## Evaluation

- `evaluation/run_eval_harness.py` — full evaluation across paths, models, and agents
- `evaluation/run_llm_sweep.py` — sweep over LLM choice counts for the default agent
- `evaluation/results/` — EVALUATION_REPORT.md, HUMAN_COMPARISON.md, aggregated statistics, compiled runs
- `evaluation/human_results/` — human baseline data

## Project Deliverables

- **[Final Paper.pdf](Final%20Paper.pdf)** — full project write-up
- **[Poster.pdf](Poster.pdf)** — conference-style poster
