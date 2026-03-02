# wikipedia-game-ai
Quarter-long project for Cal Poly SLO CSC581: Computer Support for Knowledge Management.

An AI system that autonomously plays the Wikipedia game, navigating from a start article to a target article by clicking links. Driven purely by LLM-based reasoning.

## Structure

```
src/
  agents/         # game agents (default heuristic, LangGraph planning)
  game/           # game runner and session management
  web/            # FastAPI app and routes
  retrieval.py    # BM25 + bi-encoder + cross-encoder link retrieval
  wikipedia.py    # Wikipedia API/HTML client
  tinker_llm.py   # LLM client wrapper
main.py           # entrypoint
```

## Running

```bash
pip install -r requirements.txt
python main.py
```

Opens at `http://127.0.0.1:8000` by default.
