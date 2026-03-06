from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

DEFAULT_WIKIPATH_URL = "https://wikipath.dobbel.dev"


class WikipathClient:
    """Thin wrapper around the Wikipath REST API.

    The client discovers available databases on first use, picks the newest
    English Wikipedia database, and exposes a method to fetch the true
    shortest path between two Wikipedia page IDs.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 15,
    ) -> None:
        self.base_url = (base_url or os.getenv("WIKIPATH_URL", DEFAULT_WIKIPATH_URL)).rstrip("/")
        self.timeout = timeout
        self._http = requests.Session()
        self._database: Optional[Dict[str, str]] = None  # cached {languageCode, dateCode}

    def list_databases(self) -> List[Dict[str, str]]:
        """Return the list of databases served by the Wikipath instance."""
        url = f"{self.base_url}/api/list_databases"
        resp = self._http.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_english_database(self) -> Optional[Dict[str, str]]:
        """Return metadata for the newest English database, or *None*."""
        if self._database is not None:
            return self._database

        try:
            databases = self.list_databases()
        except Exception as exc:
            log.warning("Wikipath list_databases failed: %s", exc)
            return None

        # Pick the English DB with the most recent date code.
        en_dbs = [db for db in databases if db.get("languageCode") == "en"]
        if not en_dbs:
            log.warning("No English database found on Wikipath instance at %s", self.base_url)
            return None

        best = max(en_dbs, key=lambda d: d.get("dateCode", ""))
        self._database = best
        return best

    def get_shortest_path(
        self,
        source_id: int,
        target_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Query Wikipath for the shortest path between two page IDs.

        Returns a dict with keys ``length``, ``count``, and ``path_ids``
        (a single concrete shortest path as a list of page IDs), or *None*
        if the query fails for any reason.
        """
        db = self.get_english_database()
        if db is None:
            return None

        url = f"{self.base_url}/api/shortest_paths"
        params = {
            "language-code": db["languageCode"],
            "date-code": db["dateCode"],
            "source": source_id,
            "target": target_id,
        }

        try:
            resp = self._http.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("Wikipath shortest_paths request failed: %s", exc)
            return None

        length: int = data.get("length", 0)
        count: int = data.get("count", 0)
        links: Dict[str, List[int]] = data.get("links", {})
        source = data.get("source", source_id)
        target = data.get("target", target_id)

        # Reconstruct one concrete path from the DAG of all shortest paths.
        path_ids = self._extract_one_path(source, target, links)

        return {
            "length": length,
            "count": count,
            "path_ids": path_ids,
        }

    @staticmethod
    def _extract_one_path(
        source: int,
        target: int,
        links: Dict[str, List[int]],
    ) -> List[int]:
        """Extract one path from source to target using DFS on the links DAG.
        
        The links dict maps each page ID (as string) to a list of neighbor page IDs
        that are on shortest paths. We do a simple DFS to find a path from source to target.
        """
        if source == target:
            return [source]
        
        if not links:
            return [source]
        
        # Simple DFS to find a path from source to target
        def dfs(current: int, target: int, visited: set, path: list) -> Optional[List[int]]:
            if current == target:
                return path + [current]
            if current in visited:
                return None
            visited.add(current)
            neighbors = links.get(str(current), [])
            for neighbor in neighbors:
                result = dfs(neighbor, target, visited, path + [current])
                if result is not None:
                    return result
            visited.remove(current)
            return None
        
        # Try to find a path from source to target
        result = dfs(source, target, set(), [])
        if result is not None:
            return result
        
        # Fallback: if DFS fails, just walk from source until we can't go further
        # (this shouldn't happen if the API response is correct, but handle it gracefully)
        path = [source]
        current = source
        visited = {source}
        max_depth = 20  # safety limit
        depth = 0
        while depth < max_depth and current != target:
            neighbors = links.get(str(current), [])
            if not neighbors:
                break
            next_node = None
            for n in neighbors:
                if n not in visited:
                    next_node = n
                    break
            if next_node is None:
                break
            visited.add(next_node)
            path.append(next_node)
            current = next_node
            depth += 1
            if current == target:
                break
        
        return path
