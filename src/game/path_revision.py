from __future__ import annotations

from typing import Any, Dict, List, Set


def revise_path(path: List[str], links_cache: Dict[str, Set[str]]) -> List[str]:
    """Greedy shortcut: from each page, jump to the furthest reachable page in the path.

    Example: A -> B -> C -> D -> Z with D in links(B) becomes A -> B -> D -> Z.
    """
    if len(path) <= 1:
        return path
    result: List[str] = [path[0]]
    i = 0
    while i < len(path) - 1:
        links_from_i = links_cache.get(path[i], set())
        # Find largest j > i such that path[j] is directly reachable from path[i]
        for j in range(len(path) - 1, i, -1):
            if path[j] in links_from_i:
                result.append(path[j])
                i = j
                break
        else:
            result.append(path[i + 1])
            i += 1
    return result


def moves_from_revised_path(
    revised_path: List[str],
    anchor_cache: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Build moves list from revised path using anchor_cache for anchor text."""
    moves: List[Dict[str, Any]] = []
    for step in range(1, len(revised_path)):
        from_title = revised_path[step - 1]
        to_title = revised_path[step]
        anchors = anchor_cache.get(from_title, {})
        anchor_text = anchors.get(to_title, to_title)
        moves.append({
            "step": step,
            "from_title": from_title,
            "to_title": to_title,
            "anchor_text": anchor_text,
            "analysis": "",
            "tot_score": None,
            "tot_alternatives": None,
        })
    return moves
