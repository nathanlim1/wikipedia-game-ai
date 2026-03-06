import re
import time
from typing import Any, Dict, Optional, Set, Tuple, List
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup


class WikipediaClient:
    def __init__(
        self,
        api_url: str = "https://en.wikipedia.org/w/api.php",
        user_agent: str = "CalPolySLO CSC581 Wikipedia Game AI (nlim10@calpoly.edu)",
        timeout: int = 25,
        retries: int = 2,
    ):
        self.api_url = api_url
        self.user_agent = user_agent
        self.timeout = timeout
        self.retries = retries
        self.http = requests.Session()

    def page_exists(self, page_title: str) -> bool:
        try:
            return self._get_normalized_title(page_title) is not None
        except requests.RequestException:
            return False

    def resolve_title_exact(self, title: str) -> str:
        title = title.strip()
        if not title:
            raise ValueError("Empty title")

        data = self._wiki_get({"action": "query", "format": "json", "redirects": 1, "titles": title})
        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        if page.get("missing") is None and page.get("title"):
            return page["title"]
        raise ValueError(f"Wikipedia page not found (exact): {title}")

    def resolve_title_fuzzy_start(self, title: str) -> str:
        title = title.strip()
        if not title:
            raise ValueError("Empty title")
        try:
            return self.resolve_title_exact(title)
        except Exception:
            pass

        opensearch_data = self._wiki_get(
            {"action": "opensearch", "format": "json", "search": title, "limit": 1, "namespace": 0}
        )
        if isinstance(opensearch_data, list) and len(opensearch_data) >= 2 and opensearch_data[1]:
            return self.resolve_title_exact(opensearch_data[1][0])
        raise ValueError(f"Wikipedia page not found: {title}")

    def get_extract(self, title: str, max_chars: int = 650) -> str:
        data = self._wiki_get(
            {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": 1,
                "exintro": 1,
                "redirects": 1,
                "titles": title,
            }
        )
        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        extract = (page.get("extract") or "").strip()
        extract = re.sub(r"\s+", " ", extract)
        return extract[:max_chars]

    def get_visible_outgoing_links(self, title: str, max_total: int = 6000) -> Tuple[List[str], Dict[str, str]]:
        data = self._wiki_get({"action": "parse", "format": "json", "page": title, "prop": "text", "redirects": 1})
        if "error" in data:
            raise ValueError(f"Wikipedia parse error for '{title}': {data['error']}")

        html = data["parse"]["text"]["*"]
        soup = BeautifulSoup(html, "lxml")
        root = soup.find("div", class_="mw-parser-output") or soup

        for selector in [
            "div.navbox",
            "div.vertical-navbox",
            "table.navbox",
            "div.reflist",
            "ol.references",
            "div.mw-references-wrap",
            "div.catlinks",
            "div.toc",
            "span.mw-editsection",
            "sup.reference",
        ]:
            for node in root.select(selector):
                node.decompose()

        titles: List[str] = []
        seen = set()
        title_to_anchor: Dict[str, str] = {}

        for link in root.find_all("a", href=True):
            href = link["href"]
            if not href.startswith("/wiki/"):
                continue

            classes = link.get("class") or []
            if "new" in classes:
                continue

            slug = href.split("/wiki/", 1)[1].split("#", 1)[0]
            if not slug:
                continue

            if ":" in slug:
                continue

            destination_title = link.get("title")
            if not destination_title:
                destination_title = unquote(slug).replace("_", " ")
            destination_title = destination_title.strip()
            if not destination_title or destination_title == "Main Page":
                continue

            anchor_text = link.get_text(" ", strip=True)
            if not anchor_text:
                anchor_text = destination_title

            if destination_title not in seen:
                seen.add(destination_title)
                titles.append(destination_title)
                title_to_anchor[destination_title] = anchor_text

            if len(titles) >= max_total:
                break

        return titles, title_to_anchor

    def get_page_with_structure(
        self, title: str, max_total: int = 6000
    ) -> Dict[str, Any]:
        """Return page subheadings and links annotated with which section they appear in.

        Returns a dict with:
          subheadings  - ordered list of h2/h3 heading texts (wiki-navbox headings excluded)
          links        - ordered list of unique destination titles
          link_sections - mapping from destination title to the section heading it appears under
          anchor_map   - mapping from destination title to the anchor text of the first occurrence
        """
        data = self._wiki_get({"action": "parse", "format": "json", "page": title, "prop": "text", "redirects": 1})
        if "error" in data:
            raise ValueError(f"Wikipedia parse error for '{title}': {data['error']}")

        html = data["parse"]["text"]["*"]
        soup = BeautifulSoup(html, "lxml")
        root = soup.find("div", class_="mw-parser-output") or soup

        for selector in [
            "div.navbox",
            "div.vertical-navbox",
            "table.navbox",
            "div.reflist",
            "ol.references",
            "div.mw-references-wrap",
            "div.catlinks",
            "div.toc",
            "span.mw-editsection",
            "sup.reference",
        ]:
            for node in root.select(selector):
                node.decompose()

        subheadings: List[str] = []
        seen_headings: Set[str] = set()
        links: List[str] = []
        seen_links: Set[str] = set()
        link_sections: Dict[str, str] = {}
        anchor_map: Dict[str, str] = {}
        current_section = ""

        for node in root.descendants:
            if not hasattr(node, "name"):
                continue
            if node.name in ("h2", "h3"):
                heading_text = node.get_text(" ", strip=True)
                heading_text = re.sub(r"\[edit\].*$", "", heading_text).strip()
                if heading_text and heading_text not in seen_headings:
                    subheadings.append(heading_text)
                    seen_headings.add(heading_text)
                current_section = heading_text
                continue
            if node.name != "a":
                continue
            href = node.get("href", "")
            if not href.startswith("/wiki/"):
                continue
            classes = node.get("class") or []
            if "new" in classes:
                continue
            slug = href.split("/wiki/", 1)[1].split("#", 1)[0]
            if not slug or ":" in slug:
                continue
            destination_title = node.get("title")
            if not destination_title:
                destination_title = unquote(slug).replace("_", " ")
            destination_title = destination_title.strip()
            if not destination_title or destination_title == "Main Page":
                continue
            anchor_text = node.get_text(" ", strip=True) or destination_title
            if destination_title not in seen_links:
                seen_links.add(destination_title)
                links.append(destination_title)
                link_sections[destination_title] = current_section
                anchor_map[destination_title] = anchor_text
            if len(links) >= max_total:
                break

        return {
            "subheadings": subheadings,
            "links": links,
            "link_sections": link_sections,
            "anchor_map": anchor_map,
        }

    def get_page_id(self, title: str) -> int:
        """Resolve a page title to its numeric Wikipedia page ID."""
        title = title.strip()
        if not title:
            raise ValueError("Empty title")
        data = self._wiki_get(
            {"action": "query", "format": "json", "redirects": 1, "titles": title}
        )
        pages = data.get("query", {}).get("pages", {})
        for page_id_str, page_data in pages.items():
            pid = int(page_id_str)
            if pid > 0:
                return pid
        raise ValueError(f"Could not resolve page ID for: {title}")

    def get_titles_from_ids(self, page_ids: List[int]) -> Dict[int, str]:
        """Batch-resolve numeric page IDs to their titles (max 50 per request)."""
        result: Dict[int, str] = {}
        # MediaWiki API supports up to 50 IDs per request
        for i in range(0, len(page_ids), 50):
            batch = page_ids[i : i + 50]
            ids_str = "|".join(str(pid) for pid in batch)
            data = self._wiki_get(
                {"action": "query", "format": "json", "pageids": ids_str}
            )
            pages = data.get("query", {}).get("pages", {})
            for pid_str, page_data in pages.items():
                pid = int(pid_str)
                title = page_data.get("title")
                if pid > 0 and title:
                    result[pid] = title
        return result

    def get_links_from_page(self, page_title: str) -> Set[str]:
        normalized_title = self._get_normalized_title(page_title)
        if not normalized_title:
            raise ValueError(f"Page '{page_title}' does not exist")

        links = set()
        continue_token = None
        while True:
            batch_links, continue_token = self._fetch_links_batch(normalized_title, continue_token)
            links.update(batch_links)
            if continue_token is None:
                break
        return links

    def _wiki_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"User-Agent": self.user_agent}
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                response = self.http.get(self.api_url, params=params, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_err = exc
                time.sleep(0.45 * (attempt + 1))
        raise last_err  # type: ignore[misc]

    def _get_normalized_title(self, page_title: str) -> Optional[str]:
        params = {"action": "query", "format": "json", "titles": page_title, "redirects": 1}
        response = self._wiki_get(params)
        pages = response.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if int(page_id) > 0:
                return page_data.get("title")
        return None

    def _fetch_links_batch(self, page_title: str, continue_token: Optional[str] = None) -> tuple[Set[str], Optional[str]]:
        params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "links",
            "pllimit": "max",
            "plnamespace": 0,
        }
        if continue_token:
            params["plcontinue"] = continue_token
        response = self._wiki_get(params)
        links = set()
        pages = response.get("query", {}).get("pages", {})
        for _, page_data in pages.items():
            if "links" in page_data:
                for link in page_data["links"]:
                    links.add(link["title"])
        next_continue = response.get("continue", {}).get("plcontinue")
        return links, next_continue
