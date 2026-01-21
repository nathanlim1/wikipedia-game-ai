import requests
from typing import Set, Dict, Any, Optional


class WikipediaClient:
    def __init__(self, api_url: str = "https://en.wikipedia.org/w/api.php",
                 user_agent: str = "CalPolySLO CSC581 Wikipedia Game AI (nlim10@calpoly.edu)",
                 timeout: int = 10):
        self.api_url = api_url
        self.user_agent = user_agent
        self.timeout = timeout
    
    def get_links_from_page(self, page_title: str) -> Set[str]:
        """
        Get all links from a Wikipedia page, returning a set of all page titles that are linked.
        """
        # Check if the page exists and normalize the title
        normalized_title = self._get_normalized_title(page_title)
        if not normalized_title:
            raise ValueError(f"Page '{page_title}' does not exist")
        
        # Fetch all links from the page
        links = set()
        continue_token = None
        
        while True:
            # Batch the fetching of links
            batch_links, continue_token = self._fetch_links_batch(normalized_title, continue_token)
            links.update(batch_links)
            
            if continue_token is None:
                break
        
        return links
    
    def page_exists(self, page_title: str) -> bool:
        """
        Check if a Wikipedia page exists, returning True if it does, False otherwise.
        """
        try:
            return self._get_normalized_title(page_title) is not None
        except requests.RequestException:
            return False
    
    def _get_normalized_title(self, page_title: str) -> Optional[str]:
        """
        Verifies if a page exists, then returns the normalized title of that page.
        """
        params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "redirects": 1
        }
        
        try:
            response = self._make_api_request(params)
            pages = response.get("query", {}).get("pages", {})
            
            for page_id, page_data in pages.items():
                if int(page_id) > 0:  # Missing pages have negative page IDs
                    return page_data.get("title")
            
            return None
        except requests.RequestException as e:
            raise requests.RequestException(f"Error verifying page existence: {e}")
    
    def _fetch_links_batch(self, page_title: str, continue_token: Optional[str] = None) -> tuple[Set[str], Optional[str]]:
        """
        Fetch a batch of links from a Wikipedia page, handling pagination from the MediaWiki API.
        Takes a page title, an optional continuation token, and returns a set of the linked page 
        titles as well as the continuation token for the next batch (if more calls are needed).
        """
        params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "links",
            "pllimit": "max",  # Asking for the maximum number of links per request
            "plnamespace": 0,  # Only get links to main namespace (articles)
        }
        
        if continue_token:
            params["plcontinue"] = continue_token
        
        response = self._make_api_request(params)
        
        # Extract links from response
        links = set()
        pages = response.get("query", {}).get("pages", {})
        
        for page_id, page_data in pages.items():
            if "links" in page_data:
                for link in page_data["links"]:
                    links.add(link["title"])
        
        # Check if there are more results
        next_continue = response.get("continue", {}).get("plcontinue")
        
        return links, next_continue
    
    def _make_api_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a GET request to the MediaWiki API using a dictionary of parameters. Returns the JSON response.
        """
        headers = {
            "User-Agent": self.user_agent
        }
        
        try:
            response = requests.get(self.api_url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            raise requests.RequestException("Request to Wikipedia API timed out")
        except requests.RequestException as e:
            raise requests.RequestException(f"Error making API request: {e}")
