from src.wikipedia import WikipediaClient


def main():
    wiki = WikipediaClient()
    
    page_title = "Python (programming language)"
    
    if wiki.page_exists(page_title):
        print(f"Fetching links from '{page_title}'...")
        links = wiki.get_links_from_page(page_title)
        
        print(f"\nFound {len(links)} links:")
        for i, link in enumerate(links, 1):
            print(f"  {i}. {link}")
    else:
        print(f"{page_title} does not exist")


if __name__ == "__main__":
    main()