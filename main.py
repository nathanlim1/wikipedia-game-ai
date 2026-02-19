import re
from src.wikipedia import WikipediaClient
from src.tinker_llm import TinkerLLM

def play_wikipedia_game(start_page: str, target_page: str, max_hops: int = 10, model: str = None) -> list[str]:

    wiki = WikipediaClient()
    llm = TinkerLLM(model=model) if model else TinkerLLM()
    
    # Make sure the start and target pages actualy exist
    if not wiki.page_exists(start_page):
        raise ValueError(f"Start page '{start_page}' does not exist")
    if not wiki.page_exists(target_page):
        raise ValueError(f"Target page '{target_page}' does not exist")
    
    current_page = start_page
    path = [current_page]

    # Keep track of visited pages to avoid loops
    visited_pages = set()
    visited_pages.add(start_page)
    
    print(f"\n Wikipedia Game: {start_page} → {target_page}")
    print(f"   Max hops allowed: {max_hops}")
    print("=" * 60)
    
    for hop in range(max_hops):
        print(f"\n Step {hop + 1}: Currently on '{current_page}'")
        
        # Check if we've reached the target
        if current_page.lower() == target_page.lower():
            print(f"\nReached '{target_page}' in {hop} hops!")
            print(f"   Path: {' → '.join(path)}")
            return path
        
        # Get all links from current page
        links = wiki.get_links_from_page(current_page)

        # remove visited pages from links
        links = links - visited_pages
        
        # Check if target is directly reachable
        if target_page in links:
            print(f"   Target '{target_page}' found in links")
            current_page = target_page
            path.append(current_page)
            continue

        # Ask the LLM to choose the next link
        next_page = choose_next_link(llm, current_page, target_page, links, path)
        
        if next_page is None:
            print(f"\nFAILED: Could not find a valid link to follow")
            return path
        
        print(f"   LLM chose: '{next_page}'")
        current_page = next_page
        visited_pages.add(current_page)
        path.append(current_page)
    
    # Check final page
    if current_page.lower() == target_page.lower():
        print(f"\nReached '{target_page}' in {len(path) - 1} hops!")
    else:
        print(f"\nDid not reach '{target_page}' within {max_hops} hops")
    
    return path


def choose_next_link(llm: TinkerLLM, current_page: str, target_page: str, links: set[str], path: list[str]) -> str | None:
 
    links_list = sorted(links)
    
    # Filter out pages we've already visited to avoid loops
    available_links = [link for link in links_list if link not in path]
    
    if not available_links:
        print("   No unvisited links available!")
        return None
    
    # Limit links shown to LLM to avoid context overflow 
    # TODO UPDATE THIS IN THE FUTURE TO SHOW ALL THE LINKS ***********************************************!!!!!!!!!
    # score the links based on word overlap with the target page
    target_words = set(re.findall(r"[A-Za-z]+", target_page.lower()))

    def score(link):
        link_words = set(re.findall(r"[A-Za-z]+", link.lower()))
        return len(link_words & target_words)
    
    ranked_links = sorted(available_links, key=score, reverse=True)

    display_links = ranked_links[:200]
    links_text = "\n".join(display_links)
    
    # Prompt from the paper
    system_prompt = """You are playing the WikiGame. You must output ONLY the chosen page name in the exact format specified. Do not include any reasoning, explanation, or additional text. Follow the output format precisely."""
    
    user_prompt = f"""The WikiGame (also known as Wikirace, Wikispeedia, WikiGolf, or Wikipedia Speedrun) is a game where players must navigate from one Wikipedia page to another by clicking only internal links within the article body. The goal is to reach the target page using the fewest number of clicks or in the shortest time possible.

How to play:
A start page and an end page on Wikipedia are selected. These can be chosen randomly or decided by the players.
Starting from the Start_Node, you must click only on internal links found within the main body of the article to reach the End_Node.

Your task:
The user will provide a Start_Node and an End_Node and a List_Link_From_Start_Node, a list of page name linked from Start_Node.
You must make a unique choice with a page name from those proposed in List_Link_From_Start_Node, the page you choose must get you as close as possible from Start_Node to End_Node.
Make every time a choice to reach the End_Node.
Do not explain anything.
The only output should be:
- A line containing ###
- The unique page name choice, only one from the list List_Link_From_Start_Node
- A final line containing @@@

Start_Node: {current_page}
End_Node: {target_page}
List_Link_From_Start_Node:
{links_text}

Expected output format:
###
Page_Name_Choice
@@@

Very Important Instruction:
- Write only the page titles choice.
- You must choice the page from the list List_Link_From_Start_Node
- Do not include any reasoning or explanation.
- Start your output with ### on a line by itself.
- After the page name choice write a last line with @@@
- Don't write the same page name of the Start_Node, you will lose.
- Don't write a page name that not is in the List_Link_From_Start_Node
- Don't change the case of page name, write in the same way is in the List_Link_From_Start_Node"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Get LLM's choice with stop sequences to enforce format
    # Stop after @@@ to prevent extra output
    stop_sequences = ["@@@", "\n@@@", "\n\n"]
    response = llm.chat(messages, max_tokens=128, temperature=0.0, stop_sequences=stop_sequences)
    
    print(f"   LLM response: '{response[:200]}...'")
    
    # Parse the expected format: ###\nPageName\n@@@
    # Look for content between ### and @@@
    if '###' in response and '@@@' in response:
        parts = response.split('###')
        if len(parts) > 1:
            middle = parts[1].split('@@@')[0].strip()
            if middle:
                response = middle
    
    # Make sure the choice exists in available links
    if response in available_links:
        return response
    
    print(f"   LLM response '{response}' not found in links, picking first available")
    return available_links[0] if available_links else None


def main():
 
    start = "Python (programming language)"
    target = "Holland"
    
    path = play_wikipedia_game(start, target, max_hops=15)
    
    print(f"Final Stats:")
    print(f"   Total hops: {len(path) - 1}")
    print(f"   Path length: {len(path)} pages")


if __name__ == "__main__":
    main()
