from googleapiclient.discovery import build
from config import config # Assuming you have config.py in the root

def perform_web_search(query, num_results=3):
    """
    Performs a web search using the Google Custom Search JSON API.

    Args:
        query (str): The search query.
        num_results (int): The number of search results to return.

    Returns:
        str: A formatted string of search results, or an error message.
    """
    try:
        # Check for necessary API keys and IDs in the config
        if not config.GOOGLE_API_KEY or not config.GOOGLE_CSE_ID:
            return "Web search is not configured. Missing Google API Key or CSE ID."

        # Build the custom search service
        service = build("customsearch", "v1", developerKey=config.GOOGLE_API_KEY)
        
        # Execute the search query
        res = service.cse().list(
            q=query,
            cx=config.GOOGLE_CSE_ID,
            num=num_results
        ).execute()

        # Format the results
        items = res.get('items', [])
        if not items:
            return "No relevant search results found from the web."

        # Extract title, link, and snippet for each result
        search_results = []
        for item in items:
            title = item.get('title', 'No Title')
            link = item.get('link', '#')
            snippet = item.get('snippet', 'No Snippet').replace('\n', ' ')
            search_results.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}")

        return "\n\n---\n\n".join(search_results)

    except Exception as e:
        print(f"Error during web search: {e}")
        # Provide a user-friendly error message
        return "Sorry, I couldn't perform a web search at the moment due to a configuration issue."

