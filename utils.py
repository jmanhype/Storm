import logging
import requests
import dspy

def fetch_wikipedia_links(topic):
    """Fetches links to related pages from a Wikipedia article."""
    endpoint = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": topic.replace(' ', '_'),
        "prop": "links",
        "pllimit": "max"
    }
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()  # Raises stored HTTPError, if one occurred.
        data = response.json()
        pages = next(iter(data['query']['pages'].values()))
        links = [link['title'] for link in pages.get('links', []) if 'ns' in link and link['ns'] == 0]
        logging.info(f"Fetched {len(links)} Wikipedia links for topic '{topic}'.")
        return links
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request failed: {e}")
        return []

def fetch_table_of_contents(topic):
    """Fetches the table of contents for a Wikipedia page."""
    endpoint = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": topic.replace(' ', '_'),
        "prop": "sections",
        "format": "json"
    }
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        if 'parse' in data:
            sections = [section['line'] for section in data['parse']['sections']]
            logging.info(f"Fetched Table of Contents for '{topic}': {sections}")
            return sections
        else:
            logging.warning(f"No 'parse' key in the response for topic '{topic}'.")
            return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching table of contents: {e}")
        return []

