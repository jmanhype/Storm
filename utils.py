"""
Utility functions for fetching Wikipedia data.

This module provides helper functions for interacting with the Wikipedia API
to retrieve related links and table of contents for research topics.
"""
import logging
from typing import List
import requests
import dspy


def fetch_wikipedia_links(topic: str) -> List[str]:
    """
    Fetches links to related pages from a Wikipedia article.

    Args:
        topic: The topic to search for on Wikipedia.

    Returns:
        A list of related Wikipedia page titles. Returns empty list on error.

    Example:
        >>> links = fetch_wikipedia_links("Quantum Computing")
        >>> print(len(links))
        50
    """
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

def fetch_table_of_contents(topic: str) -> List[str]:
    """
    Fetches the table of contents for a Wikipedia page.

    Args:
        topic: The Wikipedia page title to fetch sections for.

    Returns:
        A list of section titles from the page. Returns empty list on error.

    Example:
        >>> sections = fetch_table_of_contents("Machine Learning")
        >>> print(sections[0])
        "Overview"
    """
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

