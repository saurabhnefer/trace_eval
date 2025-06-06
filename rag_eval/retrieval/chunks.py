"""
Chunk retrieval module for the RAG evaluation pipeline.

This module handles the retrieval of document chunks from the API
for a given query.
"""

from typing import List
import requests

from ..config.settings import SEARCH_API_URL

def get_chunks_from_api(question: str) -> List[str]:
    """
    Fetch relevant document chunks from the Startup India search API.
    
    Makes a POST request to the API with the query and returns the text chunks.
    
    Args:
        question: The search query to find relevant chunks
        
    Returns:
        List of text chunks retrieved from the API
    """
    payload = {
        "query": question,
        "cutoff": 0.0  # Include all chunks without filtering by relevance score
    }
    
    try:
        # Make the API request
        response = requests.post(SEARCH_API_URL, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        
        # Extract chunks if results are present
        if data.get('hasResults') and data.get('results'):
            # Extract just the text content from each chunk
            chunks = [result['text'] for result in data['results']]
            return chunks
        else:
            # Return empty list if no results
            return []
    except Exception as e:
        print(f"Error fetching chunks from API: {e}")
        return [] 