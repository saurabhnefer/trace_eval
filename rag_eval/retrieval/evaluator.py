"""
Retrieval evaluator module for the RAG evaluation pipeline.

This module evaluates the performance of the retrieval component in the RAG system.
"""

import json
import logging
import requests
from typing import Dict, List, Any, Optional
import time

from ..config.settings import SEARCH_API_URL

logger = logging.getLogger(__name__)

def evaluate_retrieval(
    questions: List[str],
    conversation_ids: Optional[List[str]] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Evaluate the retrieval performance for a list of questions.
    
    Args:
        questions: List of questions to evaluate
        conversation_ids: Optional list of conversation IDs (same length as questions)
        max_retries: Maximum number of retries for API calls
        retry_delay: Delay between retries in seconds
        
    Returns:
        List of retrieval results with evaluation metrics
    """
    results = []
    
    for i, question in enumerate(questions):
        conversation_id = conversation_ids[i] if conversation_ids and i < len(conversation_ids) else None
        
        # Evaluate single question
        result = evaluate_single_retrieval(
            question=question,
            conversation_id=conversation_id,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        results.append(result)
    
    return results

def evaluate_single_retrieval(
    question: str,
    conversation_id: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Dict[str, Any]:
    """
    Evaluate the retrieval performance for a single question.
    
    Args:
        question: Question to evaluate
        conversation_id: Optional conversation ID
        max_retries: Maximum number of retries for API calls
        retry_delay: Delay between retries in seconds
        
    Returns:
        Retrieval result with evaluation metrics
    """
    if not SEARCH_API_URL:
        logger.error("Search API URL is not set")
        return {
            "question": question,
            "conversation_id": conversation_id,
            "error": "Search API URL is not set",
            "success": False
        }
    
    # Initialize result
    result = {
        "question": question,
        "conversation_id": conversation_id,
        "retrieved_documents": [],
        "success": False,
        "error": None,
        "metrics": {
            "total_documents": 0,
            "retrieval_time_ms": 0
        }
    }
    
    # Prepare request payload
    payload = {
        "query": question
    }
    
    # Add conversation ID if provided
    if conversation_id:
        payload["conversation_id"] = conversation_id
    
    # Call search API with retries
    start_time = time.time()
    success = False
    response = None
    error = None
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                SEARCH_API_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            success = True
            break
        except requests.exceptions.RequestException as e:
            error = str(e)
            logger.warning(f"Retrieval API call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    end_time = time.time()
    retrieval_time_ms = (end_time - start_time) * 1000
    
    # Process response
    if success and response:
        try:
            response_data = response.json()
            
            # Extract retrieved documents
            documents = response_data.get("documents", [])
            
            # Calculate metrics
            result["retrieved_documents"] = documents
            result["metrics"]["total_documents"] = len(documents)
            result["metrics"]["retrieval_time_ms"] = retrieval_time_ms
            result["success"] = True
            
        except Exception as e:
            error = f"Failed to parse API response: {e}"
            logger.error(error)
            result["error"] = error
    else:
        result["error"] = error or "Unknown error occurred"
    
    return result 