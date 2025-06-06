"""
Generation evaluator module for the RAG evaluation pipeline.

This module evaluates the performance of the generation component in the RAG system.
"""

import json
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
import time
import aiohttp
import asyncio

from ..config.settings import CHAT_API_URL, DEFAULT_BUSINESS_CONTEXT, OPENAI_API_KEY

logger = logging.getLogger(__name__)

def evaluate_generation(
    questions: List[str],
    expected_answers: Optional[List[str]] = None,
    conversation_ids: Optional[List[str]] = None,
    max_concurrent: int = 5,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Evaluate the generation performance for a list of questions.
    
    Args:
        questions: List of questions to evaluate
        expected_answers: Optional list of expected answers (same length as questions)
        conversation_ids: Optional list of conversation IDs (same length as questions)
        max_concurrent: Maximum number of concurrent API calls
        max_retries: Maximum number of retries for API calls
        retry_delay: Delay between retries in seconds
        
    Returns:
        List of generation results with evaluation metrics
    """
    # Prepare arguments for each question
    args_list = []
    for i, question in enumerate(questions):
        expected_answer = expected_answers[i] if expected_answers and i < len(expected_answers) else None
        conversation_id = conversation_ids[i] if conversation_ids and i < len(conversation_ids) else None
        
        args_list.append((question, expected_answer, conversation_id, max_retries, retry_delay))
    
    # Run evaluation concurrently
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        
        for args in args_list:
            task = asyncio.ensure_future(_evaluate_single_generation_async(semaphore, *args))
            tasks.append(task)
        
        results = loop.run_until_complete(asyncio.gather(*tasks))
    finally:
        loop.close()
    
    return results

async def _evaluate_single_generation_async(
    semaphore: asyncio.Semaphore,
    question: str,
    expected_answer: Optional[str] = None,
    conversation_id: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Dict[str, Any]:
    """
    Asynchronously evaluate the generation performance for a single question.
    
    Args:
        semaphore: Semaphore to limit concurrency
        question: Question to evaluate
        expected_answer: Optional expected answer
        conversation_id: Optional conversation ID
        max_retries: Maximum number of retries for API calls
        retry_delay: Delay between retries in seconds
        
    Returns:
        Generation result with evaluation metrics
    """
    async with semaphore:
        # Initialize result
        result = {
            "question": question,
            "expected_answer": expected_answer,
            "conversation_id": conversation_id,
            "generated_answer": None,
            "success": False,
            "error": None,
            "metrics": {
                "generation_time_ms": 0
            }
        }
        
        if not CHAT_API_URL:
            result["error"] = "Chat API URL is not set"
            return result
        
        # Prepare request payload
        payload = {
            "message": question,
            "businessContext": DEFAULT_BUSINESS_CONTEXT,
            "stream": False  # Disable streaming for evaluation
        }
        
        # Add conversation ID if provided
        if conversation_id:
            payload["conversationId"] = conversation_id
        
        # Call chat API with retries
        start_time = time.time()
        success = False
        response = None
        error = None
        
        headers = {
            "Content-Type": "application/json"
        }
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        CHAT_API_URL,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status == 200:
                            response_text = await response.text()
                            success = True
                            break
                        else:
                            error = f"API returned status code {response.status}"
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay)
            except Exception as e:
                error = str(e)
                logger.warning(f"Generation API call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
        
        end_time = time.time()
        generation_time_ms = (end_time - start_time) * 1000
        
        # Process response
        if success and response_text:
            try:
                # The API might return a stream or a JSON object
                # Try to parse as JSON first
                try:
                    response_data = json.loads(response_text)
                    generated_answer = response_data.get("response", "")
                except json.JSONDecodeError:
                    # If not JSON, assume it's the raw generated text
                    generated_answer = response_text.strip()
                
                # Calculate metrics
                result["generated_answer"] = generated_answer
                result["metrics"]["generation_time_ms"] = generation_time_ms
                result["success"] = True
                
                # If we have an expected answer, evaluate against it
                if expected_answer:
                    similarity = _calculate_text_similarity(generated_answer, expected_answer)
                    result["metrics"]["answer_similarity"] = similarity
                
            except Exception as e:
                error = f"Failed to parse API response: {e}"
                logger.error(error)
                result["error"] = error
        else:
            result["error"] = error or "Unknown error occurred"
        
        return result

def _calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple similarity between two texts.
    
    Note: This is a placeholder. In a real implementation, you might use:
    - Embedding similarity (e.g., cosine similarity of text embeddings)
    - ROUGE or BLEU scores
    - Model-based evaluation
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    # If OpenAI API key is available, use embedding similarity
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Get embeddings
            response1 = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text1
            )
            response2 = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text2
            )
            
            embedding1 = response1.data[0].embedding
            embedding2 = response2.data[0].embedding
            
            # Calculate cosine similarity
            import numpy as np
            
            def cosine_similarity(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            return cosine_similarity(embedding1, embedding2)
            
        except Exception as e:
            logger.warning(f"Failed to calculate embedding similarity: {e}")
            # Fall back to simple token overlap
            pass
    
    # Simple token overlap as fallback
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)