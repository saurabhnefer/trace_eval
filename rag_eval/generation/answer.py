"""
Answer generation module for the RAG evaluation pipeline.

This module handles the generation of answers based on retrieved document chunks
and user queries.
"""

import json
import re
import requests
from typing import Dict, List, Optional
from datetime import datetime

from langfuse import Langfuse

from ..config.prompts import ANSWER_GENERATION_PROMPT
from ..config.settings import CHAT_API_URL

def prepare_prompt(question: str, chunks: List[str], business_context: Optional[Dict] = None) -> str:
    """
    Prepare the prompt for answer generation based on retrieved chunks.
    
    Formats the prompt to instruct the LLM how to handle different question types,
    how to use provided chunks, and how to format the response.
    
    Args:
        question: The user's question
        chunks: List of retrieved document chunks
        business_context: Optional business context information
        
    Returns:
        Formatted prompt string for the LLM
    """
    chunks_text = []
    for chunk in chunks:
        chunks_text.append(f"[Startup Founders]\n{chunk}")

    # Format business context if available
    memory_context = ""
    if business_context:
        memory_context = f"""
    Business Context:
    - Role Or Background: {business_context.get('roleOrBackground', 'Aspiring entrepreneur')}
    - Annual Revenue: {business_context.get('annualRevenue', 'Pre-revenue')}
    - Primary Business Goal: {business_context.get('primaryBusinessGoal', 'Launch New Product')}
    - Business Stage: {business_context.get('businessStage', 'Ideation')}
    - Target Market: {business_context.get('targetMarket', 'B2B')}
    - Primary Aspiration: {business_context.get('primaryAspiration', 'Develop an innovative product/service')}"""

    # Format the prompt with the chunks and question
    prompt = ANSWER_GENERATION_PROMPT.format(
        memory_context=memory_context if business_context else '',
        question=question,
        chunks_text="\n".join(chunks_text)
    )
    
    return prompt

async def generate_answer(question: str, chunks: List[str], business_context: Optional[Dict] = None, langfuse: Optional[Langfuse] = None, trace_id: Optional[str] = None) -> str:
    """
    Generate an answer using the coach service API with Langfuse tracing.
    
    This function:
    1. Seeds the chunks to the API
    2. Makes a streaming request to the chat endpoint
    3. Collects the streamed response
    4. Logs all relevant information to Langfuse
    
    Args:
        question: The user's question
        chunks: List of retrieved chunks to use for the answer
        business_context: Optional business context
        langfuse: Langfuse instance for logging
        trace_id: Langfuse trace ID for linking spans
        
    Returns:
        Generated answer as a string, or None if generation failed
    """
    try:
        # Use Langfuse for tracing if provided
        if langfuse and trace_id:
            # Create a span for the chunk retrieval (not using context manager)
            chunk_span = langfuse.span(
                name="chunk-retrieval",
                trace_id=trace_id,
                input={"question": question},
                output={"chunks": chunks, "count": len(chunks)}
            )
            chunk_span.end()  # Explicitly end the span
        
        # Format chunks for the API
        formatted_chunks = []
        if chunks and chunks[0] != "No relevant content found in knowledge base.":
            for chunk in chunks:
                formatted_chunks.append({
                    "title": "Startup Founders",
                    "url": "",
                    "date": datetime.now().isoformat(),
                    "text": chunk,
                    "source": "Startup India Content"
                })
        
        # Create a span for the LLM call if using Langfuse
        generation_span = None
        if langfuse and trace_id:
            generation_span = langfuse.span(
                name="llm-generation",
                trace_id=trace_id,
                input={
                    "question": question,
                    "chunks": chunks,
                    "business_context": business_context
                }
            )
        
        # Call the coach service API
        try:
            # First, seed the content via the transcribe-search endpoint to make it available
            if formatted_chunks:
                seed_url = "https://coachs-production.onrender.com/service/test/transcribe-search"
                seed_payload = {
                    "query": question,
                    "results": formatted_chunks  # Send the formatted chunks to be used
                }
                
                seed_response = requests.post(seed_url, json=seed_payload)
                if not seed_response.ok:
                    print(f"Failed to seed chunks: {seed_response.status_code} - {seed_response.text}")
            
            # Set up headers for server-sent events
            headers = {
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
            
            # Parameters for the request
            params = {
                "message": question,
                "userId": "test_user",
                "chat_id": "test_chat_" + datetime.now().strftime("%Y%m%d%H%M%S")
            }
            
            # Include business context if provided
            if business_context:
                params["businessContext"] = json.dumps(business_context)
            
            # Make the request to the streaming API
            response = requests.get(CHAT_API_URL, headers=headers, params=params, stream=True)
            
            if not response.ok:
                error_msg = f"Error from API: {response.status_code} - {response.text}"
                print(error_msg)
                if generation_span:
                    generation_span.update(output={"error": error_msg})
                    generation_span.end()  # End the span before returning
                return None
            
            # Process the streaming response
            answer = ""
            try:
                # Iterate through the streaming response lines
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            try:
                                # Parse the JSON data from the stream
                                data = json.loads(line[6:])  # Skip the 'data: ' prefix
                                if data.get('type') == 'GPT' and data.get('content'):
                                    # Accumulate content from the stream
                                    answer += data.get('content')
                                    # Print progress indicator
                                    print(".", end="", flush=True)
                                elif data.get('type') == 'error':
                                    error_msg = f"API error: {data.get('content')}"
                                    print(f"\n{error_msg}")
                            except json.JSONDecodeError as json_err:
                                error_msg = f"Failed to parse JSON from line: {line}"
                                print(f"\n{error_msg}")
                        elif line == 'event: complete':
                            print("\nStream completed")
                            break
            except Exception as stream_err:
                error_msg = f"Error processing stream: {str(stream_err)}"
                print(f"\n{error_msg}")
                if not answer:  # Only return None if we haven't collected any answer yet
                    if generation_span:
                        generation_span.update(output={"error": error_msg})
                        generation_span.end()  # End the span before returning
                    return None
            
            # Log the generation in Langfuse
            if langfuse and trace_id:
                generation = langfuse.generation(
                    name="answer-generation",
                    trace_id=trace_id,
                    model="coach-service",
                    model_parameters={},
                    input={
                        "question": question,
                        "chunks": chunks,
                        "business_context": business_context
                    },
                    output=answer.strip()
                )
                
                # Update the span with the output
                if generation_span:
                    generation_span.update(output={"answer": answer.strip()})
                    generation_span.end()  # Explicitly end the span
            
            return answer.strip()
            
        except Exception as api_err:
            error_msg = f"Error calling coach API: {str(api_err)}"
            print(error_msg)
            if generation_span:
                generation_span.update(output={"error": error_msg})
                generation_span.end()  # End the span before returning
            return None
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return None

def remove_thinking_sections(text: str) -> str:
    """
    Remove any <think> or <thinking> sections from the generated text.
    
    Cleans up the response to remove any thinking/reasoning sections
    that might be included in the model's output.
    
    Args:
        text: Raw text from the model
        
    Returns:
        Cleaned text with thinking sections removed
    """
    # Remove <think>...</think> tags and their content
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also try to catch other variations like <thinking>...</thinking>
    cleaned_text = re.sub(r'<thinking>.*?</thinking>', '', cleaned_text, flags=re.DOTALL)
    # Remove any empty lines that might be left after removing sections
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    # Strip extra whitespace
    cleaned_text = cleaned_text.strip()
    return cleaned_text 