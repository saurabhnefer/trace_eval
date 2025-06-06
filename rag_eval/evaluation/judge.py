"""
Evaluation module for the RAG evaluation pipeline.

This module handles the evaluation of generated answers using a comprehensive
evaluation rubric implemented by a GPT-4o judge.
"""

import re
import json
from typing import Dict, List, Optional

import openai
from langfuse import Langfuse

from ..config.prompts import EVAL_JUDGE_PROMPT

def parse_evaluation_results(evaluation_text: str) -> Dict:
    """
    Parse the evaluation results from the LLM judge's response.
    
    Handles both JSON format (preferred) and falls back to regex parsing for older format.
    
    Args:
        evaluation_text: The raw text response from the evaluation judge
        
    Returns:
        Structured dictionary with evaluation metrics and explanations
    """
    try:
        # Try to parse as JSON first
        results = json.loads(evaluation_text)
        
        # Extract the relevant information from the JSON structure
        return {
            'accuracy': {
                'judgment': results['accuracy_and_faithfulness_evaluation']['holistic_contextual_accuracy_and_faithfulness']['judgment'],
                'score': results['accuracy_and_faithfulness_evaluation']['holistic_contextual_accuracy_and_faithfulness']['score'],
                'reasoning': results['accuracy_and_faithfulness_evaluation']['holistic_contextual_accuracy_and_faithfulness']['explanation']
            },
            'relevance': {
                'judgment': results['relevance_evaluation']['answer_chunk_relevance']['judgment'],
                'score': results['relevance_evaluation']['answer_chunk_relevance']['score'],
                'reasoning': results['relevance_evaluation']['answer_chunk_relevance']['explanation']
            },
            'coherence': {
                'judgment': results['coherence_and_clarity_evaluation']['coherence_and_clarity']['judgment'],
                'score': results['coherence_and_clarity_evaluation']['coherence_and_clarity']['score'],
                'reasoning': results['coherence_and_clarity_evaluation']['coherence_and_clarity']['explanation']
            },
            'safety': {
                'user_query': results['safety_evaluation']['user_query'],
                'ai_response': results['safety_evaluation']['ai_response']
            },
            'business_context': results['accuracy_and_faithfulness_evaluation']['business_context_adherence'],
            'factual_accuracy': results['accuracy_and_faithfulness_evaluation']['factual_accuracy_world_knowledge'],
            'full_response': results
        }
    except json.JSONDecodeError:
        # Fall back to regex parsing for old format responses
        # Create empty results structure for the older format
        results = {
            'accuracy': {'judgment': None, 'score': None, 'reasoning': None},
            'relevance': {'judgment': None, 'score': None, 'reasoning': None},
            'coherence': {'judgment': None, 'score': None, 'reasoning': None}
        }
        
        # Extract accuracy information using regex patterns
        accuracy_judgment_match = re.search(r'Accuracy Judgment:\s*(\w+)', evaluation_text)
        accuracy_score_match = re.search(r'Accuracy Score \(0–10\):\s*(\d+)', evaluation_text)
        accuracy_reasoning_match = re.search(r'Accuracy Reasoning:\s*(.+?)(?=Relevance Judgment:|$)', evaluation_text, re.DOTALL)
        
        if accuracy_judgment_match:
            results['accuracy']['judgment'] = accuracy_judgment_match.group(1).strip()
        if accuracy_score_match:
            results['accuracy']['score'] = int(accuracy_score_match.group(1))
        if accuracy_reasoning_match:
            results['accuracy']['reasoning'] = accuracy_reasoning_match.group(1).strip()
        
        # Extract relevance information
        relevance_judgment_match = re.search(r'Relevance Judgment:\s*(\w+)', evaluation_text)
        relevance_score_match = re.search(r'Relevance Score \(0–10\):\s*(\d+)', evaluation_text)
        relevance_reasoning_match = re.search(r'Relevance Reasoning:\s*(.+?)(?=Coherence Judgment:|$)', evaluation_text, re.DOTALL)
        
        if relevance_judgment_match:
            results['relevance']['judgment'] = relevance_judgment_match.group(1).strip()
        if relevance_score_match:
            results['relevance']['score'] = int(relevance_score_match.group(1))
        if relevance_reasoning_match:
            results['relevance']['reasoning'] = relevance_reasoning_match.group(1).strip()
        
        # Extract coherence information
        coherence_judgment_match = re.search(r'Coherence Judgment:\s*(\w+)', evaluation_text)
        coherence_score_match = re.search(r'Coherence Score \(0–10\):\s*(\d+)', evaluation_text)
        coherence_reasoning_match = re.search(r'Coherence Reasoning:\s*(.+?)(?=---|$)', evaluation_text, re.DOTALL)
        
        if coherence_judgment_match:
            results['coherence']['judgment'] = coherence_judgment_match.group(1).strip()
        if coherence_score_match:
            results['coherence']['score'] = int(coherence_score_match.group(1))
        if coherence_reasoning_match:
            results['coherence']['reasoning'] = coherence_reasoning_match.group(1).strip()
        
        return results

def evaluate_using_rag_prompt(question: str, chunks: List[str], answer: str, 
                            openai_client: openai.OpenAI, langfuse: Optional[Langfuse] = None, 
                            trace_id: Optional[str] = None) -> Dict:
    """
    Evaluate the generated answer using the comprehensive RAG evaluation prompt.
    
    Submits the question, chunks, and answer to a GPT-4o judge for evaluation
    across multiple dimensions and logs all metrics to Langfuse.
    
    Args:
        question: The original question
        chunks: The retrieved chunks used for the answer
        answer: The generated answer to evaluate
        openai_client: OpenAI client instance
        langfuse: Langfuse instance for logging
        trace_id: Langfuse trace ID for linking spans
        
    Returns:
        Dictionary with detailed evaluation results across all dimensions
    """
    try:
        # Ensure we have exactly 6 chunks by padding with empty strings if necessary
        chunks = (chunks + [''] * 6)[:6]
        
        # Format the business context
        business_context = """
        Business Context:
        - Role Or Background: Aspiring entrepreneur
        - Annual Revenue: Pre-revenue
        - Primary Business Goal: Launch New Product
        - Business Stage: Ideation
        - Target Market: B2B
        - Primary Aspiration: Develop an innovative product/service
        """
        
        # Prepare the prompt by formatting it with the question, chunks, and answer
        prompt = EVAL_JUDGE_PROMPT.format(
            question=question,
            business_context=business_context,
            chunk1=chunks[0],
            chunk2=chunks[1],
            chunk3=chunks[2],
            chunk4=chunks[3],
            chunk5=chunks[4],
            chunk6=chunks[5],
            answer=answer
        )
        
        # Create a span for evaluation if using Langfuse
        eval_span = None
        if langfuse and trace_id:
            eval_span = langfuse.span(
                name="evaluation-generation",
                trace_id=trace_id,
                input={
                    "prompt": prompt,
                    "model": "gpt-4o"
                }
            )
        
        # Run evaluation using OpenAI client
        messages = [
            {"role": "system", "content": "You are an evaluation judge."},
            {"role": "user", "content": prompt}
        ]
        
        # Call OpenAI API with JSON output format
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,  # Use deterministic output
            response_format={"type": "json_object"}  # Ensure JSON response
        )
        
        # Extract the evaluation text from the response
        evaluation_text = response.choices[0].message.content
        
        # Log the generation to Langfuse if available
        if langfuse and trace_id:
            generation = langfuse.generation(
                name="evaluation-generation",
                trace_id=trace_id,
                model="gpt-4o",
                model_parameters={"temperature": 0},
                input=messages,
                output=evaluation_text
            )
            
            # Update the evaluation span with the output
            if eval_span:
                eval_span.update(output={"evaluation_text": evaluation_text})
                eval_span.end()  # Explicitly end the span
        
        # Parse the evaluation results into a structured format
        results = parse_evaluation_results(evaluation_text)
        
        # Log individual dimension scores to Langfuse as observations if available
        if langfuse and trace_id:
            # Core evaluation dimensions
            langfuse.score(
                name="accuracy",
                trace_id=trace_id,
                value=results['accuracy']['score'],
                comment=f"Judgment: {results['accuracy']['judgment']}\nExplanation: {results['accuracy']['reasoning']}",
                properties={"judgment": results['accuracy']['judgment']},
                expose_props=["judgment"]
            )
            
            langfuse.score(
                name="relevance",
                trace_id=trace_id,
                value=results['relevance']['score'],
                comment=f"Judgment: {results['relevance']['judgment']}\nExplanation: {results['relevance']['reasoning']}",
                properties={"judgment": results['relevance']['judgment']},
                expose_props=["judgment"]
            )
            
            langfuse.score(
                name="coherence",
                trace_id=trace_id,
                value=results['coherence']['score'],
                comment=f"Judgment: {results['coherence']['judgment']}\nExplanation: {results['coherence']['reasoning']}",
                properties={"judgment": results['coherence']['judgment']},
                expose_props=["judgment"]
            )
            
            # Safety evaluation dimensions
            langfuse.score(
                name="safety-user-jailbreak",
                trace_id=trace_id,
                value=results['safety']['user_query']['jailbreak_attempt']['score'],
                comment=f"Judgment: {results['safety']['user_query']['jailbreak_attempt']['judgment']}\nExplanation: {results['safety']['user_query']['jailbreak_attempt']['explanation']}",
                properties={"judgment": results['safety']['user_query']['jailbreak_attempt']['judgment']},
                expose_props=["judgment"]
            )
            
            langfuse.score(
                name="safety-user-toxicity",
                trace_id=trace_id,
                value=results['safety']['user_query']['toxicity']['score'],
                comment=f"Judgment: {results['safety']['user_query']['toxicity']['judgment']}\nExplanation: {results['safety']['user_query']['toxicity']['explanation']}",
                properties={"judgment": results['safety']['user_query']['toxicity']['judgment']},
                expose_props=["judgment"]
            )
            
            langfuse.score(
                name="safety-ai-jailbreak",
                trace_id=trace_id,
                value=results['safety']['ai_response']['jailbreak_success']['score'],
                comment=f"Judgment: {results['safety']['ai_response']['jailbreak_success']['judgment']}\nExplanation: {results['safety']['ai_response']['jailbreak_success']['explanation']}",
                properties={"judgment": results['safety']['ai_response']['jailbreak_success']['judgment']},
                expose_props=["judgment"]
            )
            
            langfuse.score(
                name="safety-ai-toxicity",
                trace_id=trace_id,
                value=results['safety']['ai_response']['toxicity']['score'],
                comment=f"Judgment: {results['safety']['ai_response']['toxicity']['judgment']}\nExplanation: {results['safety']['ai_response']['toxicity']['explanation']}",
                properties={"judgment": results['safety']['ai_response']['toxicity']['judgment']},
                expose_props=["judgment"]
            )
            
            # Additional evaluation dimensions
            langfuse.score(
                name="business-context",
                trace_id=trace_id,
                value=results['business_context']['score'],
                comment=f"Judgment: {results['business_context']['judgment']}\nExplanation: {results['business_context']['explanation']}",
                properties={"judgment": results['business_context']['judgment']},
                expose_props=["judgment"]
            )
            
            langfuse.score(
                name="factual-accuracy",
                trace_id=trace_id,
                value=results['factual_accuracy']['score'],
                comment=f"Judgment: {results['factual_accuracy']['judgment']}\nExplanation: {results['factual_accuracy']['explanation']}",
                properties={"judgment": results['factual_accuracy']['judgment']},
                expose_props=["judgment"]
            )
        
        return results
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        # Return empty results structure if evaluation fails
        return {
            'accuracy': {'judgment': None, 'score': 0, 'reasoning': f"Error: {str(e)}"},
            'relevance': {'judgment': None, 'score': 0, 'reasoning': f"Error: {str(e)}"},
            'coherence': {'judgment': None, 'score': 0, 'reasoning': f"Error: {str(e)}"},
            'safety': {
                'user_query': {'jailbreak_attempt': {'judgment': None, 'score': 0, 'explanation': f"Error: {str(e)}"},
                               'toxicity': {'judgment': None, 'score': 0, 'explanation': f"Error: {str(e)}"}},
                'ai_response': {'jailbreak_success': {'judgment': None, 'score': 0, 'explanation': f"Error: {str(e)}"},
                                'toxicity': {'judgment': None, 'score': 0, 'explanation': f"Error: {str(e)}"}}
            },
            'business_context': {'judgment': None, 'score': 0, 'explanation': f"Error: {str(e)}"},
            'factual_accuracy': {'judgment': None, 'score': 0, 'explanation': f"Error: {str(e)}"},
            'full_response': {}
        } 