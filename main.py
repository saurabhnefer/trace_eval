"""
RAG (Retrieval-Augmented Generation) Evaluation Pipeline - Main Module

This script implements a comprehensive evaluation pipeline for RAG systems by:
1. Loading test questions from a MongoDB database
2. Retrieving relevant document chunks for each question
3. Generating answers using a coaching service API
4. Evaluating the answers using a structured evaluation rubric
5. Logging all metrics and results to Langfuse for analysis
6. Saving evaluation results to MongoDB

Author: Saurabh Dey
"""

import os
import asyncio
import openai
import nest_asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from langfuse import Langfuse

# Import modules from the package
from rag_eval.config.settings import (
    LANGFUSE_SECRET_KEY, 
    LANGFUSE_PUBLIC_KEY, 
    LANGFUSE_HOST,
    OPENAI_API_KEY,
    DEFAULT_BUSINESS_CONTEXT,
    parse_args, 
    init_environment
)
from rag_eval.data.mongodb import load_data_from_mongodb, save_evaluation_to_mongodb, get_business_context_from_mongodb
from rag_eval.retrieval.chunks import get_chunks_from_api
from rag_eval.generation.answer import generate_answer, remove_thinking_sections
from rag_eval.evaluation.judge import evaluate_using_rag_prompt

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Initialize the Langfuse client
langfuse = Langfuse(
    secret_key=LANGFUSE_SECRET_KEY,
    public_key=LANGFUSE_PUBLIC_KEY,
    host=LANGFUSE_HOST
)

# Initialize the OpenAI client
init_environment()
openai_client = openai.OpenAI()

async def main():
    """
    Main function that orchestrates the entire RAG evaluation pipeline using MongoDB data.
    
    For each conversation from MongoDB:
    1. Creates a Langfuse trace
    2. Uses the stored query, chunks, and answer
    3. Evaluates the answer
    4. Logs all metrics and results to Langfuse
    """
    try:
        # Parse arguments
        args = parse_args()
        
        # Check if today is Sunday (weekday() returns 6 for Sunday)
        # Only run the script on Sundays unless --force-run is specified
        today = datetime.now()
        if today.weekday() != 6 and not args.force_run:
            print(f"Today is {today.strftime('%A')}, not Sunday. The script only runs on Sundays.")
            print("Use --force-run to override this behavior.")
            return
            
        print(f"Running evaluation on {today.strftime('%A, %Y-%m-%d')}")
        
        # Load conversations from MongoDB
        conversations = await load_data_from_mongodb(
            limit=args.limit, 
            guest_mode=args.guest,
            date_filter=not args.no_date_filter,  # Use date filter unless --no-date-filter is specified
            start_date=args.start_date,  # Pass custom start date if provided
            end_date=args.end_date      # Pass custom end date if provided
        )
        if not conversations:
            print("No data loaded from MongoDB. Please check your connection and try again.")
            return
        
        # Process all conversations from MongoDB
        for conversation in conversations:
            question = conversation['query']
            created_at = conversation['created_at']
            chunks = conversation['chunks']
            answer = conversation['answer']
            chat_id = conversation['chat_id']
            user_id = conversation['user_id']
            
            print(f"\nProcessing Question: {question}")
            print(f"Chat ID: {chat_id}")
            print(f"Created At: {created_at}")
            print("-" * 50)
            
            # Retrieve business context for this user from MongoDB
            business_context = await get_business_context_from_mongodb(user_id, args.guest)
            if not business_context:
                # Fall back to default business context if none found
                print(f"No business context found for user {user_id}, using default")
                business_context = DEFAULT_BUSINESS_CONTEXT
            else:
                print(f"Retrieved business context for user {user_id} from MongoDB")
            
            # Create a single trace for the entire pipeline - ensure all data is traced
            trace = langfuse.trace(
                name="rag-evaluation",
                user_id=user_id,
                session_id=f"session_{chat_id}",
                metadata={
                    "question": question,
                    "created_at": created_at,
                    "chat_id": chat_id,
                    "business_context": business_context,
                    "evaluation_date": today.isoformat(),
                    "day_of_week": today.strftime('%A'),
                    "date_filtered": not args.no_date_filter
                },
                input={
                    "question": question, 
                    "created_at": created_at,
                    "chat_id": chat_id,
                    "business_context": business_context
                }
            )
            
            # Since we already have chunks and answers from MongoDB, we'll skip retrieval and generation
            # Just log the available chunks and answers
            
            # Log the chunks
            chunk_span = langfuse.span(
                name="mongo-chunks",
                trace_id=trace.id,
                input={"question": question, "chat_id": chat_id}
            )
            
            # Handle case where no chunks are available
            if not chunks:
                print("No chunks available for this conversation")
                chunks = ["No relevant content found in knowledge base."]
                langfuse.score(
                    name="no-chunks-found",
                    trace_id=trace.id,
                    value=0,
                    comment="No chunks were available for this conversation"
                )
            
            chunk_span.update(output={"chunks": chunks, "count": len(chunks)})
            chunk_span.end()  # Explicitly end the span
            
            print(f"Found {len(chunks)} chunks from MongoDB")
            
            # Log the answer
            answer_span = langfuse.span(
                name="mongo-answer",
                trace_id=trace.id,
                input={
                    "question": question,
                    "chat_id": chat_id,
                    "chunks": chunks,
                    "business_context": business_context
                }
            )
            
            print("\nStored Answer:")
            print(answer)
            print("-" * 50)
            
            # Log the generation in Langfuse (for historical record)
            generation = langfuse.generation(
                name="stored-answer",
                trace_id=trace.id,
                model="coach-service",
                model_parameters={},
                input={
                    "question": question,
                    "chunks": chunks,
                    "business_context": business_context
                },
                output=answer
            )
            
            answer_span.update(output={"answer": answer})
            answer_span.end()  # Explicitly end the span
            
            # Step 3: Evaluate the answer using the RAG evaluation prompt
            eval_span = langfuse.span(
                name="rag-evaluation-wrapper",
                trace_id=trace.id,
                input={
                    "question": question,
                    "created_at": created_at,
                    "chunks": chunks,
                    "answer": answer
                }
            )
            
            rag_results = evaluate_using_rag_prompt(
                question, 
                chunks, 
                answer, 
                openai_client,
                langfuse, 
                trace.id
            )
            eval_span.update(output=rag_results)
            eval_span.end()  # Explicitly end the span
            
            # Step 4: Print evaluation results for monitoring
            print("\nRAG Evaluation Results:")
            print("-" * 50)
            # Display primary evaluation metrics
            print("Accuracy:", rag_results['accuracy']['judgment'])
            print("Score:", rag_results['accuracy']['score'])
            print("Explanation:", rag_results['accuracy']['reasoning'])
            print("\nRelevance:", rag_results['relevance']['judgment'])
            print("Score:", rag_results['relevance']['score'])
            print("Explanation:", rag_results['relevance']['reasoning'])
            print("\nCoherence:", rag_results['coherence']['judgment'])
            print("Score:", rag_results['coherence']['score'])
            print("Explanation:", rag_results['coherence']['reasoning'])
            
            # Display additional safety and context metrics
            print("\nSafety Evaluation:")
            print("User Query Jailbreak Attempt:", rag_results['safety']['user_query']['jailbreak_attempt']['judgment'])
            print("Score:", rag_results['safety']['user_query']['jailbreak_attempt']['score'])
            print("User Query Toxicity:", rag_results['safety']['user_query']['toxicity']['judgment'])
            print("Score:", rag_results['safety']['user_query']['toxicity']['score'])
            print("\nAI Response Safety:")
            print("Jailbreak Success:", rag_results['safety']['ai_response']['jailbreak_success']['judgment'])
            print("Score:", rag_results['safety']['ai_response']['jailbreak_success']['score'])
            print("Toxicity:", rag_results['safety']['ai_response']['toxicity']['judgment'])
            print("Score:", rag_results['safety']['ai_response']['toxicity']['score'])
            
            print("\nBusiness Context Adherence:")
            print("Judgment:", rag_results['business_context']['judgment'])
            print("Score:", rag_results['business_context']['score'])
            
            print("\nFactual Accuracy (World Knowledge):")
            print("Judgment:", rag_results['factual_accuracy']['judgment'])
            print("Score:", rag_results['factual_accuracy']['score'])
            
            # Step 5: Create tags based on judgments for Langfuse filtering
            tags = []
            
            # Add timestamp tag
            tags.append(f"created_at:{created_at}")
            
            # Add date tag for easier filtering in Langfuse
            try:
                # Parse the ISO format timestamp to extract just the date part
                creation_date = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime('%Y-%m-%d')
                tags.append(f"date:{creation_date}")
                
                # Also add evaluation date tag
                eval_date = today.strftime('%Y-%m-%d')
                tags.append(f"eval_date:{eval_date}")
            except Exception as e:
                print(f"Error creating date tags: {e}")
            
            # Add chat ID tag
            tags.append(f"chat_id:{chat_id}")
            
            # Add judgment tags for each evaluation dimension
            # Accuracy tags
            if rag_results['accuracy']['judgment'] == "fully_correct_and_faithful":
                tags.append("accuracy:fully_correct_and_faithful")
            elif rag_results['accuracy']['judgment'] == "partially_correct_or_faithful":
                tags.append("accuracy:partially_correct_or_faithful")
            elif rag_results['accuracy']['judgment'] == "incorrect_or_unfaithful":
                tags.append("accuracy:incorrect_or_unfaithful")
                
            # Relevance tags
            if rag_results['relevance']['judgment'] == "well_supported":
                tags.append("relevance:well_supported")
            elif rag_results['relevance']['judgment'] == "partially_supported":
                tags.append("relevance:partially_supported")
            elif rag_results['relevance']['judgment'] == "unsupported":
                tags.append("relevance:unsupported")
                
            # Coherence tags
            if rag_results['coherence']['judgment'] == "coherent_and_clear":
                tags.append("coherence:coherent_and_clear")
            elif rag_results['coherence']['judgment'] == "mostly_coherent":
                tags.append("coherence:mostly_coherent")
            elif rag_results['coherence']['judgment'] == "incoherent_or_unclear":
                tags.append("coherence:incoherent_or_unclear")
                
            # Safety tags
            if rag_results['safety']['user_query']['jailbreak_attempt']['judgment'] == "attempt":
                tags.append("safety:jailbreak_attempt")
                
            if rag_results['safety']['user_query']['toxicity']['judgment'] != "none":
                tags.append(f"safety:user_toxicity_{rag_results['safety']['user_query']['toxicity']['judgment']}")
                
            if rag_results['safety']['ai_response']['jailbreak_success']['judgment'] != "none":
                tags.append(f"safety:ai_jailbreak_{rag_results['safety']['ai_response']['jailbreak_success']['judgment']}")
                
            if rag_results['safety']['ai_response']['toxicity']['judgment'] != "none":
                tags.append(f"safety:ai_toxicity_{rag_results['safety']['ai_response']['toxicity']['judgment']}")
                
            # Business context tags
            if rag_results['business_context']['judgment'] == "correct":
                tags.append("business_context:correct")
            elif rag_results['business_context']['judgment'] == "incorrect":
                tags.append("business_context:incorrect")
                
            # Factual accuracy tags
            if rag_results['factual_accuracy']['judgment'] == "correct":
                tags.append("factual_accuracy:correct")
            elif rag_results['factual_accuracy']['judgment'] == "incorrect":
                tags.append("factual_accuracy:incorrect")
                
            # Step 6: Update the Langfuse trace with all evaluation data
            trace.update(
                tags=tags,
                scores={
                    "accuracy": rag_results['accuracy']['score'],
                    "relevance": rag_results['relevance']['score'],
                    "coherence": rag_results['coherence']['score'],
                    "safety_user_query_jailbreak": rag_results['safety']['user_query']['jailbreak_attempt']['score'],
                    "safety_user_query_toxicity": rag_results['safety']['user_query']['toxicity']['score'],
                    "safety_ai_response_jailbreak": rag_results['safety']['ai_response']['jailbreak_success']['score'],
                    "safety_ai_response_toxicity": rag_results['safety']['ai_response']['toxicity']['score'],
                    "business_context": rag_results['business_context']['score'],
                    "factual_accuracy": rag_results['factual_accuracy']['score']
                },
                metadata={
                    "question": question,
                    "created_at": created_at,
                    "chat_id": chat_id,
                    "business_context": business_context,
                    "answer": answer,
                    "evaluation_scores": {
                        "accuracy": rag_results['accuracy']['score'],
                        "relevance": rag_results['relevance']['score'],
                        "coherence": rag_results['coherence']['score'],
                        "business_context": rag_results['business_context']['score'],
                        "factual_accuracy": rag_results['factual_accuracy']['score']
                    },
                    "evaluation_judgments": {
                        "accuracy": rag_results['accuracy']['judgment'],
                        "relevance": rag_results['relevance']['judgment'],
                        "coherence": rag_results['coherence']['judgment'],
                        "business_context": rag_results['business_context']['judgment'],
                        "factual_accuracy": rag_results['factual_accuracy']['judgment'],
                        "safety_user_query_jailbreak": rag_results['safety']['user_query']['jailbreak_attempt']['judgment'],
                        "safety_user_query_toxicity": rag_results['safety']['user_query']['toxicity']['judgment'],
                        "safety_ai_response_jailbreak": rag_results['safety']['ai_response']['jailbreak_success']['judgment'],
                        "safety_ai_response_toxicity": rag_results['safety']['ai_response']['toxicity']['judgment']
                    }
                },
                output={
                    "question": question,
                    "created_at": created_at,
                    "chat_id": chat_id,
                    "answer": answer,
                    "evaluation": rag_results,
                    "explanations": {
                        "accuracy": rag_results['accuracy']['reasoning'],
                        "relevance": rag_results['relevance']['reasoning'],
                        "coherence": rag_results['coherence']['reasoning'],
                        "safety_user_query_jailbreak": rag_results['safety']['user_query']['jailbreak_attempt']['explanation'],
                        "safety_user_query_toxicity": rag_results['safety']['user_query']['toxicity']['explanation'],
                        "safety_ai_response_jailbreak": rag_results['safety']['ai_response']['jailbreak_success']['explanation'],
                        "safety_ai_response_toxicity": rag_results['safety']['ai_response']['toxicity']['explanation'],
                        "business_context": rag_results['business_context']['explanation'],
                        "factual_accuracy": rag_results['factual_accuracy']['explanation']
                    }
                }
            )
            
            # Step 7: Save evaluation results to MongoDB
            message_id = conversation.get('message_id')
            aiResponseMessageid = conversation.get('aiResponseMessageid')
            guest_mode = args.guest
            
            mongodb_save_result = await save_evaluation_to_mongodb(
                rag_results,
                chat_id,
                message_id,
                aiResponseMessageid,
                guest_mode,
                query=question,
                chunks=chunks,
                answer=answer
            )
            
            if mongodb_save_result:
                print("Evaluation results saved to MongoDB successfully")
            else:
                print("Failed to save evaluation results to MongoDB")
            
            # Ensure everything is sent to Langfuse for this question
            langfuse.flush()
            
            print(f"\nCompleted evaluation for question: {question}")
            print("=" * 80)
            
    except Exception as e:
        print(f"Error processing questions: {e}")
        langfuse.flush()  # Ensure any data is flushed before exiting
    
    print("\nAll questions processed. Check your Langfuse dashboard.")

if __name__ == "__main__":
    asyncio.run(main()) 