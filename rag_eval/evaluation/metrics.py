"""
Metrics module for the RAG evaluation pipeline.

This module provides functions to calculate evaluation metrics and generate reports.
"""

import os
import json
import logging
import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from langfuse import Langfuse

from ..config.settings import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

logger = logging.getLogger(__name__)

# Initialize Langfuse client if keys are available
langfuse_client = None
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    try:
        langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )
        logger.info("Langfuse client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse client: {e}")

def calculate_metrics(conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for the given conversations.
    
    Args:
        conversations: List of conversation data
        
    Returns:
        Dictionary of calculated metrics
    """
    # Initialize metrics dictionary
    metrics = {
        "total_conversations": len(conversations),
        "total_messages": 0,
        "total_user_messages": 0,
        "total_assistant_messages": 0,
        "avg_messages_per_conversation": 0,
        "avg_conversation_duration": 0,
        "conversation_details": []
    }
    
    total_duration = datetime.timedelta()
    
    # Calculate metrics for each conversation
    for conversation in conversations:
        messages = conversation.get("messages", [])
        conv_id = conversation.get("conversationId")
        
        # Count messages by role
        user_messages = sum(1 for msg in messages if msg.get("role") == "user")
        assistant_messages = sum(1 for msg in messages if msg.get("role") == "assistant")
        
        # Calculate conversation duration
        start_time = conversation.get("startTime")
        end_time = conversation.get("endTime")
        duration = end_time - start_time if end_time and start_time else None
        
        if duration:
            total_duration += duration
        
        # Add to totals
        metrics["total_messages"] += len(messages)
        metrics["total_user_messages"] += user_messages
        metrics["total_assistant_messages"] += assistant_messages
        
        # Add conversation details
        conv_metrics = {
            "conversationId": conv_id,
            "message_count": len(messages),
            "user_message_count": user_messages,
            "assistant_message_count": assistant_messages,
            "duration_seconds": duration.total_seconds() if duration else None,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None
        }
        metrics["conversation_details"].append(conv_metrics)
    
    # Calculate averages
    if metrics["total_conversations"] > 0:
        metrics["avg_messages_per_conversation"] = metrics["total_messages"] / metrics["total_conversations"]
        if total_duration.total_seconds() > 0:
            metrics["avg_conversation_duration"] = total_duration.total_seconds() / metrics["total_conversations"]
    
    # Log to Langfuse if available
    if langfuse_client:
        try:
            trace = langfuse_client.trace(
                name="rag_evaluation",
                metadata={
                    "total_conversations": metrics["total_conversations"],
                    "total_messages": metrics["total_messages"],
                    "avg_messages_per_conversation": metrics["avg_messages_per_conversation"]
                }
            )
            
            # Add metrics as an observation
            trace.observation(
                name="evaluation_metrics",
                observation_type="metric",
                value=metrics
            )
            
            logger.info("Metrics logged to Langfuse successfully.")
        except Exception as e:
            logger.error(f"Failed to log metrics to Langfuse: {e}")
    
    return metrics

def generate_report(metrics: Dict[str, Any]) -> str:
    """
    Generate an evaluation report and save it to disk.
    
    Args:
        metrics: Dictionary of calculated metrics
        
    Returns:
        Path to the saved report file
    """
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"rag_eval_report_{timestamp}.json"
    report_path = os.path.join(reports_dir, report_filename)
    
    # Add report generation timestamp to metrics
    metrics["report_generated_at"] = datetime.datetime.now().isoformat()
    
    # Save metrics as JSON
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Generate a CSV report of conversation details
    if metrics.get("conversation_details"):
        csv_filename = f"rag_eval_details_{timestamp}.csv"
        csv_path = os.path.join(reports_dir, csv_filename)
        
        df = pd.DataFrame(metrics["conversation_details"])
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Detailed report saved to {csv_path}")
    
    return report_path 