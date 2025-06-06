"""
Data loader module for the RAG evaluation pipeline.

This module provides functions to load conversation data from MongoDB.
"""

import logging
import datetime
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

def connect_to_mongodb(uri: str, db_name: str) -> Database:
    """
    Connect to MongoDB and return the database object.
    
    Args:
        uri: MongoDB connection URI
        db_name: Name of the database to use
        
    Returns:
        MongoDB database object
    """
    client = MongoClient(uri)
    return client[db_name]

def load_conversations(
    uri: str,
    db_name: str,
    limit: int = 20,
    use_guest: bool = False,
    use_date_filter: bool = True
) -> List[Dict[str, Any]]:
    """
    Load conversation data from MongoDB.
    
    Args:
        uri: MongoDB connection URI
        db_name: Name of the database to use
        limit: Maximum number of conversations to load
        use_guest: Whether to use the Guest_Message_History collection
        use_date_filter: Whether to filter by current date
        
    Returns:
        List of conversation data dictionaries
    """
    try:
        db = connect_to_mongodb(uri, db_name)
        
        # Determine which collection to use
        collection_name = "Guest_Message_History" if use_guest else "Message_History"
        collection = db[collection_name]
        
        # Build query
        query = {}
        
        # Add date filter if needed
        if use_date_filter:
            today = datetime.datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            query["createdAt"] = {"$gte": today}
        
        # Get distinct conversation IDs
        conversation_ids = collection.distinct("conversationId", query)
        
        if not conversation_ids:
            logger.warning(f"No conversation IDs found matching the query: {query}")
            return []
        
        # Limit number of conversations
        conversation_ids = conversation_ids[:limit]
        
        # Fetch all messages for the selected conversations
        all_conversations = []
        
        for conv_id in conversation_ids:
            messages = list(collection.find({"conversationId": conv_id}).sort("createdAt", 1))
            
            if messages:
                conversation = {
                    "conversationId": conv_id,
                    "messages": messages,
                    "messageCount": len(messages),
                    "startTime": messages[0].get("createdAt"),
                    "endTime": messages[-1].get("createdAt")
                }
                all_conversations.append(conversation)
        
        return all_conversations
        
    except Exception as e:
        logger.error(f"Error loading conversations from MongoDB: {e}")
        raise

def extract_questions_and_answers(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract question-answer pairs from conversations.
    
    Args:
        conversations: List of conversation data
        
    Returns:
        List of question-answer pairs
    """
    qa_pairs = []
    
    for conversation in conversations:
        messages = conversation.get("messages", [])
        conv_id = conversation.get("conversationId")
        
        # Process messages to extract Q&A pairs
        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1]
            
            # Check if this is a user message followed by an assistant message
            if current_msg.get("role") == "user" and next_msg.get("role") == "assistant":
                qa_pair = {
                    "conversationId": conv_id,
                    "question": current_msg.get("content", ""),
                    "answer": next_msg.get("content", ""),
                    "timestamp": current_msg.get("createdAt"),
                    "message_id": current_msg.get("_id")
                }
                qa_pairs.append(qa_pair)
    
    return qa_pairs 