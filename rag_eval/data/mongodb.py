"""
MongoDB data loading module for the RAG evaluation pipeline.

This module handles all interactions with MongoDB to retrieve conversation data
for evaluation.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pymongo import MongoClient

from ..config.settings import MONGODB_URI, MONGODB_DB_NAME

async def load_data_from_mongodb(
    limit: int = 50, 
    guest_mode: bool = False, 
    date_filter: bool = True,
    start_date: str = None,
    end_date: str = None
) -> List[Dict]:
    """
    Load real user queries, answers, and chunks from MongoDB for evaluation.
    
    Args:
        limit: Maximum number of conversations to load
        guest_mode: Whether to load from guest chat history
        date_filter: Whether to filter conversations by current date
        start_date: Optional custom start date for filtering (format: YYYY-MM-DDT00:00:00)
        end_date: Optional custom end date for filtering (format: YYYY-MM-DDT00:00:00)
        
    Returns:
        List of dictionaries containing query, answer, chunks, and metadata
    """
    try:
        # Connect to MongoDB using the provided connection string
        print(f"Connecting to MongoDB (guest mode: {guest_mode}, limit: {limit}, date_filter: {date_filter})")
        
        client = MongoClient(MONGODB_URI)
        
        db = client[MONGODB_DB_NAME]
        
        # Select collection based on guest_mode
        collection_name = "Guest_Message_History" if guest_mode else "Message_History"
        collection = db[collection_name]
        
        # Query for conversations with the necessary data
        # We need messages with both a query and a response
        conversations = []
        
        # Create date filter for the last week if needed
        query_filter = {"messages": {"$exists": True}}
        if date_filter:
            # Check if custom date range is provided
            if start_date and end_date:
                # Parse the provided date strings
                try:
                    start_date_obj = datetime.fromisoformat(start_date)
                    end_date_obj = datetime.fromisoformat(end_date)
                    
                    # Add date filter to query
                    query_filter["created_at"] = {"$gte": start_date_obj, "$lt": end_date_obj}
                    print(f"Filtering conversations from {start_date} to {end_date}")
                except ValueError as e:
                    print(f"Error parsing custom dates: {e}. Using default date filter.")
                    # Fall back to default date filter
                    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    week_ago = today - timedelta(days=7)
                    query_filter["created_at"] = {"$gte": week_ago, "$lt": today}
                    print(f"Filtering conversations from {week_ago.isoformat()} to {today.isoformat()}")
            else:
                # Get today's date with time set to midnight
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                
                # Calculate the date for 7 days ago (for weekly runs on Sunday)
                week_ago = today - timedelta(days=7)
                
                # Add date filter to query
                query_filter["created_at"] = {"$gte": week_ago, "$lt": today}
                print(f"Filtering conversations from {week_ago.isoformat()} to {today.isoformat()}")
        
        # Find chats with queries and responses
        chats = collection.find(
            query_filter,
            {"chat_id": 1, "userId": 1, "messages": 1, "created_at": 1}
        ).limit(limit)
        
        for chat in chats:
            chat_id = chat.get("chat_id")
            user_id = chat.get("userId", "guest")
            
            # Process each message in the chat
            for message in chat.get("messages", []):
                # Skip if no query or aiResponseMessageid
                if not message.get("query") or not message.get("aiResponseMessageid"):
                    continue
                
                query = message.get("query")
                
                # Find GPT response in the aiResponse array
                answer = None
                for response in message.get("aiResponse", []):
                    if response.get("type") == "GPT":
                        answer = response.get("content")
                        break
                
                # Skip if no answer found
                if not answer:
                    continue
                
                # First check for retrievedChunks directly in the message
                chunks = []
                if message.get("retrievedChunks"):
                    chunks = [chunk.get("text", "") for chunk in message.get("retrievedChunks", [])]
                else:
                    # If not found, check if there are chunks in chunks_reference
                    chunks_reference = message.get("chunksReference")
                    
                    if chunks_reference:
                        # Use direct reference to chunks document
                        chunks_collection = db["SearchChunks"]
                        chunks_doc = chunks_collection.find_one({"_id": chunks_reference})
                        if chunks_doc and "chunks" in chunks_doc:
                            chunks = [chunk.get("text", "") for chunk in chunks_doc.get("chunks", [])]
                    else:
                        # Try to find chunks by query and chat_id
                        chunks_collection = db["SearchChunks"]
                        chunks_doc = chunks_collection.find_one({
                            "query": query,
                            "chat_id": chat_id,
                            "aiResponseMessageid": message.get("aiResponseMessageid")
                        })
                        if chunks_doc and "chunks" in chunks_doc:
                            chunks = [chunk.get("text", "") for chunk in chunks_doc.get("chunks", [])]
                
                # Create conversation entry
                conversation = {
                    "query": query,
                    "answer": answer,
                    "chunks": chunks,
                    "created_at": message.get("created_at", chat.get("created_at", datetime.now())).isoformat(),
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "message_id": message.get("messageid"),
                    "aiResponseMessageid": message.get("aiResponseMessageid")
                }
                
                conversations.append(conversation)
        
        print(f"Loaded {len(conversations)} conversations from MongoDB")
        return conversations
    
    except Exception as e:
        print(f"Error loading data from MongoDB: {e}")
        # Return empty list if MongoDB loading fails
        return [] 

async def save_evaluation_to_mongodb(
    evaluation_data: Dict, 
    chat_id: str,
    message_id: str,
    aiResponseMessageid: str,
    guest_mode: bool = False,
    query: str = None,
    chunks: List[str] = None,
    answer: str = None
) -> bool:
    """
    Save RAG evaluation results to MongoDB.
    
    Args:
        evaluation_data: Dictionary containing all evaluation metrics and judgments
        chat_id: The chat ID associated with the evaluation
        message_id: The message ID associated with the evaluation
        aiResponseMessageid: The AI response message ID
        guest_mode: Whether this is from guest chat history
        query: The original user query
        chunks: The retrieved document chunks used for answering
        answer: The generated answer
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DB_NAME]
        
        # Use a dedicated collection for evaluations
        collection = db["RAG_Evaluations"]
        
        # Prepare the document to insert
        evaluation_doc = {
            "chat_id": chat_id,
            "message_id": message_id,
            "aiResponseMessageid": aiResponseMessageid,
            "guest_mode": guest_mode,
            "evaluation_date": datetime.now(),
            "query": query,
            "chunks": chunks,
            "answer": answer,
            "evaluation_results": evaluation_data,
            # Add top-level fields for easier querying
            "accuracy_score": evaluation_data.get("accuracy", {}).get("score"),
            "relevance_score": evaluation_data.get("relevance", {}).get("score"),
            "coherence_score": evaluation_data.get("coherence", {}).get("score"),
            "business_context_score": evaluation_data.get("business_context", {}).get("score"),
            "factual_accuracy_score": evaluation_data.get("factual_accuracy", {}).get("score"),
            "accuracy_judgment": evaluation_data.get("accuracy", {}).get("judgment"),
            "relevance_judgment": evaluation_data.get("relevance", {}).get("judgment"),
            "coherence_judgment": evaluation_data.get("coherence", {}).get("judgment"),
            "business_context_judgment": evaluation_data.get("business_context", {}).get("judgment"),
            "factual_accuracy_judgment": evaluation_data.get("factual_accuracy", {}).get("judgment")
        }
        
        # Insert the document
        result = collection.insert_one(evaluation_doc)
        
        print(f"Saved evaluation to MongoDB with ID: {result.inserted_id}")
        return True
        
    except Exception as e:
        print(f"Error saving evaluation to MongoDB: {e}")
        return False 

async def get_business_context_from_mongodb(user_id: str, guest_mode: bool = False) -> Optional[Dict]:
    """
    Retrieve the business context for a specific user from MongoDB.
    
    Args:
        user_id: The user ID to retrieve business context for
        guest_mode: Whether to use Guest_Message_History collection
        
    Returns:
        Dictionary containing the business context or None if not found
    """
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DB_NAME]
        
        # Select collection based on guest_mode
        collection_name = "Guest_Message_History" if guest_mode else "Message_History"
        collection = db[collection_name]
        
        # Query for the user's business context
        # First try to find a document with businessContext at the root level
        result = collection.find_one(
            {"userId": user_id, "businessContext": {"$exists": True}},
            {"businessContext": 1, "_id": 0}
        )
        
        if result and "businessContext" in result:
            # Return the business context
            return result["businessContext"]
            
        # If not found at root level, check for businessContext in messages
        # This handles the case where businessContext might be stored in a message
        result = collection.find_one(
            {"userId": user_id, "messages.businessContext": {"$exists": True}},
            {"messages.businessContext": 1, "_id": 0}
        )
        
        if result and "messages" in result:
            # Find the first message with business context
            for message in result["messages"]:
                if "businessContext" in message:
                    return message["businessContext"]
        
        # No business context found
        return None
        
    except Exception as e:
        print(f"Error retrieving business context from MongoDB: {e}")
        return None 