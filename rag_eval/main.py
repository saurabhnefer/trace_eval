"""
Main entry point for the RAG evaluation pipeline.

This module orchestrates the entire evaluation process, from data loading to
metrics calculation and reporting.
"""

import os
import sys
import logging
import datetime
from typing import List, Dict, Any

from .config.settings import parse_args, init_environment, MONGODB_URI, MONGODB_DB_NAME
from .data.loader import load_conversations
from .evaluation.metrics import calculate_metrics, generate_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the RAG evaluation pipeline."""
    args = parse_args()
    
    # Check if we should run today (run only on weekdays unless forced)
    today = datetime.datetime.now().weekday()
    if today >= 5 and not args.force_run:  # 5 = Saturday, 6 = Sunday
        logger.info("Today is a weekend. Skipping evaluation. Use --force-run to override.")
        return
    
    # Initialize environment variables
    init_environment()
    
    # Validate MongoDB connection
    if not MONGODB_URI:
        logger.error("MongoDB URI is not set. Please set MONGODB_URI in your .env file.")
        return
    
    # Load conversations from MongoDB
    logger.info("Loading conversations from MongoDB...")
    try:
        conversations = load_conversations(
            uri=MONGODB_URI,
            db_name=MONGODB_DB_NAME,
            limit=args.limit,
            use_guest=args.guest,
            use_date_filter=not args.no_date_filter
        )
        logger.info(f"Loaded {len(conversations)} conversations.")
    except Exception as e:
        logger.error(f"Error loading conversations: {e}")
        return
    
    if not conversations:
        logger.warning("No conversations found. Exiting.")
        return
    
    # Calculate evaluation metrics
    logger.info("Calculating evaluation metrics...")
    try:
        metrics = calculate_metrics(conversations)
        logger.info("Metrics calculation complete.")
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return
    
    # Generate and save report
    logger.info("Generating evaluation report...")
    try:
        report_path = generate_report(metrics)
        logger.info(f"Report generated and saved to {report_path}")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return
    
    logger.info("RAG evaluation pipeline completed successfully.")

if __name__ == "__main__":
    main() 