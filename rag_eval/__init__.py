"""
RAG (Retrieval-Augmented Generation) Evaluation Pipeline

This package implements a comprehensive evaluation pipeline for RAG systems by:
1. Loading test questions from a MongoDB database
2. Retrieving relevant document chunks for each question
3. Generating answers using a coaching service API
4. Evaluating the answers using a structured evaluation rubric
5. Logging all metrics and results to Langfuse for analysis

Author: Saurabh Dey
"""

__version__ = "0.1.0" 