# VentureBox AI Evaluation

This repository contains evaluation code for VentureBox AI.

## Setup

1. Clone the repository
2. Copy `.env.template` to `.env` and fill in the required API keys and configuration
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the evaluation pipeline:

```bash
python -m rag_eval.main --limit 20
```

### Command Line Arguments

- `--guest`: Use Guest_Message_History collection instead of Message_History
- `--limit`: Maximum number of conversations to evaluate (default: 20)
- `--no-date-filter`: Disable filtering by current date (evaluate all conversations)
- `--force-run`: Force the script to run regardless of the day of the week
- `--start-date`: Specify a custom start date for filtering conversations (format: YYYY-MM-DDT00:00:00)
- `--end-date`: Specify a custom end date for filtering conversations (format: YYYY-MM-DDT00:00:00)

### Example with Custom Date Range

```bash
python main.py --force-run --start-date 2025-05-01T00:00:00 --end-date 2025-06-01T00:00:00
```

## Output

The evaluation pipeline produces two outputs:

1. **Langfuse Traces**: All evaluation metrics and scores are sent to Langfuse for visualization and analysis.
2. **MongoDB Collection**: Evaluation results are saved to the `RAG_Evaluations` collection in MongoDB for persistence and integration with other systems.

### MongoDB Collection Schema

The `RAG_Evaluations` collection stores:

- `chat_id`: ID of the chat being evaluated
- `message_id`: ID of the specific message
- `aiResponseMessageid`: ID of the AI response
- `guest_mode`: Whether the message is from guest chat history
- `evaluation_date`: When the evaluation was performed
- `query`: The original user query
- `chunks`: The retrieved document chunks used for context
- `answer`: The generated answer
- `evaluation_results`: Complete evaluation result object
- Top-level score fields for easy querying:
  - `accuracy_score`, `relevance_score`, `coherence_score`, etc.
  - `accuracy_judgment`, `relevance_judgment`, `coherence_judgment`, etc.

## Project Structure

- `rag_eval/config/`: Configuration settings
- `rag_eval/data/`: Data loading and processing
- `rag_eval/retrieval/`: Retrieval evaluation
- `rag_eval/generation/`: Generation evaluation
- `rag_eval/evaluation/`: Metrics and evaluation
- `rag_eval/utils/`: Utility functions 