# RAG Evaluation Pipeline Deployment Guide

This guide explains how to deploy the RAG Evaluation Pipeline as a scheduled cronjob on Render.

## Prerequisites

- A Render account (https://render.com)
- A GitHub repository containing your code
- MongoDB instance with connection details
- OpenAI API key
- Langfuse account with API keys

## Deployment Steps

### 1. Prepare Your Repository

Ensure your repository has the following files:
- `langfuse_eval.py` - The main evaluation script
- `requirements.txt` - Dependencies file
- `render.yaml` - Render deployment configuration
- `setup_env.py` - Optional utility for environment setup

### 2. Configure Environment Variables

You'll need to set these environment variables in Render:

- `MONGODB_URI`: Your MongoDB connection string
  - Format: `mongodb://username:password@hostname:port/dbname?options`
  - Replace the hardcoded connection string in `load_data_from_mongodb` function
  
- `OPENAI_API_KEY`: Your OpenAI API key
  - Replace the hardcoded API key in the script
  
- `LANGFUSE_SECRET_KEY`: Your Langfuse secret key
  - Replace the hardcoded key in the script
  
- `LANGFUSE_PUBLIC_KEY`: Your Langfuse public key
  - Replace the hardcoded key in the script

### 3. Update Your Code (Optional)

For better security, modify `langfuse_eval.py` to use environment variables instead of hardcoded credentials:

```python
# Langfuse Configuration
from langfuse import Langfuse
import os

# Configure Langfuse for tracking experiments and evaluation metrics
langfuse = Langfuse(
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# Setup OpenAI API key
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

# Setup the OpenAI client
openai_client = openai.OpenAI()
```

And update the MongoDB connection in `load_data_from_mongodb`:

```python
# Connect to MongoDB using the provided connection string
url = os.environ.get("MONGODB_URI")
```

### 4. Deploy to Render

#### Option 1: Using Render Dashboard

1. Log in to your Render account
2. Click "New +" and select "Cron Job"
3. Connect your GitHub repository
4. Configure the following settings:
   - Name: `rag-evaluation-weekly`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python langfuse_eval.py`
   - Schedule: `0 0 * * 0` (Runs at midnight every Sunday)
   - Environment Variables: Add all the variables listed in step 2
5. Click "Create Cron Job"

#### Option 2: Using render.yaml (Blueprint)

1. Ensure you have the `render.yaml` file in your repository
2. Log in to your Render account
3. Click "New +" and select "Blueprint"
4. Connect your GitHub repository
5. Render will automatically detect the `render.yaml` file
6. Review the configuration and click "Apply"
7. Add your secret environment variables when prompted

### 5. Monitor Your Cronjob

- View logs in the Render dashboard
- Check execution history to ensure the job is running successfully
- View evaluation results in your Langfuse dashboard

### 6. Testing Your Deployment

To test your deployment without waiting for Sunday:

1. In the Render dashboard, navigate to your cron job
2. Click "Manual Run" to trigger an immediate execution
3. Use the `--force-run` flag by modifying your Start Command temporarily:
   ```
   python langfuse_eval.py --force-run
   ```

## Troubleshooting

- **Job fails immediately**: Check environment variables and MongoDB connection
- **MongoDB connection issues**: Verify your MongoDB instance is accessible from Render
- **Missing data in Langfuse**: Ensure Langfuse API keys are correctly configured
- **Script not completing**: Check if there are timeout issues; consider optimizing for faster execution

For more help, see the [Render Cron Job documentation](https://render.com/docs/cronjobs). 