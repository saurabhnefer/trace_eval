# RAG Evaluation Pipeline - Docker Guide

This guide explains how to build and run the RAG evaluation pipeline using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose installed (optional, but recommended)
- Required API keys and credentials

## Environment Variables

Before running the container, you need to set up the following environment variables:

- `OPENAI_API_KEY` - Your OpenAI API key
- `LANGFUSE_SECRET_KEY` - Langfuse secret key
- `LANGFUSE_PUBLIC_KEY` - Langfuse public key
- `LANGFUSE_HOST` - Langfuse host URL (defaults to https://cloud.langfuse.com)
- `MONGODB_URI` - MongoDB connection string
- `MONGODB_DB_NAME` - MongoDB database name (defaults to genieai)
- `SEARCH_API_URL` - Search API URL
- `CHAT_API_URL` - Chat API URL

## Option 1: Using Docker Compose (Recommended)

1. Create a `.env` file in the project root with the required environment variables:

```
OPENAI_API_KEY=your_openai_key
LANGFUSE_SECRET_KEY=your_langfuse_secret
LANGFUSE_PUBLIC_KEY=your_langfuse_public
MONGODB_URI=your_mongodb_connection_string
```

2. Build and start the container:

```bash
docker-compose up --build
```

3. To run with custom arguments:

```bash
# Edit the command line in docker-compose.yml or override it at runtime:
docker-compose run rag-evaluator python main.py --limit 50 --force-run
```

## Option 2: Using Docker Directly

1. Build the Docker image:

```bash
docker build -t rag-evaluator .
```

2. Run the container with environment variables:

```bash
docker run -e OPENAI_API_KEY=your_key -e MONGODB_URI=your_uri -e LANGFUSE_SECRET_KEY=your_secret -e LANGFUSE_PUBLIC_KEY=your_public rag-evaluator
```

3. To run with custom arguments:

```bash
docker run rag-evaluator python main.py --limit 50 --force-run
```

## Other Docker Commands

- View logs:
```bash
docker-compose logs
```

- Stop the container:
```bash
docker-compose down
```

- Run container in background:
```bash
docker-compose up -d
``` 