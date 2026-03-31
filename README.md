# SuperSimpleRAG

`SuperSimpleRAG` is a minimal Retrieval-Augmented Generation (RAG) example built with plain Python and a local Ollama setup.

It shows the core RAG flow without frameworks:

1. Load raw text from `corpus.txt`
2. Split the text into chunks
3. Generate embeddings for each chunk
4. Store `(chunk, embedding)` pairs in memory
5. Embed the user query
6. Run cosine similarity to find relevant chunks
7. Build a prompt with the retrieved context
8. Generate an answer with a local LLM

## Requirements

- Python 3
- Ollama running locally at `http://localhost:11434`
- An embedding model available in Ollama
- A chat/generation model available in Ollama

This project is currently configured to use:

- Embedding model: `nomic-embed-text`
- LLM model: `llama3.1`


## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy requests
```

Pull the Ollama models if needed:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

Start Ollama if it is not already running.

## Test model output

Test Embedding Model

```python
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "test"
}'
```

Test LLM

```python
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1",
  "prompt": "Say hello",
  "stream": false
}'
```

## Run

From the project directory:

```bash
python main.py
```

You will be prompted for a question:

```text
Ask a question:
```

The script will retrieve the most relevant chunks from `corpus.txt` and generate an answer using the local model.

## Files

- `main.py`: the end-to-end RAG example
- `corpus.txt`: source text used for retrieval

## Notes

- This is an educational example, not a production-ready RAG system.
- The chunking is character-based and intentionally simple.
- Storage is in memory only.
- Retrieval uses cosine similarity over embeddings.
