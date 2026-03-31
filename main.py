from pathlib import Path

import numpy as np
import requests

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "corpus.txt"

# -----------------------------
# 1. Load data
# -----------------------------
text = DATA_FILE.read_text(encoding="utf-8")

# -----------------------------
# 2. Chunking
# -----------------------------
def chunk_text(text, chunk_size=100):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = chunk_text(text)

# -----------------------------
# 3. Embeddings (LOCAL)
# -----------------------------
def get_embedding(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text
        }
    )
    data = response.json()
    # print("DEBUG EMBEDDING RESPONSE:", data)  # 👈 ADD THIS

    return data.get("embedding", [])

chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

# -----------------------------
# 4. Store in memory
# -----------------------------
store = list(zip(chunks, chunk_embeddings))

# -----------------------------
# 5. Similarity
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -----------------------------
# 6. Ask question
# -----------------------------
query = input("Ask a question: ")
query_embedding = get_embedding(query)

# -----------------------------
# 7. Retrieve
# -----------------------------
scored = []

for chunk, emb in store:
    score = cosine_similarity(query_embedding, emb)
    scored.append((score, chunk))

scored.sort(reverse=True)
top_chunks = [chunk for _, chunk in scored[:2]]

# -----------------------------
# 8. Generate (LOCAL LLM)
# -----------------------------
context = "\n".join(top_chunks)

prompt = f"""
You are a helpful assistant.

Context:
{context}

Question:
{query}

Answer clearly using the context above.
"""

def generate_answer(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

answer = generate_answer(prompt)

print("\nAnswer:\n", answer)