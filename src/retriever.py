import os
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

def cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed_query(query: str):
    """Embed a single query string using Gemini's embedding API."""
    result = client.models.embed_content(model="gemini-embedding-001", contents=query)
    return np.array(result.embeddings[0].values)

def retrieve(query: str, chunks: list[str], chunk_embeddings, top_k: int = 3) -> list[tuple[str, float]]:
    """Find the top_k chunks most similar to the query."""
    query_embedding = embed_query(query)

    scores = []
    for chunk, embedding in zip(chunks, chunk_embeddings):
        score = cosine_similarity(query_embedding, embedding)
        scores.append((chunk, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

if __name__ == "__main__":
    from loader import load_pdf
    from structure_chunker import chunk_by_structure
    from embedder import embed_chunks

    text = load_pdf("data/sample.pdf")
    chunks = chunk_by_structure(text)
    chunk_embeddings = embed_chunks(chunks)

    query = "why did the availability filter test fail?"
    results = retrieve(query, chunks, chunk_embeddings)

    print(f"Query: {query}\n")
    for chunk, score in results:
        print(f"Score: {score:.4f}")
        print(chunk)
        print("---")