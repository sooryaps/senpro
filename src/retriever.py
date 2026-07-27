import numpy as np
from sentence_transformers import SentenceTransformer
from structure_chunker import chunk_by_structure
from loader import load_pdf

model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query: str, chunks: list[str], chunk_embeddings, top_k: int = 3) -> list[tuple[str, float]]:
    """Find the top_k chunks most similar to the query."""
    query_embedding = model.encode(query)

    scores = []
    for chunk, embedding in zip(chunks, chunk_embeddings):
        score = cosine_similarity(query_embedding, embedding)
        scores.append((chunk, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

if __name__ == "__main__":
    text = load_pdf("data/sample.pdf")
    chunks = chunk_by_structure(text)
    chunk_embeddings = model.encode(chunks)

    query = "why did the availability filter test fail?"
    results = retrieve(query, chunks, chunk_embeddings)

    print(f"Query: {query}\n")
    for chunk, score in results:
        print(f"Score: {score:.4f}")
        print(chunk)
        print("---")