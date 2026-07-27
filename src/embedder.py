from sentence_transformers import SentenceTransformer
from structure_chunker import chunk_by_structure
from loader import load_pdf

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks: list[str]) -> list:
    """Convert a list of text chunks into embedding vectors."""
    embeddings = model.encode(chunks)
    return embeddings

if __name__ == "__main__":
    text = load_pdf("data/sample.pdf")
    chunks = chunk_by_structure(text)
    embeddings = embed_chunks(chunks)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Each embedding has {len(embeddings[0])} dimensions")
    print(f"First 5 numbers of chunk 0's embedding: {embeddings[0][:5]}")