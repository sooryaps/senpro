import os
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

def embed_chunks(chunks: list[str]):
    """Convert a list of text chunks into embedding vectors using Gemini's embedding API."""
    result = client.models.embed_content(model="gemini-embedding-001", contents=chunks)
    return [np.array(e.values) for e in result.embeddings]

if __name__ == "__main__":
    from loader import load_pdf
    from structure_chunker import chunk_by_structure

    text = load_pdf("data/sample.pdf")
    chunks = chunk_by_structure(text)
    embeddings = embed_chunks(chunks)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Each embedding has {len(embeddings[0])} dimensions")
    print(f"First 5 numbers of chunk 0's embedding: {embeddings[0][:5]}")