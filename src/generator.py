import os
from dotenv import load_dotenv
from google import genai
from retriever import retrieve
from structure_chunker import chunk_by_structure
from loader import load_pdf
from embedder import embed_chunks

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


SYSTEM_PROMPT = """You are Sentinel, a QA assistant that answers questions using ONLY the
provided context from release notes, bug logs, and test cases. If the answer isn't
in the context, say clearly: "I don't have information about that in the provided documents."
Do not use outside knowledge. Cite which item (e.g. BUG-447, TC-108) your answer is based on."""

def generate_answer(query: str, chunks: list[str], chunk_embeddings) -> str:
    """Retrieve relevant chunks and generate a grounded answer using Gemini."""
    results = retrieve(query, chunks, chunk_embeddings, top_k=3)
    context = "\n\n".join([f"[Score: {score:.2f}] {chunk}" for chunk, score in results])

    full_prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}"
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=full_prompt
    )
    return response.text

if __name__ == "__main__":
    text = load_pdf("data/sample.pdf")
    chunks = chunk_by_structure(text)
    chunk_embeddings = embed_chunks(chunks)

    query = "why did the availability filter test fail?"
    answer = generate_answer(query, chunks, chunk_embeddings)

    print(f"Question: {query}\n")
    print(f"Answer: {answer}")