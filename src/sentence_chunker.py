import re
from loader import load_pdf

def split_sentences(text: str) -> list[str]:
    """Split raw text into a list of sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_by_sentences(text: str, max_chunk_size: int = 300) -> list[str]:
    """Group whole sentences into chunks up to max_chunk_size characters."""
    sentences = split_sentences(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

if __name__ == "__main__":
    text = load_pdf("data/sample.pdf")
    chunks = chunk_by_sentences(text)
    print(f"Document split into {len(chunks)} sentence-aware chunks")
    for i, c in enumerate(chunks):
        print(f"\n--- Chunk {i} ({len(c)} chars) ---")
        print(c)