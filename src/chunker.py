from loader import load_pdf

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping fixed-length chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    text = load_pdf("data/sample.pdf")
    chunks = chunk_text(text)
    print(f"Document split into {len(chunks)} chunks")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ({len(c)} chars) ---")
        print(c)