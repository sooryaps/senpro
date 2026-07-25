import re
from loader import load_pdf

SECTION_PATTERN = re.compile(r'(?=^\d\.\s+[A-Z])', re.MULTILINE)
ITEM_PATTERN = re.compile(r'(?=(?:BUG|TC)-\d+:)')

def chunk_by_structure(text: str) -> list[str]:
    """Split text using the document's own structure: numbered sections and BUG-/TC- item markers."""
    sections = SECTION_PATTERN.split(text)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        items = ITEM_PATTERN.split(section)
        for item in items:
            item = item.strip()
            if item:
                chunks.append(item)

    return chunks

if __name__ == "__main__":
    text = load_pdf("data/sample.pdf")
    chunks = chunk_by_structure(text)
    print(f"Document split into {len(chunks)} structure-aware chunks")
    for i, c in enumerate(chunks):
        print(f"\n--- Chunk {i} ({len(c)} chars) ---")
        print(c)