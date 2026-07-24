from pypdf import PdfReader

def load_pdf(path: str) -> str:
    """Extract raw text from a PDF file."""
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

if __name__ == "__main__":
    text = load_pdf("data/sample.pdf")
    print(f"Extracted {len(text)} characters")
    print(text[:400])