from generator import generate_answer
from structure_chunker import chunk_by_structure
from loader import load_pdf
from vector_store import index_chunks

test_cases = [
    {"query": "why did the availability filter test fail?", "must_contain": "BUG-447"},
    {"query": "what is the currency bug about?", "must_contain": "BUG-441"},
    {"query": "what's the weather today?", "must_contain": "don't have information"},
]

def run_eval():
    text = load_pdf("data/sample.pdf")
    chunks = chunk_by_structure(text)
    index_chunks(chunks)

    passed = 0
    for case in test_cases:
        answer = generate_answer(case["query"], chunks, None)
        ok = case["must_contain"].lower() in answer.lower()
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}: {case['query']}")

    print(f"\n{passed}/{len(test_cases)} passed")

if __name__ == "__main__":
    run_eval()