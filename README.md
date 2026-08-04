# Sentinel — AI QA Assistant

An AI assistant that answers questions from release notes, bug logs, and test case documents — grounded in real retrieval, not general knowledge guesses. Built as a direct extension of hands-on QA/testing experience (manual testing, regression testing, defect life cycle) into applied GenAI engineering.

**Live example:**
> **Q:** why did the availability filter test fail?
> **A:** Based on BUG-447, the 'Available Today' provider search filter failed due to a timezone mismatch between the booking service and the availability service. This mismatch causes the filter to sometimes return providers who are actually marked unavailable. This failure is also referenced in TC-108 and the Regression Test Summary.

That answer synthesizes three separate document sections into one coherent explanation — proof the system reasons over retrieved context rather than doing simple keyword lookup.

---

## What This Is

A RAG (Retrieval-Augmented Generation) pipeline: documents get loaded, split into meaningful chunks, converted into embeddings, and retrieved by semantic similarity when a question comes in — then a grounded LLM generates a cited answer using only the retrieved context.

**Why this project, specifically:** built on real QA background — release notes, bug logs, and test case documents are exactly the kind of documentation QA teams dig through manually every day. Sentinel automates that lookup.

---

## Architecture

```
PDF Document
     │
     ▼
Text Extraction (pypdf)
     │
     ▼
Structure-Aware Chunking (regex on section headers + BUG-/TC- markers)
     │
     ▼
Embeddings (Gemini embedding API)
     │
     ▼
Cosine Similarity Retrieval (top-k, with a relevance threshold)
     │
     ▼
Grounded Generation (Gemini, system-prompted to cite sources
                      and refuse to answer outside the provided context)
```

**Both embedding and generation use Google's Gemini API** — a deliberate architecture decision made after hitting a real infrastructure constraint (see "Engineering Decisions" below).

---

## The Chunking Journey — Naive to Structure-Aware

Chunking wasn't solved on the first attempt, and that iteration is intentionally preserved in this repo (see `src/chunker.py`, `src/sentence_chunker.py`, `src/structure_chunker.py`):

1. **Fixed-length chunking** (`chunker.py`) — split every 300 characters with overlap. Found: cut a bug's root-cause explanation mid-sentence.
2. **Sentence-aware chunking** (`sentence_chunker.py`) — split on sentence boundaries instead. Fixed the mid-sentence cut, but found a new issue: two unrelated test cases (TC-102 and TC-108) ended up merged into one chunk simply because they were adjacent — risking the model misattributing one test's result to a different test.
3. **Structure-aware chunking** (`structure_chunker.py`) — split using the document's own structure (numbered sections, `BUG-`/`TC-` ID markers). This version is what's actually used in the pipeline. Along the way, an overly broad regex pattern falsely matched plain numbers inside sentences (e.g. "128. Passed: 121.") as section headers — fixed by requiring a capital letter immediately after the pattern.

Each stage's bug was real, found by inspecting actual output, not anticipated in advance.

---

## Engineering Decisions Worth Knowing

**Why hosted APIs for both embedding and generation, not local models:**
Originally planned to run embeddings locally (via `sentence-transformers`) for stronger data-privacy guarantees, with only generation using a hosted API. During implementation, a Windows security policy blocked PyTorch's native library from loading on the development machine — confirmed as an OS-level issue, not a code bug, by isolating it with a standalone `import torch` test. Given the constraint, the pipeline was pivoted to use Gemini's hosted API for both embedding and generation. This remains a legitimate, common production pattern; a stricter-compliance context would warrant revisiting local embeddings.

**Why a similarity threshold, on top of prompt-based grounding:**
The system prompt instructs the model to only answer from provided context — but relying on prompt instructions alone for safety-critical behavior is fragile. A hard numeric threshold (`SIMILARITY_THRESHOLD = 0.3`) checked in code, before the LLM is ever called, guarantees irrelevant queries are rejected deterministically, and saves the cost of a wasted API call.

**Why `gemini-flash-latest` instead of a pinned model version:**
An earlier hardcoded model version (`gemini-2.5-flash`) was deprecated mid-project. Using Google's `-latest` alias avoids this specific failure mode going forward, at the cost of less strict reproducibility — an intentional trade-off for a fast-moving hosted API.

---

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root (see `.env.example`):
```
GOOGLE_API_KEY=your-key-here
```
Get a free key at [aistudio.google.com](https://aistudio.google.com).

Run the full pipeline:
```bash
python src/generator.py
```

---

## Project Structure

```
senpro/
├── data/sample.pdf          # sample QA release notes document
├── src/
│   ├── loader.py             # PDF text extraction
│   ├── chunker.py             # fixed-length chunking (superseded)
│   ├── sentence_chunker.py    # sentence-aware chunking (superseded)
│   ├── structure_chunker.py   # structure-aware chunking (active)
│   ├── embedder.py            # Gemini embedding generation
│   ├── retriever.py           # cosine-similarity retrieval
│   └── generator.py           # grounded answer generation
├── .env.example
├── .gitignore
└── requirements.txt
```

---

## Current Status

- [x] PDF loading and text extraction
- [x] Chunking (naive → sentence-aware → structure-aware, iterated with real bugs found and fixed)
- [x] Embedding generation (Gemini API)
- [x] Vector storage (ChromaDB, persistent, cosine similarity)
- [x] Semantic + exact-ID retrieval
- [x] Grounded generation with source citation and relevance threshold
- [x] Agentic tool-calling (semantic search + exact lookup)
- [x] Chat UI (Streamlit)
- [x] Automated evaluation (pass/fail test suite)
- [ ] Deployed, publicly accessible demo

---

## About

Built by Soorya Pratap Singh, drawing on hands-on QA/testing experience (manual testing, regression testing, defect life cycle, SDLC/STLC) and an academic background in applied ML (DentaGAN — dental image segmentation, published at an international conference).