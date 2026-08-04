import chromadb
from embedder import embed_chunks

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="sentinel_docs",
    metadata={"hnsw:space": "cosine"}
)

def index_chunks(chunks: list[str]):
    """Embed chunks and store them in the persistent vector database."""
    embeddings = embed_chunks(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        embeddings=[e.tolist() for e in embeddings],
        documents=chunks
    )
    return len(chunks)

def query_chunks(query: str, top_k: int = 3) -> list[tuple[str, float]]:
    """Query the vector database for the most relevant chunks."""
    from retriever import embed_query
    query_embedding = embed_query(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    similarities = [1 - d for d in distances]

    return list(zip(documents, similarities))

if __name__ == "__main__":
    from loader import load_pdf
    from structure_chunker import chunk_by_structure

def lookup_by_id(item_id: str) -> str:
    """Look up a chunk by exact BUG- or TC- ID match, bypassing similarity search."""
    all_docs = collection.get()
    for doc in all_docs["documents"]:
        if item_id.upper() in doc.upper():
            return doc
    return f"No item found matching '{item_id}'."

    text = load_pdf("data/sample.pdf")
    chunks = chunk_by_structure(text)
    count = index_chunks(chunks)
    print(f"Indexed {count} chunks into ChromaDB")

    query = "why did the availability filter test fail?"
    results = query_chunks(query)
    print(f"\nQuery: {query}\n")
    for doc, score in results:
        print(f"Score: {score:.4f}")
        print(doc)
        print("---")

    irrelevant_query = "what's the weather like today?"
    irrelevant_results = query_chunks(irrelevant_query)
    print(f"\nIrrelevant query: {irrelevant_query}\n")
    for doc, score in irrelevant_results:
        print(f"Score: {score:.4f}")

    