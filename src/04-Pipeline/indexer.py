from document_loader import load_all_pdfs
from chunker import create_chunks
from embeddings import create_embedding_model, generate_embeddings
from vector_store import create_index, save_index
from pathlib import Path


def build_index():

    print("Loading documents...")
    pages = load_all_pdfs()

    print(f"Loaded {len(pages)} pages.")

    print("\nCreating chunks...")
    chunks = create_chunks(pages)

    print(f"Created {len(chunks)} chunks.")

    print("\nLoading embedding model...")
    model = create_embedding_model()

    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks, model)

    print(f"Generated embeddings with shape: {embeddings.shape}")

    print("\nCreating FAISS index...")
    index = create_index(embeddings)

    print(f"FAISS index contains {index.ntotal} vectors.")

    # Create index directory
    Path("index").mkdir(exist_ok=True)

    # Save index and chunks
    save_index(index, chunks)

    print("\nIndex and chunks saved successfully!")

    return index, chunks


if __name__ == "__main__":

    index, chunks = build_index()

    print("\nIndexing completed successfully!")