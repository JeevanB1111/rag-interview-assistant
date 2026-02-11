import sys
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from ingestion.load_docs import load_documents
from processing.chunking import chunk_text


def build_index():
    DATA_PATH = os.path.join(BASE_DIR, "data", "raw_docs")

    print("Loading documents...")
    docs = load_documents(DATA_PATH)

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")
    embeddings = model.encode(all_chunks)

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, os.path.join(BASE_DIR, "faiss_index.index"))

    # Save chunks separately
    with open(os.path.join(BASE_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)

    print("Index built and saved successfully!")


if __name__ == "__main__":
    build_index()
