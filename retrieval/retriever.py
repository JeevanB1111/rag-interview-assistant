import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, base_dir):
        self.base_dir = base_dir

        # Load FAISS index
        index_path = os.path.join(base_dir, "faiss_index.index")
        self.index = faiss.read_index(index_path)

        # Load stored chunks
        with open(os.path.join(base_dir, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)

        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode([query])

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results
