import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from retrieval.retriever import Retriever
from llm.qa_chain import generate_answer


def main():
    retriever = Retriever(BASE_DIR)

    query = input("Enter your question: ")

    retrieved_chunks = retriever.retrieve(query)

    print("\nRetrieved Context:\n")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"--- Chunk {i} ---")
        print(chunk)
        print()

    answer = generate_answer(query, retrieved_chunks)

    print("\nFinal Answer:\n")
    print(answer)



if __name__ == "__main__":
    main()
