import sys
import os

# Add project root to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from processing.chunking import chunk_text


def load_documents(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()  # remove extra spaces

            documents.append({
                "source": file,
                "text": text
            })

    return documents


if __name__ == "__main__":
    DATA_PATH = os.path.join(BASE_DIR, "data", "raw_docs")

    docs = load_documents(DATA_PATH)

    all_chunks = []

    for doc in docs:
        text_length = len(doc["text"])
        print(f"{doc['source']} length: {text_length}")

        if text_length == 0:
            print("⚠ File is empty. Skipping...\n")
            continue

        chunks = chunk_text(doc["text"])
        print(f"{doc['source']} -> {len(chunks)} chunks created\n")

        all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")
