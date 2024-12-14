import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

class Retriever:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.index = None

    def index_documents(self, json_file):
        with open(json_file, "r") as f:
            docs = json.load(f)

        self.docs = {k: v for k, v in docs.items() if v.strip()}
        print(f"Documents loaded: {len(self.docs)}")

        embeddings = []
        for doc_name, content in self.docs.items():
            if content.strip():
                print(f"Generating embedding for: {doc_name}, Length: {len(content)} characters")
                embeddings.append(self.model.encode(content))
            else:
                print(f"Skipping empty document: {doc_name}")

        if not embeddings:
            raise ValueError("No embeddings generated. Check your documents or processing pipeline.")

        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(np.array(embeddings))
        print("Embeddings indexed successfully.")

    def retrieve(self, query, top_k=1):
        query_vec = self.model.encode(query)
        distances, indices = self.index.search(np.array([query_vec]), top_k)
        results = [(list(self.docs.keys())[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        return results
