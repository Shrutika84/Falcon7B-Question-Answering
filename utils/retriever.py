import json
from sentence_transformers import SentenceTransformer, util
import torch

def load_json_data(json_path):
    """
    Load JSON data and extract document names and texts.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        doc_names = list(data.keys())
        texts = list(data.values())
    else:
        raise ValueError("Unexpected JSON structure. Expected a dictionary.")
    return doc_names, texts, None

def embed_documents(texts):
    """
    Generate embeddings for all documents using a SentenceTransformer model.
    """
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    return embedder, embeddings

def retrieve_and_answer(question, doc_names, texts, embeddings, embedder, model, tokenizer, top_k=3):
    """
    Retrieve relevant documents based on a question and generate an answer using Mistral-7B.
    """
    question_embedding = embedder.encode([question], convert_to_tensor=True)
    similarities = util.cos_sim(question_embedding, embeddings)[0]
    top_indices = torch.topk(similarities, top_k).indices.tolist()

    relevant_contexts = [texts[idx] for idx in top_indices]
    relevant_docs = [doc_names[idx] for idx in top_indices]

    answers = []
    for context, doc in zip(relevant_contexts, relevant_docs):
        answer = get_answer_mistral(question, context, model, tokenizer)
        answers.append({"doc": doc, "answer": answer})

    return answers
