from sentence_transformers import SentenceTransformer, util
import torch
import json

def load_data_and_model():
    # Load document embeddings and texts
    with open("data/lectureNotes.json", "r") as f:
        data = json.load(f)
    doc_names = list(data.keys())
    texts = list(data.values())
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embedder.encode(texts, convert_to_tensor=True)

    # Load model and tokenizer
    from utils.model_utils import load_model
    model, tokenizer = load_model()

    return doc_names, texts, doc_embeddings, embedder, model, tokenizer

def retrieve_and_answer(question, doc_names, texts, embeddings, embedder, model, tokenizer, top_k=3):
    # Step 1: Embed the question
    question_embedding = embedder.encode([question], convert_to_tensor=True)

    # Step 2: Compute cosine similarity
    similarities = util.cos_sim(question_embedding, embeddings)[0]
    top_indices = torch.topk(similarities, top_k).indices.tolist()

    # Step 3: Retrieve contexts and generate answers
    relevant_contexts = [texts[idx] for idx in top_indices]
    relevant_docs = [doc_names[idx] for idx in top_indices]
    answers = []
    for context in relevant_contexts:
        answer = generate_answer(question, context, model, tokenizer)
        answers.append(answer)

    return relevant_docs, answers
