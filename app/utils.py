from retriever import Retriever
from summarizer import Summarizer
from qa_model import QAWithT5
import json
from sentence_transformers import SentenceTransformer
import numpy as np


def extract_relevant_context(doc_content, query, top_k=5):
    sentences = doc_content.split(". ")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_vec = model.encode(query)
    sentence_embeddings = model.encode(sentences)

    # Rank sentences by relevance
    scores = np.dot(sentence_embeddings, query_vec)
    ranked_sentences = sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)[:top_k]

    # Deduplicate and clean sentences
    unique_sentences = []
    for _, sent in ranked_sentences:
        clean_sent = sent.strip()
        if clean_sent not in unique_sentences and len(clean_sent.split()) > 5:
            unique_sentences.append(clean_sent)

    return ". ".join(unique_sentences)

def process_query(retriever, summarizer, qa_model, query):
    # Retrieve top document
    top_docs = retriever.retrieve(query, top_k=1)
    if not top_docs:
        return "No relevant documents found.", "", ""

    doc_name, _ = top_docs[0]
    with open("data/processed/lecture_notes.json", "r") as f:
        all_docs = json.load(f)
    doc_content = all_docs[doc_name]

    # Extract and clean relevant context
    relevant_context = extract_relevant_context(doc_content, query)

    # Summarize the context
    summarized_content = summarizer.summarize(relevant_context)

    # Generate answer with QA model
    answer = qa_model.answer_question(summarized_content, query)

    return relevant_context, summarized_content, answer
