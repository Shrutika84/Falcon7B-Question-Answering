import torch
from sentence_transformers import util

def encode_documents(documents, batch_size=16, max_length=512):
    embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        inputs = dpr_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = dpr_encoder(**inputs).pooler_output
        embeddings.append(outputs)
    return torch.cat(embeddings)

def retrieve_top_k(query, documents, doc_embeddings, top_k=3):
    query_inputs = dpr_tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        query_embedding = dpr_encoder(**query_inputs).pooler_output
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k]
    return " ".join([documents[idx] for idx in top_indices])
