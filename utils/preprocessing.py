import torch
from sentence_transformers import util

def encode_documents(documents, dpr_tokenizer, dpr_encoder, device, batch_size=16, max_length=512):
    """Encode all documents using DPR with truncation."""
    all_embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        inputs = dpr_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            embeddings = dpr_encoder(**inputs).pooler_output
        all_embeddings.append(embeddings)
    return torch.cat(all_embeddings)

def retrieve_top_k(query, documents, doc_embeddings, dpr_tokenizer, dpr_encoder, device, top_k=3):
    """Retrieve top-k relevant documents."""
    query_inputs = dpr_tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        query_embedding = dpr_encoder(**query_inputs).pooler_output

    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k]
    return " ".join([documents[idx] for idx in top_indices])
