from sentence_transformers import SentenceTransformer
import numpy as np

def load_documents(path):
    with open(path, "r") as f:
        text = f.read()
    return [t.strip() for t in text.split("。") if t.strip()]

def get_top_k_documents(query, documents, model_name, k):
    emb_model = SentenceTransformer(model_name)
    query_embedding = emb_model.encode([query])
    doc_embeddings = emb_model.encode(documents)
    scores = (query_embedding @ doc_embeddings.T) * 100
    top_indices = scores[0].argsort()[::-1][:k]
    return [documents[i] for i in top_indices]
