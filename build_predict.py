# scripts/predict_mappings.py

import faiss
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = "./models/bge-base-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)

def embed(text):
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def load_faiss(name):
    index = faiss.read_index(f"./models/faiss_indexes/{name}.index")
    ids = pd.read_csv(f"./models/faiss_indexes/{name}_ids.csv", header=None)[0].tolist()
    corpus = pd.read_csv(f"./models/faiss_indexes/{name}_corpus.csv", header=None)[0].tolist()
    return index, ids, corpus

def hybrid_search(query, index, ids, corpus, top_k=5):
    # Vector similarity
    vec = embed(query).astype('float32')
    _, idxs = index.search(vec, top_k)

    # TF-IDF similarity
    tfidf = TfidfVectorizer().fit(corpus)
    tfidf_matrix = tfidf.transform(corpus)
    query_vec = tfidf.transform([query])
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    tfidf_top_k = cosine_scores.argsort()[-top_k:][::-1]

    scores = {}
    for i in idxs[0]: scores[ids[i]] = scores.get(ids[i], 0) + 1.0
    for i in tfidf_top_k: scores[ids[i]] = scores.get(ids[i], 0) + 0.8
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

# === Input Obligation
title = "Ensure end-to-end encryption for customer data"
summary = "All customer-facing systems must implement 256-bit encryption and store logs securely."

combined = title + " " + summary

# === Load indexes
policy_index, policy_ids, policy_corpus = load_faiss("policy")
nfri_index, nfri_ids, nfri_corpus = load_faiss("nfri")
kpci_index, kpci_ids, kpci_corpus = load_faiss("kpci")

# === Predict
pred_policies = hybrid_search(combined, policy_index, policy_ids, policy_corpus)
pred_nfris = hybrid_search(combined, nfri_index, nfri_ids, nfri_corpus)
pred_kpcis = hybrid_search(combined, kpci_index, kpci_ids, kpci_corpus)

print("🔐 Predicted Policy IDs:", [i[0] for i in pred_policies])
print("⚠️  Predicted NFRI IDs:", [i[0] for i in pred_nfris])
print("🛡️  Predicted KPCI IDs:", [i[0] for i in pred_kpcis])
