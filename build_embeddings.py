# scripts/build_embeddings.py

import os
import pandas as pd
import numpy as np
import faiss
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import torch

# Load BGE embedding model (offline)
MODEL_PATH = "./models/bge-base-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)

def embed(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
    return embeddings.cpu().numpy()

# Load Excel data
obligations = pd.read_excel("./data/obligation_mapping.xlsx")
kpci = pd.read_excel("./data/kpci.xlsx")
nfri = pd.read_excel("./data/nfri.xlsx")

# Load policy PDFs
def read_policy_pdfs(path):
    ids, texts = [], []
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            pdf = PdfReader(os.path.join(path, file))
            text = " ".join([p.extract_text() or "" for p in pdf.pages])
            ids.append(file.replace(".pdf", ""))
            texts.append(text)
    return ids, texts

policy_ids, policy_texts = read_policy_pdfs("./data/policies")

# Combine text fields
obligations["text"] = obligations["Obligation Title"].fillna("") + " " + obligations["Obligation Summary"].fillna("")
kpci["text"] = kpci["Control Title"].fillna("") + " " + kpci["Control Description"].fillna("")
nfri["text"] = nfri["Issue Title"].fillna("") + " " + nfri["Issue Rationale"].fillna("") + " " + nfri["Resolution Summary"].fillna("")

# Embed corpora
print("Embedding policy documents...")
policy_embeddings = embed(policy_texts)
print("Embedding NFRIs...")
nfri_embeddings = embed(nfri["text"].tolist())
print("Embedding KPCIs...")
kpci_embeddings = embed(kpci["text"].tolist())

# Save FAISS + ID mapping
os.makedirs("./models/faiss_indexes", exist_ok=True)

def save_index(embeddings, ids, texts, name):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, f"./models/faiss_indexes/{name}.index")
    pd.Series(ids).to_csv(f"./models/faiss_indexes/{name}_ids.csv", index=False)
    pd.Series(texts).to_csv(f"./models/faiss_indexes/{name}_corpus.csv", index=False)

save_index(policy_embeddings, policy_ids, policy_texts, "policy")
save_index(nfri_embeddings, nfri["Issue ID"].tolist(), nfri["text"].tolist(), "nfri")
save_index(kpci_embeddings, kpci["Control ID"].tolist(), kpci["text"].tolist(), "kpci")

print("✅ Embedding and indexing completed.")
