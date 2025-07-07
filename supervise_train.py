# scripts/train_model.py

import os
import pandas as pd
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# === STEP 1: Set proxy (if needed)
os.environ["HTTPS_PROXY"] = "http://your.proxy.address:port"  # <-- Change this
os.environ["HTTP_PROXY"] = "http://your.proxy.address:port"

# === STEP 2: Load Sentence Transformer
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# === STEP 3: Paths
POLICY_FOLDER = "./data/policies/"
OUTPUT_DIR = "./models/faiss_indexes/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === STEP 4: Load Excel files
obligations = pd.read_excel("./data/obligation_mapping.xlsx")
kpci = pd.read_excel("./data/kpci.xlsx")
nfri = pd.read_excel("./data/nfri.xlsx")

# === STEP 5: Combine all meaningful obligation columns for embedding
text_cols = [
    "Obligation Title", "Obligation Summary", "Rule Citation", "Inventory Jurisdiction",
    "Key Designation", "Regulation ID", "Cross-Border/Extra-Territorial Impact",
    "GCRS info", "Mapped Risk Taxonomy", "Procedure details", "Regulatory Reporting Requirements"
]
obligations["combined_text"] = obligations[text_cols].fillna("").apply(lambda row: " ".join(row), axis=1)

# === STEP 6: Combine KPCI and NFRI fields for embedding
kpci["combined_text"] = kpci["Control Title"].fillna("") + " " + kpci["Control Description"].fillna("")
nfri["combined_text"] = (
    nfri["Issue Title"].fillna("") + " " +
    nfri["Issue Rationale"].fillna("") + " " +
    nfri["Resolution Summary"].fillna("")
)

# === STEP 7: Load and read policy PDFs
def read_policy_pdfs(folder_path):
    ids, texts = [], []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            try:
                path = os.path.join(folder_path, file)
                pdf = PdfReader(path)
                text = " ".join([page.extract_text() or "" for page in pdf.pages])
                ids.append(file.replace(".pdf", ""))
                texts.append(text)
            except Exception as e:
                print(f"Failed to read {file}: {e}")
    return ids, texts

policy_ids, policy_texts = read_policy_pdfs(POLICY_FOLDER)

# === STEP 8: Embed all text
def embed(texts):
    return MODEL.encode(texts, show_progress_bar=True).astype('float32')

print("🔄 Embedding obligations...")
obligation_embeddings = embed(obligations["combined_text"])

print("🔄 Embedding policies...")
policy_embeddings = embed(policy_texts)

print("🔄 Embedding NFRIs...")
nfri_embeddings = embed(nfri["combined_text"])

print("🔄 Embedding KPCIs...")
kpci_embeddings = embed(kpci["combined_text"])

# === STEP 9: Save FAISS Indexes
def save_index(embeddings, ids, name):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(OUTPUT_DIR, f"{name}.index"))
    pd.Series(ids).to_csv(os.path.join(OUTPUT_DIR, f"{name}_ids.csv"), index=False)

save_index(policy_embeddings, policy_ids, "policy")
save_index(nfri_embeddings, nfri["Issue ID"].tolist(), "nfri")
save_index(kpci_embeddings, kpci["Control ID"].tolist(), "kpci")

print("✅ FAISS indexes saved.")

# === STEP 10: Save Supervised Training Data (X and Y)
def expand_multilabels(value):
    if pd.isna(value):
        return []
    return [v.strip() for v in str(value).split("|")]

supervised_df = pd.DataFrame({
    "Obligation ID": obligations["Obligation ID"],
    "Embedding": list(obligation_embeddings),
    "Mapped NFRI": obligations["Mapped NFRI"].apply(expand_multilabels),
    "KPCI ID": obligations["KPCI ID"].apply(expand_multilabels),
    "Policy ID": obligations["Policy ID"].apply(expand_multilabels),
})

# Save embeddings + labels for future ML
supervised_df.to_pickle("./data/supervised_training_data.pkl")
supervised_df[["Obligation ID", "Mapped NFRI", "KPCI ID", "Policy ID"]].to_csv("./data/labels_only.csv", index=False)

print("✅ Supervised training data saved.")
