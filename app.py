import os
import pickle
import pandas as pd
import streamlit as st
import numpy as np
import faiss
import openai

# ── CONFIG ─────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
CSV_FILE       = "Master_Personal_CRM_Clay.csv"
INDEX_FILE     = "faiss_index.pkl"
DATA_FILE      = "faiss_data.pkl"
EMB_MODEL      = "text-embedding-ada-002"
BATCH_SIZE     = 500
DIM            = 1536

def build_index():
    df = pd.read_csv(CSV_FILE).astype(str)
    texts = df.agg(" | ".join, axis=1).tolist()

    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp  = openai.Embedding.create(model=EMB_MODEL, input=batch)
        embeddings.extend([d.embedding for d in resp.data])

    X = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(DIM)
    index.add(X)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index, f)
    with open(DATA_FILE, "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)

    return index, df.to_dict(orient="records")

def load_index():
    with open(INDEX_FILE, "rb") as f:
        index = pickle.load(f)
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
    return index, data

# ── LOAD OR BUILD ─────────────────────────────────────
if os.path.exists(INDEX_FILE) and os.path.exists(DATA_FILE):
    index, records = load_index()
else:
    st.info("Building FAISS index… one-time operation")
    index, records = build_index()
    st.success("Index built!")

# ── STREAMLIT UI ───────────────────────────────────────
st.title("CRM Vector Search (FAISS)")
query = st.text_input("Enter your query:")
k     = st.slider("Number of results:", 1, 20, 5)

if st.button("Search") and query:
    q_emb = np.array(
        openai.Embedding.create(model=EMB_MODEL, input=[query]).data[0].embedding,
        dtype="float32",
    ).reshape(1, -1)

    D, I = index.search(q_emb, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        rec = records[idx].copy()
        rec["Score"] = float(dist)
        results.append(rec)

    st.dataframe(pd.DataFrame(results))