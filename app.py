import os
import pandas as pd
import numpy as np
import faiss
import openai
import streamlit as st

# ── CONFIG ───────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
INDEX_FILE  = 'crm_index.faiss'
CSV_FILE    = 'Master_Personal_CRM_Clay.csv'
MODEL       = 'text-embedding-ada-002'
TOP_K       = 10

# ── LOAD DATA & INDEX ─────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_FILE)
    index = faiss.read_index(INDEX_FILE)
    return df, index

df, index = load_data()

st.title("CRM Vector Search")
st.write("Enter a natural-language query and see the top matches:")

# ── USER INPUT ────────────────────────────────────────
query = st.text_input("🔍 Search", placeholder="e.g. ‘marketing agencies in tech’")
k     = st.slider("Number of results", min_value=1, max_value=20, value=5)

if st.button("Search") and query:
    # 1) embed the query
    with st.spinner("Embedding query…"):
        resp = openai.embeddings.create(model=MODEL, input=[query])
        q_vec = np.array(resp.data[0].embedding, dtype='float32').reshape(1, -1)

    # 2) search the FAISS index
    with st.spinner("Searching index…"):
        D, I = index.search(q_vec, k)

    # 3) collect results
    results = df.iloc[I[0]].copy()
    results["Score"] = D[0]

    # 4) display
    st.success(f"Found {len(results)} results:")
    st.dataframe(results.reset_index(drop=True))