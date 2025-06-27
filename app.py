import pysqlite3  # preload a modern sqlite binding before chromadb imports
import os
import pandas as pd
import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import openai

# ── CONFIG ───────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
CSV_FILE       = 'Master_Personal_CRM_Clay.csv'
CHROMA_DIR     = './chroma_db'    # DuckDB+Parquet store here
MODEL          = 'text-embedding-ada-002'

# ── INITIALIZE CHROMA CLIENT WITH DUCKDB+PARQUET ────
settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=CHROMA_DIR
)
client = chromadb.Client(settings=settings)

# ── GET OR CREATE COLLECTION ─────────────────────────
names = [c.name for c in client.list_collections()]
if "crm" in names:
    collection = client.get_collection("crm")
else:
    collection = client.create_collection("crm")

# ── BUILD INDEX IF EMPTY ─────────────────────────────
if collection.count() == 0:
    st.info("Building vector index… this may take a minute.")
    df = pd.read_csv(CSV_FILE).astype(str)
    texts = df.agg(" | ".join, axis=1).tolist()

    ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key)
    collection.add(
        documents=texts,
        embeddings=ef(texts),
        metadatas=df.to_dict(orient="records"),
        ids=[str(i) for i in range(len(texts))]
    )
    collection.persist()
    st.success("Index built!")

# ── STREAMLIT UI ─────────────────────────────────────
st.title("CRM Vector Search (DuckDB+Parquet)")
query = st.text_input("Enter your query:")
k     = st.slider("Number of results:", 1, 20, 5)

if st.button("Search") and query:
    ef      = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key)
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["metadatas", "distances"]
    )
    rows   = results["metadatas"][0]
    scores = results["distances"][0]
    df_out = pd.DataFrame(rows)
    df_out["Score"] = scores
    st.dataframe(df_out)