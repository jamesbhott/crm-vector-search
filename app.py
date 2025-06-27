import pysqlite3                    # ensure modern sqlite before anything else
import os
import pandas as pd
import streamlit as st

from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import openai

# ── CONFIG ───────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
CSV_FILE       = 'Master_Personal_CRM_Clay.csv'
CHROMA_DIR     = './chroma_db'
MODEL          = 'text-embedding-ada-002'

# ── INITIALIZE CLIENT WITH NEW API ────────────────────
settings = Settings(
    database_impl="duckdb+parquet",
    persist_directory=CHROMA_DIR
)
client = Client(settings=settings)

# ── GET OR CREATE COLLECTION ─────────────────────────
names = [c.name for c in client.list_collections()]
collection = client.get_collection("crm") if "crm" in names else client.create_collection("crm")

# ── BUILD INDEX IF EMPTY ─────────────────────────────
if collection.count() == 0:
    st.info("Building vector index… this may take a minute.")
    df    = pd.read_csv(CSV_FILE).astype(str)
    texts = df.agg(" | ".join, axis=1).tolist()
    ef    = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key)

    collection.add(
        documents=texts,
        embeddings=ef(texts),
        metadatas=df.to_dict(orient="records"),
        ids=[str(i) for i in range(len(texts))]
    )
    collection.persist()
    st.success("Index built!")

# ── STREAMLIT UI ─────────────────────────────────────
st.title("CRM Vector Search (ChromaDB + DuckDB)")
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