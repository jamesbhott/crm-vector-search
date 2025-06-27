import os
import streamlit as st
import pandas as pd
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import openai

# ── CONFIG ───────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
DATA_FILE = "Master_Personal_CRM_Clay.csv"

# ── BUILD INDEX ──────────────────────────────────────
st.title("CRM LlamaIndex Search")

if not os.path.exists(DATA_FILE):
    st.error(f"{DATA_FILE} not found.")
    st.stop()

st.info("Loading and indexing data…")
df = pd.read_csv(DATA_FILE).astype(str)
text_data = df.apply(lambda row: " | ".join(row.values), axis=1).tolist()

# Save text lines to a temp file for ingestion
temp_dir = ".temp_docs"
os.makedirs(temp_dir, exist_ok=True)
temp_path = os.path.join(temp_dir, "data.txt")
with open(temp_path, "w") as f:
    for line in text_data:
        f.write(line + "\n")

# Build index
reader = SimpleDirectoryReader(temp_dir)
documents = reader.load_data()
service_context = ServiceContext.from_defaults(
    embed_model=OpenAIEmbedding(model="text-embedding-ada-002", api_key=openai.api_key),
    llm=OpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openai.api_key)
)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()

st.success("Index ready!")

# ── SEARCH UI ────────────────────────────────────────
query = st.text_input("Ask a question about your CRM:", "")
if st.button("Search") and query:
    response = query_engine.query(query)
    st.write(response)

# ── CLEANUP ──────────────────────────────────────────
import shutil
shutil.rmtree(temp_dir, ignore_errors=True)