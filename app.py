import os
import streamlit as st
import pandas as pd
import openai
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# ── SETUP ────────────────────────────────────────────────

# Read API key securely
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Optional: Show title
st.title("CRM Smart Search with LlamaIndex")

# Load your CSV
csv_file = "Master_Personal_CRM_Clay.csv"
df = pd.read_csv(csv_file).astype(str)

# ── EMBEDDING & INDEX SETUP ──────────────────────────────

# Combine row data for embedding
documents = df.agg(" | ".join, axis=1).tolist()

# Set up embedding function and service context
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=openai.api_key)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# Build index
index = VectorStoreIndex.from_documents(
    documents=[{"text": doc, "metadata": {"row": i}} for i, doc in enumerate(documents)],
    service_context=service_context
)

# ── UI INTERACTION ──────────────────────────────────────

query = st.text_input("Ask a question about your CRM:")
k = st.slider("Number of results:", 1, 20, 5)

if st.button("Search") and query:
    results = index.as_query_engine().query(query)
    
    # Basic display
    st.write("**Results:**")
    for res in results.sources[:k]:
        row_index = res.metadata.get("row", None)
        if row_index is not None:
            st.write(df.iloc[int(row_index)])
        st.write("---")