import os
import pandas as pd
import streamlit as st
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms import OpenAI
import faiss

# ── SETUP ──────────────────────────────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]
CSV_FILE = 'Master_Personal_CRM_Clay.csv'

st.title("CRM Vector Search (FAISS + LlamaIndex)")

# ── DATA PREP ──────────────────────────────────
df = pd.read_csv(CSV_FILE).astype(str)
docs = df.apply(lambda row: " | ".join(row), axis=1).tolist()

# ── INDEX BUILD ────────────────────────────────
embed_model = OpenAIEmbedding(api_key=openai.api_key)
llm = OpenAI(api_key=openai.api_key)

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
index = GPTVectorStoreIndex.from_documents(
    [SimpleDirectoryReader(input_files=[CSV_FILE]).load_data()],
    service_context=service_context
)

# ── SEARCH ─────────────────────────────────────
query = st.text_input("Ask a question or describe what you want:")

if query:
    chat_engine = index.as_chat_engine()
    response = chat_engine.chat(query)
    st.write(response.response)