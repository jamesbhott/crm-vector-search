import os
import streamlit as st
import pandas as pd
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI

# ── CONFIG ────────────────────────────────
st.set_page_config(page_title="CRM GPT Search", layout="wide")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

DATA_FILE = "Master_Personal_CRM_Clay.csv"

# ── SETUP LLM CONTEXT ─────────────────────
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

# ── LOAD DATA ─────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data():
    df = pd.read_csv(DATA_FILE).astype(str)
    lines = df.apply(lambda row: " | ".join(row.values), axis=1).tolist()
    return df, lines

df, lines = load_data()

# ── BUILD INDEX ───────────────────────────
@st.cache_resource(show_spinner="Building search index...")
def build_index():
    docs = [f"{line}" for line in lines]
    with open("crm_docs.txt", "w") as f:
        f.write("\n".join(docs))
    reader = SimpleDirectoryReader(input_files=["crm_docs.txt"])
    index = VectorStoreIndex.from_documents(reader.load_data(), service_context=service_context)
    return index

index = build_index()
query_engine = index.as_query_engine()

# ── STREAMLIT UI ──────────────────────────
st.title("CRM GPT Search")

query = st.text_input("Ask a question about your contacts:")

if st.button("Run Search") and query:
    with st.spinner("Thinking..."):
        response = query_engine.query(query)
    st.write(response)