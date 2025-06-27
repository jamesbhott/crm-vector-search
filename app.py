import os
import streamlit as st
from llama_index.core import SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
import openai

# ── NLTK CACHE FIX ─────────────────────────────
os.environ["NLTK_DATA"] = "/mount/src/crm-vector-search/nltk_data"

# ── CONFIG ─────────────────────────────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]
DATA_DIR = "data"
INDEX_DIR = "index_storage"

# ── LOAD OR BUILD INDEX ────────────────────────
llm = OpenAI(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

if os.path.exists(INDEX_DIR):
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context, service_context=service_context)
else:
    docs = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    index.storage_context.persist(persist_dir=INDEX_DIR)

# ── STREAMLIT UI ───────────────────────────────
st.title("CRM Q&A with LlamaIndex")
query = st.text_input("Ask a question about your CRM data:", "")

if st.button("Search") and query:
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
    response = chat_engine.chat(query)
    st.write(response.response)