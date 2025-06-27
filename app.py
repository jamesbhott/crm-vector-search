import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import openai
import pickle

# ── CONFIG ───────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")

# Local NLTK cache directory to avoid permission issues
os.environ['NLTK_DATA'] = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(os.environ['NLTK_DATA'], exist_ok=True)

DATA_DIR = "data"
INDEX_DIR = "storage"

# ── LLM + EMBEDDING SETUP ────────────────────────────
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(chunk_size=512)

# ── INDEX SETUP ──────────────────────────────────────
if not os.path.exists(INDEX_DIR):
    st.info("Building index… this may take a minute.")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    st.success("Index built!")
else:
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context, service_context=service_context)

# ── STREAMLIT UI ─────────────────────────────────────
st.title("CRM Vector Search with LlamaIndex")
query_text = st.text_input("Ask a question about your CRM data:", "")

if st.button("Search") and query_text:
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    st.write(response)

# ── OPTIONAL: DEBUG ──────────────────────────────────
# st.write(index)
