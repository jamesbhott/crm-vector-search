import os
import pandas as pd
import numpy as np
import faiss
import openai
from time import sleep

# ── CONFIG ───────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
CSV_FILE    = 'Master_Personal_CRM_Clay.csv'    # ← lowercase “.csv”

# ── LOAD & PREP DATA ─────────────────────────────────
df    = pd.read_csv(CSV_FILE).astype(str)
texts = df.agg(" | ".join, axis=1).tolist()
print(f"Concatenated {len(texts)} rows from all columns")

MODEL      = 'text-embedding-ada-002'
BATCH_SIZE = 500

# ── EMBEDDING FUNCTION ──────────────────────────────
def get_embeddings(batch):
    resp = openai.embeddings.create(model=MODEL, input=batch)
    # resp.data is a list of objects; each has an .embedding attribute
    return [record.embedding for record in resp.data]

# ── BATCHED EMBEDDING & INDEX BUILDING ──────────────
vectors = []
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    embs  = get_embeddings(batch)
    vectors.extend(embs)
    print(f"  • Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
    sleep(1)  # throttle to avoid rate limits

vectors = np.array(vectors, dtype='float32')
print("Embeddings array shape:", vectors.shape)

# ── BUILD & SAVE FAISS INDEX ───────────────────────
dim   = vectors.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(vectors)
print(f"Indexed {index.ntotal} vectors (dim={dim})")

faiss.write_index(index, 'crm_index.faiss')
print("Saved FAISS index to crm_index.faiss")