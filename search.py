import os
import pandas as pd
import numpy as np
import faiss
import openai

# ── CONFIG ───────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
INDEX_FILE  = 'crm_index.faiss'
CSV_FILE    = 'Master_Personal_CRM_Clay.csv'
MODEL       = 'text-embedding-ada-002'
TOP_K       = 5

# ── LOAD DATA & INDEX ─────────────────────────────────
df    = pd.read_csv(CSV_FILE)
index = faiss.read_index(INDEX_FILE)

# ── SEARCH FUNCTION ──────────────────────────────────
def search(query, k=TOP_K):
    # 1) Embed the query
    resp = openai.embeddings.create(model=MODEL, input=[query])
    q_vec = np.array(resp.data[0].embedding, dtype='float32').reshape(1, -1)
    # 2) Query FAISS
    D, I = index.search(q_vec, k)
    # 3) Gather results
    results = df.iloc[I[0]].copy()
    results['score'] = D[0]
    return results

# ── INTERACTIVE PROMPT ──────────────────────────────
if __name__ == "__main__":
    print(f"Loaded {index.ntotal} vectors. Top-{TOP_K} search ready!")
    while True:
        q = input("\nEnter search query (or ‘quit’): ").strip()
        if not q or q.lower().startswith('q'):
            print("Exiting search.")
            break
        out = search(q)
        print("\nTop results:")
        print(out.to_string(index=False))