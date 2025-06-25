import pandas as pd
import numpy as np
import faiss

# Load your CSV
df = pd.read_csv('Master_Personal_CRM_Clay.CSV')

print(f"Loaded {len(df)} rows")

# Create dummy vectors for now (replace with your actual embeddings later)
dimension = 128
vectors = np.random.random((len(df), dimension)).astype('float32')

# Initialize FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

print(f"Loaded {index.ntotal} vectors into FAISS index.")