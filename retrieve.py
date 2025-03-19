import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load index and text chunks
index = faiss.read_index("embeddings/vector_index.faiss")
chunks = np.load("embeddings/chunks.npy", allow_pickle=True)

# Function to retrieve relevant text chunks
def retrieve_relevant_text(query, top_k=3):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return [chunks[i] for i in indices[0]]

# Test retrieval
if __name__ == "__main__":
    query = input("Enter your query: ")
    results = retrieve_relevant_text(query)
    print("\n".join(results))
