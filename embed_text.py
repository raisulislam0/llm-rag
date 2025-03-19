import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Read text file
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Split text into chunks
def split_text(text, chunk_size=256):
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Directory path
data_dir = "data"

# Store all chunks and their source files
all_chunks = []
chunk_sources = []

# Process all text files in the data directory
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_dir, filename)
        print(f"Processing {file_path}...")
        
        # Load text and split
        text_data = read_text_file(file_path)
        file_chunks = split_text(text_data)
        
        # Add to collection
        all_chunks.extend(file_chunks)
        
        # Keep track of which file each chunk came from
        chunk_sources.extend([filename] * len(file_chunks))

# Generate embeddings for all chunks
print(f"Generating embeddings for {len(all_chunks)} chunks...")
embeddings = embed_model.encode(all_chunks)

# Convert to NumPy array
embeddings_np = np.array(embeddings, dtype=np.float32)

# Create output directory if it doesn't exist
os.makedirs("embeddings", exist_ok=True)

# Store in FAISS index
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

# Save index, chunks, and sources
faiss.write_index(index, "embeddings/vector_index.faiss")
np.save("embeddings/chunks.npy", all_chunks)
np.save("embeddings/chunk_sources.npy", chunk_sources)

print(f"✅ Embeddings for {len(all_chunks)} chunks from {len(set(chunk_sources))} files generated and stored successfully!")