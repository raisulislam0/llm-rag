rag_project/
│── data/
│   ├── data.txt  # Your text file to embed
│── embeddings/
│   ├── vector_index.faiss  # FAISS index (generated after running embedding script)
│   ├── chunks.npy  # Stored text chunks (for retrieval)
│── src/
│   ├── embed_text.py  # Script to generate embeddings
│   ├── retrieve.py  # Script to retrieve relevant text
│   ├── query_ollama.py  # Script to query Ollama with retrieved text
│── main.py  # Run the complete RAG pipeline
│── requirements.txt  # Required Python packages

llm model: llama3.1

