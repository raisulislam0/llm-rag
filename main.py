import os



print("🔹 Step 1: Generating embeddings...")
os.system("python src/embed_text.py")



print("\n🔹 Step 2: Running RAG-based Q&A...")
os.system("python src/query_ollama.py")
