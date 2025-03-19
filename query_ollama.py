import ollama
from retrieve import retrieve_relevant_text

def generate_response_with_context(query):
    retrieved_text = "\n".join(retrieve_relevant_text(query))
    prompt = f"Use the following context to answer:\n\n{retrieved_text}\n\nQuestion: {query}"
    
    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

if __name__ == "__main__":
    query = input("Ask a question: ")
    response = generate_response_with_context(query)
    print("\n🔹 Ollama Response:\n", response)
