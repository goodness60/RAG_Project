import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer


# 🔑 Configure Gemini
load_dotenv()
genai.configure(api_key= os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# 🧠 Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 🗄️ Setup Chroma (in-memory for demo)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("notes")


# 📄 Load some example documents (replace with your notes later)
documents = [
    "Python lists are mutable, meaning they can be changed after creation.",
    "Python tuples are immutable, meaning they cannot be changed after creation.",
    "Streamlit is a Python framework for building simple web apps for data science and AI."
]

# Add embeddings to Chroma
embeddings = embedder.encode(documents).tolist()
collection.add(documents=documents, embeddings=embeddings, ids=[str(i) for i in range(len(documents))])

# 🎨 Streamlit UI
st.title("📚 RAG Chatbot (Gemini + Chroma)")
st.write("Ask a question about your notes!")

user_query = st.text_input("Your question:")
if st.button("Ask") and user_query:
    # Step 1: Embed the query
    query_embedding = embedder.encode([user_query]).tolist()

    # Step 2: Retrieve top documents
    results = collection.query(query_embeddings=query_embedding, n_results=2)
    retrieved_docs = results["documents"][0]
    context = "\n".join(retrieved_docs)

    # Step 3: Send to Gemini with context
    prompt = f"Answer the question based on the following notes:\n{context}\n\nQuestion: {user_query}"
    response = model.generate_content(prompt)

    # Step 4: Show answer
    st.subheader("Answer:")
    st.write(response.text)


