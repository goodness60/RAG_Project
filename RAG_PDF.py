import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# 🔑 Configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# 🧠 Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 🗄️ Setup Chroma (in-memory)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("notes")

# 📂 File uploader in Streamlit
st.title("📚 PDF-powered RAG Chatbot (Gemini + Chroma)")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Extract text from PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Split text into chunks (simple split by paragraphs for demo)
    chunks = text.split("\n\n")
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # Embed & store in Chroma
    embeddings = embedder.encode(chunks).tolist()
    ids = [str(i) for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)

    st.success(f"Loaded {len(chunks)} chunks from {uploaded_file.name}")

    # User query input
    user_query = st.text_input("Ask a question about the PDF:")

    if st.button("Add") and user_query:
        # Step 1: Embed query
        query_embedding = embedder.encode([user_query]).tolist()

        # Step 2: Retrieve relevant chunks
        results = collection.query(query_embeddings=query_embedding, n_results=3)
        retrieved_docs = results["documents"][0]
        context = "\n".join(retrieved_docs)

        # Step 3: Ask Gemini with context
        prompt = f"Answer the question based only on the following context:\n{context}\n\nQuestion: {user_query}"
        response = model.generate_content(prompt)

        # Step 4: Show answer
        st.subheader("Answer:")
        st.write(response.text)
