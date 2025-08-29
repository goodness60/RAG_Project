import os
import time
from typing import List

from dotenv import load_dotenv
import streamlit as st
import google.api_core.exceptions as google_exceptions
import google.generativeai as genai
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch


#CONFIG
device = "cuba" if torch.cuda.is_available() else "cpu"

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)


chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=None  # disables disk persistence
    )
)


collection = chroma_client.get_or_create_collection("pdf_chunks")

#HELPERS
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file given a path."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text
    
    
def chunk_text(text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
    """Splits long text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_length
        chunk = words[start:end]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        start += max_length - overlap
    return chunks


def embed_and_store(doc_id: str, text: str):
    """Embed chunks and store them in chroma."""
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings)
    
    
def retrieve(query: str, k: int = 5) ->List[str]:
    """Retrieve top-k chunks from chroma given a query."""
    query_embedding = embedder.encode([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results["documents"][0]

def safe_generate(prompts):
    responses = []
    for prompt in prompts:
        while True:
            try:
                response = model.generate_content(prompt)
                responses.append(response.text)   # ✅ append to responses, not response
                break
            except google_exceptions.ResourceExhausted:
                wait_time = 40
                st.warning(f"⚠️ Quota exceeded. Retrying in {wait_time}s...")
                time.sleep(wait_time)
    return responses


# =============== LOAD PDFS ===============
# 👇 Add all your PDFs here
PDF_FILES = "data_files"
    

st.set_page_config(page_title="PDF RAG Chat", page_icon="💬", layout="wide")
st.title("💬 Chat with Multiple PDFs (Gemini + Chroma)")

st.sidebar.header("Settings")
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Only index PDFs once
if "indexed" not in st.session_state:
    for pdf_path in PDF_FILES:
        if os.path.exists(pdf_path):
            pdf_text = extract_text_from_pdf(pdf_path)
            embed_and_store(os.path.basename(pdf_path), pdf_text)
            st.sidebar.success(f"✅ Indexed {os.path.basename(pdf_path)}")
        else:
            st.sidebar.error(f"❌ File not found: {pdf_path}")
    st.session_state.indexed = True

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Ask a question about your PDFs..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve context
    context_chunks = retrieve(query, k=5)
    prompts = [
    f"""You are a helpful assistant. Use the following context to answer the question directly. 
Do not say things like 'based on the text' or 'from the provided context'. 
Only provide the answer as if you are responding naturally to the user. 

Context:
{chunk}

Question: {query}

Answer (concise and direct):"""
    for chunk in context_chunks
]


    # Generate answer
    answers = safe_generate(prompts)
    final_answer = "\n".join(answers)

    # Add AI response
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    with st.chat_message("assistant"):
        st.markdown(final_answer)
