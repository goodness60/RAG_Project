''''
import google.generativeai as genai

genai.configure(api_key="AIzaSyAekCZcytNtCa7qgUbf8h85CeO3CF2zxe4")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Hello, Gemini!")
print(response.text)

'''
'''
import google.generativeai as genai
print("Gemini SDK working!")
'''
'''
import faiss
print(faiss.__version__)
'''
'''
import streamlit as st

st.title("My First Streamlit App")
st.write("Hello, world! 👋")
'''

