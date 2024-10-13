import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="KF Documentation AI Helper")

# Initialize session state
if 'docs' not in st.session_state:
    st.session_state.docs = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'qa_model' not in st.session_state:
    st.session_state.qa_model = None

# Function to fetch and process documentation
@st.cache_data
def fetch_docs(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    docs = {}
    for link in soup.find_all('a', class_='reference internal'):
        sub_url = url + link['href']
        sub_response = requests.get(sub_url)
        sub_soup = BeautifulSoup(sub_response.text, 'html.parser')
        content = sub_soup.find('div', class_='body')
        if content:
            docs[link.text] = content.get_text()
    return docs

# Function to preprocess text
def preprocess(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Initialize models
@st.cache_resource
def initialize_models():
    # QA model
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=0 if torch.cuda.is_available() else -1)
    
    # Embedding model
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    return qa_model, embedding_model

# Streamlit UI
st.title("Documentation AI Helper")

# Initialize models and fetch docs
if st.session_state.qa_model is None or st.session_state.embedding_model is None:
    with st.spinner("Initializing AI models..."):
        st.session_state.qa_model, st.session_state.embedding_model = initialize_models()

if st.session_state.docs is None:
    with st.spinner("Fetching documentation..."):
        st.session_state.docs = fetch_docs("https://community.kissflow.com/category/documentation-section/")
        processed_docs = [preprocess(doc) for doc in st.session_state.docs.values()]
        st.session_state.embeddings = st.session_state.embedding_model.encode(processed_docs)

# Get user input
user_input = st.text_input("Ask a question about Kissflow:")

if user_input:
    with st.spinner("Generating response..."):
        # Find most relevant document
        query_embedding = st.session_state.embedding_model.encode([user_input])
        similarities = cosine_similarity(query_embedding, st.session_state.embeddings)[0]
        most_similar_idx = np.argmax(similarities)
        most_similar_doc = list(st.session_state.docs.values())[most_similar_idx]
        
        # Generate response
        response = st.session_state.qa_model(question=user_input, context=most_similar_doc)
        
        # Display response
        st.write("AI Helper:", response['answer'])

        # Display source document
        with st.expander("Source Document"):
            st.write(most_similar_doc)
