!pip install transformers
import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

@st.cache(allow_output_mutation=True)
def load_model():
    return BertTokenizer.from_pretrained('bert-base-multilingual-cased'), BertModel.from_pretrained('bert-base-multilingual-cased')

@st.cache(allow_output_mutation=True)
def get_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

@st.cache(allow_output_mutation=True)
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['text'] = data['text'].str.lower().str.replace('[^\w\s]', '')
    tokenizer, model = load_model()
    data['embeddings'] = data['text'].apply(lambda x: get_embeddings(x, tokenizer, model))
    return data

data = pd.read_csv('malay_text_data.csv')

def find_most_similar(query, data, tokenizer, model):
    query_embedding = get_embeddings(query, tokenizer, model)
    similarities = cosine_similarity(query_embedding, np.vstack(data['embeddings'].values))
    most_similar_idx = np.argmax(similarities)
    return data.iloc[most_similar_idx]['text']

# Streamlit interface
st.title("Malay Text Chatbot")

st.write("Enter a query to get a response from the chatbot.")

user_query = st.text_input("Your query:", "")

if user_query:
    tokenizer, model = load_model()
    response = find_most_similar(user_query, data, tokenizer, model)
    st.write("Chatbot response:", response)

# Upload your CSV file
from google.colab import files
uploaded = files.upload()
