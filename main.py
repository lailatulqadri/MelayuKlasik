import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Example function to get embeddings from a model
def get_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to find the most similar text
def find_most_similar(query, data, tokenizer, model):
    # Check if 'embeddings' column exists
    if 'embeddings' not in data.columns:
        raise KeyError("'embeddings' column not found in the DataFrame")
    
    query_embedding = get_embeddings(query, tokenizer, model)
    
    # Reshape query_embedding if necessary
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Convert list of embeddings to numpy array
    embeddings = np.vstack(data['embeddings'].values)
    
    similarities = cosine_similarity(query_embedding, embeddings)
    most_similar_idx = np.argmax(similarities)
    
    return data.iloc[most_similar_idx]['text']

# Sample data
data = pd.DataFrame({
    'text': ['sample text 1', 'sample text 2', 'sample text 3'],
    'embeddings': [np.random.rand(768), np.random.rand(768), np.random.rand(768)]
})

# Print DataFrame columns to verify the structure
print(data.columns)

# Load a pre-trained tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Query for similarity search
query = "example query"

# Find the most similar text
try:
    result = find_most_similar(query, data, tokenizer, model)
    print("Most similar text:", result)
except KeyError as e:
    print(e)
