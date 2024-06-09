import streamlit as st
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Load and preprocess your data
@st.cache(allow_output_mutation=True)
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['text'] = data['text'].str.lower().str.replace('[^\w\s]', '')
    return data

# Load the BERTopic model
@st.cache(allow_output_mutation=True)
def load_model():
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    topic_model = BERTopic(embedding_model=model)
    return topic_model

data = load_data('Salatus_salatin.csv')
topic_model = load_model()

# Fit the BERTopic model
@st.cache(allow_output_mutation=True)
def fit_model(data):
    topics, _ = topic_model.fit_transform(data['text'].tolist())
    return topics

topics = fit_model(data)

# Streamlit interface
st.title("Malay Text Topic Modeling with BERTopic")

st.write("This application performs topic modeling on Malay text using BERTopic.")

if st.checkbox("Show Raw Data"):
    st.write(data)

st.write("Topics discovered in the text:")

topic_info = topic_model.get_topic_info()
st.write(topic_info)

topic_id = st.number_input("Enter a topic ID to see the top words:", min_value=0, max_value=len(topic_info)-1, step=1)
if st.button("Show Topic"):
    st.write(topic_model.get_topic(topic_id))
