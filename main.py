# Import necessary libraries
import streamlit as st
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Load the BERTopic model
topic_model = BERTopic()

# Streamlit app
st.title("Topic Modeling App")
text = st.text_input("Enter some text here:")
if text:
    # Fit the BERTopic model on the input text
    topics, _ = topic_model.fit_transform([text])

    # Display the predicted topic
    st.write(f"Predicted topic: {topics[0]}")
