import streamlit as st
from transformers import pipeline

# Load the sentiment analysis pipeline
@st.cache(allow_output_mutation=True)
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

# Streamlit app
st.title("Sentiment Analysis with BERT")
st.write("Enter some text and the model will analyze the sentiment.")

# User input
user_input = st.text_area("Enter your text here:")

# Model inference
if st.button("Analyze"):
    if user_input:
        results = model(user_input)
        for result in results:
            st.write(f"**Label:** {result['label']}")
            st.write(f"**Score:** {result['score']:.4f}")
    else:
        st.write("Please enter some text to analyze.")
