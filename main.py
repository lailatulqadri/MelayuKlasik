import streamlit as st
#from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# Load the sentiment analysis pipeline
@st.cache_resource
def load_model():
   
    return pipeline("sentiment-analysis")
@st.cache_resource
def load_model_malay():
    #model_name = "huggingface/bert-base-bahasa-cased"
    model_name = "mesolitica/bert-base-standard-bahasa-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, from_pt=True)


model = load_model_malay()

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
