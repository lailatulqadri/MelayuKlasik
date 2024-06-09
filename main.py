# Import necessary libraries
import streamlit as st
import transformers
import torch

# Load the BERT model and tokenizer
model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

# Define the function for sentiment analysis
def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Pass the tokenized input through the model
    outputs = model(**inputs)

    # Get the predicted class and return the corresponding sentiment
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    if predicted_class == 0:
        return "Negative"
    elif predicted_class == 1:
        return "Neutral"
    else:
        return "Positive"

# Streamlit app
st.title("Sentiment Analysis App")
text = st.text_input("Enter some text here:")
if text:
    sentiment = predict_sentiment(text)
    st.write(f"Sentiment: {sentiment}")
