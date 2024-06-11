import streamlit as st
#from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# 
model = transformers.AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-cased")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    
# User input
user_input = st.text_area("Enter your text here:")

# Tokenize input
input_ids = tokenizer.encode(user_input, add_special_tokens=True, return_tensors="pt")

# Get model predictions
with torch.no_grad():
    logits = model(input_ids).logits
    predicted_class = torch.argmax(logits, dim=1).item()

# Display prediction
st.write(f"Predicted class: {predicted_class}")
