import streamlit as st 
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from transformers import pipeline

st.title("Welcome to the Fine-Tuned Tiny-Bert Sentiment Classifier Page")
text = st.text_input("Please Enter Your Thoughts of The Movie: ")
