import streamlit as st
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

@st.cache
def load():
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    return pipeline('question-answering', model=model, tokenizer=tokenizer)

bert = load()

st.title("BERT Q&A App")

question = st.text_input("Enter your question:")
context = st.text_area("Enter your passage:")

if st.button("Get Answer"):
    # Process input text and get answer using BERT model
    input_dict = {"question": question, "context": context}
    answer = bert(input_dict)
    st.write("Answer:", answer['answer'])
