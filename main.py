import streamlit as st

st.title("BERT Q&A App")

input_text = st.text_area("Enter your question and passage:")

if st.button("Get Answer"):
    # Process input text and get answer using BERT model
    answer = bert(input_text)
    st.write("Answer:", answer)
