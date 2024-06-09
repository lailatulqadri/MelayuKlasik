import streamlit as st
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

@st.cache
def load():
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    return pipeline('question-answering', model=model, tokenizer=tokenizer)

bert = load()

st.title("BERT Q&A App")
st.caption("Created using BERT model")

col1, col2 = st.columns(2)

with col1:
    with st.form("form1", clear_on_submit=False):
        context = st.text_area("Enter your context here")
        question = st.text_input("Enter your question here")
        submit = st.form_submit_button("Submit", type="primary")

        if submit:
            with col2:
                st.success('Done!')
                st.write(bert({
                    "question": question,
                    "context": context
                }))
