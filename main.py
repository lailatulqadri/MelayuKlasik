import streamlit as st
st.set_page_config(layout="wide")

from transformers import BertForQuestionAnswering, AutoTokenizer,pipeline
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Q&A App")
st.caption("Created using BERT model")

footer="""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made by Dhruven.....<a href="https://github.com/dhruven-god"><i class="fa-brands fa-github"></i>
 <a href="https://www.linkedin.com/in/dhruven-god/"> <i class="fa-brands fa-linkedin"></i></a></p>
 
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
@st.cache
def load():
    return BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = load()

@st.cache

def token_load():
    return AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

token = token_load()

bert = pipeline('question-answering',model = model, tokenizer= token)

col1, col2 = st.columns(2)
with col1:
    with st.form("form1",clear_on_submit=False):
        context = st.text_area("Enter your context here")
        question = st.text_input("Enter your question here")

        submit = st.form_submit_button("Submit",type="primary")
        
        if submit:
            with col2:
                st.success('Done!')
                st.write(bert({
                    "question":question,
                    "context":context
                }))




