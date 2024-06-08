import torch
from transformers import BertTokenizer, BertForQuestionAnswering

@st.cache
def load():
    return BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

model = load()

@st.cache
def token_load():
    return AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

token = token_load()
bert = pipeline('question-answering', model=model, tokenizer=token)
