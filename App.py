import numpy as np
import pandas as pd
import streamlit as st
import numpy as np
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
import torch

@st.cache(hash_funcs={"tokenizers.AddedToken": lambda _: None})
def get_model():
    tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
    model = AutoModelForSeq2SeqLM.from_pretrained("riajul/FineTunedNLLBModelLatest")
    return tokenizer,model


tokenizer,model = get_model()
st.title('English to Rohingya Translator App')
user_input = st.text_area('Enter Text to translate')
button = st.button("Translate")

if user_input and button :
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    output = model.generate(**inputs)
    translated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write(translated_text)