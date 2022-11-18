#Importing Libraries

import streamlit as st
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

#Load tokenizer and model

@st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
def load_data():    
 tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
 model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
 return tokenizer, model
tokenizer, model = load_data()

#Setup Streamlit

st.header("Conversational AI Chatbot")
st.sidebar.image("logo.gif")
st.image("images.gif")
st.sidebar.header("Conversational AI")
st.sidebar.write("A State-of-the-Art Large-scale Pretrained Response generation model (DialoGPT)")
st.sidebar.write("DialoGPT is a SOTA large-scale pretrained dialogue response generation model for multiturn conversations.")
st.sidebar.write("The human evaluation results indicate that the response generated from DialoGPT is comparable to human response quality under a single-turn conversation Turing test.")
st.sidebar.write(" The model is trained on 147M multi-turn dialogue from Reddit discussion thread.")

st.caption("I am still learning, Please be patient")
input = st.text_input('Talk to me:')

if 'count' not in st.session_state or st.session_state.count == 6:
 st.session_state.count = 0 
 st.session_state.chat_history_ids = None
 st.session_state.old_response = '1'
else:
 st.session_state.count += 1

#Tokenizing user input

new_user_input_ids = tokenizer.encode(input  + tokenizer.eos_token, return_tensors='pt')

#Appending user inputs

bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.count > 1 else new_user_input_ids

#Generating inputs
st.session_state.chat_history_ids = model.generate(bot_input_ids,do_sample=True, max_length=1000,pad_token_id=tokenizer.eos_token_id, temperature=0.6, repetition_penalty=1.3)

#Decoding Response
response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

#Regenerating Response
if st.session_state.old_response == response:
   bot_input_ids = new_user_input_ids
 
st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

#Display Response on UI
st.write(f"Conversational AI: {response}")

#Updating old response variable
st.session_state.old_response = response

st.write("Developed by Shahnab")