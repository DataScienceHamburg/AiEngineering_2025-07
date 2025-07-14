#%% streamlit app
import streamlit as st
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# %%
st.title('Nostradamus')

user_prompt = st.chat_input(placeholder="What do you want to know about the future?")

if user_prompt:
    # set up prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are Nostradamus, a famous prophet. Answer the user's question truthfully and in a way that is easy to understand."),
        ("user", "{user_prompt}")
    ])
    # create model instance
    model = ChatGroq(model="llama3-8b-8192")
    
    # create chain
    chain = prompt_template | model | StrOutputParser()
    
    # invoke chain
    result = chain.invoke({"user_prompt": user_prompt})
    with st.container():
        message_user = st.chat_message("human")
        message_user.write(user_prompt)
    with st.container():
        message = st.chat_message("assistant")
        message.write(result)