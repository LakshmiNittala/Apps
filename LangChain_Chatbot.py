import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import os

# Set up OpenRouter API key and LLM
llm = ChatOpenAI(
    model="openai/gpt-4o",
    openai_api_key= st.secrets.api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Initialize memory and conversation chain
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Streamlit UI
st.title("LangChain Chat App")
st.write("Chat with an AI powered by OpenRouter!")

# User input
user_input = st.text_input("You:", key="input")
if user_input:
    response = conversation.invoke({"input": user_input})
    st.text_area("AI:", value=response["response"], height=200, key="output")
