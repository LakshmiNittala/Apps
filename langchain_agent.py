# --- 1. Installation ---
# First, you need to install the required Python libraries.
# Open your terminal or command prompt and run these commands:
# pip install streamlit langchain langchain-openai python-dotenv google-search-results

import os
import streamlit as st
import langchain_community
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType


# --- 2. API Key Setup ---
# For local development, we can still use a .env file.
# Create a file named ".env" in the same directory as this script
# and add your API keys like this:
#
# OPENROUTER_API_KEY="sk-or-..."
# SERPAPI_API_KEY="..."
#
# Define API keys
OPENROUTER_API_KEY = st.secrets.OPENROUTER_API_KEY
serpapi = st.secrets.serpapi

# --- App UI Configuration ---
st.set_page_config(page_title="LangChain Agent with DeepSeek", page_icon="🤖")
st.title("🤖 LangChain Agent with DeepSeek")
st.caption("This app allows you to ask questions to an AI agent powered by DeepSeek and LangChain.")

if not OPENROUTER_API_KEY or not serpapi:
    st.warning(
        "API keys not found. Please create a `.env` file in the project root "
        "and add your `OPENROUTER_API_KEY` and `SERPAPI_API_KEY`."
    )
    st.stop()

# --- Agent Initialization ---
# We initialize the agent once and store it in session state to avoid re-initializing on every interaction.
if "agent_executor" not in st.session_state:
    with st.spinner("Initializing the agent... Please wait."):
        # Initialize the LLM (The Agent's Brain)
        llm = ChatOpenAI(
            model="deepseek/deepseek-r1-0528:free",
            temperature=0,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:8501", # Streamlit's default port
                "X-Title": "LangChain Agent Streamlit App",
            }
        )

        # Load the Tools (The Agent's Skills)
        tools = load_tools(["serpapi"], llm=llm)
       # tools = load_tools(["serpapi"], llm=llm, serpapi_api_key=serpapi)

        # Initialize the Agent
        # We use a conversational agent type to allow for chat history.
        st.session_state.agent_executor = initialize_agent(
            tools,
            llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True
        )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main App Logic ---
# Accept user input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Use an expander to show the agent's "thinking" process
        with st.expander("Agent's Thought Process"):
            try:
                # The st.write_stream function is not standard.
                # A common way to capture verbose output is to redirect stdout.
                # However, for a cleaner approach in Streamlit, we'll just run the agent
                # and the verbose output will appear in the terminal.
                # The final answer will be displayed in the app.
                response = st.session_state.agent_executor.run(prompt)
                st.markdown(response)
            except Exception as e:
                response = f"An error occurred: {e}"
                st.error(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

