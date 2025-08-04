# --- 1. Installation & Imports ---
# This section imports all the necessary libraries to make our application work.
# Think of it as gathering all the ingredients before you start cooking.

import os  # The 'os' library is not directly used in this script but is often needed for environment variable management.
import streamlit as st  # 'streamlit' is the framework we use to create the web app's user interface (UI).
import langchain_community # A core part of the LangChain ecosystem.
from langchain_openai import ChatOpenAI  # This class allows us to connect to OpenAI-compatible models like DeepSeek.
from langchain.agents import load_tools, initialize_agent, AgentType # These are key functions from LangChain for building agents.
from langchain_core.messages import AIMessage, HumanMessage # These are special classes to structure the conversation history for the agent.
from langchain.agents import AgentExecutor # The runtime environment for an agent.

# --- 2. API Key Setup ---
# Agents need access to external services (like language models and search tools).
# This section securely retrieves the necessary API keys.

# We get the API keys from Streamlit's secret management. This is a secure way to handle sensitive information.
OPENROUTER_API_KEY = st.secrets.OPENROUTER_API_KEY
serpapi = st.secrets.serpapi

# --- 3. App UI Configuration ---
# This section sets up the basic appearance and information for our web app.

# `set_page_config` configures the browser tab's title and icon.
st.set_page_config(page_title="LangChain Agent with DeepSeek", page_icon="�")
# `st.title` sets the main title displayed on the web page.
st.title("🤖 LangChain Agent with DeepSeek")
# `st.caption` provides a short description under the title.
st.caption("This app allows you to ask questions to an AI agent powered by DeepSeek and LangChain.")

# This is a check to ensure the API keys were successfully loaded. If not, it displays a warning and stops the app.
if not OPENROUTER_API_KEY or not serpapi:
    st.warning(
        "API keys not found. Please add your `OPENROUTER_API_KEY` and `SERPAPI_API_KEY` to your Streamlit secrets."
    )
    st.stop() # Halts the execution of the app.

# --- 4. Agent Initialization ---
# This is the most important part. We create the AI agent here.
# An "agent" is a program that uses a Large Language Model (LLM) to reason and decide which actions to take.

# Streamlit reruns the script from top to bottom on every user interaction.
# To avoid re-creating the agent every single time (which is slow), we store it in `st.session_state`.
# `st.session_state` is a special dictionary that persists across reruns.
if "agent_executor" not in st.session_state:
    # `st.spinner` shows a loading message to the user while the agent is being created.
    with st.spinner("Initializing the agent... Please wait."):
        # Step A: Initialize the LLM (The Agent's "Brain")
        # We're using the ChatOpenAI class to connect to a model hosted on OpenRouter.
        llm = ChatOpenAI(
            model="deepseek/deepseek-r1-0528:free",  # Specifies the exact model we want to use.
            temperature=0,  # A temperature of 0 makes the model's responses more deterministic and less random.
            openai_api_key=OPENROUTER_API_KEY,  # The API key for authentication.
            openai_api_base="https://openrouter.ai/api/v1",  # The server endpoint for OpenRouter.
            default_headers={ # These headers are required by OpenRouter for identification.
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "LangChain Agent Streamlit App",
            }
        )

        # Step B: Load the Tools (The Agent's "Skills")
        # "Tools" are functions the agent can use to get information or perform actions.
        # Here, we're giving it the "serpapi" tool, which allows it to search the internet.
        tools = load_tools(["serpapi"], llm=llm, serpapi_api_key=serpapi)

        # Step C: Initialize the Agent
        # This step combines the LLM (brain) and the tools (skills) into a functional agent.
        st.session_state.agent_executor = initialize_agent(
            tools,
            llm,
            # `CONVERSATIONAL_REACT_DESCRIPTION` is a specific agent type.
            # It's designed for conversations and uses a "ReAct" (Reasoning and Acting) framework
            # to decide which tool to use based on the conversation history.
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            # `verbose=True` makes the agent print its step-by-step thought process to the terminal, which is great for debugging.
            verbose=True,
            # This tells the agent to try and gracefully handle errors if it can't understand a tool's output.
            handle_parsing_errors=True
        )

# --- 5. Chat History Management ---
# We need to store the conversation so the agent has context about what was said before.

# We initialize an empty list called `messages` in the session state to store the chat history.
if "messages" not in st.session_state:
    st.session_state.messages = []

# This loop runs every time the app reruns. It goes through the chat history
# and displays each message on the screen, so the conversation is always visible.
for message in st.session_state.messages:
    with st.chat_message(message["role"]): # `role` can be "user" or "assistant"
        st.markdown(message["content"])

# --- 6. Main App Logic ---
# This is where the app interacts with the user.

# `st.chat_input` creates a text input box at the bottom of the screen.
# The code inside this `if` block only runs when the user types something and hits Enter.
if prompt := st.chat_input("Ask a question..."):
    # Add the user's new message to our chat history list.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display the user's message on the screen immediately.
    with st.chat_message("user"):
        st.markdown(prompt)

    # Now, it's the assistant's turn to respond.
    with st.chat_message("assistant"):
        # We use an expander to hide the agent's complex thought process unless the user wants to see it.
        with st.expander("Agent's Thought Process"):
            st.info("The agent's step-by-step reasoning appears in your terminal because `verbose=True` is set.")
            try:
                # The agent needs the chat history in a specific format (HumanMessage, AIMessage).
                # This loop converts our simple list of messages into that format.
                formatted_chat_history = []
                for msg in st.session_state.messages[:-1]: # We process all messages except the latest user prompt
                    if msg["role"] == "user":
                        formatted_chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        formatted_chat_history.append(AIMessage(content=msg["content"]))

                # This is the key step: we call the agent with the user's new question and the formatted history.
                # The agent will now think, potentially use its tools (like searching the web), and come up with a final answer.
                result = st.session_state.agent_executor.invoke(
                    {
                        "input": prompt,
                        "chat_history": formatted_chat_history,
                    }
                )
                # The agent's final answer is stored in the 'output' key of the result dictionary.
                response = result['output']
                # We can write the full, raw result inside the expander for debugging.
                st.write(result)

            except Exception as e:
                # If anything goes wrong, we display an error message.
                response = f"An error occurred: {e}"
                st.error(response)

        # Display the agent's final, clean response to the user.
        st.markdown(response)

    # Finally, add the assistant's response to the chat history so it's remembered for the next turn.
    st.session_state.messages.append({"role": "assistant", "content": response})
