# ==============================================================================
#
#           🤖 AI Research Agent with Deepseek, LangChain & Streamlit 🤖
#
#
# 📖 TUTORIAL & SETUP
#
# This single Python file contains a complete Streamlit application that runs
# an AI agent powered by LangChain and the Deepseek model via OpenRouter.
#
# To run this file:
#
# 1.  SAVE THE FILE:
#     Save this code as `app.py` in a new folder.
#
# 2.  INSTALL DEPENDENCIES:
#     Open your terminal in the same folder and run the following command to
#     install the necessary Python libraries:
#     pip install streamlit langchain langchain-community google-search-results python-dotenv numexpr
#
# 3.  GET API KEYS:
#     This agent needs two API keys to function:
#     - OpenRouter API Key: For accessing the Deepseek LLM.
#       (Get it from: https://openrouter.ai/keys)
#     - SerpApi API Key: For the web search tool.
#       (Get it from: https://serpapi.com/)
#
# 4.  SET UP ENVIRONMENT VARIABLES:
#     Create a file named `.env` in the same folder as `app.py`.
#     Add your API keys to this file like this:
#
#     # .env file
#     OPENROUTER_API_KEY="sk-or-..."
#     SERPAPI_API_KEY="..."
#
# 5.  RUN THE APP:
#     Go back to your terminal and run the following command:
#     streamlit run app.py
#
#     A new tab should open in your browser with the running application!
#
# ==============================================================================

# --- Step 1: Import all necessary libraries ---
import streamlit as st
import os
from dotenv import load_dotenv

# Core LangChain imports for building the agent
from langchain_community.chat_models.openrouter import ChatOpenRouter
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# --- Step 2: Load Environment Variables and Check for API Keys ---
# load_dotenv() will automatically find and load the variables from the .env file.
load_dotenv()

# --- Step 3: Define the Streamlit User Interface ---

# Set the page title and icon for the browser tab
st.set_page_config(page_title="🤖 Deepseek Research Agent", page_icon="�")

# Display the main title of the application
st.title("🤖 Deepseek Research Agent")

# Provide a welcome message and instructions
st.info("Hello! I am an AI Research Assistant powered by Deepseek and OpenRouter. I can search the web and do math. What can I help you with today?")

# --- Step 4: API Key Validation and Conditional Execution ---
# We wrap the main logic in an 'if' block to ensure it only runs if the keys are valid.
# This prevents errors and provides clear instructions to the user.

if not os.getenv("OPENROUTER_API_KEY", "").startswith("sk-or-"):
    st.error("🚨 Your OpenRouter API key is missing or invalid.")
    st.info("Please create a `.env` file in the project directory and add your key: `OPENROUTER_API_KEY='sk-or-...'`")
    st.stop() # Halts the app if the key is not found

if not os.getenv("SERPAPI_API_KEY"):
    st.error("🚨 Your SerpApi API key is missing.")
    st.info("Please create a `.env` file in the project directory and add your key: `SERPAPI_API_KEY='...'`")
    st.stop() # Halts the app if the key is not found

# --- Step 5: Initialize the LLM, Tools, and Agent ---
# This section is executed only if the API keys are valid.

# Initialize the LLM (Large Language Model) from OpenRouter.
# We specify the Deepseek model, which is excellent for reasoning tasks.
# Temperature is set to a low value (e.g., 0.1) to make the agent's responses
# more deterministic and factual, which is ideal for a research assistant.
llm = ChatOpenRouter(
    model_name="deepseek/deepseek-chat-v2",
    temperature=0.1,
    max_tokens=1024
)

# Load the pre-built tools for the agent.
# - "serpapi": A tool for performing live Google searches.
# - "llm-math": A tool that uses an LLM to solve mathematical problems.
# The `llm` is passed to the tools so they can use it if needed.
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# --- Step 6: Create the Agent Interaction Logic ---

# Create a text input box for the user to ask a question.
user_query = st.text_input("Your question:")

# This block executes only when the user has entered a query.
if user_query:
    # An expander provides a neat, collapsible section for the agent's thought process.
    with st.expander("🤔 Thought Process", expanded=True):

        # The StreamlitCallbackHandler is a key component. It "listens" to the
        # agent's execution and renders each step (thought, action, observation)
        # inside the specified Streamlit container.
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True, max_thought_containers=100)

        # Initialize the agent.
        # - `tools`: The set of tools the agent can use.
        # - `llm`: The "brain" of the agent.
        # - `agent`: We use ZERO_SHOT_REACT_DESCRIPTION, a standard and effective
        #   agent type that reasons using the ReAct (Reason+Act) framework based on
        #   the descriptions of the tools.
        # - `verbose=True`: This is crucial for the callback handler to receive events.
        # - `callbacks`: We pass our Streamlit callback handler here.
        # - `handle_parsing_errors=True`: Makes the agent more robust if it
        #   generates a response that it can't parse.
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            callbacks=[st_callback],
            handle_parsing_errors=True
        )

        try:
            # The `agent.run()` method is the main execution call.
            # It takes the user's query and starts the ReAct loop.
            response = agent.run(user_query)

            # Display the final answer from the agent.
            st.success("✅ Final Answer:")
            st.write(response)
        except Exception as e:
            # Provide a user-friendly error message if something goes wrong.
            st.error(f"An error occurred during agent execution: {e}")
�
