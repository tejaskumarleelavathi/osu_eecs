import streamlit as st
from src.ui.components import Components
from src.ui.state import SessionState
from config.logger import Logger
import torch
from dotenv import load_dotenv
from langsmith import Client
from langsmith.run_helpers import traceable

torch.classes.__path__ = []

def main():
    """
    Initializes and runs the RAG System Streamlit application, including the chat interface and sidebar.

    Args:
        None
        
    Returns:
        None
        
    This function:
        - Loads environment variables using `load_dotenv`.
        - Sets up Streamlit page configuration.
        - Initializes necessary components and session states.
        - Renders the sidebar with configuration options.
        - Renders the chat interface for user interaction.
        - Provides a "Clear Chat" button to reset chat history.
        

    """
    load_dotenv()
    client = Client()


    

    st.set_page_config(
        page_title="OSU EECS Chatbot",
        page_icon="🧙‍♂️",
        layout="wide"
    )
    # st.title("OSU EECS Chatbot")

    st.markdown(
    """
    <h1 style='text-align: center; color: #FF5C00;'>OSU EECS Chatbot</h1>
    """,
    unsafe_allow_html=True
)

    # Render sidebar and get configuration
    
    if "components" not in st.session_state:
        st.session_state["components"] = Components()
        st.session_state["session_state"] = SessionState()
        print("This should happen once ")

    # Render sidebar and get configuration
    st.session_state.components.render_sidebar()

    # Render chat interface
    st.session_state.components.render_chat_interface()
    
    # Render chat interface
    # Add clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state["session_state"].clear_chat_history()
        st.rerun()

if __name__ == "__main__":
    main()
