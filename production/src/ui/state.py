import streamlit as st
from typing import Optional, Dict, Any
from config.logger import Logger
import traceback as tb

class SessionState:
    """Manages Streamlit session state"""
    
    def __init__(self):
        """
        Initializes the SessionState instance and sets up the logger.

        Args:
            None
        
        Returns:
            None
        """
        try:
            self.logger = Logger().get_logger()
        except Exception as e:
            logger = Logger().get_logger()
            logger.error(f"Error initializing SessionState: {tb.format_exc()}")
            raise RuntimeError("Failed to initialize SessionState. Check logger configuration.") from e

    def get_chat_history(self) -> list:
        """
        Retrieves the chat history from Streamlit session state.

        If the 'messages' key does not exist in the session state, it initializes it as an empty list.

        Args:
            None
        
        Returns:
            list: A list of messages from the chat history.
        """
        try:
            if "messages" not in st.session_state:
                st.session_state.messages = []
                self.logger.info("Initialized new chat history")
            return st.session_state.messages
        except Exception as e:
            self.logger.error(f"Error getting chat history: {tb.format_exc()}")
            raise RuntimeError("Failed to retrieve chat history from session state. Check Streamlit configuration.") from e
    
    def add_message(self,role: str, content: str):
        """
        Adds a message to the chat history in Streamlit session state.

        If the 'messages' key does not exist in the session state, it initializes it as an empty list.
        
        Args:
            role (str): The role of the sender (e.g., 'user', 'assistant').
            content (str): The content of the message.
        
        Returns:
            None
        """
        try:
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"role": role, "content": content})
            self.logger.info(f"Added message with role {role}")
        except Exception as e:
            self.logger.error(f"Error adding message: {tb.format_exc()}")
            raise RuntimeError(f"Failed to add message with role '{role}' to chat history. Check session state.") from e
    
    def clear_chat_history(self):
        """
        Clears the chat history from the Streamlit session state.

        Args:
            None
        
        Returns:
            None
        """
        try:
            st.session_state.messages = []
            self.logger.info("Cleared chat history")
        except Exception as e:
            self.logger.error(f"Error clearing chat history: {tb.format_exc()}")
            raise RuntimeError("Failed to clear chat history from session state. Check Streamlit configuration.") from e