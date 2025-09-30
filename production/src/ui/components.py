import streamlit as st
import os
import tempfile
import traceback as tb
from typing import Tuple, Optional
from config.config import Config, LLMProvider
from ..rag_system import RAGSystem
from config.logger import Logger
from utils import constants

class Components:
    def __init__(self):
        """
        Initialize the Components class with configuration, logger, and RAG system.

        Args:
            None
        
        Returns:
            None
        """
        self.logger = Logger().get_logger()
        self.config=Config()
        self.rag_system = None
        self.temp_dir = None
        print("Current working directory:", os.getcwd())

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []


    def render_sidebar(self) -> Tuple[Optional[Config], Optional[RAGSystem]]:
        """
        Render the sidebar and return the configuration and RAG system.
        
        This method handles LLM provider selection, model selection, 
        advanced settings, and RAG system initialization.

        Args:
            None
        
        Returns:
            Tuple[Optional[Config], Optional[RAGSystem]]: The configuration and RAG system objects.
        """
        try:
            st.sidebar.title("RAG System Configuration")
            
            # Initialize the session state variable if it doesn't exist
            if "llm_config_changed" not in st.session_state:
                st.session_state.llm_config_changed = False
                
            # Store current settings to detect changes
            if "current_llm_provider" not in st.session_state:
                st.session_state.current_llm_provider = self.config.llm_provider
                
            if "current_llm_model" not in st.session_state:
                if self.config.llm_provider == LLMProvider.OLLAMA.value:
                    st.session_state.current_llm_model = self.config.ollama_model_name
                else:
                    st.session_state.current_llm_model = self.config.groq_model_name

            # LLM Provider selection
            provider = st.sidebar.selectbox(
                "Select LLM Provider",
                options=[LLMProvider.OLLAMA.value, LLMProvider.GROQ.value],
                index=0 if self.config.llm_provider =="ollama" else 1
            )

            # Check if provider changed
            if provider != st.session_state.current_llm_provider:
                st.session_state.llm_config_changed = True
                st.session_state.current_llm_provider = provider


            # Set the selected provider in config
            self.config.llm_provider = provider

            # Log the provider switch
            self.logger.info(f"LLM provider switched to: {provider}")
            
            
            # Model selection based on provider
            if provider == LLMProvider.OLLAMA.value:
                try:
                    model_name = st.sidebar.selectbox(
                        "Ollama Model Name",
                        options=list(self.config.ollama_models),
                        index=0
                    )
                    base_url = st.sidebar.text_input(
                        "Ollama Base URL",
                        value="http://localhost:11434"
                    )


                    # Check if model changed
                    if st.session_state.current_llm_provider == LLMProvider.OLLAMA.value and model_name != st.session_state.current_llm_model:
                        st.session_state.llm_config_changed = True
                    
                    st.session_state.current_llm_model = model_name
                    self.config.ollama_model_name = model_name
                    self.logger.info(f"Current ollama model: {model_name}")

                except Exception as e:
                    self.logger.error(f"Error configuring Ollama: {str(e)}\n{tb.format_exc()}")
                    st.sidebar.error("Error configuring Ollama. Check logs for details.")
                    st.stop()
            else:
                try:
                    model_name = st.sidebar.selectbox(
                        "Groq Model Name",
                        options = list(self.config.groq_models),
                        index = 0
                    )

                    # Check if model changed
                    if st.session_state.current_llm_provider == LLMProvider.GROQ.value and model_name != st.session_state.current_llm_model:
                        st.session_state.llm_config_changed = True
                        
                    st.session_state.current_llm_model = model_name
                    
                    self.config.groq_model_name = model_name
                    self.logger.info(f"Current groq model : {model_name}")

                except Exception as e:
                    self.logger.error(f"Error configuring Groq: {str(e)}\n{tb.format_exc()}")
                    st.sidebar.error("Error configuring Groq. Check logs for details.")
                    st.stop()
            
            try:    
                # Advanced settings collapsible
                with st.sidebar.expander("Advanced Settings"):
                    self.config.chunk_size = st.number_input(
                        "Chunk Size",
                        value=self.config.chunk_size,
                        min_value=100,
                        max_value=10000
                    )
                    self.config.chunk_overlap = st.number_input(
                        "Chunk Overlap",
                        value=self.config.chunk_overlap,
                        min_value=0,
                        max_value=self.config.chunk_size
                    )
                    self.config.retriever_k = st.number_input(
                        "Number of Retrieved Chunks",
                        value=self.config.retriever_k,
                        min_value=1,
                        max_value=30
                    )
            except Exception as e:
                self.logger.error(f"Error configuring advanced settings: {tb.format_exc()}")
                st.sidebar.error("Error configuring advanced settings. Check logs for details.")
                st.stop()

            # Initialize RAG system with config
            try:

                self.rag_system = RAGSystem(self.config)

                with st.sidebar:
                    st.title("Update vectore store")

                    if st.button("Vectors Update"):
                        with st.spinner("Processing..."):
                            self.rag_system.initialize_knowledge_base(constants.DATA_PATH)
                            
                    st.text(f"Embedding used {self.config.embedding_model}")
                # self.logger.info("RAG system initialized successfully")
            except Exception as e:
                print(f"Exception {tb.format_exc()}")
                self.logger.error(f"Error initializing/updating Vector Store: {tb.format_exc()}")
                st.sidebar.error("Error initializing RAG system. Check logs for details.")
                st.stop()
               
            
        except Exception as e:
            self.logger.error(f"Error in render_sidebar: {tb.format_exc()}")
            st.sidebar.error("An error occurred in the sidebar. Check logs for details.")
            st.stop()

    def render_chat_interface(self):
        """
        Render the chat interface for interacting with the system.

        Args:
            None
        
        Returns:
            None
        """

        try:
            if self.rag_system is None:
                self.logger.warning("RAG system not initialized. Chat interface disabled.")
                st.warning("System not properly initialized. Please check configuration.")
                return
                
            # st.subheader("Chat Interface")

            
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("What is your question?"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user",avatar="🖊"):
                    st.markdown(prompt)
                
                # Get AI response
                with st.chat_message("assistant",avatar="🦫"):
                    with st.spinner("Thinking..."):
                        self.logger.info(f"Message passed to model to rag system query")
                        response = self.rag_system.query(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            self.logger.error(f"Error in render_chat_interface: {tb.format_exc()}")
            st.error("An error occurred in chat interface. Check logs for details.")
            st.stop()

