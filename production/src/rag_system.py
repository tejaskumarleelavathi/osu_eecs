from typing import Optional
import os
from config.config import Config
from config.logger import Logger
from utils.constants import DATA_PATH
from .document_processor.document_processor_factory import DocumentProcessorFactory
from .vector_store_code.vector_processor_factory import VectorProcessorFactory
from utils.faculty_extractor import extract_faculty_names_from_filenames
from .rag_pipeline import RAGPipeline
from .query_generator import QueryGenerator
import traceback as tb
from langsmith import traceable
import streamlit as st
from .chroma_pipeline import ChromaPipeline

@traceable
class RAGSystem:
    """Main system class that orchestrates the entire RAG process"""
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the RAGSystem with the given configuration, faculty information, and pipelines.

        Args:
            config (Config, optional): Configuration object. If not provided, a default `Config` object is used.
        
        Returns:
            None
        """
        self.logger = Logger().get_logger()
        self.config = config or Config()
        self.chroma_pipeline = ChromaPipeline(self.config)
        
        self.config.faculty_info = extract_faculty_names_from_filenames(DATA_PATH)
        # self.logger.info(f"Detected {len(self.config.faculty_info)} faculty members.") 

        self.document_processor = DocumentProcessorFactory.create_processor(
            processor_type=self.config.processor_type,  # e.g., "faiss" or "chroma"
            config=self.config,
            faculty_info=self.config.faculty_info  # Pass the extracted faculty info directly
        )

        print(f"Vector store {self.config.processor_type}")
        self.vector_store = VectorProcessorFactory.create_vector(self.config.processor_type,self.config)
        # self.query_generator = QueryGenerator(self.config)
        
        # Check if we need to reinitialize the pipeline
        if "llm_config_changed" in st.session_state and st.session_state.llm_config_changed:
            self.logger.info("LLM configuration changed, reinitializing pipeline")
            if "rag_pipeline" in st.session_state:
                del st.session_state.rag_pipeline
            st.session_state.llm_config_changed = False  # Reset the flag
            
        # Initialize pipeline if it doesn't exist
        if "rag_pipeline" not in st.session_state:
            self.logger.info("Initializing new RAG pipeline in session state")
            st.session_state.rag_pipeline = RAGPipeline(self.config, self.vector_store)
            # Initialize the pipeline right away
            st.session_state.rag_pipeline.setup()
        else:
            self.logger.info("Reusing existing RAG pipeline from session state")
        
        self.rag_pipeline = st.session_state.rag_pipeline

    def initialize_knowledge_base(self, file_path: str) -> None:
        """
        Initialize the knowledge base with documents from the provided file path.

        Args:
            file_path (str): The path to the document directory that will be used to initialize the knowledge base.
        
        Returns:
            None
        =
        """
        try:
            self.logger.info(f"Initializing knowledge base from {file_path}")
            document_processor = DocumentProcessorFactory.create_processor(self.config.processor_type,self.config)
            
            documents = document_processor.load(file_path)
            chunks = document_processor.split(documents)
            self.vector_store.create_and_save(chunks)  ## need to probably return something from faiss vector store
            
        except Exception as e:
            self.logger.error(f"Error initializing knowledge base: {tb.format_exc()}")
            raise RuntimeError("Failed to initialize the knowledge base. Ensure documents are valid and paths are correct.") from e  

    def query(self, question: str) -> str:
        """
        Query the RAG system to generate a response.

        Args:
            question (str): The query string to ask the system.
        
        Returns:
            str: The generated response from the RAG system.
   
        """

        try:
            return self.rag_pipeline.query(question)
        except Exception as e:
            self.logger.error(f"Error querying the RAG system: {tb.format_exc()}")
            raise RuntimeError("Failed to process the query. Ensure the pipeline is ready and question is valid.") from e