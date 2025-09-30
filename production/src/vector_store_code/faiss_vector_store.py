from .vector_store import VectorStore

import os
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langsmith import traceable
from ..embeddings_model import EmbeddingsModel
from config.config import Config
from config.logger import Logger
from utils.constants import FAISS_PATH
import traceback as tb


class FaissVectorStore(VectorStore):
    """Handles vector store operations"""
    def __init__(self, config: Config):
        """
        Initialize FaissVectorStore.

        Args:
            config (Config): Configuration object containing embedding client and model information.
        
        Returns:
            None
        """
        self.logger = Logger().get_logger()
        self.config = config
        # Use the singleton pattern for embeddings
        self.embeddings = EmbeddingsModel.get_instance(self.config.embedding_client,self.config.embedding_model)  ## need to make an Embedding model file and use it here.
        
    @traceable(name="create_vector_store")
    def create_and_save(self, chunks: List[Document]) -> None:
        """
        Create and save a FAISS vector store using the provided document chunks.

        Args:
            chunks (List[Document]): A list of `Document` objects that represent the data to be stored.
                - Each `Document` contains the text data and its associated metadata.
        
        Returns:
            None
        
        Raises:
            Exception: If there is any error during the creation or saving of the vector store.
        """
        try:
            self.logger.info(f"Saving to vector database")
            if not os.path.exists(FAISS_PATH):
                os.makedirs(FAISS_PATH)
                
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local(FAISS_PATH)
            print("✅ FAISS vector store saved successfully!")
        
        except Exception as e:
            self.logger.error(f"Error creating vector store: {tb.format_exc()}")
            raise RuntimeError("Failed to create and save the FAISS vector store.") from e

    @traceable(name="load_vector_store")
    def load_vector_store(self) -> FAISS:
        """
        Load the FAISS vector store from disk.

        Args:
            None
        
        Returns:
            FAISS: The FAISS vector store loaded from disk.
   
        """
        try:
            self.logger.info(f"Loading FAISS vector store from {FAISS_PATH}")
            vector_store = FAISS.load_local(
                FAISS_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.logger.info("FAISS vector store loaded successfully")
            return vector_store
            
        except Exception as e:
            self.logger.error(f"Error loading vector store: {tb.format_exc()}")
            raise RuntimeError("Failed to load the FAISS vector store from disk.") from e