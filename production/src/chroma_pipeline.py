
import traceback as tb
from typing import Dict, List, Any, Optional, Tuple

import chromadb
from langchain_chroma import Chroma

from config.config import Config
from config.logger import Logger
from .embeddings_model import EmbeddingsModel
from utils.constants import CHROMA_PATH
import re

class ChromaPipeline:
    def __init__(self, config: Config):
        """
        Initialize ChromaPipeline with the given configuration.

        Args:
            config (Config): Configuration object containing necessary settings for the pipeline.
        
        Returns:
            None
        """
        self.logger = Logger().get_logger()
        self.config = config

    def find_matched_faculty(self, query: str) -> Optional[str]:
        """
        Determine which faculty is mentioned in the query.
        
        Args:
            query (str): User query
            self.config.faculty_info (dict): Dictionary of faculty information
            
        Returns:
            str: Matched faculty name or None
        """
        try:
            query_lower = query.lower()
            
            for faculty_name, variations in self.config.faculty_info.items():
                # Check existing variations first
                if any(variation in query_lower for variation in variations):
                    return faculty_name
                    
                # Check for first and last name individually
                try:
                    name_parts = faculty_name.split()
                    if len(name_parts) >= 2:
                        first_name, last_name = name_parts[0], name_parts[-1]
                        
                        # Check for whole word matches of first or last name
                        if re.search(r'\b' + re.escape(first_name) + r'\b', query_lower) or \
                        re.search(r'\b' + re.escape(last_name) + r'\b', query_lower):
                            return faculty_name
                except Exception as e:
                    # Continue checking other faculty if there's an error with one name
                    self.logger.error(f"Error checking name parts for {faculty_name}: {str(e)}")
                    continue
                    
            return None
        except Exception as e:
            self.logger.error(f"Error in find_matched_faculty: {str(e)}")
            return None

    def initialize_search_clients(self) -> Tuple[Any, Any]:
        """
        Initialize embeddings and ChromaDB client for search.

        Args:
            None

        Returns:
            tuple: A tuple containing:
                - embeddings (EmbeddingsModel): The embeddings model used for document embeddings.
                - client (chromadb.PersistentClient): The ChromaDB client for interacting with the database.
            If initialization fails, returns (None, None).
        """
        # Initialize embeddings
        
        try:
            embeddings = EmbeddingsModel.get_instance(
                self.config.embedding_client, 
                self.config.embedding_model
            )
        except Exception as e:
            self.logger.error(f"Error creating embeddings: {tb.format_exc()}")
            raise RuntimeError("Failed to initialize the embeddings component.") from e
            
        # Initialize ChromaDB client
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            return embeddings, client
        except Exception as e:
            self.logger.error(f"Error creating ChromaDB client: {tb.format_exc()}")
            raise RuntimeError("Failed to initialize ChromaDB client. Check database path and permissions.") from e

    def search_faculty_collection(self, client, embeddings, collection_name: str, query: str, k: int = 15,  filters: Optional[Dict[str, Any]] = None) -> List:
        """
        Search in a faculty-specific collection.
        
        Args:
            client: ChromaDB client
            embeddings: Embedding function
            collection_name: Name of the collection to search
            query: Query string
            k: Number of results to return
            
        Returns:
            list: Retrieved documents or empty list if search fails
        """
        try:
            self.logger.info(f"Searching in collection: {collection_name}")
            
            # Check if collection exists
            try:
                # Create the Langchain Chroma wrapper for similarity search
                collection = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=CHROMA_PATH,
                    client=client
                )
                
                # Perform similarity search
                docs = collection.similarity_search(
                        query=query,
                        k=k,
                        filter=filters if filters else None
                    )

                
                if docs:
                    self.logger.info(f"Found {len(docs)} results in {collection_name} collection")
                return docs
                
            except Exception as e:
                self.logger.error(f"Error initializing collection {collection_name}: {tb.format_exc()}")
                raise RuntimeError(f"Failed to initialize or search collection '{collection_name}'.") from e

                
        except Exception as e:
            self.logger.error(f"Error searching collection {collection_name}: {tb.format_exc()}")
            raise RuntimeError(f"Failed to search faculty collection: {collection_name}") from e

    def search_combined_collection(self, client, embeddings, query: str, matched_faculty: Optional[str] = None, k: int = 10,  filters: Optional[Dict[str, Any]] = None) -> List:
        """
        Search in the combined collection, optionally with faculty filtering.
        
        Args:
            client: ChromaDB client
            embeddings: Embedding function
            query: Query string
            matched_faculty: Faculty to filter results (optional)
            k: Number of results to return
            
        Returns:
            list: Retrieved documents
        """
        combined_docs = []
        
        try:
            self.logger.info(f"Searching in combined collection with query: {query}")
            
            # Create the Langchain Chroma wrapper for similarity search
            combined_collection = Chroma(
                collection_name="combined",
                embedding_function=embeddings,
                persist_directory=CHROMA_PATH,
                client=client
            )
            
            # If we have a matched faculty, use metadata filtering to focus on their documents
            if matched_faculty:
                try:
                    # Try with metadata filter
                    faculty_k = min(k * 2, 20)  # Get more results from faculty-filtered search
                    
                    # Perform filtered search
                    specific_docs = combined_collection.similarity_search(
                        query=query,
                        k=faculty_k,
                        filter=filters if filters else {"faculty": matched_faculty}
                    )
                    
                    if specific_docs:
                        self.logger.info(f"Found {len(specific_docs)} faculty-specific results in combined collection")
                        combined_docs.extend(specific_docs)
                except Exception as e:
                    self.logger.error(f"Error during faculty-specific filtered search: {tb.format_exc()}")
                    raise RuntimeError("Failed to perform filtered search with faculty metadata.") from e
            
            # Also get general results if we need more
            if len(combined_docs) < k:
                remaining_k = k - len(combined_docs)
                general_docs = combined_collection.similarity_search(
                    query=query,
                    k=remaining_k,
                    filter=filters if filters else None
                )
                combined_docs.extend(general_docs)
                
            self.logger.info(f"Found {len(combined_docs)} total results in combined collection")
            
        except Exception as e:
            self.logger.error(f"Error searching combined collection: {tb.format_exc()}")
            raise RuntimeError("Failed to search in the combined collection.") from e
        
        return combined_docs

    def merge_search_results(self, faculty_docs: List, combined_docs: List) -> List:
        """
        Merge results from faculty and combined collections, avoiding duplicates.
        
        Args:
            faculty_docs: List of faculty-specific documents
            combined_docs: List of documents from combined collection
            
        Returns:
            list: Merged list of documents without duplicates
        """

        try:
            unique_docs = {}

            for doc in faculty_docs:
                unique_docs[doc.page_content] = doc

            for doc in combined_docs:
                if doc.page_content not in unique_docs:
                    unique_docs[doc.page_content] = doc

            all_docs = list(unique_docs.values())

            self.logger.info(
                f"Total unique documents: {len(all_docs)} ({len(faculty_docs)} from faculty collection, "
                f"{len(combined_docs)} from combined collection)"
            )

            return all_docs

        except Exception as e:
            self.logger.error(f"Error merging search results: {tb.format_exc()}")
            raise RuntimeError("Failed to merge faculty and combined search results.") from e


    def improved_search_faculty_vectorstore(self, query: str, enhanced_query: str, tag: str, k_value: int =5) -> List:
        """
        Faster search function that only searches either faculty-specific or combined collection.

        Args:
            query (str): The original user query.
            enhanced_query (str): The enhanced query for better search results.

        Returns:
            list: Relevant documents retrieved from the appropriate collection (either faculty-specific or combined).
        """
        # Step 1: Determine which faculty is mentioned in the query
        matched_faculty = self.find_matched_faculty(query)
        self.logger.info(f"Matched faculty: {matched_faculty}")
        filters = {}
        if tag:
            filters[tag] = True

        
        # Step 2: Initialize search clients
        embeddings, client = self.initialize_search_clients()
        if not embeddings or not client:
            return []
        
        # Step 3: Search in faculty-specific collection if applicable, otherwise use combined
        k_value = 10
        
        if matched_faculty and matched_faculty in self.config.faculty_collections:
            # We found a faculty match, use only the faculty-specific collection
            collection_name = self.config.faculty_collections[matched_faculty]
            self.logger.info(f"Using faculty-specific collection: {collection_name}")
            
            return self.search_faculty_collection(
                client,
                embeddings,
                collection_name,
                enhanced_query,
                k=k_value,
                filters=filters if filters else None
            )

        else:
            # No faculty match, use only the combined collection
            self.logger.info("No faculty match, using combined collection")
            
            return self.search_combined_collection(
                client,
                embeddings,
                enhanced_query,
                matched_faculty=None,
                k=k_value,
                filters=filters if filters else None
            )

       