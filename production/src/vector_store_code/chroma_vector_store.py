from .vector_store import VectorStore

import os
import re
import datetime
import traceback as tb
import chromadb
from langchain_chroma import Chroma
from config.config import Config
from config.logger import Logger
from ..embeddings_model import EmbeddingsModel
from ..document_processor.chroma_document_processor import ChromaDocumentProcessor
from ..metadata.process_metadata_collection import process_documents
from ..metadata.process_metadata_collection import process_metadata_for_faculty
from ..metadata.process_metadata_collection import process_metadata_for_combined
from utils.pdf_files import get_pdf_files
from utils.constants import CHROMA_PATH



class ChromaVectorStore(VectorStore):
    """Implements vector store operations for ChromaDB"""
    
    def __init__(self, config=None):
        """
        Initialize ChromaVectorStore
        
        Args:
            config (Config, optional): Configuration object containing faculty information and other settings.
                - Defaults to a new instance of Config if not provided.
        
        Returns:
            None
        """
        self.config = config or Config()
        self.logger = Logger().get_logger()
        self.data_logger = Logger().get_logger()  # Using same logger for data processing
        self.client = None
        self.embeddings = None
        self.document_processor = ChromaDocumentProcessor(self.config)
        
        # Initialize ChromaDB client
        self._init_client()
        

        


    def _init_client(self):
        """
        Initialize the ChromaDB client.
        
        Args:
            None
        
        Returns:
            None
        """
        try:
            os.makedirs(CHROMA_PATH, exist_ok=True)
            self.client = chromadb.PersistentClient(path=CHROMA_PATH)
            self.logger.info(f"ChromaDB client initialized at {CHROMA_PATH}")
        
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB client: {tb.format_exc()}")
            raise RuntimeError("Failed to initialize ChromaDB client.") from e


    # def process_chunked_docs(self, tempdocs):
    #     """
    #     Process pre-chunked documents and assign to faculty collections based on metadata.

    #     Args:
    #         tempdocs (list): List of already chunked documents, each containing metadata.
    #             - Each document is expected to have metadata fields like 'source', 'email', and 'faculty_mentions'.
            
    #     Returns:
    #         tuple: A tuple containing:
    #             - faculty_docs (dict): A dictionary mapping faculty names to lists of documents assigned to each faculty.
    #             - combined_docs (list): A list of documents that could not be assigned to a specific faculty, stored in a combined collection.
    #     """
    #     faculty_docs = {faculty: [] for faculty in self.config.faculty_info}
    #     combined_docs = []
        
    #     for i, doc in enumerate(tempdocs):
    #         try:
    #             # Access metadata from the document
    #             metadata = doc.metadata
    #             # print(f"Processing document {i} with metadata: {metadata}")
    #             faculty_assigned = False
                
    #             # Extract email from source or email field
    #             email_username = None
    #             if 'source' in metadata:
    #                 # Look for email pattern in the source path
    #                 source = metadata['source']
    #                 email_match = re.search(r'([^/]+@[^/]+)', source)
    #                 if email_match:
    #                     email_username = email_match.group(1).split('@')[0]
                
    #             if not email_username and 'email' in metadata:
    #                 email_match = re.search(r'([^/]+@[^/]+)', metadata['email'])
    #                 if email_match:
    #                     email_username = email_match.group(1).split('@')[0]
                
    #             # Try to match faculty by email username
    #             if email_username:
    #                 self.logger.info(f"Document {i} has extracted email username: '{email_username}'")
                    
    #                 for faculty_name, variations in self.config.faculty_info.items():
    #                     # Get faculty first and last name
    #                     name_parts = faculty_name.lower().split()
                        
    #                     # Check if email contains faculty name or variations
    #                     if any(part in email_username.lower() for part in name_parts) or \
    #                     any(var.lower() in email_username.lower() for var in variations):
    #                         faculty_docs[faculty_name].append(doc)
    #                         combined_docs.append(doc)
    #                         faculty_assigned = True
    #                         # Add faculty to metadata
    #                         metadata['faculty'] = faculty_name
    #                         self.data_logger.info(f"Added chunk to {faculty_name}'s collection based on email.")
    #                         break
                
                
                
    #             # If still not assigned, add to combined only
    #             if not faculty_assigned:
    #                 combined_docs.append(doc)
    #                 self.data_logger.info("Added chunk to combined collection (no faculty match).")
                    
    #         except Exception as e:
    #             self.data_logger.error(f"Error processing document chunk: {tb.format_exc()}")
    #             raise RuntimeError("Failed to process one of the document chunks.") from e

                    
    #     return faculty_docs, combined_docs

    def process_chunked_docs(self, tempdocs):
        """
        Assign pre-chunked documents to faculty collections using 'faculty' in metadata.

        Args:
            tempdocs (list): List of already chunked documents with metadata['faculty'].

        Returns:
            tuple: (faculty_docs, combined_docs)
        """
        faculty_docs = {faculty: [] for faculty in self.config.faculty_info}
        combined_docs = []

        for i, doc in enumerate(tempdocs):
            try:
                metadata = doc.metadata
                faculty_name = metadata.get('faculty')
                print(f"Processing document {i} with faculty name: {faculty_name}")
            

                if faculty_name and faculty_name in faculty_docs:
                    faculty_docs[faculty_name].append(doc)
                    combined_docs.append(doc)
                    self.data_logger.info(f"Added chunk to {faculty_name}'s collection based on metadata.")
                else:
                    combined_docs.append(doc)

            except Exception as e:
                self.logger.error(f"Error processing document {i}: {e}")
                combined_docs.append(doc)

        return faculty_docs, combined_docs


    def create_and_save(self, tempdocs):
        """
        Create vector stores for each faculty and a combined collection in ChromaDB
        using pre-chunked documents.

        Args:
            tempdocs: List of already chunked documents with metadata

        Returns:
            Dictionary mapping faculty to collection names
        """
        # Set up a persistent directory for ChromaDB
        os.makedirs(CHROMA_PATH, exist_ok=True)
        
        # Process pre-chunked documents
        faculty_docs, combined_docs = self.process_chunked_docs(tempdocs)
        
        # Initialize embeddings
        bedrock_embeddings = EmbeddingsModel.get_instance(self.config.embedding_client, self.config.embedding_model)
        if not bedrock_embeddings:
            self.logger.error("Failed to initialize embeddings.")
            return {}
        
        # Create faculty-specific collections
        for faculty, docs in faculty_docs.items():
            if docs:  # Only create if we have documents
                try:
                    # Clean faculty name for collection name
                    safe_faculty_name = re.sub(r'[^a-zA-Z0-9_]', '_', faculty.lower())
                    collection_name = f"faculty_{safe_faculty_name}"
                    
                    self.logger.info(f"Creating collection for {faculty} with {len(docs)} documents: {collection_name}")
                    
                    # Process documents for this faculty
                    processed_docs = process_documents(docs, process_metadata_for_faculty, faculty=faculty)
                    
                    # Collection metadata
                    collection_metadata = {
                        "faculty_name": faculty,
                        "document_count": len(processed_docs),
                        "creation_time": datetime.datetime.now().isoformat()
                    }
                    
                    # Create collection
                    _, name = self.create_chroma_collection(
                        collection_name, 
                        processed_docs, 
                        bedrock_embeddings, 
                        self.client, 
                        collection_metadata
                    )
                    
                    if name:
                        self.config.faculty_collections[faculty] = name
                    
                except Exception as e:
                    self.logger.error(f"Error creating collection for {faculty}: {tb.format_exc()}")
                    raise RuntimeError(f"Failed to create collection for faculty '{faculty}'.") from e

        
        # Create combined collection
        try:
            if combined_docs:
                self.logger.info(f"Creating combined collection with {len(combined_docs)} documents")
                
                # Process documents for combined collection
                processed_combined_docs = process_documents(combined_docs, process_metadata_for_combined)
                
                # Collection metadata
                combined_metadata = {
                    "type": "combined",
                    "faculty_count": len(self.config.faculty_info),
                    "document_count": len(processed_combined_docs),
                    "creation_time": datetime.datetime.now().isoformat()
                }
                
                # Create collection
                _, name = self.create_chroma_collection(
                    "combined", 
                    processed_combined_docs, 
                    bedrock_embeddings, 
                    self.client, 
                    combined_metadata
                )
                
                if name:
                    self.config.faculty_collections["combined"] = name
                
            else:
                self.logger.warning("No documents found for combined collection")
        except Exception as e:
            self.logger.error(f"Error creating combined collection: {tb.format_exc()}")
            raise RuntimeError("Failed to create combined collection.") from e

        
        # Log summary
        self.logger.info("Faculty collections mapping:")
        for faculty, collection_name in self.config.faculty_collections.items():
            self.logger.info(f"Faculty: {faculty} -> Collection: {collection_name}")

        print("\n--- FACULTY COLLECTIONS MAPPING ---")
        for faculty, collection_name in self.config.faculty_collections.items():
            print(f"Faculty: {faculty} -> Collection: {collection_name}")
            
        self.logger.info(f"Created {len(self.config.faculty_collections)} collections in ChromaDB at {CHROMA_PATH}")
        return self.config.faculty_collections

    
    def create_chroma_collection(self, name, docs, embeddings, client, metadata=None):
        """
        Create or update a ChromaDB collection.
        
        Args:
            name: Collection name
            docs: Documents to add
            embeddings: Embeddings model
            client: ChromaDB client
            metadata: Collection metadata
            
        Returns:
            tuple: (collection object, collection name)
        """
        try:
            # Prepare collection metadata
            collection_metadata = metadata or {
                "document_count": len(docs),
                "creation_time": datetime.datetime.now().isoformat()
            }
            
            # First, delete existing collection if it exists
            try:
                client.delete_collection(name=name)
                self.logger.info(f"Deleted existing collection {name} for update")
            except Exception:
                # Collection doesn't exist, which is fine for new creation
                pass
                
            # Create the collection first with metadata
            client.create_collection(
                name=name,
                metadata=collection_metadata
            )
            
            # Now use LangChain's Chroma to add documents
            collection = Chroma(
                collection_name=name,
                embedding_function=embeddings,
                persist_directory=CHROMA_PATH,
                client=client
            )
            
            # Add documents to the collection
            collection.add_documents(docs)
            
            self.logger.info(f"Collection created: {name} with {len(docs)} documents")
            return collection, name
            
        except Exception as e:
            self.logger.error(f"Error creating collection {name}: {tb.format_exc()}")
            raise RuntimeError(f"Failed to create collection: {name}") from e

    
    # def load_vector_store(self):
    #     """
    #     Load existing faculty collections from ChromaDB.
    #     Maps faculty names to their collection names for later retrieval.

    #     Args:
    #         None

    #     Returns:
    #         None
    #     """
        
        
    #     try:
    #         # Initialize ChromaDB client if not already done

    #         self.logger.info("Using ChromaDB to load vector store")
    #         if not self.client:
    #             self._init_client()
                
    #         # Get all collections - in v0.6.0 this returns only names
    #         collection_names = self.client.list_collections()
            
    #         if not collection_names:
    #             self.logger.warning(f"No collections found in ChromaDB at {CHROMA_PATH}")
    #             return
                
    #         self.logger.info(f"Found {len(collection_names)} collections in ChromaDB")
            
    #         # Process each collection name
    #         for collection_name in collection_names:
    #             # Get the actual collection object
    #             coll_name = collection_name.name
    #             collection = self.client.get_collection(name=coll_name)
                
    #             # Handle combined collection
    #             if collection_name == "combined":
    #                 self.config.faculty_collections["combined"] = "combined"
    #                 self.logger.info(f"Loaded combined collection with {collection.count()} documents")
    #                 continue
                
    #             # Handle faculty-specific collections
    #             if collection_name.startswith("faculty_"):
    #                 # Default to extracting name from collection name
    #                 faculty_name = self.extract_faculty_name(collection_name)
    #                 name_source = "extracted from name"
                    
    #                 # Try to get a better name from metadata if available
    #                 metadata = getattr(collection, 'metadata', None)
    #                 if metadata and metadata.get("faculty_name"):
    #                     faculty_name = metadata["faculty_name"]
    #                     name_source = "from metadata"
                    
    #                 # Store the mapping
    #                 self.config.faculty_collections[faculty_name] = collection_name
    #                 self.logger.info(
    #                     f"Loaded collection for {faculty_name} ({name_source}) with {collection.count()} documents"
    #                 )
            
    #         self.logger.info(f"Loaded {len(self.config.faculty_collections)} collections from ChromaDB")
            
    #     except Exception as e:
    #         self.logger.error(f"Error loading faculty collections: {tb.format_exc()}")
    #         raise RuntimeError("Failed to load faculty collections from ChromaDB.") from e

    def load_vector_store(self):
        """
        Load existing faculty collections from ChromaDB.
        Maps faculty names to their collection names for later retrieval.
        Args:
            None
        Returns:
            None
        """
        
        try:
            # Initialize ChromaDB client if not already done
            self.logger.info("Using ChromaDB to load vector store")
            if not self.client:
                self._init_client()
                
            # Get all collections
            collection_objects = self.client.list_collections()
            
            if not collection_objects:
                self.logger.warning(f"No collections found in ChromaDB at {CHROMA_PATH}")
                return
                
            self.logger.info(f"Found {len(collection_objects)} collections in ChromaDB")
            
            # Process each collection
            for collection_info in collection_objects:
                # Extract the collection name string
                coll_name = collection_info.name
                
                # Get the actual collection object
                collection = self.client.get_collection(name=coll_name)
                
                # Handle combined collection
                if coll_name == "combined":  # Use coll_name (string) instead of collection_info (object)
                    self.config.faculty_collections["combined"] = "combined"
                    self.logger.info(f"Loaded combined collection with {collection.count()} documents")
                    continue
                
                # Handle faculty-specific collections
                if coll_name.startswith("faculty_"):  # Use coll_name (string)
                    # Default to extracting name from collection name
                    faculty_name = self.extract_faculty_name(coll_name)  # Use coll_name (string)
                    name_source = "extracted from name"
                    
                    # Try to get a better name from metadata if available
                    metadata = getattr(collection, 'metadata', None)
                    if metadata and metadata.get("faculty_name"):
                        faculty_name = metadata["faculty_name"]
                        name_source = "from metadata"
                    
                    # Store the mapping - use coll_name (string)
                    self.config.faculty_collections[faculty_name] = coll_name
                    self.logger.info(
                        f"Loaded collection for {faculty_name} ({name_source}) with {collection.count()} documents"
                    )
            
            self.logger.info(f"Loaded {len(self.config.faculty_collections)} collections from ChromaDB")
            
        except Exception as e:
            self.logger.error(f"Error loading faculty collections: {tb.format_exc()}")
            raise RuntimeError("Failed to load faculty collections from ChromaDB.") from e

    
    # Helper function to extract faculty name from collection name
    def extract_faculty_name(self, coll_name):
        """
        Extract faculty name from collection name.

        Args:
            coll_name (str): Collection name (format: faculty_name_here)
        
        Returns:
            str: Formatted faculty name with proper capitalization.
        """
        faculty_suffix = coll_name[8:]  # Remove "faculty_"
        return " ".join(word.capitalize() for word in faculty_suffix.split("_"))
    
   
    
    