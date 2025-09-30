from .document_processor import DocumentProcessor

from typing import *
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langsmith import traceable
from config.config import Config
from config.logger import Logger
import traceback as tb
import platform
import os
import re


class ChromaDocumentProcessor(DocumentProcessor):
    """Document processor implementation for ChromaDB with enhanced metadata processing"""
    
    def __init__(self, config):
        """
        Initialize the ChromaDocumentProcessor with the given configuration.

        Args:
            config (Config): The configuration object containing chunk_size, chunk_overlap, and faculty_info.
        
        Returns:
            None
        """
        self.logger = Logger().get_logger()
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.logger.info(f"Initiailzing Chroma vector document processor")
        try:
            self.logger = Logger().get_logger()
            self.config = config
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            self.logger.info(f"Initiailzing Chroma vector document processor")
        except Exception as e:
            logger = Logger().get_logger()
            logger.error(f"Error initializing ChromaDocumentProcessor: {tb.format_exc()}")
            raise RuntimeError("Failed to initialize ChromaDocumentProcessor. Check configuration parameters.") from e

    
    def load(self, source: str) -> List[Document]:
        """
        Load documents from a PDF file located at the specified source.

        Args:
            source (str): The path to the directory containing PDF files to load.
        
        Returns:
            List[Document]: A list of processed documents after loading and enhancing metadata.
        """
        try:
            self.logger.info("Loading to chroma")
            self.logger.info(f"Loading documents from: {source}")
            
            loader = PyPDFDirectoryLoader(path=source)
            
            documents = loader.load()
            self.logger.info(f"Documents loaded: {len(documents)}")

            processed_documents = self.process_metadata(documents)
            self.logger.info(f"length of processed document {len(processed_documents)}")
            
            return processed_documents
        except Exception as e:
            self.logger.error(f"Error loading documents: {tb.format_exc()}")
            raise RuntimeError(f"Failed to load documents from {source}. Check file permissions and format.") from e
    
    def split(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        Split documents into chunks using the configured text splitter.

        Args:
            documents (List[Document]): The list of documents to split.
        
        Returns:
            List[Document]: A list of chunked documents.
        """
        try:
            self.logger.info("splitting in chroma")
            self.logger.info(f"Received {len(documents)} documents to split")
            self.logger.info("Splitting documents into chunks")
            chunks = self.text_splitter.split_documents(documents)
            self.logger.info(f"Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            self.logger.error(f"Error splitting documents: {tb.format_exc()}")
            raise RuntimeError("Failed to split documents into chunks. Check text splitter configuration.") from e
    
    def process_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Process and enhance metadata for ChromaDB documents.

        Args:
            documents (List[Document]): The list of documents to process.
        
        Returns:
            List[Document]: The list of documents with enhanced metadata.
        """
        try:
            self.logger.info("Processing metadata in chroma")
            processed_docs = []
            for doc in documents:
                processed_docs.append(self._preprocess_document(doc))
            return processed_docs
        except Exception as e:
            self.logger.error(f"Error processing document metadata: {tb.format_exc()}")
            raise RuntimeError("Failed to process document metadata. Check document structure.") from e
    
    def _preprocess_document(self, doc: Document) -> Document:
        """
        Preprocess a document to enhance the metadata and content for better retrieval.

        Args:
            doc (Document): The document to preprocess.
        
        Returns:
            Document: The preprocessed document with enhanced metadata.
        """
        try:
            content = doc.page_content
            metadata = doc.metadata.copy()
            self.logger.info("Processing metadata in chroma db")
            # Add page information if available
            if 'page' in metadata:
                content = f"Page {metadata['page']}: {content}"

            # Add source information if available
            # if 'source' in metadata:
            #     source = metadata['source']
            #     content = f"Source: {source}\n{content}"

            #     # Extract faculty from source filename
            #     match = re.search(r'([^@]+@[^_]+)_', source)
            #     if match:
            #         email = match.group(1)
            #         print(f"email in chroma metadata: {email}")
            #         raw_name_part = email.split('@')[0]
                    
            #         # Use corrected name if present, otherwise default to raw
            #         corrected_name = self.config.email_to_name_corrections.get(raw_name_part.lower(), raw_name_part)
            #         name_part = corrected_name.lower().replace(" ", ".") if corrected_name != raw_name_part else raw_name_part
            #         metadata['email'] = email
            #         metadata['name_part'] = name_part
            #         print(f"name part in chroma metadata: {name_part}")
            #         #it should be pushed
            #         # Try to find which faculty this belongs to
            #         for faculty_name, variations in self.config.faculty_info.items():
            #             if name_part.lower() in faculty_name or any(name_part.lower() in var for var in variations):
            #                 metadata['faculty'] = faculty_name
            #                 content = f"[FACULTY: {faculty_name}] {content}"
            #                 break
            if 'source' in metadata:
                source = metadata['source']
                content = f"Source: {source}\n{content}"
                
                # Extract email from source filename
                if platform.system() == "Windows":
                    match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]+)', source)
                elif platform.system() == "Darwin":
                    match = re.search(r'([^/]+@[^/_]+)', source)  #MAC CODE

                if match:
                    email = match.group(1)
                    self.logger.info(f"Email in Chroma metadata: {email}")
                    raw_name_part = email.split('@')[0]

                    # Use corrected name if present, otherwise default to raw
                    corrected_name = self.config.email_to_name_corrections.get(raw_name_part.lower(), raw_name_part)
                    name_part = corrected_name.lower().replace(" ", ".") if corrected_name != raw_name_part else raw_name_part
                   

                    metadata['email'] = email
                    metadata['name_part'] = name_part

                    # Try to find which faculty this belongs to
                    
                    # Normalize the name_part for matching
                    normalized_name_part = name_part.lower().replace('.', ' ')

                    for faculty_name, variations in self.config.faculty_info.items():
                        if normalized_name_part in faculty_name or any(normalized_name_part in var for var in variations):
                            print(f"faculty name in chroma metadata: {faculty_name}")
                            metadata['faculty'] = faculty_name
                            print(f"faculty name in chroma metadata: {faculty_name}")
                            content = f"[FACULTY: {faculty_name}] {content}"
                            break


     
            # Check for faculty mentions in content
            for faculty_name, variations in self.config.faculty_info.items():
                content_lower = content.lower()
                if any(variation in content_lower for variation in variations):
                    # Add faculty tag to metadata
                    if 'faculty_mentions' not in metadata:
                        
                        metadata['faculty_mentions'] = []
                    if faculty_name not in metadata['faculty_mentions']:
                        metadata['faculty_mentions'].append(faculty_name)

            # Check for publication indicators
            if any(term in content.lower() for term in [
                "publication", "journal", "conference", "proceedings",
                "published", "paper", "article", "research"
            ]):
                metadata['has_publications'] = True
                content = f"[CONTAINS: PUBLICATIONS] {content}"
                
            # Check for software indicators
            if any(term in content.lower() for term in [
                "software", "package", "tool", "app", "library", "framework",
                "code", "developed", "github", "repository", "implementation"
            ]):
                metadata['has_software'] = True
                content = f"[CONTAINS: SOFTWARE] {content}"

            # Check for patent indicators
            if any(term in content.lower() for term in [
                "patent", "patented", "provisional application", "uspto", "intellectual property"
            ]):
                metadata['has_patents'] = True
                content = f"[CONTAINS: PATENTS] {content}"


            # Update document
            doc.page_content = content
            doc.metadata = metadata
            return doc
        except Exception as e:
            self.logger.error(f"Error preprocessing document: {tb.format_exc()}")
            raise RuntimeError("Failed to preprocess document. Check document format and metadata structure.") from e