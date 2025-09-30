from .document_processor import DocumentProcessor
from typing import *
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langsmith import traceable
from config.config import Config
from config.logger import Logger
import traceback as tb


class FaissDocumentProcessor(DocumentProcessor):
    """Document processor implementation for FAISS-based vector storage"""
    
    def __init__(self, config):
        """
        Initialize the FaissDocumentProcessor with the given configuration.

        Args:
            config (Config): The configuration object containing chunk_size, chunk_overlap, and other settings.
        
        Returns:
            None
        """
        try:
            self.logger = Logger().get_logger()
            self.config = config
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            self.logger.info(f"Initiailzing Faiss vector document processor")
        except Exception as e:
            logger = Logger().get_logger()
            logger.error(f"Error initializing FaissDocumentProcessor: {tb.format_exc()}")
            raise RuntimeError("Failed to initialize FAISS document processor. Check configuration parameters.") from e
    
    @traceable(name="load_documents")
    def load(self, source: str) -> List[Document]:
        """
        Load PDF documents from a directory, merging content from files with the same source.

        Args:
            source (str): The path to the directory containing the PDF files to load.
        
        Returns:
            List[Document]: A list of loaded PDF documents.
        """        
        
        try:
            self.logger.info(f"Loading documents from: {source}")
            loader = PyPDFDirectoryLoader(path=source)
            pdfs = []
            pdfs_lazy = loader.lazy_load()

            for pdf in pdfs_lazy:
                if pdfs and pdfs[-1].metadata['source'] == pdf.metadata['source']:
                    pdfs[-1].page_content += pdf.page_content
                    continue
                pdfs.append(pdf)

            self.logger.info(f"Documents loaded: {len(pdfs)}")
            return pdfs
        except Exception as e:
            self.logger.error(f"Error loading documents: {tb.format_exc()}")
            raise RuntimeError(f"Failed to load documents from {source}. Check file permissions and format.") from e

    @traceable(name="split_documents")
    def split(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        Split documents into smaller chunks for processing.

        Args:
            documents (List[Document]): The list of documents to be split.
        
        Returns:
            List[Document]: A list of chunked documents.
        """
        try:
            self.logger.info("Splitting documents into chunks")
            chunks = self.text_splitter.split_documents(documents)
            self.logger.info(f"Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            self.logger.error(f"Error splitting documents: {tb.format_exc()}")
            raise RuntimeError("Failed to split documents into chunks. Check text splitter configuration.") from e
    
    def process_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Process metadata for FAISS documents. This implementation is minimal since FAISS 
        doesn't require extensive metadata processing.

        Args:
            documents (List[Document]): The list of documents whose metadata needs to be processed.
        
        Returns:
            List[Document]: The list of documents with processed metadata.
        """
        try:
            return documents
        except Exception as e:
            self.logger.error(f"Error processing document metadata: {tb.format_exc()}")
            raise RuntimeError("Failed to process document metadata for FAISS.") from e