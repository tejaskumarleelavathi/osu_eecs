from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langsmith import traceable
from config.config import Config
from config.logger import Logger
import traceback as tb

class DocumentProcessor:
    """Handles document loading and splitting operations"""
    def __init__(self, config: Config):
        """
        Initialize DocumentProcessor with the given configuration.

        Args:
            config (Config): Configuration object containing chunk size and chunk overlap for document splitting.
        
        Returns:
            None
        """
        self.logger = Logger().get_logger()
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    @traceable(name="load_documents")
    def load_documents(self, file_path: str) -> List[Document]:
        """
        Load PDF documents from a specified directory.

        Args:
            file_path (str): The directory path containing PDF files to be loaded.
        
        Returns:
            list: A list of `Document` objects loaded from the specified directory.
       
        """
        try:
            self.logger.info(f"Loading documents from: {file_path}")
            loader = PyPDFDirectoryLoader(path=file_path)
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
            raise RuntimeError("Failed to load documents. Check the input directory and file format.") from e

    @traceable(name="split_documents")
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split the loaded documents into chunks based on the configuration settings.

        Args:
            documents (List[Document]): A list of `Document` objects to be split into chunks.
        
        Returns:
            list: A list of `Document` objects, each representing a chunk of the original document.
      
        """
        try:
            self.logger.info("Splitting documents into chunks")
            chunks = self.text_splitter.split_documents(documents)
            self.logger.info(f"Created {len(chunks)} chunks")
            return chunks
        
        except Exception as e:
            self.logger.error(f"Error splitting documents: {tb.format_exc()}")
            raise RuntimeError("Failed to split documents. Check document structure or text splitter config.") from e  
