from typing import Optional
import traceback as tb
from config.config import Config
from config.logger import Logger
from utils.constants import DATA_PATH
from .document_processor.document_processor_factory import DocumentProcessorFactory
from .vector_store_code.vector_processor_factory import VectorProcessorFactory
from utils.faculty_extractor import extract_faculty_names_from_filenames
from .rag_pipeline import RAGPipeline
from .chroma_pipeline import ChromaPipeline

class RAGSystem:
    """Main system class that orchestrates the entire RAG process"""

    def __init__(self, config: Optional[Config] = None):
        self.logger = Logger().get_logger()
        self.config = config or Config()
        self.chroma_pipeline = ChromaPipeline(self.config)
        self.config.faculty_info = extract_faculty_names_from_filenames(DATA_PATH)

        self.document_processor = DocumentProcessorFactory.create_processor(
            processor_type=self.config.processor_type,
            config=self.config,
            faculty_info=self.config.faculty_info
        )

        self.vector_store = VectorProcessorFactory.create_vector(self.config.processor_type, self.config)

        self.rag_pipeline = RAGPipeline(self.config, self.vector_store)
        self.rag_pipeline.setup()

    def initialize_knowledge_base(self, file_path: str) -> None:
        try:
            self.logger.info(f"Initializing knowledge base from {file_path}")
            document_processor = DocumentProcessorFactory.create_processor(self.config.processor_type, self.config)
            documents = document_processor.load(file_path)
            chunks = document_processor.split(documents)
            self.vector_store.create_and_save(chunks)
        except Exception as e:
            self.logger.error(f"Error initializing knowledge base: {tb.format_exc()}")
            raise RuntimeError("Failed to initialize the knowledge base.") from e

    def query(self, question: str) -> str:
        try:
            return self.rag_pipeline.query(question)
        except Exception as e:
            self.logger.error(f"Error querying the RAG system: {tb.format_exc()}")
            raise RuntimeError("Failed to process the query.") from e
