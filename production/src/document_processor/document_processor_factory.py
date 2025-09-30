from .faiss_document_processor import FaissDocumentProcessor
from .chroma_document_processor import ChromaDocumentProcessor
import traceback as tb
from config.logger import Logger

logger = Logger().get_logger()

class DocumentProcessorFactory:
    @staticmethod
    def create_processor(processor_type: str, config, **kwargs):
        """
        Factory method to create the appropriate document processor
        
        Args:
            processor_type: Type of processor ('faiss' or 'chroma')
            config: Configuration object
            **kwargs: Additional arguments needed for specific processor types
        
        Returns:
            DocumentProcessor: An instance of the requested document processor
        """
        try:
            if processor_type.lower() == 'faiss':
                return FaissDocumentProcessor(config)
            elif processor_type.lower() == 'chroma':
                return ChromaDocumentProcessor(config)
            else:
                raise ValueError(f"Unknown processor type: {processor_type}")
        except Exception as e:
            logger.error(f"Error creating document processor: {tb.format_exc()}")
            raise RuntimeError(f"Failed to initialize document processor of type '{processor_type}'. Check configuration and dependencies.") from e