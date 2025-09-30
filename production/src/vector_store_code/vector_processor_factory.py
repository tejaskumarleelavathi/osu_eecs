from .faiss_vector_store import FaissVectorStore
from .chroma_vector_store import ChromaVectorStore
from config.config import Config
from config.logger import Logger
import traceback as tb

logger = Logger().get_logger()

class VectorProcessorFactory:
    @staticmethod
    def create_vector(processor_type: str, config: Config, **kwargs):
        """
        Factory method to create the appropriate vector store processor
        
        Args:
            processor_type: Type of processor ('faiss' or 'chroma')
            config: Configuration object
            **kwargs: Additional arguments needed for specific processor types
        
        Returns:
            VectorStore: An instance of the requested vector store
        """
        try:
            if processor_type.lower() == 'faiss':
                # No need to pass embeddings because FaissVectorStore takes care of it internally
                return FaissVectorStore(config)
            
            elif processor_type.lower() == 'chroma':
                faculty_info = kwargs.get('faculty_info', {})
                return ChromaVectorStore(config)
            
            else:
                raise ValueError(f"Unknown processor type: {processor_type}")
        except Exception as e:
            logger.error(f"Error creating vector store: {tb.format_exc()}")
            raise RuntimeError(f"Failed to initialize vector store of type '{processor_type}'. Check configuration and dependencies.") from e