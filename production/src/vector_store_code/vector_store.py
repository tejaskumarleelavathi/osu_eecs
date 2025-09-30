from abc import ABC, abstractmethod



class VectorStore(ABC):
    @abstractmethod
    def load_vector_store(self, source):
        """Load documents from a source"""
        pass
        
    @abstractmethod
    def create_and_save(self, documents, **kwargs):
        """Split documents into chunks"""
        pass
        
    