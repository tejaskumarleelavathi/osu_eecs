from abc import ABC, abstractmethod



class DocumentProcessor(ABC):
    @abstractmethod
    def load(self, source):
        """Load documents from a source"""
        pass
        
    @abstractmethod
    def split(self, documents, **kwargs):
        """Split documents into chunks"""
        pass
        
    @abstractmethod
    def process_metadata(self, documents):
        """Process metadata for documents"""
        pass