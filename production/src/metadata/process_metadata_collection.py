import datetime
import traceback as tb
from config.logger import Logger

logger = Logger().get_logger()

def process_metadata_for_faculty(doc, faculty):
    """
    Process document metadata for faculty-specific collection.
    
    Args:
        doc: Document to process
        faculty: Faculty name
        
    Returns:
        dict: Clean metadata dictionary
    """
    try:
        clean_metadata = {}
        
        for key, value in doc.metadata.items():
            # Convert lists to strings
            if isinstance(value, list):
                clean_metadata[f"{key}_str"] = ", ".join(str(item) for item in value)
            # Keep simple types
            elif isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
                
        # Add required metadata
        clean_metadata["faculty"] = faculty
        clean_metadata["is_faculty_document"] = True
        clean_metadata["created_at"] = datetime.datetime.now().isoformat()
        
        return clean_metadata
    except Exception as e:
        logger.error(f"Error processing faculty metadata: {tb.format_exc()}")
        raise RuntimeError(f"Failed to process metadata for faculty '{faculty}'. Check document structure.") from e


def process_metadata_for_combined(doc):
    """
    Process document metadata for combined collection.
    
    Args:
        doc: Document to process
        
    Returns:
        dict: Clean metadata dictionary
    """
    try:
        clean_metadata = {}
        
        for key, value in doc.metadata.items():
            # Convert lists to strings
            if isinstance(value, list):
                clean_metadata[f"{key}_str"] = ", ".join(str(item) for item in value)
            # Keep simple types
            elif isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
                
        # Add required metadata
        if "faculty" not in clean_metadata:
            clean_metadata["faculty"] = "unknown"
        clean_metadata["collection_type"] = "combined"
        clean_metadata["created_at"] = datetime.datetime.now().isoformat()
        
        return clean_metadata
    except Exception as e:
        logger.error(f"Error processing combined metadata: {tb.format_exc()}")
        raise RuntimeError("Failed to process metadata for combined collection. Check document structure.") from e


def process_documents(docs, metadata_processor, **kwargs):
    """
    Process documents with the specified metadata processor.
    
    Args:
        docs: List of documents to process
        metadata_processor: Function to process metadata
        kwargs: Additional arguments for metadata processor
        
    Returns:
        list: Processed documents
    """
    try:
        processed_docs = []
        for doc in docs:
            # Process metadata
            clean_metadata = metadata_processor(doc, **kwargs)
            
            # Update document with clean metadata
            doc.metadata = clean_metadata
            processed_docs.append(doc)
        
        return processed_docs
    except Exception as e:
        logger.error(f"Error processing documents: {tb.format_exc()}")
        raise RuntimeError("Failed to process documents. Check metadata processor and document structure.") from e