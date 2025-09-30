import os

from config.config import Config
from config.logger import Logger

def get_pdf_files():
    """
    Find all PDF files in the configured directory.
    
    Args:
        None
        
    Returns:
        list: A list of file paths for all PDF files found in the configured directory (PDF_DIRECTORY).
        
    """
    config = Config
    pdf_files = []
    if not os.path.exists(config.PDF_DIRECTORY):
        logger_data_processing.error(f"PDF directory '{config.PDF_DIRECTORY}' does not exist.")
        return []
        
    for root, _, files in os.walk(config.PDF_DIRECTORY):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
                
    logger_data_processing.info(f"Found {len(pdf_files)} PDF files.")
    return pdf_files