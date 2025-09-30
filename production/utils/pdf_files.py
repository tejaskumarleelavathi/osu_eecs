import os
import traceback as tb
from config.logger import Logger
from .constants import DATA_PATH

def get_pdf_files():
    """
    Find all PDF files in the configured directory.

    Args:
        None
        
    Returns:
        list: A list of file paths for all PDF files found in the configured directory (DATA_PATH).
        
    """
    logger = Logger().get_logger()

    try:
        if not os.path.exists(DATA_PATH):
            logger.error(f"PDF directory '{DATA_PATH}' does not exist.")
            raise RuntimeError(f"PDF directory '{DATA_PATH}' does not exist.")

        pdf_files = []
        for root, _, files in os.walk(DATA_PATH):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))

        logger.info(f"Found {len(pdf_files)} PDF files.")
        return pdf_files

    except Exception as e:
        logger.error(f"Error finding PDF files: {tb.format_exc()}")
        raise RuntimeError("Failed to retrieve PDF files from the configured directory.") from e
