# data/faculty_extractor.py
# Functions for extracting faculty information from filenames

import os
import re
import traceback as tb
from config.logger import Logger
from config.config import Config

def extract_faculty_names_from_filenames(pdf_directory):
    """
    Automatically extracts faculty names from PDF filenames and creates a dictionary
    with variations of the names.

    Args:
        pdf_directory (str): Directory containing the faculty PDF files

    Returns:
        dict: Dictionary mapping faculty names to variations of their names
    """
    # Will store faculty names and variations
    faculty_names = {}
    # Prevents processing duplicate emails
    emails_seen = set()
    logger_main = Logger().get_logger()
    config = Config()
    
    try:
        
        
        # Get all PDF files in the directory
        if not os.path.exists(pdf_directory):
            logger_main.warning(f"PDF directory '{pdf_directory}' does not exist.")
            return {}

        try:
            pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
            
        except OSError as e:
            logger_main.error(f"Error accessing directory {pdf_directory}: {str(e)}")
            return {}

        # Process each PDF filename
        for filename in pdf_files:
           
            try:
                # Extract email from filename
                match = re.match(r'([^@]+@[^_]+)_', filename)
                if not match:
                    continue

                email = match.group(1)
                # Skip if we've already processed this email
                if email in emails_seen:
                    continue

                emails_seen.add(email)
                

                # Convert email to faculty name
                # Get the part before @
                name_part = email.split('@')[0]
                

                # Apply corrections if available
                if name_part in config.email_to_name_corrections:
                    full_name = config.email_to_name_corrections[name_part]
                elif '.' in name_part:
                    # Format: firstname.lastname@domain
                    parts = name_part.split('.')
                    first_name = parts[0].capitalize()
                    # Use last part as last name
                    last_name = parts[-1].capitalize()
                    full_name = f"{first_name} {last_name}"
                elif '_' in name_part:
                    # Format: firstname_lastname@domain
                    parts = name_part.split('_')
                    first_name = parts[0].capitalize()
                    # Use last part as last name
                    last_name = parts[-1].capitalize()
                    full_name = f"{first_name} {last_name}"
                else:
                    # For other formats, just use the email name part
                    # This might need manual correction
                    full_name = name_part.capitalize()

                # Create variations for the name
                if full_name:
                    try:
                        variations = [
                            # alan fern
                            full_name.lower(),
                            # professor fern
                            f"professor {full_name.split()[-1].lower()}",
                            # dr. fern
                            f"dr. {full_name.split()[-1].lower()}",
                            f"prof. {full_name.split()[-1].lower()}"
                        ]
                        faculty_names[full_name.lower()] = variations
                    
                    except Exception as e:
                        logger_main.error(f"Error creating name variations for {full_name}: {tb.format_exc()}")
                        raise RuntimeError(f"Failed to create name variations for {full_name}") from e
            except Exception as e:
                logger_main.error(f"Error processing filename {filename}: {tb.format_exc()}")
                raise RuntimeError(f"Failed to process filename: {filename}") from e

    except Exception as e:
        logger_main.error(f"Error in extract_faculty_names_from_filenames: {tb.format_exc()}")
        raise RuntimeError("Failed to extract faculty names from filenames.") from e
    # print (f"Extracted faculty names: {faculty_names}")

    # Return the faculty names dictionary outside the try-except block
    return faculty_names