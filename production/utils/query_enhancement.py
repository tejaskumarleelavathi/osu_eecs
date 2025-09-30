from typing import Dict, Any
from config.logger import Logger
from config.config import Config

def query_enhancement(query: str):
    """
    Enhances the query with additional context before searching.

    Args:
        query (str): The original user query.
        
    Returns:
        tuple: A tuple containing:
            - enhanced_query (str): The improved version of the original query with additional context.
            - search_kwargs (dict): A dictionary containing search parameters like 'k' for the number of documents to retrieve.
 
    """

    try:
        # Initialize logger
        logger = Logger().get_logger()
        logger.info("Enhancing the query if applicable")
        
        
        
        # Normalize query for matching
        query_lower = query.lower().strip()
        
        # Load configuration
        config = Config()
        
        # Initialize enhanced query with original query as fallback
        enhanced_query = query
        tag = None
        
        # Identify query type and enhance accordingly
        if any(term in query_lower for term in ["publication", "paper", "publish", "journal", "conference"]):
            # This is a publication-related query
           
            
            # Check for faculty mentions
            faculty_mentioned = None
            for faculty_name, variations in config.faculty_info.items():
                if any(variation.lower() in query_lower for variation in variations):
                    faculty_mentioned = faculty_name
                    break
            
            # Enhance query with publication-specific terms
            enhanced_query = f"publications papers {query}"
            
            # Add faculty-specific enhancement if a faculty is mentioned
            if faculty_mentioned:
                enhanced_query = f"publications authored by {faculty_mentioned} papers journal conference {query}"
                logger.info(f"Enhanced publication query for faculty: {faculty_mentioned}")
            else:
                logger.info("Enhanced general publication query")
            tag = "has_publications"


        elif any(term in query_lower for term in ["research", "interest"]):
            # Research interests query
            enhanced_query = f"research interests expertise focus areas {query}"
            tag = "has_publications"
            logger.info("Enhanced query for research interests")

        elif any(term in query_lower for term in ["contact", "email", "phone"]):
            # Contact information query
            enhanced_query = f"contact information email address phone office {query}"
            logger.info("Enhanced query for contact information")

        elif any(term in query_lower for term in ["education", "degree", "phd"]):
            # Education query
            enhanced_query = f"education background degrees phd university {query}"
            logger.info("Enhanced query for education background")

        elif any(term in query_lower for term in ["software", "package", "tool", "application", "code", "develop", "github", "library"]):
            # Software/development query
            enhanced_query = f"software development tool code GitHub implementation package library {query}"
            tag = "has_software"
            logger.info("Enhanced query for software-related information")

        elif any(term in query_lower for term in ["patent", "patented", "provisional", "uspto", "intellectual property"]):
            # Patent-related query
            enhanced_query = f"patents filed intellectual property USPTO provisional patent applications {query}"
            tag = "has_patents"
            logger.info("Enhanced query for patent-related information")

        
        else:
            # Default case - return the original query with minor enhancement
            logger.info("No specific query enhancement applied")
        
        return enhanced_query, tag
    
    
    except Exception as e:
        logger.error(f"Error enhancing query: {tb.format_exc()}")
        raise RuntimeError("Failed to enhance the query. Check input and faculty configuration.")
