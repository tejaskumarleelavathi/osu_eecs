from typing import Optional
from langchain.llms.base import LLM
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from config.config import Config, LLMProvider
from config.logger import Logger
import traceback as tb

class LLMFactory:
    """Factory class for creating different LLM instances"""
    
    @staticmethod
    def create_llm() -> LLM:
        """
        Create an LLM instance based on the config
        
        Args:
            config: The configuration object
            
        Returns:
            An instance of the specified LLM
        """

        config = Config()
        logger = Logger().get_logger()
        try:
            logger.info(f"LLM provider checked in llm factory invoked from rag_pipeline : {config.llm_provider}")
            if config.llm_provider == LLMProvider.GROQ:
                logger.info(f"Creating Groq LLM with model: {config.groq_model_name}")

                return ChatGroq(
                    model_name=config.groq_model_name,
                )
            elif config.llm_provider == LLMProvider.OLLAMA:
                logger.info(f"Creating Ollama LLM with model: {config.ollama_model_name}")

                return ChatOllama(
                    model=config.ollama_model_name,
                    base_url=config.ollama_base_url,
                )
            
            elif config.llm_provider == LLMProvider.BEDROCK:
                try:
                    # Initialize AWS Bedrock client
                    bedrock = boto3.client(service_name=config.SERVICE_NAME)
                    
                    # Create Bedrock LLM
                    model_kwargs = {
                        'max_tokens': config.MAX_GEN_LEN_LLAMA3,
                        'temperature': config.llm_params.get('temperature', 0.7),
                        'top_p': config.llm_params.get('top_p', 0.9)
                    }
                    
                    llm = BedrockLLM(
                        model_id=config.LLAMA3_MODEL,
                        client=bedrock,
                        model_kwargs=model_kwargs
                    )
                    
                    logger.info(f"Successfully initialized Bedrock model: {config.LLAMA3_MODEL}")
                    return llm
                    
                except Exception as e:
                    logger.error(f"Error initializing primary Bedrock model: {str(e)}\n{tb.format_exc()}")
                    
                    # Try alternative models if primary fails
                    try:
                        alternate_model = "anthropic.claude-v2"  # Fallback model
                        
                        logger.info(f"Trying alternative Bedrock model: {alternate_model}")
                        
                        alt_model_kwargs = {
                            "max_tokens_to_sample": 2000,
                            "temperature": config.llm_params.get('temperature', 0.7),
                            "top_p": config.llm_params.get('top_p', 0.9)
                        }
                        
                        alt_llm = BedrockLLM(
                            model_id=alternate_model,
                            client=bedrock,
                            model_kwargs=alt_model_kwargs
                        )
                        
                        logger.info(f"Successfully initialized alternative Bedrock model: {alternate_model}")
                        return alt_llm
                        
                    except Exception as e:
                        logger.error(f"Failed to initialize alternative Bedrock model: {tb.format_exc()}")
                        raise RuntimeError("Failed to initialize any Bedrock model. Check credentials, model ID, and configuration.") from e

        except Exception as e:
            logger.error(f"Error creating LLM: {tb.format_exc()}")
            raise RuntimeError("Failed to initialize the LLM component. Check configuration and dependencies.") from e
