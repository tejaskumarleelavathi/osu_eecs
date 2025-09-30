from enum import Enum
from typing import Optional, Dict, Any, List
import os

class LLMProvider(str, Enum):
    """Enum for supported LLM providers"""
    GROQ = "groq"
    OLLAMA = "ollama"
    BEDROCK = "bedrock"

class Config:
    """Singleton configuration class to store all parameters"""
    _instance: Optional["Config"] = None  # Singleton instance

    def __new__(cls) -> "Config":
        """
        Creates a new instance of the Config class if one doesn't exist.

        Args:
            cls (type): The class type to create an instance of.
        
        Returns:
            Config: The singleton instance of the Config class.
        """
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """
        Initialize configuration parameters.

        Args:
            None
        
        Returns:
            None
        """
        # Vector store settings

        self.faculty_info = {} 
        self.faculty_collections ={}
 


        # # Document processing settings
        self.chunk_size: int = 1500
        self.chunk_overlap: int = 400
        self.use_prompt_chaining = True  

        self.email_to_name_corrections = {
            "chenliz": "Chen Lizhong",
            "leestef": "Stefan Lee",
            "tadepall": "Prasad Tadepalli",
            "raich": "Raviv Raich",
            "ramseyst": "Stephan Ramsey",
            "sinisa": "Sinisa Todorovic",
            "termehca": "Arash Termehchy",
            "tgd": "Thomas G Dietterich",
            "wong": "Weng Keen Wong",
            "hamdaoui": "Bechir Hamdaoui",
            "Mathews": "V John Mathews",
            "julie.a.adams": "Julie Adams"
            # Add other corrections as needed
        }

        #Path to Data Folder 

        

        # Processor type : faiss or chroma
        self.processor_type = "chroma"
        # self.processor_type = "chroma"

        # Embedding model settings
        self.embedding_client = "huggingface" #bedrock
        
        # self.EMBEDDING_MODEL = "amazon.titan-embed-image-v1"
        
        # self.embedding_model_name: str = "WhereIsAI/UAE-Large-V1"
        # self.embedding_model_name: str = "Snowflake/snowflake-arctic-embed-m"
        # self.embedding_model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0"
        # self.embedding_model_name: str = "Snowflake/snowflake-arctic-embed-m-v1.5"
        # self.embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
        # self.embedding_model_name: str = "Linq-AI-Research/Linq-Embed-Mistral"
        # self.embedding_model_name: str = "Lajavaness/bilingual-embedding-small"
        # self.embedding_model_name: str = "intfloat/e5-base"

        # LLM settings
        self.llm_provider: LLMProvider = LLMProvider.GROQ
        
        # self.groq_model_name: str = "llama3-70b-8192"  # DEPRECATED - model decommissioned
        self.groq_model_name: str ="llama-3.3-70b-versatile"
        self.groq_models: List[str] = [
            # "llama3-70b-8192",  # DEPRECATED - model decommissioned
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "llama3-8b-8192",
            "gemma2-9b-it",      
            "llama-guard-3-8b",
            
            "llama-3.3-70b-versatile",        
            "llama-3.1-8b-instant",
     
            "llama-3.3-70b-specdec",
            "qwen-2.5-32b",
            "deepseek-r1-distill-llama-70b"
        ]

        # Ollama model settings 
        self.ollama_model_name: str = "llama3"
        self.ollama_base_url: str = "http://localhost:11434"
        self.ollama_models: List[str] = [
            "llama3",
            "llama2:13b",
            "llama3.2",
            "deepseek-r1:14b"
        ]

        # Bedrock model settings
        self.LLAMA3_MODEL = "meta.llama3-70b-instruct-v1"
        self.MAX_GEN_LEN_LLAMA3 = 2048
        self.bedrock_models: List[str] = [
            "meta.llama3-70b-instruct-v1",
            "anthropic.claude-v2",
            "amazon.titan-text-express-v1"
        ]

        # Retriever settings
        self.retriever_k: int = 20

        # LLM-specific parameters
        self.llm_params: Dict[str, Any] = {
            "temperature": 0,
            "top_p": 0.6
        }
        # Multi-query setting
        self.num_queries: int = 3


    @property
    def embedding_model(self) -> str:
        """
        Get the appropriate embedding model based on the selected embedding client.

        Args:
            self.embedding_client (str): The embedding client in use ('huggingface' or 'bedrock').
        
        Returns:
            str: The name of the embedding model.
        """

        try:    
            if self.embedding_client=='huggingface':
                return "BAAI/bge-m3"
                # return "Snowflake/snowflake-arctic-embed-m-long"
                # return "Snowflake/snowflake-arctic-embed-l-v2.0"
                # return "BAAI/bge-m3"
            
            elif self.embedding_client == 'bedrock':
                return "amazon.titan-embed-image-v1"
            
            else:
                raise ValueError(f"Unsupported embedding client: {self.embedding_client}")
            
        except Exception as e:
            self.logger.error(f"Error retrieving embedding model: {tb.format_exc()}")
            raise RuntimeError(f"Failed to determine embedding model for client '{self.embedding_client}'. Check configuration.") from e
