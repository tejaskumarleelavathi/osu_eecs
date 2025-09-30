
from langchain_community.embeddings import BedrockEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Optional
from langchain.schema import Document

from langsmith import traceable
from config.config import Config
from config.logger import Logger
import traceback as tb
import streamlit as st
import numpy as np
import boto3
import torch

class EmbeddingsModel:
    _instance = None
    _model_name = None

    @classmethod
    def get_instance(cls, embedding_client: str ="huggingface", model_name: str = None):
        """
        Returns a singleton embedding model instance.

        Args:
            embedding_client (str): "huggingface" or "bedrock"
            model_name (str): name or ID of the embedding model

        Returns:
            Embeddings object
        """
        if cls._model_name == model_name and cls._instance is not None:
            return cls._instance
        print(f"Model name {model_name} from embedding_models")
        logger = Logger().get_logger()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            if embedding_client == "huggingface":
                logger.info(f"Initializing HuggingFace embedding model: {model_name}")
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"trust_remote_code": True, "device": device}
                )

            elif embedding_client == "bedrock":
                logger.info(f"Initializing Bedrock embedding model: {model_name}")
                bedrock_client = boto3.client(service_name="bedrock-runtime")
                embeddings = BedrockEmbeddings(model_id=model_name, client=bedrock_client)

            else:
                raise ValueError(f"Unsupported embedding model type: {embedding_client}")

            # Cache in singleton + optionally Streamlit session state
            cls._instance = embeddings
            cls._model_name = model_name

            if "embedding_model" not in st.session_state:
                logger.info(f"Initializing embedding model in session: {model_name}")
            else:
                logger.info(f"Replacing existing embedding model with: {model_name}")

            st.session_state.embedding_model_name = model_name
            st.session_state.embedding_model = embeddings

            return embeddings

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {tb.format_exc()}")
            raise RuntimeError("Failed to initialize the embedding model. Check model name, client type, and dependencies.") from e
