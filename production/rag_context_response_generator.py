import os
import pandas as pd
import time
import traceback as tb
from typing import List, Dict, Any, Literal
from dotenv import load_dotenv

# Import your existing RAG system components
from config.config import Config
from config.logger import Logger
from src.rag_system import RAGSystem
from utils.constants import PROJECT_ROOT, EMBEDDING_DEPARTMENT_NAME
from langchain.retrievers.document_compressors import FlashrankRerank
from utils.clean_context import clean_context
from utils.query_enhancement import query_enhancement

class RAGContextResponseGenerator:
    """
    Generates and saves RAG context and responses for a given set of questions.
    """
    def __init__(self, config=None):
        """
        Initialize the RAG Context and Response Generator.
        
        Args:
            config (Config, optional): Configuration object.
        """
        load_dotenv()
        self.config = config or Config()
        self.logger = Logger(log_file="rag_context_response_generation.log").get_logger()
        
        # Log detailed model information
        self.logger.info(f"Initializing RAG system with:")
        self.logger.info(f"  LLM Provider: {self.config.llm_provider}")
        self.logger.info(f"  LLM Model: {self.config.groq_model_name}")
        self.logger.info(f"  Embedding Client: {self.config.embedding_client}")
        self.logger.info(f"  Embedding Model: {self.config.embedding_model}")
        
        # Initialize RAG system
        self.logger.info("Initializing RAG system...")
        self.rag_system = RAGSystem(self.config)
        self.logger.info("RAG system initialized successfully")

    def load_questions(self, excel_path: str, sheet_name: str) -> pd.DataFrame:
        """
        Load questions from the specified Excel file and sheet.
        
        Args:
            excel_path (str): Path to the Excel file.
            sheet_name (str): Name of the sheet containing questions.
            
        Returns:
            pd.DataFrame: DataFrame containing the questions.
        """
        try:
            self.logger.info(f"Loading questions from {excel_path}, sheet: {sheet_name}")
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # Log the columns found in the Excel file
            self.logger.info(f"Columns found in Excel file: {df.columns.tolist()}")
            
            # Check if expected columns are present
            if "Question" not in df.columns:
                self.logger.warning("'Question' column not found in Excel file")
                
            return df
        except Exception as e:
            self.logger.error(f"Error loading questions: {tb.format_exc()}")
            raise RuntimeError(f"Failed to load questions from {excel_path}") from e

    def get_contexts_with_faiss_reranker(self, question: str):
        """
        Get contexts using FAISS retriever with reranker.
        
        Args:
            question (str): The query question.
            
        Returns:
            List[str]: List of relevant context texts.
        """
        self.logger.info("Using FAISS + Reranker retrieval method")
        
        # Ensure FAISS retriever is available
        if not hasattr(self.rag_system.rag_pipeline, 'retriever'):
            self.logger.info("FAISS retriever not found, initializing...")
            # Make sure FAISS vector store is loaded
            self.config.processor_type = 'faiss'
            vector_store = self.rag_system.vector_store.load_vector_store()
            self.rag_system.rag_pipeline.retriever = vector_store.as_retriever(
                search_kwargs={'k': self.config.retriever_k, 'fetch_k': 25}
            )
        
        # Get documents using FAISS retriever
        retriever = self.rag_system.rag_pipeline.retriever
        contexts = retriever.get_relevant_documents(question)
        
        # Apply reranking
        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)
        reranked_contexts = compressor.compress_documents(contexts, query=question)
        
        # Extract text from documents
        context_texts = [context.page_content for context in reranked_contexts]
        return context_texts

    def get_contexts_with_chroma_metadata(self, question: str):
        """
        Get contexts using Chroma with metadata filtering but without reranking.
        
        Args:
            question (str): The query question.
            
        Returns:
            List[str]: List of relevant context texts.
        """
        self.logger.info("Using Chroma + Metadata retrieval method (without reranker)")
        
        # Generate enhanced query 
        query_enhanced, tag = query_enhancement(question)
        
        # Get documents from Chroma with metadata filtering
        contexts = self.rag_system.chroma_pipeline.improved_search_faculty_vectorstore(
            question, query_enhanced, tag, k_value = 5
        )
        
        # Extract top 5 contexts without reranking
        contexts = contexts[:5]  # Take top 5 directly without reranking
        context_texts = [context.page_content for context in contexts]
        # context_texts = clean_context(context_texts)
        
        return context_texts

    def get_contexts_with_chroma_metadata_reranker(self, question: str):
        """
        Get contexts using Chroma with metadata filtering and reranking.
        
        Args:
            question (str): The query question.
            
        Returns:
            List[str]: List of relevant context texts.
        """
        self.logger.info("Using Chroma + Metadata + Reranker retrieval method")
        
        # Generate enhanced query 
        query_enhanced, tag = query_enhancement(question)
        
        # Get documents from Chroma with metadata filtering
        contexts = self.rag_system.chroma_pipeline.improved_search_faculty_vectorstore(
            question, query_enhanced, tag
        )
        
        # Apply reranking
        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)
        reranked_contexts = compressor.compress_documents(contexts, query=question)
        
        # Extract text from documents
        context_texts = [context.page_content for context in reranked_contexts]
        # context_texts = clean_context(context_texts)
        
        return context_texts

    def generate_contexts_and_responses(
        self, 
        questions: List[str],
        ground_truth: List[str],
        retrieval_method: Literal["faiss_reranker", "chroma_metadata", "chroma_metadata_reranker"] = "chroma_metadata_reranker",
        output_path: str = os.path.join(PROJECT_ROOT, "data", "rag_contexts_responses.xlsx")
    ) -> pd.DataFrame:
        """
        Generate contexts and responses for given questions.
        
        Args:
            questions (List[str]): List of questions to process.
            ground_truth (List[str]): List of ground truth answers.
            retrieval_method (str): Which retrieval method to use:
                - "faiss_reranker": FAISS retriever with reranking
                - "chroma_metadata": Chroma with metadata filtering (no reranking)
                - "chroma_metadata_reranker": Chroma with metadata filtering and reranking
            output_path (str, optional): Path to save the contexts and responses.
        
        Returns:
            pd.DataFrame: DataFrame with questions, contexts, and responses.
        """
        results = []
        
        # Configure processor type based on retrieval method
        if retrieval_method.startswith("faiss"):
            self.config.processor_type = "faiss"
        elif retrieval_method.startswith("chroma"):
            self.config.processor_type = "chroma"
        
        # Update RAG pipeline configuration
        self.logger.info(f"Using retrieval method: {retrieval_method}")
        
        for i, question in enumerate(questions):
            try:
                self.logger.info(f"Processing question {i+1}/{len(questions)}")
                
                # Get contexts based on selected retrieval method
                if retrieval_method == "faiss_reranker":
                    context_texts = self.get_contexts_with_faiss_reranker(question)
                elif retrieval_method == "chroma_metadata":
                    context_texts = self.get_contexts_with_chroma_metadata(question)
                elif retrieval_method == "chroma_metadata_reranker":
                    context_texts = self.get_contexts_with_chroma_metadata_reranker(question)
                else:
                    raise ValueError(f"Unknown retrieval method: {retrieval_method}")
                
                # Generate response
                # Temporarily update RAG pipeline for proper method if needed
                original_processor_type = self.config.processor_type
                self.config.processor_type = "faiss" if retrieval_method == "faiss_reranker" else "chroma"
                
                # Reset RAG pipeline to ensure it uses correct method
                if retrieval_method != self.rag_system.rag_pipeline.config.processor_type:
                    self.logger.info(f"Updating RAG pipeline to use {retrieval_method}")
                    self.rag_system.rag_pipeline.config.processor_type = "faiss" if retrieval_method == "faiss_reranker" else "chroma"
                    self.rag_system.rag_pipeline.setup()
                
                # Generate response using the RAG pipeline
                response = self.rag_system.query(question)
                
                # Restore original processor type
                self.config.processor_type = original_processor_type
                
                # Store results
                results.append({
                    "Question": question,
                    "Contexts": context_texts,
                    "Response": response,
                    "Reference": ground_truth[i],
                    "Retrieval Method": retrieval_method
                })
                
                # Add delay before next iteration
                self.logger.info("Waiting 8 seconds before processing next question...")
                time.sleep(8)
                
            except Exception as e:
                self.logger.error(f"Error processing question {i+1}: {tb.format_exc()}")
                # Add an entry with error information
                results.append({
                    "Question": question,
                    "Contexts": "Error generating contexts",
                    "Response": "Error generating response",
                    "Reference": ground_truth[i] if i < len(ground_truth) else "",
                    "Retrieval Method": retrieval_method
                })
                continue
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)   
             
        # Save to Excel
        try:
            self.logger.info(f"Saving contexts and responses to {output_path}")
            results_df.to_excel(output_path, index=False)
            self.logger.info("Saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving results: {tb.format_exc()}")
        
        return results_df

    def run_generation(
        self, 
        excel_path: str = os.path.join(PROJECT_ROOT, "data", "Rag.xlsx"), 
        # sheet_name: str = "Question List", this was for embedding evaluation
        sheet_name: str = "Question List full",

        retrieval_method: Literal["faiss_reranker", "chroma_metadata", "chroma_metadata_reranker"] = "chroma_metadata_reranker",
        output_path: str = os.path.join(PROJECT_ROOT, "data", "rag_contexts_responses.xlsx")
    ) -> pd.DataFrame:
        """
        Run the complete context and response generation process.
        
        Args:
            excel_path (str, optional): Path to the Excel file with questions.
            sheet_name (str, optional): Name of the sheet containing questions.
            retrieval_method (str): Which retrieval method to use:
                - "faiss_reranker": FAISS retriever with reranking
                - "chroma_metadata": Chroma with metadata filtering (no reranking)
                - "chroma_metadata_reranker": Chroma with metadata filtering and reranking
            output_path (str, optional): Path to save the contexts and responses.
        
        Returns:
            pd.DataFrame: DataFrame with questions, contexts, and responses.
        """
        try:
            self.logger.info(f"Starting RAG context and response generation with questions from {excel_path}")
            self.logger.info(f"Using retrieval method: {retrieval_method}")
            
            # Load questions
            df = self.load_questions(excel_path, sheet_name)
            questions = df["Question"].tolist()
            ground_truth = df["Ground Truth"].tolist() if "Ground Truth" in df.columns else [""] * len(questions)
            
            # Generate contexts and responses
            results_df = self.generate_contexts_and_responses(
                questions, 
                ground_truth,
                retrieval_method=retrieval_method,
                output_path=output_path
            )
            
            self.logger.info("RAG context and response generation completed successfully")
            return results_df
            
        except Exception as e:
            self.logger.error(f"Error during RAG context and response generation: {tb.format_exc()}")
            raise RuntimeError("Failed to complete RAG context and response generation process") from e


if __name__ == "__main__":
    # Example usage with retrieval method selection
    generator = RAGContextResponseGenerator()
    embedding_department_name = EMBEDDING_DEPARTMENT_NAME

    # For FAISS + Reranker
    # results = generator.run_generation(retrieval_method="faiss_reranker", output_path=os.path.join(PROJECT_ROOT, "data", embedding_department_name,"rag_faiss_reranker_results.xlsx"))
    results_final = generator.run_generation(retrieval_method="faiss_reranker", output_path=os.path.join(PROJECT_ROOT, "data", embedding_department_name,"rag_faiss_reranker_results_final.xlsx"))

    # For Chroma + Metadata (no reranker)
    
    # results = generator.run_generation(retrieval_method="chroma_metadata",output_path=os.path.join(PROJECT_ROOT, "data",embedding_department_name, "rag_chroma_metadata_results.xlsx"))
    # results_final = generator.run_generation(retrieval_method="chroma_metadata",output_path=os.path.join(PROJECT_ROOT, "data",embedding_department_name, "rag_chroma_metadata_results_final.xlsx"))
    
    # For Chroma + Metadata + Reranker (default)
    # results = generator.run_generation(retrieval_method="chroma_metadata_reranker",output_path=os.path.join(PROJECT_ROOT, "data", embedding_department_name, "rag_chroma_metadata_reranker_results.xlsx"))
    # results_final = generator.run_generation(retrieval_method="chroma_metadata_reranker",output_path=os.path.join(PROJECT_ROOT, "data", embedding_department_name, "rag_chroma_metadata_reranker_results_final.xlsx"))
    
    # Print summary
    print("\nGeneration Summary:")
    print(f"Total Questions Processed: {len(results_final)}")

    #output would be stored in folder name mentioned above in results call