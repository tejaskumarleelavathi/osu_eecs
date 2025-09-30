import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any
import time
from datetime import datetime

# Import necessary components from your system
from config.config import Config
from config.logger import Logger
from .document_processor import DocumentProcessor
from .vector_store import VectorStore, EmbeddingsModel
from .rag_pipeline import RAGPipeline
from .query_generator import QueryGenerator
from langchain.schema import Document

class RAGTester:
    """Test harness for evaluating RAG system without Streamlit UI"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the RAGTester with the given configuration path.

        Args:
            config_path (str, optional): Path to the configuration file. If not provided, the default `Config` is used.
        
        Returns:
            None
        """
        self.logger = Logger('tester.log').get_logger()
        self.config = Config() if config_path is None else Config(config_path)
        
        # Initialize components
        self.logger.info("Initializing RAG components for testing")
        self.vector_store = VectorStore(self.config)
        self.rag_pipeline = RAGPipeline(self.config)
        
        # Set up pipeline without Streamlit
        self.rag_pipeline.setup()
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing problematic characters and normalizing whitespace.

        Args:
            text (str): The raw text to be cleaned.
        
        Returns:
            str: The cleaned text.
        """
        if not text:
            return ""
        
        # Replace Unicode control characters
        text = re.sub(r'[\u0000-\u001F\u007F-\u009F\u200B-\u200D\uFEFF]', ' ', text)
        
        # Replace Unicode icons and other special characters (like \uf002)
        text = re.sub(r'[\uf000-\uffff]', ' ', text)
        
        # Normalize whitespace: convert multiple spaces, tabs, newlines to single space
        text = re.sub(r'\s+', ' ', text)
        
        # Trim leading/trailing whitespace
        text = text.strip()
        
        return text
        
    def run_test(self, questions: List[str], output_dir: str = "test_results") -> Dict[str, Any]:
        """
        Run tests on a list of questions and save results.

        Args:
            questions (List[str]): List of questions to test.
            output_dir (str, optional): Directory to save results. Defaults to "test_results".
        
        Returns:
            Dict[str, Any]: A dictionary with test results.
        """
        results = []
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Timestamp for this test run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info(f"Starting test run with {len(questions)} questions")
        
        # Process each question
        for i, question in enumerate(questions):
            self.logger.info(f"Processing question {i+1}/{len(questions)}: {question}")
            
            try:
                # Get start time
                start_time = time.time()
                
                # Run standard query to get answer
                response = self.rag_pipeline.query(question)
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                print(f"Response {response}")
                # Extract context docs (this depends on how your pipeline is structured)
                # If you're using RetrievalQA or a similar pattern, you might get this from:
                if hasattr(self.rag_pipeline, 'reranker'):
                    context_docs = self.rag_pipeline.reranker.invoke(question)
                    
                else:
                    # For multi-query retrieval
                    context_docs = self.rag_pipeline.query_generator.retrieve_with_multiple_queries(question)
                
                # Format context for readability
                formatted_context = []
                full_context = []
                for doc in context_docs:
                # Clean the text content
                    cleaned_content = self._clean_text(doc.page_content)
                    
                    formatted_context.append({
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "Unknown"),
                        "content": cleaned_content
                    })
                    full_context.append(cleaned_content)
                full_context = " ".join(full_context)
                # Append results
                results.append({
                    "question": question,
                    "answer": response,
                    "context": formatted_context,
                    "full_context":full_context,
                    "retrieval_time_seconds": elapsed_time
                })
                
                self.logger.info(f"Completed question {i+1}: retrieved {len(context_docs)} documents in {elapsed_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error processing question '{question}': {str(e)}")
                results.append({
                    "question": question,
                    "error": str(e),
                    "status": "failed"
                })
        
        # Save results to JSON file
        result_file = os.path.join(output_dir, f"rag_test_results_{timestamp}.json")
        with open(result_file, 'w') as f:
            json.dump({
                "test_date": timestamp,
                "config": {
                    "model": self.config.groq_model_name,
                    "embedding_model": self.config.embedding_model,
                    "retriever_k": self.config.retriever_k,
                    "num_queries": self.config.num_queries if hasattr(self.config, "num_queries") else None,
                },
                "results": results
            }, f, indent=2)
            
        self.logger.info(f"Test results saved to {result_file}")
        
        # Create a summary report
        self._create_summary_report(results, output_dir, timestamp)
        
        return results
    
    def _create_summary_report(self, results: List[Dict], output_dir: str, timestamp: str):
        """
        Generate a markdown summary report of test results.

        Args:
            results (List[Dict]): List of dictionaries containing test results.
            output_dir (str): Directory to save the summary report.
            timestamp (str): Timestamp of the test run.
        
        Returns:
            None
        """
        
        # Count successful queries
        successful = sum(1 for r in results if "error" not in r)
        
        # Calculate average retrieval time
        avg_time = sum(r.get("retrieval_time_seconds", 0) for r in results if "retrieval_time_seconds" in r) / successful if successful else 0
        
        # Create report content
        report = f"""# RAG System Test Results

Test Date: {timestamp}

## Summary
- Total Questions: {len(results)}
- Successful Responses: {successful}
- Failed Responses: {len(results) - successful}
- Average Retrieval Time: {avg_time:.2f} seconds

## Configuration
- LLM Model: {self.config.groq_model_name}
- Embedding Model: {self.config.embedding_model}
- Retriever K: {self.config.retriever_k}
- Number of Queries (if multi-query): {self.config.num_queries if hasattr(self.config, "num_queries") else "N/A"}

## Detailed Results

"""
        
        # Add details for each question
        for i, result in enumerate(results):
            question = result["question"]
            report += f"### Question {i+1}: {question}\n\n"
            
            if "error" in result:
                report += f"**ERROR**: {result['error']}\n\n"
                continue
                
            # Add answer
            report += f"**Answer**:\n{result['answer']}\n\n"
            
            # Add context summary
            report += f"**Context Documents**: {len(result['context'])}\n\n"
            report += "**Top Context Sources**:\n"
            
            for j, ctx in enumerate(result['context'][:3]):  # Just show top 3 for brevity
                report += f"- {ctx['source']} (page {ctx['page']})\n"
                
            report += f"\n**Retrieval Time**: {result.get('retrieval_time_seconds', 0):.2f} seconds\n\n"
            report += "---\n\n"
            
        # Save report to file
        report_file = os.path.join(output_dir, f"rag_test_summary_{timestamp}.md")
        with open(report_file, 'w') as f:
            f.write(report)
            
        self.logger.info(f"Summary report saved to {report_file}")


