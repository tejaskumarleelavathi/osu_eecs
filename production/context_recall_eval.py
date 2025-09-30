import os
import pandas as pd
from dotenv import load_dotenv

# Ragas imports
from ragas import EvaluationDataset, evaluate, SingleTurnSample
from ragas.metrics import LLMContextRecall
from ragas import RunConfig

# Local project imports
from config.config import Config
from config.logger import Logger
from production.src.rag_system_main import RAGSystem
from src.embeddings_model import EmbeddingsModel
from utils.constants import PROJECT_ROOT, EMBEDDING_DEPARTMENT_NAME
import openpyxl
import ast
import time
from typing import Literal

class LLMContextRecallEvaluator:
    def __init__(self, config=None):
        load_dotenv()
        self.rag_system = RAGSystem()

        self.config = config or Config()
        
        self.logger = Logger(log_file="context_recall_eval.log").get_logger()

        self.logger.info("Initializing LLM Context Recall Evaluator")

        self.embeddings = EmbeddingsModel.get_instance(
            self.config.embedding_client,
            self.config.embedding_model
        )

        self.logger.info("LLM Context Recall Evaluator initialized successfully")

    def load_dataset(self, context_response_path) -> EvaluationDataset:
        df = pd.read_excel(context_response_path)

        if "Reference" not in df.columns:
            self.logger.warning("Column 'Reference' not found in input file. Filling with empty strings.")
            df["Reference"] = ["" for _ in range(len(df))]

        samples = []
        for _, row in df.iterrows():
            # Ensure contexts is a list, even if it's a single string
            contexts = ast.literal_eval(row["Contexts"])
            # if isinstance(contexts, str):
            #     contexts = [contexts]
            
            samples.append(SingleTurnSample(
                user_input=row["Question"],
                retrieved_contexts=contexts,
                response=row["Response"],
                reference=row["Reference"]
            ))

        return EvaluationDataset(samples=samples)

    def get_input_output_paths(self, retrieval_method: Literal["faiss_reranker", "chroma_metadata", "chroma_metadata_reranker"]):
        """
        Get the appropriate input and output file paths based on the retrieval method.
        
        Args:
            retrieval_method: The retrieval method used
            
        Returns:
            tuple: (context_response_path, output_path)
        """
        # Map of retrieval methods to file names
        file_mapping = {
            "faiss_reranker": "rag_faiss_reranker_results.xlsx",
            "chroma_metadata": "rag_chroma_metadata_results.xlsx",
            "chroma_metadata_reranker": "rag_chroma_metadata_reranker_results.xlsx"
        }
        embedding_department_name = EMBEDDING_DEPARTMENT_NAME

        # Get the appropriate input file based on retrieval method
        input_file = file_mapping.get(retrieval_method, "rag_contexts_responses.xlsx")
        context_response_path = os.path.join(PROJECT_ROOT, "data",embedding_department_name, input_file)
        
        # Output path remains the same, we'll use different sheet names
        output_path = os.path.join(PROJECT_ROOT, "data", embedding_department_name, "rag_evaluation_results.xlsx")
        
        return context_response_path, output_path

    def run_context_recall(self, retrieval_method: Literal["faiss_reranker", "chroma_metadata", "chroma_metadata_reranker"] = "chroma_metadata_reranker"):
        """
        Run context recall evaluation for a specific retrieval method.
        
        Args:
            retrieval_method: The retrieval method to evaluate
            
        Returns:
            DataFrame: Results of the evaluation
        """
        self.logger.info(f"Running context recall evaluation for method: {retrieval_method}")
        
        # Get the appropriate input and output paths
        context_response_path, output_path = self.get_input_output_paths(retrieval_method)
        
        if not os.path.exists(context_response_path):
            self.logger.error(f"Input file not found: {context_response_path}")
            raise FileNotFoundError(f"Input file not found: {context_response_path}")

        # Map method to sheet name
        sheet_name_mapping = {
            "faiss_reranker": "context_recall_faiss",
            "chroma_metadata": "context_recall_chroma",
            "chroma_metadata_reranker": "context_recall_chroma_rerank"
        }
        sheet_name = sheet_name_mapping.get(retrieval_method, "context_recall")

        # Load the dataset
        dataset = self.load_dataset(context_response_path)

        run_config = RunConfig(
            max_workers=1,
            timeout=180,
            max_retries=3,
            max_wait=60,
            exception_types=(Exception,)
        )

        samples = dataset
        results_list = []

        for i, sample in enumerate(samples):
            try:
                self.logger.info(f"Evaluating sample {i+1}/{len(samples)}")
                mini_dataset = EvaluationDataset(samples=[sample])
                result = evaluate(
                    dataset=mini_dataset,
                    metrics=[LLMContextRecall()],
                    llm=self.rag_system.rag_pipeline.llm,
                    embeddings=self.embeddings,
                    raise_exceptions=False,
                    show_progress=False
                )
                self.logger.info(f"Question being evaluated : {sample.user_input} ")
                score = result.scores[0]
                results_list.append({
                    'Question': sample.user_input,
                    'Contexts': sample.retrieved_contexts,
                    'Reference': sample.reference,
                    'Response': sample.response,
                    'Context Recall': score.get('context_recall') if isinstance(score, dict) else score,
                    'Retrieval Method': retrieval_method
                })

                self.logger.info(f"Sample {i+1}: Score = {score}")
                
                time.sleep(12)  # Add delay to prevent rate limiting
                self.logger.info(f"Added delay in llm call for question {i+1}")
            except Exception as e:
                self.logger.warning(f"Error evaluating sample {i+1}: {e}")
                results_list.append({
                    'Question': sample.user_input,
                    'Contexts': sample.retrieved_contexts,
                    'Reference': sample.reference,
                    'Response': sample.response,
                    'Context Recall': None,
                    'Retrieval Method': retrieval_method
                })

        results_df = pd.DataFrame(results_list)

        # Log individual context recall scores
        for idx, row in results_df.iterrows():
            self.logger.info(f"Question {idx + 1}: {row['Question']}")
            self.logger.info(f"Context Recall: {row.get('Context Recall', 'N/A')}")
            self.logger.info("-" * 50)

        # Calculate and log average score
        avg_score = results_df['Context Recall'].mean()
        self.logger.info(f"Average Context Recall score for {retrieval_method}: {avg_score}")

        try:
            # Save results to Excel
            if os.path.exists(output_path):
                try:
                    book = openpyxl.load_workbook(output_path)
                    if sheet_name in book.sheetnames:
                        book.remove(book[sheet_name])
                        book.save(output_path)
                except Exception as e:
                    self.logger.warning(f"Error modifying existing workbook: {e}")
                    # If there's an error, we'll just overwrite the sheet when saving

            with pd.ExcelWriter(output_path, engine='openpyxl', mode='a' if os.path.exists(output_path) else 'w') as writer:
                results_df.to_excel(writer, sheet_name=sheet_name, index=False)

            self.logger.info(f"Context Recall evaluation results saved to sheet '{sheet_name}' in {output_path}")
        except Exception as e:
            self.logger.error(f"Error writing evaluation results: {e}")
            raise

        # If this is the first time running evaluations, create a summary sheet
        try:
            self.update_summary_sheet(output_path, retrieval_method, avg_score)
        except Exception as e:
            self.logger.error(f"Error updating summary sheet: {e}")

        return results_df

    def update_summary_sheet(self, output_path, retrieval_method, avg_score):
        """Update or create a summary sheet with results from all methods"""
        summary_data = {
            'Retrieval Method': [retrieval_method],
            'Average Context Recall': [avg_score]  # Replace [Metric Name] with appropriate metric name
        }
        new_summary_df = pd.DataFrame(summary_data)
        
        try:
            # Check if the summary sheet already exists
            if os.path.exists(output_path):
                try:
                    # Read existing summary
                    existing_summary = pd.read_excel(output_path, sheet_name='summary')
                    
                    # Check if this method already exists in summary
                    method_exists = False
                    for idx, row in existing_summary.iterrows():
                        if row['Retrieval Method'] == retrieval_method:
                            # Replace 'Average Context Recall' with the appropriate column name
                            if 'Average Context Recall' not in existing_summary.columns:
                                existing_summary['Average Context Recall'] = None
                            existing_summary.at[idx, 'Average Context Recall'] = avg_score
                            method_exists = True
                            break
                    
                    if not method_exists:
                        # Add new entry
                        existing_summary = pd.concat([existing_summary, new_summary_df], ignore_index=True)
                    
                    # Remove the existing summary sheet
                    book = openpyxl.load_workbook(output_path)
                    if 'summary' in book.sheetnames:
                        book.remove(book['summary'])
                        book.save(output_path)
                    
                    # Save the updated summary
                    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a') as writer:
                        existing_summary.to_excel(writer, sheet_name='summary', index=False)
                    
                except Exception as e:
                    self.logger.warning(f"Error reading existing summary, creating new one: {str(e)}")
                    # Create a new summary sheet if there was an error
                    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a') as writer:
                        new_summary_df.to_excel(writer, sheet_name='summary', index=False)
            else:
                # Create a new file with summary
                with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
                    new_summary_df.to_excel(writer, sheet_name='summary', index=False)
            
            self.logger.info(f"Updated summary sheet with results for {retrieval_method}")
            
        except Exception as e:
            self.logger.error(f"Error updating summary sheet: {str(e)}")


if __name__ == "__main__":
    evaluator = LLMContextRecallEvaluator()
    
    # Specify which retrieval method to evaluate
    # Options: "faiss_reranker", "chroma_metadata", "chroma_metadata_reranker"
    # method = "faiss_reranker"  # Change this to evaluate different methods
    method = "chroma_metadata"  # Change this to evaluate different methods
    # method = "chroma_metadata_reranker"  # Change this to evaluate different methods

    results = evaluator.run_context_recall(retrieval_method=method)
    print(f"Completed context recall evaluation for {method}")
    print(f"Average score: {results['Context Recall'].mean()}")