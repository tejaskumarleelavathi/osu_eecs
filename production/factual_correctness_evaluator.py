import os
import pandas as pd
from dotenv import load_dotenv

# Ragas imports
from ragas import EvaluationDataset, evaluate, SingleTurnSample
from ragas.metrics._factual_correctness import FactualCorrectness
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

class FactualCorrectnessEvaluator:
    def __init__(self, config=None):
        load_dotenv()
        self.config = config or Config()
        self.rag_system = RAGSystem(self.config)
        
        self.logger = Logger(log_file="factual_correctness_eval.log").get_logger()

        self.logger.info("Initializing Factual Correctness Evaluator")

        self.embeddings = EmbeddingsModel.get_instance(
            self.config.embedding_client,
            self.config.embedding_model
        )

        self.logger.info("Factual Correctness Evaluator initialized successfully")

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
        output_path = os.path.join(PROJECT_ROOT, "data",embedding_department_name, "rag_evaluation_results.xlsx")
        
        return context_response_path, output_path

    def run_factual_correctness(self, retrieval_method: Literal["faiss_reranker", "chroma_metadata", "chroma_metadata_reranker"]):
        """
        Run factual correctness evaluation for a specific retrieval method.
        
        Args:
            retrieval_method: The retrieval method to evaluate
            
        Returns:
            DataFrame: Results of the evaluation
        """
        self.logger.info(f"Running factual correctness evaluation for method: {retrieval_method}")
        
        # Get the appropriate input and output paths
        context_response_path, output_path = self.get_input_output_paths(retrieval_method)
        
        if not os.path.exists(context_response_path):
            self.logger.error(f"Input file not found: {context_response_path}")
            raise FileNotFoundError(f"Input file not found: {context_response_path}")

        # Map method to sheet name
        sheet_name_mapping = {
            "faiss_reranker": "factual_correctness_faiss",
            "chroma_metadata": "factual_correctness_chroma",
            "chroma_metadata_reranker": "factual_correctness_chroma_rerank"
        }
        sheet_name = sheet_name_mapping.get(retrieval_method, "factual_correctness")

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
                    metrics=[FactualCorrectness(    mode="precision",atomicity="low")],
                    llm=self.rag_system.rag_pipeline.llm,
                    embeddings=self.embeddings,
                    raise_exceptions=False,
                    show_progress=False
                )

                score = result.scores[0]
                results_list.append({
                    'Question': sample.user_input,
                    'Contexts': sample.retrieved_contexts,
                    'Reference': sample.reference,
                    'Response': sample.response,
                    'Factual Correctness': score.get('factual_correctness') if isinstance(score, dict) else score,
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
                    'Factual Correctness': None,
                    'Retrieval Method': retrieval_method
                })

        results_df = pd.DataFrame(results_list)

        # Log individual factual correctness scores
        for idx, row in results_df.iterrows():
            self.logger.info(f"Question {idx + 1}: {row['Question']}")
            self.logger.info(f"Factual Correctness: {row.get('Factual Correctness', 'N/A')}")
            self.logger.info("-" * 50)

        # Calculate and log average score
        avg_score = results_df['Factual Correctness'].mean()
        self.logger.info(f"Average Factual Correctness score for {retrieval_method}: {avg_score}")

        # Save results to the appropriate sheet
        self.save_results(output_path, sheet_name, results_df)
        
        # Update the summary sheet with the new results
        self.update_summary_sheet(output_path, retrieval_method, avg_score)

        return results_df
        
    def save_results(self, output_path, sheet_name, results_df):
        """Save results to the specified Excel sheet, handling existing sheets properly"""
        try:
            # Check if the file exists
            if os.path.exists(output_path):
                # Try to remove the sheet if it exists
                try:
                    book = openpyxl.load_workbook(output_path)
                    if sheet_name in book.sheetnames:
                        book.remove(book[sheet_name])
                        book.save(output_path)
                        self.logger.info(f"Removed existing sheet '{sheet_name}'")
                except Exception as e:
                    self.logger.warning(f"Error modifying existing workbook: {e}")
            
            # Create or append to the Excel file
            # Use 'a' mode if file exists, 'w' if it doesn't
            with pd.ExcelWriter(
                output_path, 
                engine='openpyxl', 
                mode='a' if os.path.exists(output_path) else 'w'
            ) as writer:
                results_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
            self.logger.info(f"Results saved to sheet '{sheet_name}' in {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise

    def update_summary_sheet(self, output_path, retrieval_method, avg_score):
        """Update or create a summary sheet with results from all methods"""
        summary_data = {
            'Retrieval Method': [retrieval_method],
            'Average Factual Correctness': [avg_score]
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
                            if 'Average Factual Correctness' not in existing_summary.columns:
                                existing_summary['Average Factual Correctness'] = None
                            existing_summary.at[idx, 'Average Factual Correctness'] = avg_score
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
    evaluator = FactualCorrectnessEvaluator()
    
    # Specify which retrieval method to evaluate
    # Options: "faiss_reranker", "chroma_metadata", "chroma_metadata_reranker"
    retrieval_method = "chroma_metadata"
    results = evaluator.run_factual_correctness(retrieval_method=retrieval_method)
    print(f"Completed factual correctness evaluation for {retrieval_method}")
    print(f"Average score: {results['Factual Correctness'].mean()}")