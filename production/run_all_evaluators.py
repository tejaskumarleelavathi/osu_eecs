import time
import importlib
import sys
import os
from context_precision_eval import ContextPrecisionEvaluator
from context_recall_eval import LLMContextRecallEvaluator
from faithfullness_eval import FaithfulnessEvaluator
from response_relevancy_eval import ResponseRelevancyEvaluator

def run_evaluator(evaluator_instance, method_name, retrieval_methods):
    """Run an evaluator with all retrieval methods"""
    print(f"\n{'-'*80}")
    print(f"Running {method_name} evaluator")
    print(f"{'-'*80}")
    
    for method in retrieval_methods:
        print(f"\nEvaluating with method: {method}")
        
        # Get the correct method to call
        run_method = getattr(evaluator_instance, method_name)
        
        # Run the evaluation
        try:
            results = run_method(retrieval_method=method)
            metric_column = results.columns[4]  # The metric column is typically at index 4
            avg_score = results[metric_column].mean()
            print(f"Completed {method_name} for {method}")
            print(f"Average score: {avg_score}")
        except Exception as e:
            print(f"Error running {method_name} with {method}: {str(e)}")
        
        # Wait 1 minute before next method
        if method != retrieval_methods[-1]:
            print(f"Waiting 60 seconds before next method...")
            time.sleep(60)
    
    return True

def main():
    # Retrieval methods to evaluate
    retrieval_methods = ["faiss_reranker", "chroma_metadata", "chroma_metadata_reranker"]
    
    # Create all evaluator instances
    context_precision = ContextPrecisionEvaluator()
    context_recall = LLMContextRecallEvaluator()
    faithfulness = FaithfulnessEvaluator()
    response_relevancy = ResponseRelevancyEvaluator()
    
    # List of evaluators and their method names
    evaluators = [
        (context_precision, "run_context_precision"),
        (context_recall, "run_context_recall"),
        (faithfulness, "run_faithfulness"),
        (response_relevancy, "run_response_relevancy"),
    ]
    
    # Run all evaluators
    for i, (evaluator, method_name) in enumerate(evaluators):
        run_evaluator(evaluator, method_name, retrieval_methods)
        
        # Wait 1 minute before next evaluator
        if i < len(evaluators) - 1:
            print(f"\nWaiting 60 seconds before next evaluator...")
            time.sleep(60)
    
    print("\nAll evaluations completed!")

if __name__ == "__main__":
    main()