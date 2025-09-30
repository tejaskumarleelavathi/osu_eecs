import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import os
import PyPDF2
import fitz  # PyMuPDF - better for text extraction
from pathlib import Path
import re

class RAGEvaluationSystem:
    def __init__(self, excel_file_path="data/Rag.xlsx", sheet_name="Question List", 
                 pdf_folder_path="C:/Coding/Projects/osu_eecs_chatbot/suyash/pipeline_output/pdfs"):
        self.excel_file_path = excel_file_path
        self.sheet_name = sheet_name
        self.pdf_folder_path = pdf_folder_path
        self.questions_df = None
        self.pdf_texts = {}  # Store extracted PDF texts
        self.pdf_files = []  # Store PDF file names
        
        self.load_questions()
        self.load_pdf_documents()

    def load_questions(self):
        """Load questions and ground truth from Excel file"""
        try:
            self.questions_df = pd.read_excel(self.excel_file_path, sheet_name=self.sheet_name)
            print(f"✅ Loaded {len(self.questions_df)} questions from {self.excel_file_path}")
            print("Columns found:", list(self.questions_df.columns))
            
            # Clean column names
            self.questions_df.columns = [col.strip() for col in self.questions_df.columns]
            
            # Check for expected columns
            if 'Question' in self.questions_df.columns and 'Ground Truth' in self.questions_df.columns:
                print("✅ Found 'Question' and 'Ground Truth' columns")
            else:
                print("⚠️ Expected columns 'Question' and 'Ground Truth' not found")
                print("Available columns:", list(self.questions_df.columns))
            
            # Show first few rows
            print("\nFirst 3 questions:")
            for idx, row in self.questions_df.head(3).iterrows():
                print(f"Q{idx+1}: {row.get('Question', 'N/A')[:100]}...")
                print(f"A{idx+1}: {row.get('Ground Truth', 'N/A')[:100]}...")
                print()
                
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # Clean the text
            text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
            text = text.strip()
            return text
            
        except Exception as e:
            print(f"⚠️ Error extracting text from {pdf_path}: {e}")
            return ""
    
    def load_pdf_documents(self):
        """Load and extract text from all PDF files in the folder"""
        print(f"\n📁 Loading PDF documents from: {self.pdf_folder_path}")
        
        if not os.path.exists(self.pdf_folder_path):
            print(f"❌ PDF folder not found: {self.pdf_folder_path}")
            print("Creating sample documents for demonstration...")
            self.create_sample_documents()
            return
        
        pdf_files = list(Path(self.pdf_folder_path).glob("*.pdf"))
        
        if not pdf_files:
            print("❌ No PDF files found in the folder")
            self.create_sample_documents()
            return
        
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            print(f"  📄 Processing: {pdf_file.name}")
            text = self.extract_text_from_pdf(str(pdf_file))
            
            if text:
                self.pdf_texts[pdf_file.name] = text
                self.pdf_files.append(pdf_file.name)
                print(f"    ✅ Extracted {len(text)} characters")
            else:
                print(f"    ❌ Failed to extract text")
        
        print(f"\n✅ Successfully loaded {len(self.pdf_texts)} PDF documents")
        print("PDF files loaded:")
        for i, filename in enumerate(self.pdf_files):
            print(f"  {i}: {filename}")
    
    def calculate_text_similarity(self, question, ground_truth, document_text):
        """Calculate how well a document matches the question and expected answer"""
        # Simple keyword matching approach
        question_words = set(question.lower().split())
        truth_words = set(ground_truth.lower().split())
        doc_words = set(document_text.lower().split())
        
        # Calculate overlap
        question_overlap = len(question_words.intersection(doc_words)) / len(question_words) if question_words else 0
        truth_overlap = len(truth_words.intersection(doc_words)) / len(truth_words) if truth_words else 0
        
        # Combined relevance score
        relevance_score = (question_overlap + truth_overlap) / 2
        return relevance_score
    
    def determine_ground_truth_relevance(self, question, ground_truth):
        """Determine which PDFs are relevant based on question and ground truth answer"""
        relevance_scores = {}
        
        for filename, doc_text in self.pdf_texts.items():
            score = self.calculate_text_similarity(question, ground_truth, doc_text)
            relevance_scores[filename] = score
        
        # Sort by relevance
        sorted_docs = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Consider top 30% as relevant (minimum 1, maximum 3)
        num_relevant = max(1, min(3, len(sorted_docs) // 3))
        relevant_docs = [filename for filename, score in sorted_docs[:num_relevant] if score > 0.1]
        
        return relevant_docs, relevance_scores
    
    def evaluate_model(self, model_name, max_questions=None):
        """Evaluate a single embedding model"""
        if self.questions_df is None or not self.pdf_texts:
            print("❌ No questions or documents loaded!")
            return None
        
        print(f"\n🔍 Evaluating Model: {model_name}")
        print("-" * 50)
        
        # Load model
        try:
            model = SentenceTransformer(model_name)
            
            # Get model characteristics
            total_params = sum(p.numel() for p in model.parameters())
            total_params_m = total_params / 1_000_000
            test_embedding = model.encode(["test"])
            embedding_dim = test_embedding.shape[1]
            
            print(f"📊 Model Info: {total_params_m:.1f}M parameters, {embedding_dim}D embeddings")
            
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {e}")
            return None
        
        # Encode all PDF documents
        print("🔄 Encoding PDF documents...")
        pdf_texts_list = [self.pdf_texts[filename] for filename in self.pdf_files]
        start_time = time.time()
        pdf_embeddings = model.encode(pdf_texts_list, show_progress_bar=True)
        encoding_time = time.time() - start_time
        print(f"⏱️ Document encoding time: {encoding_time:.2f}s")
        
        # Test questions (limit if specified)
        questions_to_test = self.questions_df.copy()
        if max_questions:
            questions_to_test = questions_to_test.head(max_questions)
        
        # Evaluate each question
        results = []
        total_query_time = 0
        
        print(f"\n🧪 Testing {len(questions_to_test)} questions...")
        
        for idx, row in questions_to_test.iterrows():
            question = row['Question']
            ground_truth = row['Ground Truth']
            
            print(f"\nQ{idx+1}: {question[:80]}...")
            
            # Determine ground truth relevance
            relevant_docs, relevance_scores = self.determine_ground_truth_relevance(question, ground_truth)
            relevant_indices = [self.pdf_files.index(doc) for doc in relevant_docs if doc in self.pdf_files]
            
            print(f"📋 Relevant documents: {relevant_docs}")
            
            # Encode question and find similarities
            start_time = time.time()
            question_embedding = model.encode([question])
            query_time = time.time() - start_time
            total_query_time += query_time
            
            # Calculate similarities
            similarities = cosine_similarity(question_embedding, pdf_embeddings)[0]
            ranked_indices = np.argsort(similarities)[::-1]
            
            # Calculate metrics
            precision_at_1 = 1.0 if len(relevant_indices) > 0 and ranked_indices[0] in relevant_indices else 0.0
            
            # Precision@3
            top_3 = ranked_indices[:3]
            relevant_in_top_3 = sum(1 for idx in top_3 if idx in relevant_indices)
            precision_at_3 = relevant_in_top_3 / min(3, len(relevant_indices)) if relevant_indices else 0.0
            
            # MRR
            mrr = 0.0
            for rank, doc_idx in enumerate(ranked_indices):
                if doc_idx in relevant_indices:
                    mrr = 1.0 / (rank + 1)
                    break
            
            # Show top results
            print(f"🎯 Top 3 retrieved documents:")
            for i in range(min(3, len(ranked_indices))):
                doc_idx = ranked_indices[i]
                doc_name = self.pdf_files[doc_idx]
                similarity = similarities[doc_idx]
                is_relevant = "✅" if doc_idx in relevant_indices else "❌"
                print(f"   {i+1}. {doc_name} (sim: {similarity:.3f}) {is_relevant}")
            
            print(f"📈 Metrics: P@1={precision_at_1:.3f}, P@3={precision_at_3:.3f}, MRR={mrr:.3f}")
            
            result = {
                'question_id': idx,
                'question': question,
                'relevant_docs': relevant_docs,
                'top_retrieved': self.pdf_files[ranked_indices[0]],
                'top_similarity': similarities[ranked_indices[0]],
                'precision_at_1': precision_at_1,
                'precision_at_3': precision_at_3,
                'mrr': mrr,
                'query_time_ms': query_time * 1000
            }
            results.append(result)
        
        # Calculate averages
        avg_metrics = {
            'model_name': model_name,
            'parameters_m': total_params_m,
            'embedding_dim': embedding_dim,
            'num_questions': len(results),
            'num_documents': len(self.pdf_files),
            'document_encoding_time_s': encoding_time,
            'avg_precision_at_1': np.mean([r['precision_at_1'] for r in results]),
            'avg_precision_at_3': np.mean([r['precision_at_3'] for r in results]),
            'avg_mrr': np.mean([r['mrr'] for r in results]),
            'avg_query_time_ms': np.mean([r['query_time_ms'] for r in results]),
            'total_evaluation_time_s': total_query_time + encoding_time
        }
        
        print(f"\n📊 SUMMARY RESULTS:")
        print(f"   Questions: {avg_metrics['num_questions']}")
        print(f"   Documents: {avg_metrics['num_documents']}")
        print(f"   Avg Precision@1: {avg_metrics['avg_precision_at_1']:.3f}")
        print(f"   Avg Precision@3: {avg_metrics['avg_precision_at_3']:.3f}")
        print(f"   Avg MRR: {avg_metrics['avg_mrr']:.3f}")
        print(f"   Avg Query Time: {avg_metrics['avg_query_time_ms']:.1f}ms")
        
        return avg_metrics, results
    
    def compare_models(self, model_names, max_questions=None):
        """Compare multiple embedding models"""
        print("🏁 STARTING MODEL COMPARISON")
        print("=" * 80)
        
        all_results = []
        
        for model_name in model_names:
            try:
                metrics, detailed_results = self.evaluate_model(model_name, max_questions)
                if metrics:
                    all_results.append(metrics)
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
        
        if not all_results:
            print("❌ No models were successfully evaluated!")
            return None
        
        # Create comparison table
        comparison_df = pd.DataFrame(all_results)
        
        # Calculate combined quality score
        comparison_df['quality_score'] = (
            comparison_df['avg_precision_at_1'] * 0.4 +  # Weight top result heavily
            comparison_df['avg_precision_at_3'] * 0.4 +  # Weight top 3 results
            comparison_df['avg_mrr'] * 0.2               # Weight first relevant result
        )
        
        # Sort by quality score
        comparison_df = comparison_df.sort_values('quality_score', ascending=False)
        
        # Display results
        print(f"\n🏆 MODEL COMPARISON RESULTS")
        print("=" * 100)
        print(f"{'Rank':<4} {'Model':<35} {'Quality':<8} {'P@1':<6} {'P@3':<6} {'MRR':<6} {'Speed(ms)':<10} {'Params':<8}")
        print("-" * 100)
        
        for rank, (_, row) in enumerate(comparison_df.iterrows(), 1):
            print(f"{rank:<4} {row['model_name']:<35} {row['quality_score']:.3f}    {row['avg_precision_at_1']:.3f}  {row['avg_precision_at_3']:.3f}  {row['avg_mrr']:.3f}  {row['avg_query_time_ms']:<8.1f}  {row['parameters_m']:.1f}M")
        
        # Recommendations
        best = comparison_df.iloc[0]
        fastest = comparison_df.loc[comparison_df['avg_query_time_ms'].idxmin()]
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"   🥇 Best Overall: {best['model_name']} (Quality: {best['quality_score']:.3f})")
        print(f"   ⚡ Fastest: {fastest['model_name']} (Speed: {fastest['avg_query_time_ms']:.1f}ms)")
        
        if best['model_name'] != fastest['model_name']:
            print(f"   💡 Trade-off: {best['model_name']} is {fastest['avg_query_time_ms']/best['avg_query_time_ms']:.1f}x slower but {best['quality_score']/fastest['quality_score']:.1f}x better quality")
        
        return comparison_df

# Usage example
if __name__ == "__main__":
    # Initialize evaluation system
    evaluator = RAGEvaluationSystem(
        excel_file_path="data/Rag.xlsx",
        sheet_name="Sentence_embedding",
        pdf_folder_path="C:/Coding/Projects/osu_eecs_chatbot/suyash/pipeline_output/pdfs"
    )
    
    # Models to test
    models_to_test = [
        "sentence-transformers/all-MiniLM-L6-v2",    # Fast, small
        "BAAI/bge-small-en-v1.5",                   # Good balance
        "BAAI/bge-large-en-v1.5",
        "intfloat/e5-small-v2",       
        "intfloat/e5-large-v2",                      # Larger E5 model
    "intfloat/e5-mistral-7b-instruct",              # Recent, good quality
        "sentence-transformers/all-mpnet-base-v2",   # Higher quality
        "BAAI/bge-base-en-v1.5",                     # Larger, better quality
            "WhereIsAI/UAE-Large-V1",
    "ibm-granite/granite-embedding-125m-english",
    "NovaSearch/stella_en_400M_v5",
    "Snowflake/snowflake-arctic-embed-m-v1.5",
"sentence-transformers/allenai-specter",
"allenai/scibert_scivocab_uncased",

    "allenai/specter2_base",
"sentence-transformers/all-MiniLM-L6-v2",
"Lajavaness/bilingual-embedding-small",
"intfloat/e5-small-v2",
"Snowflake/snowflake-arctic-embed-l-v2.0",
    "mixedbread-ai/mxbai-embed-large-v1",        # Top MTEB performer
    "thenlper/gte-large",                        # Strong general embeddings
    
            ]
    
    # Run comparison (use max_questions=5 for quick testing, None for all questions)
    results = evaluator.compare_models(models_to_test, max_questions=None)
    
    if results is not None:
        print(f"\n🎉 Evaluation complete!")
        print(f"📄 Tested {results.iloc[0]['num_questions']} questions against {results.iloc[0]['num_documents']} PDF documents")
        
        # Save results
        results.to_csv('model_comparison_results.csv', index=False)
        print(f"💾 Results saved to 'model_comparison_results.csv'")