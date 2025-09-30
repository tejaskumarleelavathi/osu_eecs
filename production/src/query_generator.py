import os
from typing import List, Any
from langchain.schema import Document
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import trace, traceable
from config.config import Config
from config.logger import Logger
from langchain.load import dumps, loads
import traceback as tb

class QueryGenerator:
    """Handles generation of multiple search queries for RAG-fusion"""
    def __init__(self, config: Config, llm=None, retriever=None):
        """
        Initialize the QueryGenerator with configuration, LLM, and retriever.

        Args:
            config (Config): Configuration object containing embedding model, number of queries, etc.
            llm: Language model used for generating queries (optional).
            retriever: Retriever used for document search (optional).
        
        Returns:
            None
        """
        self.logger = Logger().get_logger()
        self.config = config
        self.llm = llm
        self.retriever = retriever
        
        # Use embeddings from config if not already initialized
        if not hasattr(self, 'embeddings'):
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"trust_remote_code": True}  # Enable loading of sharded models
            )

    @traceable(name="generate_queries")
    def generate_queries(self, question: str, num_queries: int = 4):
        """
        Generate multiple search queries from a single input query.

        Args:
            question (str): The original question from which to generate multiple queries.
            num_queries (int, optional): The number of queries to generate. Defaults to 4.
        
        Returns:
            list: A list of generated search queries.
        
        Raises:
            Exception: If there is any error during query generation.
        """
        try:
            template = """Generate {num_queries} different search queries related to: {question}

These queries should:

Explore different aspects of the original question within the context of university professors' webpages
Use different wording and phrasing while maintaining relevance
Focus on different potential interpretations, such as biography, research, publications, students, lab affiliation, and news
Include different keywords and entities related to academia
Output exactly {num_queries} queries, one per line:

Example input:
Question: "What are Professor John Doe's latest research publications?"

Example output:

"Recent research papers published by Professor John Doe"
"Latest publications and academic work of John Doe"
"Professor John Doe's research articles and journals"
"New studies and conference papers by John Doe"
"John Doe's contributions to academic research in recent years"
Output exactly {num_queries} queries, one per line:
"""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            @traceable(name="parse_queries")
            def parse_queries(text: str) -> List[str]:
                """
                Parse the generated query text and return the requested number of queries.

                Args:
                    text (str): The generated query text to parse.
                
                Returns:
                    list: A list of parsed queries, limited to the requested number.
                
                """
                queries = [q.strip() for q in text.split('\n') if q.strip()]
                
                # Ensure we have the requested number of queries (or close to it)
                if len(queries) < num_queries:
                    self.logger.warning(f"Generated only {len(queries)} queries instead of {num_queries}")
                
                return queries[:num_queries]  # Limit to the requested number
            
            generate_queries_chain = (
                prompt 
                | self.llm
                | StrOutputParser() 
                | parse_queries
            )
            
            with trace(name="execute_query_generation"):
                queries = generate_queries_chain.invoke({
                    "question": question, 
                    "num_queries": num_queries
                })
                
            self.logger.info(f"Generated {len(queries)} queries from original question")
            self.logger.info(f"Generated queries: {queries}")
            
            return queries
        
        except Exception as e:
            self.logger.error(f"Error generating queries: {tb.format_exc()}")
            raise RuntimeError("Failed to generate search queries. Check prompt structure or LLM configuration.") from e
    
    @traceable(name="execute_multi_query_retrieval")
    def retrieve_with_multiple_queries(self, question: str):
        """
        Execute multiple queries and merge the results using reciprocal rank fusion.

        Args:
            question (str): The original question for which multiple queries will be generated.
        
        Returns:
            list: A list of documents retrieved using reciprocal rank fusion from multiple queries.
        
    
        """
        # Generate multiple queries
        try:
            self.logger.info(f"Generating multiple queries for {question} ")
            num_queries = self.config.num_queries
            queries = self.generate_queries(question, num_queries)
            
            # Get results for each query
            all_results = []
            for query in queries:
                with trace(name=f"retrieve_docs_for_query"):
                    docs = self.retriever.invoke(query)
                    all_results.append(docs)
            
            # Fuse the results
            with trace(name="fuse_retrieval_results"):
                fused_results = reciprocal_rank_fusion(all_results)
            
            # Return just the documents without scores
            return [doc for doc, _ in fused_results]
        
        except Exception as e:
            self.logger.error(f"Error in multi-query retrieval: {tb.format_exc()}")
            raise RuntimeError("Failed during multi-query document retrieval. Check retriever or query generation logic.") from e



@traceable(name="reciprocal_rank_fusion")
def reciprocal_rank_fusion(results: List[List[Document]], k: int = 60) -> List[tuple]:
    """
    Implements the Reciprocal Rank Fusion algorithm to combine multiple result lists.
    
    Args:
        results: List of document lists, where each list contains ranked retrieval results
        k: Constant in the RRF formula (default: 60)
        
    Returns:
        List of (document, score) tuples sorted by descending score
    """

    try:
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_key = hash(doc.page_content + str(doc.metadata.get('source', '')))
                if doc_key not in fused_scores:
                    fused_scores[doc_key] = {"doc": doc, "score": 0}
                fused_scores[doc_key]["score"] += 1 / (rank + k)

        reranked_results = sorted(
            [(item["doc"], item["score"]) for item in fused_scores.values()],
            key=lambda x: x[1],
            reverse=True
        )
        return reranked_results

    except Exception as e:
        logger = Logger().get_logger()
        logger.error(f"Error during reciprocal rank fusion: {tb.format_exc()}")
        raise RuntimeError("Failed to fuse retrieval results. Check document structure or fusion logic.") from e