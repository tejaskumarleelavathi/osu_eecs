from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from config.config import Config
from config.logger import Logger
from .query_generator import QueryGenerator
from .llm_factory import LLMFactory
from utils.query_enhancement import query_enhancement
import traceback as tb
from langsmith import trace,traceable
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
# from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationTokenBufferMemory
from ragatouille import RAGPretrainedModel
from typing import Dict
from typing import Any
from .chroma_pipeline import ChromaPipeline
from utils.constants import FAISS_PATH
import time
import json
import os


@traceable
class RAGPipeline:
    """Manages the RAG pipeline setup and execution"""
    def __init__(self, config: Config,vector_store):
        """
        Initialize the RAGPipeline with configuration and vector store.

        Args:
            config (Config): Configuration object containing settings for the RAG pipeline.
            vector_store: The vector store (e.g., Chroma or FAISS) used for document retrieval.
        
        Returns:
            None
        """
        self.logger = Logger().get_logger()
        self.config = config
        self.vector_store = vector_store
        self.query_generator = QueryGenerator(config)
        self.qa_chain = None
        self.chroma_pipeline = ChromaPipeline(config)
        self.logger.info("The Rag pipeline is initialzed now")



       
        self.llm = LLMFactory.create_llm()
        self.memory = ConversationTokenBufferMemory(
            llm=self.llm,
            max_token_limit=8192,
            memory_key="history",
            return_messages=True,
            input_key="question"
        ) if self.config.use_prompt_chaining else None

    

    @traceable(name="setup_pipeline")
    def setup(self, use_multi_query: bool = False) -> RetrievalQA:
        """
        Set up the RAG pipeline by defining prompt templates, query generation, and retrieval methods.

        Args:
            use_multi_query (bool, optional): Whether to use multi-query retrieval (default is False).
            faculty_info (Dict[str, Any], optional): A dictionary of faculty-related information (default is None).
            faculty_collections (Dict[str, str], optional): A dictionary of faculty collection names (default is None).
        
        Returns:
            RetrievalQA: The set-up QA chain for querying.
        
        """
       
        try: 
            self.logger.info("Setting up RAG pipeline")
            # self.llm = LLMFactory.create_llm()
            if self.config.processor_type == 'faiss':
            # Check if FAISS index exists
                faiss_index_path = os.path.join(FAISS_PATH, "index.faiss")
                if not os.path.exists(faiss_index_path):
                    self.logger.warning("FAISS vector store not initialized")
                    # Don't raise an error, just return None
                    return None
            vector_store = self.vector_store.load_vector_store()
            
            # Define shared prompt templates to be used for both Chroma and FAISS
            system_prompt = SystemMessagePromptTemplate.from_template(
                """You are the primary point of contact for all information related to research conducted at Oregon State University. 
                You have extensive knowledge about professor profiles, their research areas, publications, collaborations, 
                and departmental information. Whether users are prospective students, researchers, or industry collaborators, 
                you provide precise, helpful, and friendly answers. Always prioritize clarity, relevance, and accuracy. Please ensure the response is structured and easy to understand."""
            )

            # Define user query format
            # user_prompt = HumanMessagePromptTemplate.from_template(
            #     "Context:\n{context}\n\nUser Question:\n{question}\n\nAnswer:"
            # )

            user_prompt = HumanMessagePromptTemplate.from_template(#history removed
                """

                Context:
                {context}

                User Question:
                {question}

                Answer:"""
            ) if self.config.use_prompt_chaining else HumanMessagePromptTemplate.from_template(
                "Context:\n{context}\n\nUser Question:\n{question}\n\nAnswer:"
            )

            # Combine prompts
            prompt_template = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
            
            if self.config.processor_type == 'chroma':
                self.logger.info("Setting up Chroma-based RAG pipeline")
                
                # Define a function to retrieve documents using Chroma
                @traceable(name="chroma_retrieve")
                def chroma_retrieve(question:str,**kwargs):
                    self.logger.info(f"chroma_retrieve received: question={question}, kwargs={kwargs}")
                    # Generate enhanced query if needed
                    self.logger.info("Entered in chroma retrieve to get context to be sent to llm")
                    query_enhanced, tag = query_enhancement(question)
                    # Use the improved search function to get context
                    contexts = self.chroma_pipeline.improved_search_faculty_vectorstore(question, query_enhanced, tag, k_value=10)

                    # # Apply reranking to the retrieved contexts
                    # compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)
                    # # # Convert contexts to a format suitable for reranker
                    # reranked_contexts = compressor.compress_documents(contexts, query=question)

                    context_texts = [context.page_content for context in contexts]
                    context_texts = "\n\n".join(context_texts)
                    # print("\nContext for the question\n")
                    # print(context_texts)
                    # print("\n\n\n\n")
                    return context_texts
                

                # Build QA chain with Chroma retrieval

                #Uncomment this if we don't want prompt chaining code

                with trace(name="setup_chroma_qa_chain"):
                    self.qa_chain = (

                        RunnableMap({"context": chroma_retrieve, "question": RunnablePassthrough()})
                        | prompt_template
                        | self.llm
                    )

                #Comment this if we don't want prompt chaining

                # with trace(name="setup_chroma_qa_chain"):
                #     self.qa_chain = RunnableMap({
                #         "context": chroma_retrieve,
                #         "question": RunnablePassthrough(),
                #         "history": RunnableLambda(lambda inputs: self.memory.load_memory_variables(inputs)["history"] if self.memory else "")
                #         # "history": RunnableLambda(lambda inputs: self._log_and_return_history(inputs))

                #     }) | prompt_template | self.llm 

                self.logger.info("QA Chain setup for chromadb is done")  
                return self.qa_chain
            
            
            # Below setup is for FAISS
            
            # Trace the retriever setup
            with trace(name="setup_retriever"):
                self.retriever = vector_store.as_retriever(
                    # search_type="mmr",
                    search_kwargs={'k': self.config.retriever_k, 'fetch_k': 25}
                )

            # Set up query generator
            self.query_generator = QueryGenerator(
                self.config, 
                llm=self.llm, 
                retriever=self.retriever,
            )

            # Set up appropriate retrieval method
            if use_multi_query:
                self.logger.info("Using multi-query retrieval")
                
                # Define a function to retrieve documents using multi-query generation
                @traceable(name="multi_query_retrieve")
                def multi_query_retrieve(question):
                    """
                    Retrieve documents using multiple queries.

                    Args:
                        question (str): The question to retrieve documents for.
                    
                    Returns:
                        list: A list of documents retrieved for the multi-query.
                    
                    """
                    return self.query_generator.retrieve_with_multiple_queries(question)
                
                # Build QA chain with multi-query retrieval

                #Uncomment this if we don't want prompt chaining code

                # with trace(name="setup_qa_chain"):
                #     self.qa_chain = (
                #         RunnableMap({
                #             "context": RunnablePassthrough() | multi_query_retrieve,
                #             "question": RunnablePassthrough()
                #         })
                #         | prompt_template
                #         | self.llm
                #     )

                #Comment this if we don't want prompt chaining

                # with trace(name="setup_qa_chain"):
                #     self.qa_chain = RunnableMap({
                #         "context": RunnablePassthrough() | multi_query_retrieve,
                #         "question": RunnablePassthrough(),
                #         "history": RunnableLambda(lambda inputs: self.memory.load_memory_variables(inputs)["history"] if self.memory else "")
                #         # "history": RunnableLambda(lambda inputs: self._log_and_return_history(inputs))

                #     }) | prompt_template | RunnableLambda(lambda inputs: self._log_and_pass(inputs)) | self.llm 

            else:
                compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)
                self.reranker = ContextualCompressionRetriever(
                    base_compressor=compressor, 
                    base_retriever=self.retriever
                )

                self.logger.info("Using standard retrieval along with Reranker")
                # Trace the prompt template setup

                #Uncomment this if we don't want prompt chaining code

                with trace(name="setup_qa_chain"):
                    self.qa_chain = (
                        RunnableMap({"context": self.reranker, "question": RunnablePassthrough()})
                        | prompt_template
                        | self.llm
                    )

                #Comment this if we don't want prompt chaining code

                # with trace(name="setup_qa_chain"):
                #     self.qa_chain = RunnableMap({
                #         "context": self.reranker,
                #         "question": RunnablePassthrough(),
                #         "history": RunnableLambda(lambda inputs: self.memory.load_memory_variables(inputs)["history"] if self.memory else "")
                #         # "history": RunnableLambda(lambda inputs: self._log_and_return_history(inputs))

                #     }) | prompt_template |  self.llm 
                    
            self.logger.info("Rag pipeline setup is done (QA chain)")        
            return self.qa_chain
            
        except Exception as e:
            self.logger.error(f"Error setting up RAG pipeline: {tb.format_exc()}")
            raise RuntimeError("Failed to set up the RAG pipeline. Verify LLM config, vector store, and retriever settings.") from e

    @traceable(name="execute_query")
    def query(self, question: str) -> str:
        """
        Execute a query through the RAG pipeline.

        Args:
            question (str): The query string to be processed by the pipeline.
        
        Returns:
            str: The response generated by the RAG pipeline.
        

        """
        try:
            self.logger.info(f"Processing query: {question}")
            start_time = time.time()
            self.logger.info(f"Status of qa chain before setting up {self.qa_chain==None}")
            if self.qa_chain is None:
                self.setup()

            # Trace the actual query execution
            with trace(name="process_query") as query_trace:
                self.logger.info("Invoking the chain")
                response = self.qa_chain.invoke(question)
                self.logger.info("Query processed successfully")

            #for history Comment this if we don't want prompt chaining code
            if self.memory:
                self.memory.save_context({"question": question}, {"answer": response.content})

            stop_time = time.time()

            self.logger.info(f"Total Time taken:  {stop_time - start_time}")
            return response.content

        except Exception as e:
            self.logger.error(f"Error processing query: {tb.format_exc()}")
            raise RuntimeError("Failed to execute query. Ensure the pipeline is properly set up and input is valid.") from e