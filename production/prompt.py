from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from config.config import Config
from config.logger import Logger
from .query_generator import QueryGenerator
from .llm_factory import LLMFactory
from utils.query_enhancement import query_enhancement
import traceback as tb
from langsmith import trace, traceable
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.memory import ConversationBufferMemory
from ragatouille import RAGPretrainedModel
from typing import Dict, Any
from .chroma_pipeline import ChromaPipeline
import time

@traceable
class RAGPipeline:
    def __init__(self, config: Config, vector_store):
        self.logger = Logger().get_logger()
        self.config = config
        self.vector_store = vector_store
        self.query_generator = QueryGenerator(config)
        self.qa_chain = None
        self.chroma_pipeline = ChromaPipeline(config)
        self.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            input_key="question"
        ) if config.use_prompt_chaining else None
        self.logger.info("The Rag pipeline is initialized now")

    @traceable(name="setup_pipeline")
    def setup(self, use_multi_query: bool = False) -> RetrievalQA:
        print(f"Self qa chain status {self.qa_chain}")

        try:
            self.logger.info("Setting up RAG pipeline")
            self.llm = LLMFactory.create_llm()
            vector_store = self.vector_store.load_vector_store()

            system_prompt = SystemMessagePromptTemplate.from_template(
                """You are the primary point of contact for all information related to research conducted at Oregon State University. 
                You have extensive knowledge about faculty profiles, their research areas, publications, collaborations, 
                and departmental information. Whether users are prospective students, researchers, or industry collaborators, 
                you provide precise, helpful, and friendly answers. Always prioritize clarity, relevance, and accuracy."""
            )

            user_prompt = HumanMessagePromptTemplate.from_template(
                """{history}

Context:
{context}

User Question:
{question}

Answer:"""
            ) if self.config.use_prompt_chaining else HumanMessagePromptTemplate.from_template(
                "Context:\n{context}\n\nUser Question:\n{question}\n\nAnswer:"
            )

            prompt_template = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

            if self.config.processor_type == 'chroma':
                self.logger.info("Setting up Chroma-based RAG pipeline")

                @traceable(name="chroma_retrieve")
                def chroma_retrieve(question: str, **kwargs):
                    self.logger.info(f"chroma_retrieve received: question={question}, kwargs={kwargs}")
                    self.logger.info("Entered in chroma retrieve to get context to be sent to llm")
                    query_enhanced, search_kwargs = query_enhancement(question)
                    contexts = self.chroma_pipeline.improved_search_faculty_vectorstore(question, query_enhanced)
                    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)
                    reranked_contexts = compressor.compress_documents(contexts, query=question)
                    context_texts = [context.page_content for context in reranked_contexts]
                    context_texts = "\n\n".join(context_texts)
                    return context_texts

                with trace(name="setup_chroma_qa_chain"):
                    self.qa_chain = RunnableMap({
                        "context": chroma_retrieve,
                        "question": RunnablePassthrough(),
                        "history": RunnableLambda(lambda inputs: self.memory.load_memory_variables(inputs)["history"] if self.memory else "")
                    }) | prompt_template | self.llm

                self.logger.info("QA Chain setup for chromadb is done")
                return self.qa_chain

            with trace(name="setup_retriever"):
                self.retriever = vector_store.as_retriever(
                    search_kwargs={'k': self.config.retriever_k, 'fetch_k': 25}
                )

            self.query_generator = QueryGenerator(
                self.config,
                llm=self.llm,
                retriever=self.retriever,
            )

            if use_multi_query:
                self.logger.info("Using multi-query retrieval")

                @traceable(name="multi_query_retrieve")
                def multi_query_retrieve(question):
                    return self.query_generator.retrieve_with_multiple_queries(question)

                with trace(name="setup_qa_chain"):
                    self.qa_chain = RunnableMap({
                        "context": RunnablePassthrough() | multi_query_retrieve,
                        "question": RunnablePassthrough(),
                        "history": RunnableLambda(lambda inputs: self.memory.load_memory_variables(inputs)["history"] if self.memory else "")
                    }) | prompt_template | self.llm

            else:
                compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)
                self.reranker = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.retriever
                )

                self.logger.info("Using standard retrieval along with Reranker")
                with trace(name="setup_qa_chain"):
                    self.qa_chain = RunnableMap({
                        "context": self.reranker,
                        "question": RunnablePassthrough(),
                        "history": RunnableLambda(lambda inputs: self.memory.load_memory_variables(inputs)["history"] if self.memory else "")
                    }) | prompt_template | self.llm

            self.logger.info("Rag pipeline setup is done (QA chain)")
            return self.qa_chain

        except Exception as e:
            self.logger.error(f"Error setting up RAG pipeline: {tb.format_exc()}")
            raise RuntimeError("Failed to set up the RAG pipeline. Verify LLM config, vector store, and retriever settings.") from e

    @traceable(name="execute_query")
    def query(self, question: str) -> str:
        try:
            self.logger.info(f"Processing query: {question}")
            start_time = time.time()
            if self.qa_chain is None:
                self.setup()

            with trace(name="process_query"):
                self.logger.info("Invoking the chain")
                response = self.qa_chain.invoke(question)
                self.logger.info("Query processed successfully")

            if self.memory:
                self.memory.save_context({"question": question}, {"answer": response.content})

            stop_time = time.time()
            self.logger.info(f"Total Time taken:  {stop_time - start_time}")
            return response.content

        except Exception as e:
            self.logger.error(f"Error processing query: {tb.format_exc()}")
            raise RuntimeError("Failed to execute query. Ensure the pipeline is properly set up and input is valid.") from e
