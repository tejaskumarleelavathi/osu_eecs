## [Unreleased]
### To Do
- Experiment with different Reranking (Flash rank change default model) and Ragatouille


### Added
For now fixed on UAE-Large-V1 embedding model based on results from sentence_transformer.py
Performance of the model is improved. From 15s now it takes less than 3s to generate answers.
The Streamlit app runs from top to down on any clicks. Implemented Singleton Pattern to avoid Reinstantiation of loading vector data, setting up the pipeline,etc.
Fixed on particular prompt for now

### Changed
Query Performance Improved

### Fixed
N/A
---

## [1.0.1] - 2025-03-02
### Added
- Creation of Query Generator
- Using WhereIsAI/UAE-Large-V1 for model embedding
- Created sentence_transformer.py for testing different embedding models

---

## [Unreleased]
### To Do
- Improve performance when handling a large number of PDFs  
- Experiment with different prompts to optimize retrieval quality  
- Test alternative embedding models to enhance relevance and accuracy  

### Added
N/A

### Changed
N/A

### Fixed
N/A
---

## [1.0.0] - YYYY-MM-DD
### Added
- Initial implementation of document processing pipeline
- PDF-to-text conversion using `PyPDFDirectoryLoader`
- Document chunking with LangChain `text_splitter`
- Vector storage using `vector_store.create_and_save()`
- Logging and error handling improvements

---