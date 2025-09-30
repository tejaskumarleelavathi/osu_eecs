import os

# Get the current script's directory
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) # goes two directories up wrt cwd to get /production/
# output SCRIPT DIR  C:\Coding\Projects\osu_eecs_chatbot\production\utils

# Go up one level (..) then into suyash/pipeline_output/pdfs
DATA_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..","..", "suyash", "pipeline_output", "pdfs")
)

VECTOR_STORE_PATH = os.path.join(SCRIPT_DIR,"..","vector_store")

FAISS_PATH = os.path.join(VECTOR_STORE_PATH,"faiss_db")
CHROMA_PATH = os.path.join(VECTOR_STORE_PATH,"chroma_db")


QUESTION_SET_PATH = os.path.join(SCRIPT_DIR,"../data/Rag.xlsx")
EMBEDDING_DEPARTMENT_NAME = "stella_artificial_intelligence"
# "bge-m3_artificial_intelligence"
# "snow_arctic-embed-m-long_artificial_intelligence"
# "snow_arctic-embed-l-v2_artificial_intelligence"
# "stella_artificial_intelligence" #maybe one more  for embedding model used..
    
#  "mpnet_artificial_intelligence" 