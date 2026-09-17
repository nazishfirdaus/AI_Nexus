CHUNK_SIZE = 400 
CHUNK_OVERLAP = 80 
BASE_K_RETRIEVAL = 10 
TOP_K_RERANK = 5 
 
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en" 
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
 
CHROMA_PERSIST_DIRECTORY = "chroma_db" 
COLLECTION_NAME = "anexus_rag_collection" 
 
MODEL_SETTINGS = { 
    "GROQ": "groq", 
    "GEMINI": "gemini_flash", 
    "NVIDIA": "nvidia", 
} 
 
FALLBACK_CHAIN = ["nvidia", "gemini_flash", "groq"] 
MIN_TEXT_LENGTH_FOR_OCR = 100
OCR_DPI = 300