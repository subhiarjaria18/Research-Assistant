import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # LLM Provider
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "together")  # "together" or "groq"

    # Together AI Models
    TOGETHER_MODELS = {
        "deepseek-ai/DeepSeek-V3": "DeepSeek V3",
        "Meta Llama 3.1 8B Instruct Awq Int4": "LLaMA 3.1 8B Int4",
        "Mistral (7B) Instruct": "Mistral 7B"
    }

    # Groq Models
    GROQ_MODELS = {
        "llama-3.1-8b-instant": "LLaMA 3.1 8B Instant",
        "llama-3.3-70b-versatile": "LLaMA 3.3 70B Versatile",
        "deepseek-r1-distill-llama-70b": "DeepSeek R1 LLaMA 70B",
        "mistral-saba-24b": "Mistral Saba 24B"
    }

    # Embeddings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Vector Store
    VECTOR_STORE_PATH = "./vector_store"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Search
    MAX_SEARCH_RESULTS = 10
    SIMILARITY_THRESHOLD = 0.7
