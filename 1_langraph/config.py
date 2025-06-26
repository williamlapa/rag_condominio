import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../0_base_conhecimento"))

DOCS_DIR = os.path.join(BASE_DIR, "docs_condominio")
CACHE_DIR = os.path.join(BASE_DIR, "processed_docs_cache")

# Configurações de modelos
SUPPORTED_PROVIDERS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash-exp",
    "claude": "claude-3-7-sonnet-latest",
    "groq_llama3": "llama3-8b-8192",
    "groq_gemma": "gemma-7b-it",
    "groq_mistral": "mixtral-8x7b-32768",
    "deepseek": "deepseek-chat"
}

EMBEDDING_MODEL = "text-embedding-3-small"  # Modelo de embedding padrão

# print("BASE_DIR:", BASE_DIR)
# print("DOCS_DIR:", DOCS_DIR)
# print("CACHE_DIR:", CACHE_DIR)

